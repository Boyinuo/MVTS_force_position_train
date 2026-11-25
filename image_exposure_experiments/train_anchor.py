"""Independent-image exposure-anchor fine tuning and causal motion compensation.

Run ``python -m image_exposure_experiments.train_anchor`` from the repository.
Only the saved two-cache partitions are used. Selection precedes test loading;
all final accuracy comparisons use the original current-time labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from mvts.engine import load_checkpoint, seed_everything
from .prepare import BASELINE, DELAYS_MS, contiguous_segments, sha256
from .run import load_pack, metrics, paired_bootstrap, save_predictions


SMOOTHING = (0.0, 0.25, 0.5, 1.0)
VARIANTS = {
    'image_exposure_nooffset_anchor': {'delay_ms': 0.0, 'averaged': False},
    'image_exposure_midpoint_anchor': {'delay_ms': 7.8, 'averaged': False},
    'image_exposure_average_anchor': {'delay_ms': 7.8, 'averaged': True},
    'image_exposure_estimated_delay_anchor': {'delay_ms': 23.4, 'averaged': False,
                                              'interpretation': 'estimated total delay, including unknown readout; not exposure duration'},
}


def causal_extrapolate(anchors, records, image_ns, host_ages, delay_ms, smoothing, max_speed=200.0):
    """Only current/past predicted image anchors determine event positions.

    A missing predecessor or order gap resets velocity. A zero smoothing value
    is the prespecified no-motion control. Speed is capped by vector norm.
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    image_ns = np.asarray(image_ns, dtype=np.int64)
    ages = np.asarray(host_ages, dtype=np.float64)
    if anchors.shape != (len(records), 3) or image_ns.shape != (len(records),) or ages.shape[0] != len(records):
        raise ValueError('Prediction/timing shapes do not match records')
    if not 0 <= smoothing <= 1 or delay_ms < 0 or max_speed <= 0:
        raise ValueError('Invalid causal motion settings')
    if not np.isfinite(anchors).all() or not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError('Predicted anchors/ages must be finite and causal')
    velocity = np.zeros_like(anchors)
    for segment in contiguous_segments(records):
        previous_velocity = np.zeros(3)
        for previous, current in zip(segment[:-1], segment[1:]):
            dt = (int(image_ns[current]) - int(image_ns[previous])) * 1e-9
            if dt <= 0:
                raise ValueError('Nonmonotonic image timestamps')
            observed = (anchors[current] - anchors[previous]) / dt
            speed = float(np.linalg.norm(observed))
            if speed > max_speed:
                observed *= max_speed / speed
            previous_velocity = smoothing * observed + (1 - smoothing) * previous_velocity
            velocity[current] = previous_velocity
    prediction = anchors[:, None] + velocity[:, None] * (ages + delay_ms * .001)[..., None]
    return prediction


def read_images(records):
    def read(record):
        with Image.open(Path(record['cache_root']) / record['image']) as image:
            rgb = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)
            return np.array(rgb.transpose(2, 0, 1), copy=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return torch.from_numpy(np.stack(list(pool.map(read, records))))


def images_to_device(images, indices, device):
    rgb = images[indices].to(device=device, dtype=torch.float32) / 255.0
    mean = rgb.new_tensor((.485, .456, .406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((.229, .224, .225)).view(1, 3, 1, 1)
    return (rgb - mean) / std


@torch.no_grad()
def predict_anchors(model, images, stats, device, batch_size=48):
    model.eval()
    predictions = []
    mean = np.asarray(stats['target_mean'][:3])
    std = np.asarray(stats['target_std'][:3])
    for start in range(0, len(images), batch_size):
        batch = images_to_device(images, slice(start, start + batch_size), device)
        normalized = model.forward_sequence(batch, sequence_length=1)['position'][:, 0]
        predictions.append(normalized.cpu().numpy() * std + mean)
    return np.concatenate(predictions)


def select_smoothing(anchors, data, delay_ms):
    candidates = []
    for smoothing in SMOOTHING:
        p = causal_extrapolate(anchors, data['records'], data['image_ns'], data['host_ages'], delay_ms, smoothing)
        candidates.append({'smoothing': smoothing, 'val_xy_mean_mm': metrics(p, data['target'])['xy_mean_mm']})
    return min(candidates, key=lambda c: c['val_xy_mean_mm']), candidates


def train_variant(name, settings, train, val, train_images, val_images, args, source_checkpoint):
    seed_everything(args.seed)
    model, original, _ = load_checkpoint(BASELINE, args.device)
    assert model.config.image_only
    for head in (model.force_head, model.slip_head):
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    stats = original['stats']
    mean = torch.tensor(stats['target_mean'][:3], device=args.device)
    std = torch.tensor(stats['target_std'][:3], device=args.device)
    anchor_index = DELAYS_MS.index(settings['delay_ms'])
    label_key, valid_key = ('averaged_anchors', 'averaged_valid') if settings['averaged'] else ('anchors', 'anchor_valid')
    valid = np.flatnonzero(train[valid_key][:, anchor_index])
    if not len(valid):
        raise ValueError(f'No valid training anchors for {name}')
    labels = torch.tensor(np.nan_to_num(train[label_key][:, anchor_index]).astype(np.float32), device=args.device)
    normalized_labels = (labels - mean) / std
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    rng = np.random.default_rng(args.seed)
    best, history, best_state, best_epoch, best_smoothing = float('inf'), [], None, 0, None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(valid)
        total_loss, total_items = 0.0, 0
        for offset in range(0, len(order), args.batch_size):
            idx = order[offset:offset + args.batch_size]
            images = images_to_device(train_images, idx, args.device)
            result = model.forward_sequence(images, sequence_length=1)['position'][:, 0]
            loss = F.smooth_l1_loss(result, normalized_labels[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite anchor loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
            total_items += len(idx)
        scheduler.step()
        anchors = predict_anchors(model, val_images, stats, args.device)
        selected, smoothing_candidates = select_smoothing(anchors, val, settings['delay_ms'])
        score = selected['val_xy_mean_mm']
        entry = {'epoch': epoch, 'training_anchor_loss': total_loss / total_items,
                 'val_xy_mean_mm': score, 'selected_smoothing': selected['smoothing'],
                 'smoothing_candidates': smoothing_candidates, 'elapsed_seconds': time.perf_counter() - started}
        history.append(entry)
        if score < best:
            best, best_epoch, best_smoothing = score, epoch, selected['smoothing']
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            elapsed = time.perf_counter() - started
            remaining = elapsed / epoch * (args.epochs - epoch)
            print(f'{name} epoch {epoch}/{args.epochs} valXY={score:.5f} best={best:.5f} '
                  f'smoothing={selected["smoothing"]} elapsed={elapsed:.1f}s eta={remaining:.1f}s', flush=True)
    model.load_state_dict(best_state)
    best_val_anchors = predict_anchors(model, val_images, stats, args.device)
    config = {**settings, 'smoothing': best_smoothing, 'max_speed_mm_s': 200.0,
              'location_inference': 'independent current image, GRU reset every frame',
              'event_inference': 'causal finite difference of predicted anchors; zero velocity at first/gap'}
    # Original loader-compatible weights; anchor_config describes the additional
    # causal time propagation needed for CURRENT-time event predictions.
    checkpoint = {'format_version': 1, 'model_name': original['model_name'],
                  'model_config': original['model_config'], 'feature_config': original['feature_config'],
                  'model': best_state, 'stats': stats, 'epoch': best_epoch,
                  'split_ids': original['split_ids'], 'anchor_config': config,
                  'experiment_type': 'image_exposure_independent_anchor_v2',
                  'baseline_sha256': source_checkpoint, 'validation_xy_mm': best,
                  'training': vars(args), 'variant': name}
    out = Path(args.output)
    torch.save(checkpoint, out / f'{name}.pt')
    np.savez(out / f'{name}_val_anchors.npz', anchors=best_val_anchors,
             ids=np.array([r['id'] for r in val['records']]))
    (out / f'{name}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    result = {'settings': config, 'best_epoch': best_epoch, 'validation_xy_mm': best,
              'valid_training_anchors': len(valid), 'seconds': time.perf_counter() - started}
    del model, optimizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/image_exposure_v1')
    parser.add_argument('--output', default='runs/image_exposure_anchor_v2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    args = parser.parse_args()
    out = Path(args.output)
    if not out.name.startswith('image_exposure_'):
        parser.error('Output directory must start with image_exposure_')
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error('epochs/lr must be positive and batch size at least 2')
    if out.exists():
        # An empty user-created destination is okay; existing artifacts are never
        # replaced or reused as an implicit continuation.
        if any(out.iterdir()):
            raise ValueError('Experiment output is not empty; choose a fresh directory')
    else:
        out.mkdir(parents=True)
    source = Path(args.source)
    torch.set_num_threads(4)
    source_protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    source_hashes = source_protocol['source_sha256']
    assert all(sha256(path) == expected for path, expected in source_hashes.items())
    baseline_hash = sha256(BASELINE)
    train, val = [load_pack(source, split) for split in ('train', 'val')]
    assert set(r['sequence'] for r in train['records']).isdisjoint(r['sequence'] for r in val['records'])
    print('Preloading resized images for train/validation', flush=True)
    train_images, val_images = read_images(train['records']), read_images(val['records'])
    baseline_val = metrics(val['baseline'][..., :3], val['target'], val['records'])
    protocol = {'baseline': str(BASELINE), 'baseline_sha256': baseline_hash, 'source': str(source),
                'source_sha256': source_hashes, 'training': vars(args), 'variants': VARIANTS,
                'selection': 'minimum validation current-time XY mean across epoch/smoothing; original baseline remains fallback',
                'test': 'loaded only after all variant checkpoints and selection have been committed',
                'inference': 'current/past images and host timestamps only; no measured position/velocity/audio/force',
                'fine_tuning': 'all vision/fusion/GRU/position parameters, original pretrained two-cache model; no wrench/slip losses',
                'known_exposure_ms': 15.6, 'unknown_readout_delay_ms': 'not measured; midpoint/average assume zero, estimated23.4ms variant assumes an additional15.6ms',
                'raw_data': 'used only to build train anchor supervision; no original cache/raw file writes'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    results = {}
    for name in args.variants:
        results[name] = train_variant(name, VARIANTS[name], train, val, train_images, val_images, args, baseline_hash)
    candidates = {'original_baseline': baseline_val['xy_mean_mm'],
                  **{name: result['validation_xy_mm'] for name, result in results.items()}}
    selected = min(candidates, key=candidates.get)
    selection = {'baseline_validation': baseline_val, 'variants': results, 'selection_scores': candidates,
                 'selected_model': selected, 'test_labels': 'original, unchanged current event times'}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('All choices fixed; validation selected', selected, flush=True)
    # No test data is loaded above this line.
    test = load_pack(source, 'test')
    assert set(r['sequence'] for r in test['records']).isdisjoint(r['sequence'] for r in train['records'])
    test_images = read_images(test['records'])
    summary = {'selection': selection, 'baseline': metrics(test['baseline'][..., :3], test['target'], test['records']),
               'models': {}, 'primary_selected_model': selected}
    winner_predictions = test['baseline'][..., :3]
    for name in args.variants:
        model, checkpoint, _ = load_checkpoint(out / f'{name}.pt', args.device)
        anchors = predict_anchors(model, test_images, checkpoint['stats'], args.device)
        np.savez(out / f'{name}_test_anchors.npz', anchors=anchors,
                 ids=np.array([r['id'] for r in test['records']]))
        config = checkpoint['anchor_config']
        p = causal_extrapolate(anchors, test['records'], test['image_ns'], test['host_ages'],
                              config['delay_ms'], config['smoothing'], config['max_speed_mm_s'])
        summary['models'][name] = metrics(p, test['target'], test['records'])
        if name == selected:
            winner_predictions = p
            torch.save(checkpoint, out / 'image_exposure_best.pt')
        del model
    if selected == 'original_baseline':
        # Keep an explicit reviewable fallback artifact rather than calling a
        # validation-worse refinement the replacement.
        checkpoint = torch.load(BASELINE, map_location='cpu', weights_only=True)
        checkpoint['image_exposure_selection'] = {'selected_model': selected, 'reason': 'baseline had best validation XY mean'}
        torch.save(checkpoint, out / 'image_exposure_best.pt')
    save_predictions(out / 'image_exposure_test_predictions.csv', winner_predictions, test)
    summary['paired_trajectory_bootstrap'] = paired_bootstrap(test['baseline'], winner_predictions, test['target'], test['records'])
    summary['original_integrity'] = {path: sha256(path) == expected for path, expected in source_hashes.items()}
    assert all(summary['original_integrity'].values())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'selected_model': selected, 'baseline_test': summary['baseline'],
                      'selected_test': summary['baseline'] if selected == 'original_baseline' else summary['models'][selected],
                      'ci': summary['paired_trajectory_bootstrap']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
