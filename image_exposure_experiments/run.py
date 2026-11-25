"""Controlled position experiments on frozen, two-cache visual features.

python -m image_exposure_experiments.run --output runs/image_exposure_v1
Test metrics are computed only after validation fixes every selected model.
"""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from mvts.engine import seed_everything
from .prepare import contiguous_segments, sha256, DELAYS_MS, BASELINE
from .model import ExposurePositionModel, from_checkpoint
from .compensation import compensate


def metrics(pred, target, records=None):
    error = np.asarray(pred, np.float64) - target[..., :3]
    distance = np.linalg.norm(error[..., :2], axis=-1)
    result = {'events': int(distance.size), 'xy_mean_mm': float(distance.mean()),
              'xy_rmse_mm': float(np.sqrt((distance ** 2).mean())),
              'xy_median_mm': float(np.median(distance)), 'xy_p95_mm': float(np.percentile(distance, 95)),
              'mae_mm': dict(zip(('x', 'y', 'z'), np.abs(error).mean(axis=(0, 1)).tolist()))}
    if records is not None:
        result['sources'] = {}
        for name in ('slide', 'polyline'):
            idx = [i for i, r in enumerate(records) if f'v6_{name}_contact' in r['cache_root']]
            if idx:
                result['sources'][name] = metrics(pred[idx], target[idx])
        result['by_subevent_xy_mean_mm'] = distance.mean(0).tolist()
    return result


def load_pack(root, split):
    with np.load(root / f'{split}_features.npz') as z:
        data = {k: z[k] for k in z.files}
    records = json.loads((root / f'{split}_records.json').read_text(encoding='utf-8'))
    data['records'], data['segments'] = records, contiguous_segments(records)
    return data


def batch(data, segments, anchor_index, averaged, device):
    length = max(map(len, segments))
    out = {}
    for key, src in [('features', 'features'), ('dt', 'frame_dt'), ('ages', 'host_ages')]:
        values = data[src]
        array = np.zeros((len(segments), length, *values.shape[1:]), np.float32)
        if key == 'dt':
            array.fill(1 / 15.)
        for i, seg in enumerate(segments):
            array[i, :len(seg)] = values[seg]
        out[key] = torch.tensor(array, device=device)
    mask = np.zeros((len(segments), length), bool)
    target = np.zeros((len(segments), length, 10, 3), np.float32)
    base = np.zeros((len(segments), length, 3), np.float32)
    anchor = np.zeros_like(base)
    valid = np.zeros_like(mask)
    label_key, valid_key = ('averaged_anchors', 'averaged_valid') if averaged else ('anchors', 'anchor_valid')
    for i, seg in enumerate(segments):
        n = len(seg)
        mask[i, :n] = True
        target[i, :n] = data['target'][seg, :, :3]
        base[i, :n] = data['baseline'][seg, 0, :3]
        anchor[i, :n] = np.nan_to_num(data[label_key][seg, anchor_index])
        valid[i, :n] = data[valid_key][seg, anchor_index]
    out.update({k: torch.tensor(v, device=device) for k, v in
                [('mask', mask), ('target', target), ('base', base), ('anchor_target', anchor), ('anchor_valid', valid)]})
    return out


def forward(model, b):
    return model(b['features'], b['dt'], b['ages'], b['base'])


@torch.no_grad()
def predict(model, data, device='cuda'):
    model.eval()
    pred = np.zeros_like(data['target'][..., :3])
    for offset in range(0, len(data['segments']), 16):
        segments = data['segments'][offset:offset + 16]
        b = batch(data, segments, 0, False, device)
        values = forward(model, b)['position'].cpu().numpy()
        for i, seg in enumerate(segments):
            pred[seg] = values[i, :len(seg)]
    return pred


def train_variant(name, settings, train, val, stats, args):
    seed_everything(args.seed)
    model = ExposurePositionModel(target_mean=tuple(stats['target_mean'][:3]),
        target_std=tuple(stats['target_std'][:3]), exposure_ms=settings['exposure_ms'],
        readout_delay_ms=settings.get('readout_delay_ms', 0)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    best, history, best_state = float('inf'), [], None
    generator = np.random.default_rng(args.seed)
    ai = DELAYS_MS.index(settings['delay_ms'])
    started = time.perf_counter()
    std = model.target_std
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(len(train['segments']))
        losses = []
        for offset in range(0, len(order), args.batch_size):
            segments = [train['segments'][i] for i in order[offset:offset + args.batch_size]]
            b = batch(train, segments, ai, settings.get('averaged', False), 'cuda')
            result = forward(model, b)
            current = F.smooth_l1_loss(result['position'][b['mask']] / std,
                                     b['target'][b['mask']] / std)
            am = b['mask'] & b['anchor_valid']
            aux = F.smooth_l1_loss(result['anchor'][am] / std, b['anchor_target'][am] / std)
            loss = current + settings['anchor_weight'] * aux + 1e-5 * (result['velocity'][b['mask']] / 150).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        values = predict(model, val)
        score = metrics(values, val['target'])['xy_mean_mm']
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'val_xy_mean_mm': score})
        if score < best:
            best, best_state, best_epoch = score, deepcopy(model.state_dict()), epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(name, epoch, 'val_xy_mm', round(score, 5), 'best', round(best, 5), flush=True)
    model.load_state_dict(best_state)
    metadata = {'variant': name, 'settings': settings, 'epoch': best_epoch, 'validation_xy_mm': best,
                'frozen_visual_checkpoint': str(BASELINE), 'frozen_visual_sha256': sha256(BASELINE),
                'training': vars(args), 'stats': stats,
                'split_ids': {k: [r['id'] for r in d['records']] for k, d in [('train', train), ('val', val)]}}
    torch.save(model.export_checkpoint(**metadata), Path(args.output) / f'{name}.pt')
    (Path(args.output) / f'{name}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return model, {'settings': settings, 'best_epoch': best_epoch, 'validation_xy_mm': best,
                   'seconds': time.perf_counter() - started}


def paired_bootstrap(baseline, candidate, truth, records, seed=42):
    sequences = sorted({r['sequence'] for r in records})
    differences = []
    # Trajectory-cluster bootstrap, not falsely treating adjacent events independent.
    a = np.linalg.norm(baseline[..., :2] - truth[..., :2], axis=-1)
    b = np.linalg.norm(candidate[..., :2] - truth[..., :2], axis=-1)
    sums, counts = [], []
    for seq in sequences:
        idx = [i for i, r in enumerate(records) if r['sequence'] == seq]
        sums.append(float((a[idx] - b[idx]).sum()))
        counts.append(a[idx].size)
    sums, counts = np.array(sums), np.array(counts)
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        draw = rng.integers(len(sequences), size=len(sequences))
        differences.append(sums[draw].sum() / counts[draw].sum())
    return {'trajectories': len(sequences), 'resamples': 2000,
            'baseline_minus_candidate_xy_mm_95ci': np.percentile(differences, [2.5, 97.5]).tolist()}


def save_predictions(path, predictions, data):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'sequence', 'subevent', 'event_pc_ns', 'true_x', 'true_y', 'true_z',
                         'baseline_x', 'baseline_y', 'baseline_z', 'pred_x', 'pred_y', 'pred_z'])
        for i, r in enumerate(data['records']):
            for j in range(10):
                writer.writerow([r['id'], r['sequence'], j, int(data['event_ns'][i, j]),
                                 *data['target'][i, j, :3], *data['baseline'][i, j, :3], *predictions[i, j]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/image_exposure_v1')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8, help='independent trajectory segments per batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='new small head only, backbone frozen')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = Path(args.output)
    if not root.name.startswith('image_exposure_'):
        parser.error('Output directory must start with image_exposure_')
    if (root / 'selection.json').exists():
        raise ValueError('Already selected/tested; use a separate experiment output')
    torch.set_num_threads(4)
    prep = json.loads((root / 'prepared.json').read_text())
    train, val = [load_pack(root, s) for s in ('train', 'val')]
    val_baseline = metrics(val['baseline'][..., :3], val['target'], val['records'])
    # Prespecified simple physics-only controls; all tested against unchanged labels.
    compensations = []
    for delay in (0., 7.8, 15.6, 23.4, 31.2):
        for smoothing in (.25, .5, 1.):
            p = compensate(val['baseline'][..., :3], val['records'], val['event_ns'], delay, smoothing)
            compensations.append({'delay_ms': delay, 'smoothing': smoothing,
                                  'val_xy_mean_mm': metrics(p, val['target'])['xy_mean_mm']})
    best_comp = min(compensations, key=lambda x: x['val_xy_mean_mm'])
    variants = {
        'image_exposure_time_control': dict(exposure_ms=0., delay_ms=0., anchor_weight=0.),
        'image_exposure_anchor_control': dict(exposure_ms=0., delay_ms=0., anchor_weight=.3),
        'image_exposure_midpoint': dict(exposure_ms=15.6, delay_ms=7.8, anchor_weight=.3),
        'image_exposure_average': dict(exposure_ms=15.6, delay_ms=7.8, anchor_weight=.3, averaged=True),
        'image_exposure_full_delay': dict(exposure_ms=15.6, readout_delay_ms=7.8, delay_ms=15.6, anchor_weight=.3),
    }
    models, results = {}, {}
    for name, settings in variants.items():
        models[name], results[name] = train_variant(name, settings, train, val, prep['stats'], args)
    winner = min(results, key=lambda k: results[k]['validation_xy_mm'])
    # Persist the complete decision before the test pack is ever loaded here.
    selection = {'baseline_val': val_baseline, 'physics_compensation_candidates': compensations,
                 'selected_compensation': best_comp, 'variants': results, 'selected_model': winner,
                 'test_targets': 'unchanged current-time labels; no oracle robot velocity at inference'}
    (root / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('Validation selected', winner, flush=True)
    test = load_pack(root, 'test')
    summary = {'selection': selection, 'baseline': metrics(test['baseline'][..., :3], test['target'], test['records']),
               'models': {}, 'primary_selected_model': winner}
    for name, model in models.items():
        p = predict(model, test)
        summary['models'][name] = metrics(p, test['target'], test['records'])
        if name == winner:
            summary['paired_trajectory_bootstrap'] = paired_bootstrap(test['baseline'], p, test['target'], test['records'])
            save_predictions(root / 'image_exposure_test_predictions.csv', p, test)
            # Final artifact bundles the unmodified frozen visual weights too.
            c = torch.load(BASELINE, map_location='cpu', weights_only=True)
            final = model.export_checkpoint(variant=name, settings=results[name]['settings'],
                epoch=results[name]['best_epoch'], frozen_visual_model=c['model'],
                frozen_visual_config=c['model_config'], stats=c['stats'],
                feature_config=c['feature_config'], split_ids=c['split_ids'],
                source_sha256=json.loads((root/'protocol.json').read_text())['source_sha256'],
                training=vars(args), validation_xy_mm=results[name]['validation_xy_mm'])
            torch.save(final, root / 'image_exposure_best.pt')
    p = compensate(test['baseline'][..., :3], test['records'], test['event_ns'],
                   best_comp['delay_ms'], best_comp['smoothing'])
    summary['selected_physics_compensation'] = metrics(p, test['target'], test['records'])
    summary['exposure_7p8ms_compensation'] = metrics(compensate(test['baseline'][..., :3], test['records'],
        test['event_ns'], 7.8, .5), test['target'], test['records'])
    hashes = json.loads((root/'protocol.json').read_text())['source_sha256']
    summary['original_integrity'] = {p: sha256(p) == h for p,h in hashes.items()}
    assert all(summary['original_integrity'].values())
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'baseline': summary['baseline'], 'selected': winner,
                     'selected_test': summary['models'][winner], 'ci': summary['paired_trajectory_bootstrap']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
