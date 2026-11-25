"""Position-only exposure-anchor adaptation of existing multimodal models.

These functions never load a test partition or choose settings from test data.
Each anchor uses one current image and ONLY its first causal audio window. The
remaining nine event positions use predicted-anchor motion, so this experiment
does not preserve the baseline's 150 Hz acoustic updates or other-head accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from mvts.engine import load_checkpoint, seed_everything
from .multimodal_data import batch
from .audio_cache import freeze_audio, forward_cached, prepare_audio_tokens
from .prepare import DELAYS_MS, contiguous_segments, sha256
from .run import metrics
from .train_anchor import SMOOTHING, causal_extrapolate


SUPPORTED_MODELS = ('model_slip_gated', 'model_contact_gated')


def _setting_delay(settings):
    delay = float(settings.get('delay_ms', 0.0))
    if delay not in (0.0, 7.8) or settings.get('averaged', False):
        raise ValueError('This controlled adaptation supports instantaneous anchors at 0 or 7.8 ms only')
    return delay


def _argument(args, key, default):
    return getattr(args, key, default)


def _first_forward(model, inputs, delay_ms):
    """Reset GRU for each image without feeding later acoustic windows."""
    acoustic = inputs['audio_tokens'][:, :1] if 'audio_tokens' in inputs else inputs['audio'][:, :1]
    kwargs = {}
    if getattr(model, 'requires_timing', False):
        dt = inputs.get('delta_t', inputs.get('event_dt'))
        if dt is None:
            raise ValueError('Slip-gated anchors require actual event delta_t')
        dt = dt[:, :1]
        age = inputs['image_age'][:, :1] + delay_ms / 1000.0
        kwargs.update(delta_t=dt, image_age=age,
                      has_history=torch.zeros(acoustic.shape[0], dtype=torch.bool, device=acoustic.device))
    if 'audio_tokens' in inputs:
        return forward_cached(model, inputs['image'], acoustic, hidden=None, sequence_length=1, **kwargs)
    return model.forward_sequence(inputs['image'], acoustic, hidden=None, sequence_length=1, **kwargs)


@torch.no_grad()
def predict_anchors(model, data, stats=None, device='cuda', batch_size=16, delay_ms=0.0):
    """Return physical anchors, all raw first-event heads and diagnostics.

    No gating is applied to anchors; the caller must retain coverage and contact
    error metrics separately. Contact hysteresis is evaluated by
    :func:`first_event_metrics`, preserving state across consecutive images.
    """
    stats = stats or data['stats']
    model.eval()
    freeze_audio(model)
    if 'audio_tokens' not in data:
        prepare_audio_tokens(model, data, device=device)
    mean, std = np.asarray(stats['target_mean']), np.asarray(stats['target_std'])
    outputs = {'anchors': [], 'raw_first': []}
    for offset in range(0, len(data['records']), batch_size):
        indices = np.arange(offset, min(offset + batch_size, len(data['records'])))
        inputs = batch(data, indices, device, first_only=True)
        result = _first_forward(model, inputs, delay_ms)
        raw = torch.cat([result[key][:, 0] for key in ('position', 'wrench', 'slip')], dim=-1)
        physical = raw.cpu().numpy() * std + mean
        outputs['anchors'].append(physical[:, :3])
        outputs['raw_first'].append(physical)
        for key in ('contact_probability', 'contact_logits', 'slip_prior', 'slip_prior_mm',
                    'visual_weight', 'audio_weight', 'prior_speed_mm_s', 'delta_t', 'image_age'):
            if key in result:
                outputs.setdefault(key, []).append(result[key][:, 0].cpu().numpy())
    return {key: np.concatenate(values, axis=0) for key, values in outputs.items()}


def first_event_metrics(predicted, data, model_config=None, current_positions=None):
    """Side effects measured only at image arrival, never claimed as 150 Hz.

    Position gate errors use GT-valid labels, not predicted-valid filtering.
    Every slide/polyline sample is contact-positive; background specificity
    cannot be evaluated on these two datasets.
    """
    raw = predicted['raw_first']
    target = np.asarray(data['target'])[:, 0]
    valid = np.asarray(data.get('target_valid', np.ones_like(data['target'], dtype=bool)))[:, 0]
    names = ('x_mm', 'y_mm', 'delta_z_mm', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'slip_x_mm', 'slip_y_mm')
    result = {'scope': 'raw first acoustic event only; other heads were not optimized',
              'raw_position_note': 'raw anchor position is estimated exposure-time position, before current-time propagation',
              'raw_mae': {name: float(np.abs(raw[valid[:, j], j] - target[valid[:, j], j]).mean())
                          if valid[:, j].any() else None for j, name in enumerate(names)}}
    for key in ('visual_weight', 'audio_weight', 'prior_speed_mm_s'):
        if key in predicted:
            result[key + '_mean'] = float(np.asarray(predicted[key]).mean())
    if 'slip_prior_mm' in predicted:
        result['slip_prior_first_mae_mm'] = np.abs(predicted['slip_prior_mm'] - target[:, 9:11]).mean(0).tolist()
    if 'contact_probability' in predicted:
        config = model_config or {}
        probability = predicted['contact_probability']
        off, initial, on = (float(config.get(key, default)) for key, default in (
            ('contact_off_threshold', .4), ('contact_initial_threshold', .5), ('contact_on_threshold', .6)))
        state = np.zeros(len(probability), bool)
        for segment in contiguous_segments(data['records']):
            previous = None
            for i in segment:
                current = probability[i] >= initial if previous is None else previous
                if probability[i] >= on:
                    current = True
                elif probability[i] <= off:
                    current = False
                state[i], previous = current, current
        truth = np.asarray(data.get('contact', np.ones((len(raw), 10), bool)))[:, 0].astype(bool)
        cv = np.asarray(data.get('contact_valid', np.ones((len(raw), 10), bool)))[:, 0].astype(bool)
        tp, fn = int((cv & truth & state).sum()), int((cv & truth & ~state).sum())
        tn, fp = int((cv & ~truth & ~state).sum()), int((cv & ~truth & state).sum())
        posvalid = valid[:, :3].all(-1)
        current = raw[:, :3] if current_positions is None else np.asarray(current_positions)[:, 0, :3]
        gated = np.where(state[:, None], current, 0.)
        result['contact'] = {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
                             'recall': tp / max(tp + fn, 1),
                             'specificity': tn / (tn + fp) if tn + fp else None,
                             'note': 'hysteresis retained across images; no background evaluation if all-positive'}
        result['position_output_coverage'] = float(state[posvalid].mean()) if posvalid.any() else None
        result['gated_first_xy_mean_mm'] = float(np.linalg.norm(gated[posvalid, :2] - target[posvalid, :2], axis=-1).mean()) if posvalid.any() else None
    return result


def _validate_data(data, original, split):
    records = data['records']
    if any(record.get('kind') != 'slide' or record.get('events') != 10 for record in records):
        raise ValueError('Only ten-event slide/polyline windows are allowed')
    if any(not any(f'data_final_v6_{name}_contact' in str(record['cache_root']) for name in ('slide', 'polyline')) for record in records):
        raise ValueError('An unauthorized dataset was supplied to the anchor experiment')
    saved_ids = original.get('split_ids', {}).get(split)
    if saved_ids is not None and [record['id'] for record in records] != saved_ids:
        raise ValueError(f'{split} records differ from the clean baseline partition')
    if 'stats' in data:
        for key in ('audio_mean', 'audio_std'):
            if not np.allclose(data['stats'][key], original['stats'][key], atol=1e-7, rtol=1e-7):
                raise ValueError(f'Audio normalization differs from baseline for {key}; reload data with baseline stats')


def train_variant(model_name, baseline_path, train, val, args, settings, out):
    """Train one independently initialized variant; never access test data.

    ``train``/``val`` come from ``multimodal_data.load_data`` with baseline stats.
    ``settings`` contains delay_ms=0 or 7.8 and optional ``name``. ``out`` is an
    independent image_exposure_* experiment directory. Returns checkpoint path,
    JSON-safe metadata, and the best validation anchor/diagnostic arrays.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f'Unsupported architecture {model_name}')
    delay_ms = _setting_delay(settings)
    device, seed = _argument(args, 'device', 'cuda'), _argument(args, 'seed', 42)
    epochs, batch_size = _argument(args, 'epochs', 30), _argument(args, 'batch_size', 8)
    lr, weight_decay = _argument(args, 'lr', 1e-5), _argument(args, 'weight_decay', 1e-4)
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid training configuration')
    seed_everything(seed)
    baseline_path, out = Path(baseline_path), Path(out)
    if not out.name.startswith('image_exposure_'):
        raise ValueError('Output directory must begin image_exposure_')
    name = settings.get('name', f'image_exposure_{model_name.removeprefix("model_")}_{"midpoint" if delay_ms else "nooffset"}')
    if not name.startswith('image_exposure_') or Path(name).name != name:
        raise ValueError('Variant name must be a single image_exposure_* component')
    path = out / f'{name}.pt'
    if path.exists():
        raise ValueError(f'Never overwrite an existing variant: {path}')
    out.mkdir(parents=True, exist_ok=True)
    original_hash = sha256(baseline_path)
    model, original, _ = load_checkpoint(baseline_path, device)
    if model.model_name != model_name or model.config.image_only:
        raise ValueError('Baseline architecture/multimodal configuration mismatch')
    _validate_data(train, original, 'train')
    _validate_data(val, original, 'val')
    if {r['sequence'] for r in train['records']} & {r['sequence'] for r in val['records']}:
        raise ValueError('Trajectory overlap between train and validation')
    stats = original['stats']
    model.set_normalization(stats)
    freeze_audio(model)
    for data in (train, val):
        if 'audio_tokens' not in data:
            prepare_audio_tokens(model, data, device=device)
    frozen_heads = []
    for key in ('force_head', 'slip_head', 'contact_head', 'slip_prior_head'):
        if hasattr(model, key):
            frozen_heads.append(key)
            for parameter in getattr(model, key).parameters():
                parameter.requires_grad_(False)
    index = DELAYS_MS.index(delay_ms)
    valid = np.flatnonzero(np.asarray(train['anchor_valid'])[:, index])
    if not len(valid):
        raise ValueError('No interpolation-valid training anchors')
    labels = torch.as_tensor(np.nan_to_num(train['anchors'][:, index]).astype(np.float32), device=device)
    mean = torch.tensor(stats['target_mean'][:3], device=device)
    std = torch.tensor(stats['target_std'][:3], device=device)
    labels = (labels - mean) / std
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * .1)
    rng = np.random.default_rng(seed)
    history, best_state, best_score, best_epoch, best_smoothing = [], None, float('inf'), 0, None
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        freeze_audio(model)
        order = rng.permutation(valid)
        total_loss, count = 0.0, 0
        for offset in range(0, len(order), batch_size):
            ids = order[offset:offset + batch_size]
            inputs = batch(train, ids, device, first_only=True)
            predicted = _first_forward(model, inputs, delay_ms)['position'][:, 0]
            loss = F.smooth_l1_loss(predicted, labels[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite multimodal anchor loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ids)
            count += len(ids)
        scheduler.step()
        prediction = predict_anchors(model, val, stats, device, batch_size=16, delay_ms=delay_ms)
        candidates = []
        for smoothing in SMOOTHING:
            p = causal_extrapolate(prediction['anchors'], val['records'], val['image_ns'], val['host_ages'], delay_ms, smoothing)
            candidates.append({'smoothing': smoothing, 'val_xy_mean_mm': metrics(p, val['target'])['xy_mean_mm'],
                               'val_image_arrival_xy_mean_mm': float(np.linalg.norm(p[:, 0, :2] - val['target'][:, 0, :2], axis=-1).mean())})
        selected = min(candidates, key=lambda c: c['val_xy_mean_mm'])
        score = selected['val_xy_mean_mm']
        history.append({'epoch': epoch, 'training_anchor_loss': total_loss / count,
                        'val_xy_mean_mm': score, 'selected_smoothing': selected['smoothing'],
                        'smoothing_candidates': candidates, 'elapsed_seconds': time.perf_counter() - start})
        if score < best_score:
            best_score, best_epoch, best_smoothing = score, epoch, selected['smoothing']
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.perf_counter() - start
            print(f'{name} {epoch}/{epochs} loss={total_loss / count:.5f} valXY={score:.5f} '
                  f'best={best_score:.5f} smoothing={selected["smoothing"]} elapsed={elapsed:.1f}s '
                  f'eta={elapsed / epoch * (epochs - epoch):.1f}s', flush=True)
    model.load_state_dict(best_state)
    prediction = predict_anchors(model, val, stats, device, batch_size=16, delay_ms=delay_ms)
    p = causal_extrapolate(prediction['anchors'], val['records'], val['image_ns'], val['host_ages'], delay_ms, best_smoothing)
    anchor_config = {**settings, 'delay_ms': delay_ms, 'smoothing': best_smoothing, 'max_speed_mm_s': 200.0,
                     'location_inference': 'one current image plus first causal audio window; recurrent state reset per image',
                     'event_inference': 'causal finite difference of predicted anchors; later nine audio windows ignored',
                     'slip_gate_timing': 'actual first event dt; host image age plus assumed midpoint delay',
                     'audio_training': 'audio encoder and projection frozen; cached real projected acoustic tokens',
                     'trainable': 'vision, multimodal fusion, recurrent state and position head',
                     'other_heads': 'frozen heads on changed shared features; no preservation or 150Hz accuracy claim'}
    safe_args = json.loads(json.dumps(vars(args) if hasattr(args, '__dict__') else {}, default=str))
    checkpoint = {'format_version': 1, 'model_name': model_name, 'model_config': original['model_config'],
                  'feature_config': original['feature_config'], 'model': best_state, 'stats': stats,
                  'split_ids': original['split_ids'], 'epoch': best_epoch, 'anchor_config': anchor_config,
                  'experiment_type': 'multimodal_anchor_position_only', 'baseline_path': str(baseline_path),
                  'baseline_sha256': original_hash, 'training': safe_args, 'variant': name,
                  'validation_xy_mm': best_score, 'frozen_heads': frozen_heads}
    checkpoint['frozen_audio_modules'] = ['audio_backbone', 'audio_projection']
    torch.save(checkpoint, path)
    np.savez(out / f'{name}_val_anchors.npz', **prediction, ids=np.array([r['id'] for r in val['records']]))
    (out / f'{name}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    metadata = {'name': name, 'checkpoint_path': str(path), 'settings': anchor_config,
                'best_epoch': best_epoch, 'validation_xy_mm': best_score,
                'validation_current': metrics(p, val['target'], val['records']),
                'validation_image_arrival_xy_mm': float(np.linalg.norm(p[:, 0, :2] - val['target'][:, 0, :2], axis=-1).mean()),
                'validation_first_event_side_effects': first_event_metrics(prediction, val, original['model_config'], current_positions=p),
                'valid_training_anchors': len(valid), 'seconds': time.perf_counter() - start,
                'baseline_integrity': sha256(baseline_path) == original_hash}
    assert metadata['baseline_integrity']
    (out / f'{name}_result.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    del model, optimizer
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return {'metadata': metadata, 'checkpoint_path': str(path), 'val_anchors': prediction['anchors'],
            'val_diagnostics': prediction}


train_anchor_variant = train_variant
