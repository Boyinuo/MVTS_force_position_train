"""Read-only audiovisual inputs on the fixed two-cache exposure partitions.

``load_data(source, split, stats=None)`` keeps all arrays on CPU and reuses the
reference pack's original times/labels/anchor supervision. It reads cached
Log-Mel (never raw ADC), normalizes with the supplied checkpoint statistics,
and checks that train/val/test IDs are disjoint and all records come from the
slide/polyline contact caches. No press or synthetic contact-negative samples
are permitted. If stats is omitted, use the common two-cache audiovisual
``model_triple_head`` checkpoint with microphone #5 disabled.

``batch(data, indices, device, first_only=False)`` transfers only the selected
items to the requested device. ``target`` is normalized [B,T,11], ``physical``
is the original measured label, and ``event_dt``/``delta_t``/``dt`` are actual
event-minus-predecessor durations in seconds. These differ from camera frame
intervals. ``image_age`` is elapsed time since host read completion, excluding
any assumed exposure/readout delay. ``first_only`` selects event 0 while
retaining every candidate image anchor [B,5,3].
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import numpy as np
import torch

from mvts.contact_data import SUPERVISION_KEYS, supervision_digest
from .run import load_pack
from .train_anchor import images_to_device, read_images


MULTIMODAL_BASELINE = Path('runs/v6_triple_head_slide_polyline_no_mic5_trim0p029_xyfilter/best.pt')
ALLOWED_CACHE_NAMES = {'data_final_v6_slide_contact', 'data_final_v6_polyline_contact'}
EXPECTED_SPLITS = {'train': 1891, 'val': 443, 'test': 368}


def _validate_partitions(source):
    """Check identity/scope without loading features from another partition."""
    source = Path(source)
    seen, all_records = {}, {}
    for split in ('train', 'val', 'test'):
        records = json.loads((source / f'{split}_records.json').read_text(encoding='utf-8'))
        if not records:
            raise ValueError(f'Empty {split} partition')
        for record in records:
            identifier = record['id']
            if identifier in seen:
                raise ValueError(f'Duplicate/overlapping ID {identifier!r} in {seen[identifier]} and {split}')
            seen[identifier] = split
            if Path(record['cache_root']).name not in ALLOWED_CACHE_NAMES:
                raise ValueError(f'Only original slide/polyline caches are allowed: {identifier}')
            if record.get('kind') != 'slide' or record.get('events') != 10:
                raise ValueError(f'Only ten-event slide/polyline windows are allowed: {identifier}')
        all_records[split] = records
    # Complete trajectories must not cross a partition, regardless of ID spelling.
    sequences = {}
    for split, records in all_records.items():
        for record in records:
            sequence = record['sequence']
            if sequence in sequences and sequences[sequence] != split:
                raise ValueError(f'Trajectory crosses partitions: {sequence}')
            sequences[sequence] = split
    return all_records


def _statistics(stats):
    if stats is None:
        if not MULTIMODAL_BASELINE.is_file():
            raise FileNotFoundError(MULTIMODAL_BASELINE)
        checkpoint = torch.load(MULTIMODAL_BASELINE, map_location='cpu', weights_only=True)
        roots = checkpoint['args'].get('prepared_root', [])
        if {Path(root).name for root in roots} != ALLOWED_CACHE_NAMES:
            raise ValueError('Default normalization checkpoint must use exactly two original caches')
        config = checkpoint['model_config']
        if config.get('image_only') or list(config.get('disabled_mic_channels', [])) != [5]:
            raise ValueError('Default checkpoint must be audiovisual with microphone #5 disabled')
        stats = checkpoint['stats']
    arrays = {key: np.asarray(stats[key], dtype=np.float64)
              for key in ('audio_mean', 'audio_std', 'target_mean', 'target_std')}
    for key, shape in [('audio_mean', (6,)), ('audio_std', (6,)),
                       ('target_mean', (11,)), ('target_std', (11,))]:
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid normalization {key}')
    if (arrays['audio_std'] <= 0).any() or (arrays['target_std'] <= 0).any():
        raise ValueError('Normalization standard deviations must be positive')
    return stats, arrays


def load_data(source='runs/image_exposure_v1', split='train', stats=None,
              workers=4, load_images=True):
    """Read one fixed partition, cached Log-Mel and actual event timing on CPU.

    Full-dataset audio stays on CPU; call :func:`batch` for GPU transfers.
    Image-only anchor packs and original cache targets must agree to 1e-6 mm.
    All slide/polyline contact supervision is positive by acquisition protocol;
    it therefore supports no estimate of no-contact detection performance.
    """
    if split not in ('train', 'val', 'test') or type(workers) is not int or workers < 1:
        raise ValueError('Choose train/val/test and a positive integer worker count')
    source = Path(source)
    partitions = _validate_partitions(source)
    default_statistics = stats is None
    stats, norm = _statistics(stats)
    data = load_pack(source, split)
    records = data['records']
    if [r['id'] for r in records] != [r['id'] for r in partitions[split]]:
        raise ValueError('Reference partition changed while loading')
    n, steps = len(records), 10
    data['audio'] = np.empty((n, steps, 6, 64, 28), np.float32)
    data['event_dt'] = np.empty((n, steps), np.float32)
    for key in SUPERVISION_KEYS:
        shape = (n, steps, 11) if key == 'target_valid' else (n, steps)
        dtype = np.int8 if key in ('contact', 'previous_contact') else bool
        data[key] = np.empty(shape, dtype=dtype)
    am, ast = norm['audio_mean'][None, :, None, None], norm['audio_std'][None, :, None, None]

    def read(record):
        with np.load(Path(record['cache_root']) / record['file'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ('audio', 'target', 'event_pc_ns', 'previous_pc_ns', *SUPERVISION_KEYS)}
        if arrays['audio'].shape != (steps, 6, 64, 28) or not np.isfinite(arrays['audio']).all():
            raise ValueError(f"Invalid Log-Mel: {record['id']}")
        if arrays['target'].shape != (steps, 11) or not np.isfinite(arrays['target']).all():
            raise ValueError(f"Invalid measured labels: {record['id']}")
        for key in SUPERVISION_KEYS:
            expected = (steps, 11) if key == 'target_valid' else (steps,)
            if arrays[key].shape != expected:
                raise ValueError(f"Invalid {key}: {record['id']}")
        if (not np.all(arrays['contact'] == 1) or not arrays['contact_valid'].all()
                or not arrays['target_valid'].all()):
            raise ValueError(f"Slide/polyline-only pack requires positive valid contact: {record['id']}")
        if record.get('supervision_sha256') != supervision_digest(arrays):
            raise ValueError(f"Contact supervision changed: {record['id']}")
        event = np.asarray(arrays['event_pc_ns'])
        previous = np.asarray(arrays['previous_pc_ns'])
        if event.shape != (steps,) or previous.shape != (steps,) or event.dtype.kind not in 'iu' or previous.dtype.kind not in 'iu':
            raise ValueError(f"Invalid event timestamps: {record['id']}")
        dt = (event.astype(np.int64) - previous.astype(np.int64)).astype(np.float64) * 1e-9
        if not np.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError(f"Nonpositive event duration: {record['id']}")
        if not np.array_equal(previous[1:], event[:-1]):
            raise ValueError(f"Nonadjacent event history: {record['id']}")
        arrays['audio'] = ((arrays['audio'].astype(np.float64) - am) / ast).astype(np.float32)
        arrays['event_dt'] = dt.astype(np.float32)
        return arrays

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, arrays in enumerate(pool.map(read, records)):
            if not np.allclose(arrays['target'], data['target'][i], rtol=0, atol=1e-6):
                raise ValueError(f"Reference measured targets changed: {records[i]['id']}")
            if not np.array_equal(arrays['event_pc_ns'], data['event_ns'][i]):
                raise ValueError(f"Reference timestamps changed: {records[i]['id']}")
            data['audio'][i] = arrays['audio']
            data['event_dt'][i] = arrays['event_dt']
            for key in SUPERVISION_KEYS:
                data[key][i] = arrays[key]
    if not np.isfinite(data['host_ages']).all() or (data['host_ages'] < 0).any():
        raise ValueError('Reference image ages must be finite and causal')
    data['image_age'] = data['host_ages']
    data['delta_t'] = data['event_dt']
    data['target_normalized'] = ((data['target'].astype(np.float64) - norm['target_mean']) /
                                  norm['target_std']).astype(np.float32)
    data['stats'] = stats
    data['source_checkpoint'] = str(MULTIMODAL_BASELINE) if default_statistics else None
    data['reference_source'] = str(source)
    data['images'] = read_images(records) if load_images else None
    data['contact_scope'] = 'all-positive slide/polyline acquisition; no negative-contact evaluation'
    return data


def batch(data, idx, device, first_only=False):
    """Transfer a batch while retaining all six stored microphone channels."""
    if data.get('images') is None:
        raise ValueError('Images were not loaded; call load_data(load_images=True)')
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().numpy()
    if isinstance(idx, (int, np.integer)):
        idx = [int(idx)]
    output = {'image': images_to_device(data['images'], idx, device)}
    for key, source in [('audio', 'audio'), ('audio_tokens', 'audio_tokens'),
                        ('target', 'target_normalized'), ('physical', 'target'),
                        ('event_dt', 'event_dt'), ('image_age', 'image_age'),
                        *[(key, key) for key in SUPERVISION_KEYS]]:
        if source not in data:
            if source in ('audio', 'audio_tokens'):
                continue
            raise KeyError(source)
        value = data[source][idx]
        if first_only:
            value = value[:, :1]
        output[key] = torch.as_tensor(value, device=device)
    output['dt'] = output['event_dt']
    output['delta_t'] = output['event_dt']
    for key in ('anchors', 'anchor_valid', 'averaged_anchors', 'averaged_valid', 'image_ns'):
        output[key] = torch.as_tensor(data[key][idx], device=device)
    return output
