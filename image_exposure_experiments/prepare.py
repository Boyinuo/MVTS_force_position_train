"""Extract frozen image features and independent timing metadata, never mutate v6.

Run from repository root: python -m image_exposure_experiments.prepare
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mvts.engine import load_checkpoint
from mvts.features import load_image
from mvts.prepared_selection import load_prepared_caches
from .alignment import load_timeline

BASELINE = Path('runs/v6_triple_head_slide_polyline_image_only_trim0p029_xyfilter/best.pt')
ROOTS = [Path(f'.cache/data_final_v6_{s}_contact') for s in ('slide', 'polyline')]
DELAYS_MS = [0., 7.8, 15.6, 23.4, 31.2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Images(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        return load_image(Path(r['cache_root']) / r['image'])


def contiguous_segments(records):
    """Independent ordered segments; gaps reset both original and new models."""
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r['sequence']].append(i)
    segments = []
    for sequence in sorted(groups):
        ordered = sorted(groups[sequence], key=lambda i: records[i]['order'])
        current = []
        for i in ordered:
            if current and records[i]['order'] != records[current[-1]]['order'] + 1:
                segments.append(current)
                current = []
            current.append(i)
        if current:
            segments.append(current)
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/image_exposure_v1')
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output)
    if not out.name.startswith('image_exposure_'):
        parser.error('Output directory must start with image_exposure_')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'prepared.json').exists():
        raise ValueError('Preparation already completed; choose a new directory or reuse it')
    torch.set_num_threads(4)
    model, ckpt, _ = load_checkpoint(BASELINE, 'cuda')
    model.eval()
    assert model.config.image_only
    all_records, _, report = load_prepared_caches(ROOTS)
    by_id = {r['id']: r for r in all_records}
    hashes = {str(p): sha256(p) for p in [BASELINE, *[r / 'manifest.json' for r in ROOTS],
              Path('mvts/model_triple_head/model_triple_head.py')]}
    protocol = {
        'baseline': str(BASELINE), 'source_sha256': hashes,
        'roots': list(map(str, ROOTS)), 'exposure_ms': 15.6,
        'candidate_center_delays_ms': DELAYS_MS,
        'selection': 'reuse exact saved baseline split IDs; no re-splitting or new trimming',
        'primary_metric': 'mean Euclidean XY distance in mm at ORIGINAL current event times',
        'selection_metric': 'validation primary metric only; test after model and settings frozen',
        'inference_inputs': 'current/past images, their host read times, elapsed event time; no robot labels, force, audio or future image',
        'training': 'frozen two-cache pretrained visual backbone/attention; independent position/motion head',
        'auxiliary_labels': 'raw position at estimated exposure midpoint or mean over 15.6ms exposure; no extrapolation',
        'limitations': ['readout/transport delay unknown', 'shutter type unknown',
                       'robot position timestamps are software estimates; exposure averaging is approximate physical supervision'],
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    raw_roots = {str(Path(r['cache_root'])): Path(r['source_root']) for r in report['cache_reports']}
    sizes = {}
    for split, ids in ckpt['split_ids'].items():
        records = [by_id[i] for i in ids]
        assert all(r['kind'] == 'slide' and r['events'] == 10 for r in records)
        t0 = time.perf_counter()
        features = []
        with torch.inference_mode():
            for images in DataLoader(Images(records), batch_size=args.batch_size,
                                     num_workers=args.workers, pin_memory=True):
                fused = model.fuse(model.encode_visual(images.to('cuda')))
                features.append(fused[:, 0].cpu().numpy())
        features = np.concatenate(features)
        n = len(records)
        predictions = np.zeros((n, 10, 11), np.float32)
        targets = np.zeros_like(predictions)
        times = np.zeros((n, 10), np.int64)
        image_ns = np.array([r['image_pc_ns'] for r in records], dtype=np.int64)
        frame_dt = np.full(n, 1 / 15., np.float32)
        segments = contiguous_segments(records)
        mean = np.asarray(ckpt['stats']['target_mean'])
        std = np.asarray(ckpt['stats']['target_std'])
        for i, r in enumerate(records):
            with np.load(Path(r['cache_root']) / r['file']) as z:
                targets[i] = z['target']
                times[i] = z['event_pc_ns']
            assert np.allclose(targets[i], ckpt['split_labels'][split][i], atol=1e-6)
        with torch.inference_mode():
            for seg in segments:
                frame_dt[seg[1:]] = np.diff(image_ns[seg]) * 1e-9
                f = torch.from_numpy(features[seg]).to('cuda')[None].repeat_interleave(10, dim=1)
                states, _ = model.gru(f)
                result = model._predict(states)
                p = torch.cat([result[k] for k in ('position', 'wrench', 'slip')], -1)
                predictions[seg] = (p.cpu().numpy().reshape(-1, 10, 11) * std + mean)
        anchors = np.full((n, len(DELAYS_MS), 3), np.nan, np.float64)
        averaged = np.full_like(anchors, np.nan)
        valid = np.zeros((n, len(DELAYS_MS)), bool)
        average_valid = np.zeros_like(valid)
        # Diagnostic only: compare baseline predictions against past truth, never
        # claim this changed-target comparison improves current-time accuracy.
        past_targets = np.full((n, len(DELAYS_MS), 10, 3), np.nan, np.float64)
        for sequence, seqrows in __import__('itertools').groupby(
                sorted(enumerate(records), key=lambda ir: ir[1]['sequence']), key=lambda ir: ir[1]['sequence']):
            entries = list(seqrows)
            r = entries[0][1]
            timeline = load_timeline(raw_roots[str(Path(r['cache_root']))], r)
            idx = [i for i, _ in entries]
            for j, delay in enumerate(DELAYS_MS):
                q = image_ns[idx] - int(round(delay * 1e6))
                anchors[idx, j], valid[idx, j] = timeline.evaluate(q)
                averaged[idx, j], average_valid[idx, j] = timeline.evaluate(q, exposure_ms=15.6)
                shifted, _ = timeline.evaluate(times[idx] - int(round(delay * 1e6)))
                past_targets[idx, j] = shifted
        host_ages = ((times - image_ns[:, None]) * 1e-9).astype(np.float32)
        np.savez(out / f'{split}_features.npz', features=features, baseline=predictions,
                 target=targets, event_ns=times, image_ns=image_ns, host_ages=host_ages,
                 frame_dt=frame_dt, anchors=anchors, anchor_valid=valid,
                 averaged_anchors=averaged, averaged_valid=average_valid, past_targets=past_targets)
        (out / f'{split}_records.json').write_text(json.dumps(records), encoding='utf-8')
        sizes[split] = {'windows': n, 'events': n * 10, 'segments': len(segments),
                        'trajectories': len({r['sequence'] for r in records}),
                        'anchor_valid': valid.sum(0).tolist(), 'average_valid': average_valid.sum(0).tolist()}
        print(split, sizes[split], 'seconds', round(time.perf_counter() - t0, 2), flush=True)
    integrity = {p: sha256(p) == h for p, h in hashes.items()}
    assert all(integrity.values())
    (out / 'prepared.json').write_text(json.dumps({'splits': sizes, 'integrity': integrity,
        'stats': ckpt['stats'], 'baseline_epoch': ckpt['epoch']}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
