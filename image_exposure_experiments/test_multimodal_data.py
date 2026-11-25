"""Input integrity, timing and normalization tests without real cache writes."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mvts.contact_data import add_contact_supervision
from image_exposure_experiments import multimodal_data as module


def fixtures(monkeypatch):
    mean = np.arange(6, dtype=np.float64)
    std = np.arange(1, 7, dtype=np.float64)
    stats = {'audio_mean': mean.tolist(), 'audio_std': std.tolist(),
             'target_mean': np.arange(11).tolist(), 'target_std': np.full(11, 2.).tolist()}
    event = np.cumsum(np.arange(1, 11, dtype=np.int64) * 1_000_000) + 100_000_000
    previous = np.r_[100_000_000, event[:-1]]
    target = np.arange(110, dtype=np.float32).reshape(10, 11)
    audio = np.broadcast_to((mean + 2 * std)[None, :, None, None], (10, 6, 64, 28)).copy()
    arrays, metadata = add_contact_supervision(
        {'audio': audio, 'target': target, 'event_pc_ns': event, 'previous_pc_ns': previous}, 'slide')
    records = {}
    for i, split in enumerate(('train', 'val', 'test')):
        records[split] = [{'id': f'source/trajectory{i}/0000', 'sequence': f'source/trajectory{i}',
                          'order': 0, 'kind': 'slide', 'events': 10,
                          'cache_root': '/fake/data_final_v6_slide_contact' if i != 1 else '/fake/data_final_v6_polyline_contact',
                          'file': f'events/{i}.npz', 'image': 'images/fake.png', **metadata}]
    monkeypatch.setattr(Path, 'read_text', lambda self, **kwargs: json.dumps(records[self.name.split('_')[0]]))

    def pack(source, split):
        return {'records': copy.deepcopy(records[split]), 'segments': [[0]],
                'target': target[None].copy(), 'event_ns': event[None].copy(),
                'host_ages': ((event - event[0]) * 1e-9)[None].astype(np.float32),
                'image_ns': event[:1], 'anchors': np.zeros((1, 5, 3), np.float64),
                'anchor_valid': np.ones((1, 5), bool), 'averaged_anchors': np.ones((1, 5, 3), np.float64),
                'averaged_valid': np.ones((1, 5), bool)}
    monkeypatch.setattr(module, 'load_pack', pack)
    monkeypatch.setattr(module, 'read_images', lambda records: torch.full((len(records), 3, 224, 224), 128, dtype=torch.uint8))

    class Archive(dict):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: Archive({k: v.copy() for k, v in arrays.items()}))
    return records, arrays, stats


def test_actual_event_durations_and_checkpoint_normalization(monkeypatch):
    _, arrays, stats = fixtures(monkeypatch)
    data = module.load_data('/unused', 'train', stats=stats, workers=1)
    np.testing.assert_allclose(data['audio'], 2.)
    np.testing.assert_allclose(data['event_dt'][0], np.arange(1, 11) / 1000.)
    np.testing.assert_allclose(data['target_normalized'][0], (arrays['target'] - np.arange(11)) / 2)
    assert data['images'].dtype == torch.uint8
    assert data['delta_t'] is data['event_dt']
    assert data['contact'].all() and data['target_valid'].all()


def test_first_event_batch_retains_all_anchor_candidates_and_six_channels(monkeypatch):
    _, _, stats = fixtures(monkeypatch)
    data = module.load_data('/unused', 'train', stats=stats, workers=1)
    data['audio_tokens'] = np.ones((1, 10, 28, 128), np.float32)
    batch = module.batch(data, 0, 'cpu', first_only=True)
    assert batch['audio'].shape == (1, 1, 6, 64, 28)
    assert batch['audio_tokens'].shape == (1, 1, 28, 128)
    assert batch['target'].shape == batch['physical'].shape == (1, 1, 11)
    assert batch['anchors'].shape == (1, 5, 3)
    assert batch['anchor_valid'].shape == (1, 5)
    assert batch['image_age'].shape == batch['delta_t'].shape == (1, 1)
    assert batch['dt'] is batch['event_dt']
    del data['audio']
    assert 'audio_tokens' in module.batch(data, [0], 'cpu')


def test_partition_overlap_and_press_sources_are_rejected(monkeypatch):
    records, _, _ = fixtures(monkeypatch)
    records['test'][0]['id'] = records['train'][0]['id']
    with pytest.raises(ValueError, match='overlapping ID'):
        module._validate_partitions('/unused')
    records['test'][0]['id'] = 'different'
    records['test'][0]['cache_root'] = '/fake/data_final_v6_press_contact'
    with pytest.raises(ValueError, match='slide/polyline caches'):
        module._validate_partitions('/unused')


def test_supervision_changes_and_noncausal_event_timing_rejected(monkeypatch):
    _, arrays, stats = fixtures(monkeypatch)
    arrays['previous_contact'][0] = 1
    with pytest.raises(ValueError, match='supervision changed'):
        module.load_data('/unused', 'train', stats=stats, workers=1)
    arrays['previous_contact'][0] = -1
    arrays['previous_pc_ns'][0] = arrays['event_pc_ns'][0]
    with pytest.raises(ValueError, match='Nonpositive event duration'):
        module.load_data('/unused', 'train', stats=stats, workers=1)


def test_invalid_standard_deviations_rejected(monkeypatch):
    _, _, stats = fixtures(monkeypatch)
    stats['audio_std'][0] = 0
    with pytest.raises(ValueError, match='standard deviations'):
        module.load_data('/unused', 'train', stats=stats, workers=1)
