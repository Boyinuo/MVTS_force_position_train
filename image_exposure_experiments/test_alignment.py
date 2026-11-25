"""Physical timing tests independent of the training model."""
import json

import numpy as np
import pytest

from image_exposure_experiments.alignment import PositionTimeline, load_timeline


def test_linear_trajectory_exposure_average_equals_center():
    t = np.arange(21, dtype=np.int64) * 1_000_000
    p = np.column_stack((2 * t / 1e9, -3 * t / 1e9, np.ones(len(t))))
    stream = PositionTimeline(t, p)
    q = np.array([8_000_000, 10_000_000, 12_000_000])
    averaged, valid = stream.evaluate(q, exposure_ms=15.6)
    instant, _ = stream.evaluate(q)
    assert valid.all()
    np.testing.assert_allclose(averaged, instant, atol=1e-14)


def test_piecewise_exposure_average_integrates_corner_exactly():
    stream = PositionTimeline(np.array([0, 1, 2], dtype=np.int64) * 1_000_000_000,
                              np.array([[0.], [1.], [0.]]))
    position, valid = stream.evaluate(np.int64(1_000_000_000), exposure_ms=2000)
    assert bool(valid)
    np.testing.assert_allclose(position, [.5], atol=1e-14)


def test_integer_subtraction_preserves_nanoseconds():
    origin = 9_000_000_000_000_000_000
    stream = PositionTimeline(np.array([origin, origin + 10], dtype=np.int64), [[0.], [10.]])
    position, valid = stream.evaluate(np.array([origin + 1, origin + 7], dtype=np.int64))
    assert valid.all()
    np.testing.assert_allclose(position[:, 0], [1., 7.], atol=1e-12)


def test_no_extrapolation_including_exposure_edges():
    stream = PositionTimeline(np.array([0, 10_000_000], dtype=np.int64), [[0.], [10.]])
    position, valid = stream.evaluate(np.array([-1, 0, 5_000_000, 10_000_001], dtype=np.int64))
    assert valid.tolist() == [False, True, True, False]
    assert np.isnan(position[[0, 3]]).all()
    position, valid = stream.evaluate(np.array([0, 5_000_000, 10_000_000]), exposure_ms=10)
    assert valid.tolist() == [False, True, False]
    assert np.isnan(position[[0, 2]]).all()


def test_validation_rejects_nonmonotonic_and_bad_exposure():
    with pytest.raises(ValueError):
        PositionTimeline(np.array([3, 3]), [[0.], [1.]])
    stream = PositionTimeline(np.array([0, 1]), [[0.], [1.]])
    for exposure in (-1., np.nan, np.inf):
        with pytest.raises(ValueError):
            stream.evaluate([0], exposure_ms=exposure)


def test_source_loading_only_requires_position_arrays(tmp_path):
    sequence = tmp_path / "data_final_slide" / "source" / "trajectories" / "t0001"
    sequence.mkdir(parents=True)
    groups = []
    for i in range(2):
        name = f"interval_{i + 1:03d}.npz"
        t = np.array([i * 2, i * 2 + 1], dtype=np.int64) * 1_000_000
        np.savez(sequence / name, force_pc_est_ns=t,
                 force_local_xyz_mm=np.column_stack((t / 1e6, np.zeros((2, 2)))),
                 force_depth_mm=np.full(2, 1.2))
        groups.append({"data_file": name})
    (sequence / "groups.jsonl").write_text("\n".join(json.dumps(x) for x in groups), encoding="utf-8")
    record = {"source": "source\\trajectories\\t0001\\interval_001.npz"}
    stream = load_timeline(tmp_path, record)
    positions, valid = stream.evaluate([500_000, 2_500_000])
    assert valid.all()
    np.testing.assert_allclose(positions, [[.5, 0., 1.2], [2.5, 0., 1.2]])
