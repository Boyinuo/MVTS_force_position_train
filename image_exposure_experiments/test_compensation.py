"""Behavioral checks for prediction-only, causal motion compensation."""
import numpy as np
import pytest

from image_exposure_experiments.compensation import compensate


def examples(anchors, *, sequences=None, orders=None):
    anchors = np.asarray(anchors, dtype=float)
    positions = np.repeat(anchors[:, None, :], 10, axis=1)
    n = len(positions)
    records = [{"sequence": (sequences or ["a"] * n)[i],
                "order": (orders or list(range(n)))[i]} for i in range(n)]
    times = 10**15 + np.arange(n, dtype=np.int64)[:, None] * 100_000_000 + (
        np.arange(10, dtype=np.int64)[None, :] * 10_000_000)
    return positions, records, times


def test_constant_speed_known_horizon_and_no_feedback():
    p, r, t = examples([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    corrected = compensate(p, r, t, 7.8)
    np.testing.assert_array_equal(corrected[0], p[0])
    np.testing.assert_allclose(corrected[1:, :, 0], p[1:, :, 0] + .078)
    # Frame 3 still uses (2-1)/.1, not (2-corrected_frame_2)/.1.
    np.testing.assert_array_equal(p[:, 0, 0], [0, 1, 2])


def test_future_frames_and_later_subevents_do_not_affect_current_correction():
    p, r, t = examples([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    first = compensate(p, r, t, 15.6)
    changed = p.copy()
    changed[2] += 1000
    changed[1, 1:] += 50
    second = compensate(changed, r, t, 15.6)
    np.testing.assert_array_equal(first[:1], second[:1])
    np.testing.assert_array_equal(first[1, 0], second[1, 0])
    np.testing.assert_allclose(first[1] - p[1], second[1] - changed[1], atol=1e-12)


def test_sequence_and_order_gap_reset_history():
    p, r, t = examples([[0, 0, 0], [1, 0, 0], [9, 0, 0], [10, 0, 0], [90, 0, 0]],
                       sequences=["a", "a", "a", "a", "b"], orders=[0, 1, 3, 4, 0])
    result = compensate(p, r, t, 10)
    np.testing.assert_array_equal(result[[0, 2, 4]], p[[0, 2, 4]])
    np.testing.assert_allclose(result[3, :, 0], 10.1)


def test_nonincreasing_anchor_resets_without_using_invalid_velocity():
    p, r, t = examples([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    t[1] = t[0]
    result = compensate(p, r, t, 10)
    np.testing.assert_array_equal(result[1], p[1])
    np.testing.assert_allclose(result[2, :, 0], 2.05)


def test_speed_clipping_uses_vector_norm():
    p, r, t = examples([[0, 0, 0], [30, 40, 0]])
    result = compensate(p, r, t, 10, max_speed=100)
    np.testing.assert_allclose(result[1, 0] - p[1, 0], [.6, .8, 0])


def test_smoothing_uses_only_previous_estimated_velocity():
    p, r, t = examples([[0, 0, 0], [1, 0, 0], [4, 0, 0]])
    result = compensate(p, r, t, 10, smoothing=.25)
    np.testing.assert_allclose(result[1, 0, 0], 1.1)
    np.testing.assert_allclose(result[2, 0, 0], 4.15)


def test_zero_horizon_is_exact_copy():
    p, r, t = examples([[0, 0, 0], [1, 0, 0]])
    result = compensate(p, r, t, 0)
    np.testing.assert_array_equal(result, p)
    assert not np.shares_memory(result, p)


@pytest.mark.parametrize("setting", [{"smoothing": 0}, {"max_speed": 0},
                                     {"horizon_ms": float("nan")}])
def test_invalid_parameters_fail(setting):
    p, r, t = examples([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        compensate(p, r, t, **({"horizon_ms": 7.8} | setting))
