"""Causal motion compensation for frozen, image-only position predictions.

The current image's first prediction and the preceding image's first prediction
estimate frame-to-frame velocity. No later prediction, robot measurement, label,
or corrected output participates in that estimate. A frame's velocity estimate
is held across its subevents, as no new image arrives during those subevents.

This is a hypothesis test for residual lag, not a measurement of camera latency.
In particular, an already-trained model may have learned delay compensation.
"""
from collections.abc import Mapping, Sequence

import numpy as np


def compensate(predictions, records: Sequence[Mapping], event_ns, horizon_ms,
               smoothing=1.0, max_speed=200):
    """Return ``predictions + horizon * causal_predicted_velocity`` in mm.

    Args:
        predictions: Uncorrected physical positions, shape [frames, events, 3].
        records: One mapping per frame, with ``sequence`` and integer ``order``.
            Frames belonging to a sequence must arrive in acquisition order.
            Interleaved independent sequences maintain separate histories.
        event_ns: Integer host event timestamps, shape [frames, events]. The
            first event in each frame supplies the frame-anchor time.
        horizon_ms: Signed finite prediction horizon. Positive values extrapolate
            forward; zero is an exact no-op. Choose horizons on validation only.
        smoothing: EMA weight for the newest measured frame velocity, in (0,1].
            The first valid consecutive frame initializes directly from velocity;
            a sequence/order/timestamp discontinuity clears the estimate.
        max_speed: Positive Euclidean speed limit in mm/s, applied before EMA.

    A first frame or a discontinuity receives zero correction. State always uses
    original predictions, preventing corrections from feeding back into velocity.
    The returned float64 array is independent of the input arrays.
    """
    positions = np.asarray(predictions, dtype=np.float64)
    times = np.asarray(event_ns)
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] < 1:
        raise ValueError("predictions must have shape [frames, events, 3]")
    if len(records) != positions.shape[0] or times.shape != positions.shape[:2]:
        raise ValueError("records and event_ns must match prediction dimensions")
    if times.dtype.kind not in "iu":
        raise ValueError("event_ns must contain integer nanosecond timestamps")
    if not np.isfinite(positions).all():
        raise ValueError("predictions must be finite")
    horizon_ms, smoothing, max_speed = map(float, (horizon_ms, smoothing, max_speed))
    if not np.isfinite(horizon_ms):
        raise ValueError("horizon_ms must be finite")
    if not np.isfinite(smoothing) or not 0 < smoothing <= 1:
        raise ValueError("smoothing must be in (0, 1]")
    if not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("max_speed must be finite and positive")
    if positions.shape[1] > 1 and (np.diff(times, axis=1) <= 0).any():
        raise ValueError("event_ns must increase strictly within each frame")

    output = positions.copy()
    state = {}
    for index, record in enumerate(records):
        sequence = record["sequence"]
        order = record["order"]
        if not isinstance(order, (int, np.integer)) or isinstance(order, bool):
            raise ValueError("record order must be an integer")
        anchor_ns = int(times[index, 0])
        anchor = positions[index, 0]
        previous = state.get(sequence)
        velocity = None
        if previous is not None:
            previous_order, previous_ns, previous_anchor, previous_velocity = previous
            if order == previous_order + 1 and anchor_ns > previous_ns:
                # Subtract integer timestamps before converting to seconds.
                elapsed = (anchor_ns - previous_ns) * 1e-9
                measured = (anchor - previous_anchor) / elapsed
                speed = np.linalg.norm(measured)
                if speed > max_speed:
                    measured = measured * (max_speed / speed)
                velocity = measured if previous_velocity is None else (
                    smoothing * measured + (1 - smoothing) * previous_velocity)
        if velocity is not None and horizon_ms != 0:
            output[index] += (horizon_ms * 1e-3) * velocity
        state[sequence] = (order, anchor_ns, anchor.copy(), velocity)
    return output
