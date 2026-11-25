"""Read-only position timelines for exposure alignment experiments.

``PositionTimeline(times_ns, position_mm).evaluate(query_ns, exposure_ms=0)``
returns ``(positions, valid)``. Positions have shape ``query.shape + (D,)``;
invalid queries are NaN. A positive exposure uses the exact average of the
piecewise-linear measured trajectory over a centered exposure interval. This
is a supervision approximation, not an assertion that image appearance is
linear in position. Absolute host timestamps are subtracted in integer space
before conversion to floating point, preserving nanosecond differences.

``load_timeline(data_root, record)`` reads all original interval files in one
slide/polyline sequence, including boundaries excluded from training. Only
timestamps, measured XY and measured depth are loaded; ADC is never read.
``data_root`` can be the source root from a cache's report.json or the common
``data/data_final`` directory. Neither cache nor source is modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PositionTimeline:
    """A strictly ordered measured position stream with no extrapolation."""

    def __init__(self, times_ns, position_mm):
        times = np.asarray(times_ns)
        positions = np.asarray(position_mm, dtype=np.float64)
        if times.ndim != 1 or times.dtype.kind not in "iu" or len(times) < 2:
            raise ValueError("times_ns must be at least two integer timestamps")
        if times.dtype.kind == "u" and np.any(times > np.iinfo(np.int64).max):
            raise ValueError("timestamps exceed signed nanosecond range")
        times = times.astype(np.int64, copy=False)
        if np.any(np.diff(times) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if positions.ndim != 2 or positions.shape[0] != len(times) or not np.isfinite(positions).all():
            raise ValueError("position_mm must be finite [N,D] measurements")
        self.times_ns = times
        self.position_mm = positions
        self.origin_ns = int(times[0])
        self._times = (times - times[0]).astype(np.float64) / 1e9
        dt = np.diff(self._times)
        self._slopes = np.diff(positions, axis=0) / dt[:, None]
        trapezoids = .5 * (positions[:-1] + positions[1:]) * dt[:, None]
        self._areas = np.vstack((np.zeros((1, positions.shape[1])), np.cumsum(trapezoids, axis=0)))

    def _relative(self, query_ns):
        q = np.asarray(query_ns)
        if q.dtype.kind in "iu":
            # Do not cast large absolute nanosecond values to float first.
            if q.dtype.kind == "u" and np.any(q > np.iinfo(np.int64).max):
                raise ValueError("query timestamps exceed signed nanosecond range")
            return (q.astype(np.int64) - self.origin_ns).astype(np.float64) / 1e9
        if q.dtype.kind != "f":
            raise ValueError("query_ns must be numeric timestamps")
        return (q - self.origin_ns) / 1e9

    def _integral(self, t):
        indices = np.searchsorted(self._times, t, side="right") - 1
        indices = np.clip(indices, 0, len(self._times) - 2)
        elapsed = t - self._times[indices]
        return (self._areas[indices] + elapsed[:, None] * self.position_mm[indices]
                + .5 * elapsed[:, None] ** 2 * self._slopes[indices])

    def evaluate(self, query_ns, exposure_ms=0.):
        """Interpolate or exposure-average; reject incomplete exposure windows."""
        if not np.isscalar(exposure_ms) or not np.isfinite(exposure_ms) or exposure_ms < 0:
            raise ValueError("exposure_ms must be finite and nonnegative")
        relative = self._relative(query_ns)
        shape = relative.shape
        t = relative.reshape(-1)
        half = float(exposure_ms) / 2000.
        valid = np.isfinite(t) & (t - half >= self._times[0]) & (t + half <= self._times[-1])
        out = np.full((len(t), self.position_mm.shape[1]), np.nan)
        if valid.any():
            q = t[valid]
            if exposure_ms:
                out[valid] = (self._integral(q + half) - self._integral(q - half)) / (2 * half)
            else:
                out[valid] = np.stack([np.interp(q, self._times, values)
                                       for values in self.position_mm.T], axis=-1)
        return out.reshape(shape + (self.position_mm.shape[1],)), valid.reshape(shape)


def _source_file(data_root, record):
    root = Path(data_root).resolve()
    relative = Path(str(record["source"]).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("record source must be a relative path without parent traversal")
    direct = root / relative
    if direct.is_file():
        return direct
    candidates = [category / relative for category in root.glob("data_final_*")
                  if category.is_dir() and (category / relative).is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot uniquely resolve source {relative} under {root}")
    return candidates[0]


def load_timeline(data_root, record):
    """Load the full original position stream for this record's sequence."""
    sequence_dir = _source_file(data_root, record).parent
    manifest = sequence_dir / "groups.jsonl"
    if manifest.is_file():
        groups = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines()
                  if line.strip()]
        filenames = list(dict.fromkeys(group["data_file"] for group in groups))
        paths = []
        for name in filenames:
            path = (sequence_dir / name).resolve()
            if not path.is_relative_to(sequence_dir.resolve()):
                raise ValueError("group data file escapes sequence directory")
            paths.append(path)
    else:
        paths = sorted(sequence_dir.glob("interval_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No interval files in {sequence_dir}")
    timestamps, positions = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            t = archive["force_pc_est_ns"]
            xy = archive["force_local_xyz_mm"][:, :2]
            depth = archive["force_depth_mm"]
            timestamps.append(t)
            positions.append(np.column_stack((xy, depth)))
    t, p = np.concatenate(timestamps), np.concatenate(positions)
    if len(t) < 2:
        raise ValueError(f"Insufficient position measurements in {sequence_dir}")
    return PositionTimeline(t, p)
