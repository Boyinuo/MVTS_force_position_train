"""Combine prepared caches and select trajectory prefixes without copying features."""
from collections import Counter
import math
from pathlib import Path

import numpy as np

from .data import read_json
from .dual_rate_data import read_prepared, split_prepared

FIXED_XY_RANGE_MM = {"x": [-10., 10.], "y": [-8., 10.]}


def filter_xy_range(records, enabled=False):
    """Drop entire slide sequences containing ANY out-of-range XY truth."""
    rejected_sequences, rejected_press = set(), set()
    for record in records:
        if enabled:
            target = np.asarray(record["target"], dtype=np.float64)
            if target.ndim != 2 or not len(target) or target.shape[1] != 11 or not np.isfinite(target[:, :2]).all():
                raise ValueError(f"Invalid XY truth: {record['id']}")
            inside = ((target[:, 0] >= FIXED_XY_RANGE_MM["x"][0])
                      & (target[:, 0] <= FIXED_XY_RANGE_MM["x"][1])
                      & (target[:, 1] >= FIXED_XY_RANGE_MM["y"][0])
                      & (target[:, 1] <= FIXED_XY_RANGE_MM["y"][1]))
            if not inside.all():
                if record["kind"] == "slide":
                    rejected_sequences.add(record["sequence"])
                elif record["kind"] == "press":
                    rejected_press.add(record["id"])
                else:
                    raise ValueError(f"Unknown acquisition kind: {record['kind']}")
    retained, removed = [], []
    for record in records:
        reject = record["sequence"] in rejected_sequences if record["kind"] == "slide" else record["id"] in rejected_press
        (removed if reject else retained).append(record)
    return retained, {"enabled": bool(enabled), "bounds_mm": FIXED_XY_RANGE_MM,
                      "rule": "inclusive rectangle; remove an ENTIRE slide trajectory if ANY event is outside; press filtered per sample",
                      "source_windows": len(records), "retained_windows": len(retained),
                      "removed_windows": len(removed), "retained_events": sum(r["events"] for r in retained),
                      "removed_events": sum(r["events"] for r in removed),
                      "removed_trajectories": len(rejected_sequences),
                      "retained_trajectories": len({r["sequence"] for r in retained if r["kind"] == "slide"}),
                      "removed_press_samples": len(rejected_press)}


def xy_diameter_mm(record):
    """Largest pairwise measured XY distance inside one camera interval."""
    target = np.asarray(record["target"], dtype=np.float64)
    if target.ndim != 2 or not len(target) or target.shape[1] != 11 or not np.isfinite(target).all():
        raise ValueError(f"Invalid target: {record['id']}")
    xy = target[:, :2]
    return float(np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).max())


def select_records(records, threshold_mm):
    """Keep press and slide prefixes before the FIRST strictly stationary window."""
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("threshold_mm must be finite and positive")
    retained, removed, distances, cutoffs = [], [], {}, {}
    for record in records:
        if record["kind"] not in ("press", "slide"):
            raise ValueError(f"Unknown acquisition kind: {record['kind']}")
        if record["kind"] == "slide":
            distance = distances[record["id"]] = xy_diameter_mm(record)
            sequence, order = record["sequence"], record["order"]
            if distance < threshold_mm and order < cutoffs.get(sequence, float("inf")):
                cutoffs[sequence] = order
    for record in records:
        cutoff = cutoffs.get(record["sequence"]) if record["kind"] == "slide" else None
        if cutoff is not None and record["order"] >= cutoff:
            removed.append({"id": record["id"], "sequence": record["sequence"],
                            "order": record["order"], "events": record["events"],
                            "xy_diameter_mm": distances[record["id"]], "file": record["file"],
                            "image": record["image"], "cutoff_order": cutoff,
                            "reason": "first_stationary_xy" if record["order"] == cutoff
                                      else "after_first_stationary_xy"})
        else:
            retained.append(dict(record))
    return retained, removed


def trim_prepared_records(records, threshold_mm=0.):
    """Selection uses existing manifest targets; NPZ, history and features stay intact."""
    if not math.isfinite(threshold_mm) or threshold_mm < 0:
        raise ValueError("trim-stationary-xy-mm must be finite and nonnegative (0 disables trimming)")
    retained, removed = select_records(records, threshold_mm) if threshold_mm else (records, [])
    cutoffs = {r["sequence"]: r["cutoff_order"] for r in removed}
    return retained, {"threshold_mm": threshold_mm, "enabled": bool(threshold_mm),
                      "metric": "maximum pairwise XY distance within each camera window; Z ignored",
                      "rule": "first distance < threshold: exclude that window and ALL later windows in the trajectory",
                      "source_windows": len(records), "retained_windows": len(retained),
                      "removed_windows": len(removed), "removed_events": sum(r["events"] for r in removed),
                      "retained_events": sum(r["events"] for r in retained), "trajectory_cutoffs": cutoffs}


def load_prepared_caches(roots, *, contact_only=False):
    """Combine disjoint IDs while keeping each file relative to its own cache."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    roots = [Path(root).resolve() for root in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Specify distinct prepared cache directories")
    if contact_only:
        from .contact_data import read_contact_prepared
        reader = read_contact_prepared
    else:
        reader = read_prepared
    records, manifests, reports, by_id = [], [], [], {}
    for root in roots:
        part, manifest = reader(root)
        for record in part:
            if record["id"] in by_id:
                raise ValueError(f"Overlapping sample ID {record['id']!r} in {by_id[record['id']]} and {root}; "
                                 "choose non-overlapping caches")
            by_id[record["id"]] = root
        records.extend(part)
        manifests.append({"cache_root": str(root), **{k: v for k, v in manifest.items() if k != "records"}})
        reports.append({"cache_root": str(root), **read_json(root / "report.json")})
    # Mixed v4/v5 input is valid for the triple model. V4 supplies valid regression
    # labels, but does not invent per-event contact supervision.
    if any("supervision_sha256" in r for r in records):
        records = [{**r, "mixed_contact_cache": True} for r in records]
    combined = {"smoke_test": any(m.get("smoke_test", False) for m in manifests), "caches": manifests}
    report = {"prepared_roots": [str(root) for root in roots], "cache_reports": reports,
              "windows": len(records), "events": sum(r["events"] for r in records),
              "counts": dict(Counter(r["kind"] for r in records))}
    return records, combined, report


def prepare_training_data(args, default_root, checkpoint=None, *, contact_only=False):
    saved = checkpoint["args"] if checkpoint else {}
    roots = args.prepared_root or saved.get("prepared_root") or default_root
    requested = getattr(args, "trim_stationary_xy_mm", None)
    previous = saved.get("trim_stationary_xy_mm", 0.) or 0.
    threshold = previous if requested is None else requested
    if checkpoint and threshold != previous:
        raise ValueError("Resume trim-stationary-xy-mm differs from checkpoint; start a new run")
    requested_range = getattr(args, "filter_xy_range", None)
    previous_range = saved.get("filter_xy_range", False)
    use_range = previous_range if requested_range is None else requested_range
    if checkpoint and use_range != previous_range:
        raise ValueError("Resume filter-xy-range differs from checkpoint; start a new run")
    records, manifest, report = load_prepared_caches(roots, contact_only=contact_only)
    args.prepared_root = report["prepared_roots"]
    args.trim_stationary_xy_mm = threshold
    args.filter_xy_range = bool(use_range)
    # Assign complete locations/trajectories BEFORE filtering so changing the
    # threshold cannot move a surviving trajectory between train/val/test.
    original = split_prepared(records, args.seed, args.val_fraction, args.test_fraction, args.max_groups)
    split, selection, range_selection = {}, {}, {}
    for key, part in original.items():
        in_range, range_selection[key] = filter_xy_range(part, use_range)
        split[key], selection[key] = trim_prepared_records(in_range, threshold)
        if not split[key]:
            raise ValueError(f"No {key} windows remain after data selection; reduce the trim threshold or disable --filter-xy-range")
    report["training_selection"] = selection
    report["xy_range_selection"] = range_selection
    return split, manifest, report


def prepared_test_records(roots, checkpoint, *, contact_only=False):
    """Reapply the saved selection before looking up the saved test partition."""
    records, _, _ = load_prepared_caches(roots, contact_only=contact_only)
    records, _ = filter_xy_range(records, checkpoint["args"].get("filter_xy_range", False))
    records, _ = trim_prepared_records(records, checkpoint["args"].get("trim_stationary_xy_mm", 0.) or 0.)
    by_id = {r["id"]: r for r in records}
    missing = set(checkpoint["split_ids"]["test"]) - by_id.keys()
    if missing:
        raise ValueError(f"Prepared test samples missing after selection: {sorted(missing)[:3]}")
    return [by_id[key] for key in checkpoint["split_ids"]["test"]]
