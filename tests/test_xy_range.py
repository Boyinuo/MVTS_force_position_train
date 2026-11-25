"""Fixed XY selection removes whole slide trajectories and individual press samples."""
from types import SimpleNamespace

import numpy as np
import pytest

from mvts.prepared_selection import filter_xy_range, prepare_training_data


def record(sequence, order, xy, kind="slide"):
    target = np.zeros((len(xy), 11))
    target[:, :2] = xy
    target[:, 2] = 1000.  # Only XY constrains selection.
    return {"id": f"{sequence}/{order}", "kind": kind, "sequence": sequence, "order": order,
            "events": len(xy), "target": target.tolist(), "file": "unused.npz", "image": "unused.png",
            "group": sequence}


def test_whole_trajectory_filter_keeps_boundaries_and_removes_inside_fragments():
    good = [record("good", 0, [(-10., -8.), (10., 10.)]), record("good", 1, [(0., 0.)])]
    bad = []
    for i, outside in enumerate(((-10.00001, 0.), (10.00001, 0.), (0., -8.00001), (0., 10.00001))):
        bad.extend([record(f"bad{i}", 0, [(0., 0.)]), record(f"bad{i}", 1, [outside]),
                    record(f"bad{i}", 2, [(0., 0.)])])
    press = [record("press_in", 0, [(10., -8.)], "press"),
             record("press_out", 0, [(10.00001, -8.)], "press")]
    records = good + bad + press
    selected, report = filter_xy_range(records, True)
    assert selected == good + press[:1]
    assert report["removed_trajectories"] == 4 and report["retained_trajectories"] == 1
    assert report["removed_press_samples"] == 1
    assert report["removed_events"] == 13 and report["retained_events"] == 4
    unchanged, report = filter_xy_range(records, False)
    assert unchanged == records and report["removed_windows"] == report["removed_trajectories"] == 0


def test_range_is_applied_before_stationary_tail_trimming_and_after_partition(monkeypatch):
    import mvts.prepared_selection as selection
    original = {}
    for key in ("train", "val", "test"):
        original[key] = [record(f"{key}/good", 0, [(0., 0.), (.1, 0.)]),
                         record(f"{key}/good", 1, [(0., 0.), (.001, 0.)]),
                         record(f"{key}/bad", 0, [(0., 0.), (.1, 0.)]),
                         record(f"{key}/bad", 1, [(0., 0.), (.001, 0.)]),
                         record(f"{key}/bad", 2, [(11., 0.)]),
                         record(f"{key}/press_in", 0, [(0., 0.)], "press"),
                         record(f"{key}/press_out", 0, [(11., 0.)], "press")]
    monkeypatch.setattr(selection, "load_prepared_caches", lambda *a, **k: (
        sum(original.values(), []), {"smoke_test": True}, {"prepared_roots": ["fixture"]}))
    monkeypatch.setattr(selection, "split_prepared", lambda *a: original)
    args = SimpleNamespace(prepared_root=["fixture"], filter_xy_range=True, trim_stationary_xy_mm=.03,
                           seed=42, val_fraction=.15, test_fraction=.15, max_groups=None)
    selected, _, report = prepare_training_data(args, "unused")
    for key, part in selected.items():
        assert [r["id"] for r in part] == [f"{key}/good/0", f"{key}/press_in/0"]
        assert report["xy_range_selection"][key]["removed_trajectories"] == 1
        assert report["xy_range_selection"][key]["removed_press_samples"] == 1
        assert report["training_selection"][key]["removed_windows"] == 1
    # Omitting the switch on resume inherits the saved policy; changing it fails.
    args.filter_xy_range = None
    checkpoint = {"args": {"prepared_root": ["fixture"], "filter_xy_range": True, "trim_stationary_xy_mm": .03}}
    resumed, _, _ = prepare_training_data(args, "unused", checkpoint)
    assert resumed == selected and args.filter_xy_range is True
    args.filter_xy_range = False
    with pytest.raises(ValueError, match="Resume filter-xy-range differs"):
        prepare_training_data(args, "unused", checkpoint)
