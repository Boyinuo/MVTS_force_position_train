"""Regression checks for causal inference, selection and playback (no trained weights needed)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from deployment.replay import (ROOT, ImageOnlyReplay, ReplayData, default_checkpoint, infer_events,
                               infer_image_only, list_sequences, model_name, precompute, save_results, select_sequence)


def test_model_argument_required_and_checkpoint_automatic():
    from deployment.deployment_viewer import parser
    with pytest.raises(SystemExit) as error:
        parser().parse_args([])
    assert error.value.code == 2
    args = parser().parse_args(["--model", "mvts/model_contact_gated/model_contact_gated.py"])
    assert model_name(args.model) == "model_contact_gated"
    assert not hasattr(args, "checkpoint")


def test_checkpoint_beside_selected_model(tmp_path, monkeypatch):
    import deployment.replay as replay
    directory = tmp_path / "model_a"
    directory.mkdir()
    monkeypatch.setattr(replay, "get_model_module", lambda name: SimpleNamespace(__file__=str(directory / "__init__.py")))
    with pytest.raises(FileNotFoundError, match="Place best.pt beside"):
        default_checkpoint("model_a")
    weights = directory / "best.pt"
    weights.touch()
    assert default_checkpoint("model_a") == weights


def test_select_full_example_slide():
    rows = select_sequence(ROOT / "deployment/test_data/deployment_test_150hz")
    assert len(rows) == 290
    assert rows[0]["sequence"] == "slide_t0242"
    assert sum(row["reset_state"] for row in rows) == 1
    assert sum(row["image_update"] for row in rows) == 29


def test_reject_partial_trajectory(tmp_path):
    rows = [{"kind": "slide", "sequence": "a", "sequence_index": 1,
             "reset_state": True, "image_update": True}]
    (tmp_path / "events.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="complete and ordered"):
        select_sequence(tmp_path)


def test_select_converted_t0232_subset():
    root = ROOT / "deployment/test_data/deployment_test_150hz_t0232"
    rows = select_sequence(root)
    assert len(rows) == 80
    assert rows[0]["sequence"] == "slide_t0232"
    assert sum(row["reset_state"] for row in rows) == 1
    assert sum(row["image_update"] for row in rows) == 8
    assert list(dict.fromkeys(row["source_window_order"] for row in rows)) == list(range(2, 10))


@pytest.mark.parametrize("gated", [False, True])
def test_ordered_steps_cache_and_physical_units(gated):
    calls = []

    class Model:
        def reset_state(self):
            calls.append("reset")

        def step(self, audio, image, *, frame_id, reset):
            calls.append((image is not None, frame_id, reset, float(audio[0, 0, 0, 0])))
            if gated:
                return {"physical_prediction": torch.full((1, 11), 9.), "position_valid": torch.tensor([True]),
                    "slip_valid": torch.tensor([len(calls) > 2]), "contact_probability": torch.tensor([.8])}
            return {"position": torch.ones(1, 3), "wrench": torch.ones(1, 6), "slip": torch.ones(1, 2)}

    rows = [{"image": "a", "image_update": i == 0, "reset_state": i == 0} for i in range(3)]
    stats = {"target_mean": [2.] * 11, "target_std": [3.] * 11}
    result = infer_events(Model(), stats, torch.device("cpu"), rows,
        np.stack([np.full((6, 64, 28), i, np.float32) for i in range(3)]), {"a": np.zeros((1, 3, 224, 224), np.float32)})
    assert calls == ["reset", (True, "a", True, 0.), (False, "a", False, 1.), (False, "a", False, 2.)]
    np.testing.assert_array_equal(result[0], np.full((3, 11), 9. if gated else 5.))
    np.testing.assert_array_equal(result[2], [False, True, True])


def test_image_only_15hz_has_no_audio_or_extra_hidden_updates():
    calls = []

    class ImageModel:
        config = SimpleNamespace(image_only=True)
        audio_backbone = None

        def eval(self):
            return self

        def reset_state(self):
            calls.append("reset")

        # Keyword-only signature rejects any audio supplied by the caller.
        def step(self, *, image, frame_id, reset):
            calls.append((frame_id, reset, tuple(image.shape)))
            return {"position": torch.ones(1, 3), "wrench": torch.ones(1, 6), "slip": torch.ones(1, 2)}

    rows = [{"image": str(i // 10), "reset_state": i == 0} for i in range(290)]
    inputs = {str(i): np.zeros((1, 3, 224, 224), np.float32) for i in range(29)}
    stats = {"target_mean": [2.] * 11, "target_std": [3.] * 11}
    baseline = infer_image_only(ImageModel(), stats, torch.device("cpu"), rows, inputs, 150.)
    np.testing.assert_array_equal(baseline.event_indices, np.arange(0, 290, 10))
    np.testing.assert_allclose(np.diff(baseline.times), 1 / 15)
    assert len(calls) == 30  # One reset, exactly 29 recurrent updates.
    assert calls[1] == ("0", True, (1, 3, 224, 224))
    assert calls[-1] == ("28", False, (1, 3, 224, 224))
    np.testing.assert_array_equal(baseline.prediction, np.full((29, 11), 5.))
    # An AV checkpoint cannot silently serve as the camera-only comparison.
    model = ImageModel()
    model.config = SimpleNamespace(image_only=False)
    with pytest.raises(ValueError, match="image_only=True"):
        infer_image_only(model, stats, torch.device("cpu"), rows, inputs, 150.)


def fixture_data():
    n = 30
    values = np.zeros((n, 11), np.float32)
    values[:, 0] = np.linspace(-5, 15, n)
    values[:, 1] = np.sin(np.linspace(0, 3, n))
    values[:, 5] = 1.
    metadata = [{"sequence_index": i, "event_pc_ns": i * 6666667, "image": "a", "reset_state": i == 0} for i in range(n)]
    baseline = ImageOnlyReplay(np.arange(0, n, 10), np.arange(3) / 15, values[::10] + .1,
        np.ones(3, bool), np.ones(3))
    return ReplayData(metadata, {"a": np.full((224, 224, 3), 150, np.uint8)},
        np.full((n, 6, 64, 28), -6., np.float32), values, values + .02,
        np.ones(n, bool), np.ones(n, bool), np.full(n, np.nan), np.ones(n),
        {"random_weights": True, "sequence": "SYNTHETIC_UI_TEST", "image_updates": 1, "device": "cpu",
         "precompute_seconds": 1., "disabled_mic_channels": [5], "step_mean_ms": 1., "step_p95_ms": 1., "step_over_budget": 0}, baseline)


def test_export_preserves_all_events_and_validity(tmp_path):
    data = fixture_data()
    save_results(data, tmp_path)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["prediction"], data.prediction)
        np.testing.assert_array_equal(z["position_valid"], data.position_valid)
        np.testing.assert_array_equal(z["image_only_prediction"], data.image_only.prediction)
        np.testing.assert_array_equal(z["image_only_event_indices"], [0, 10, 20])
    assert len((tmp_path / "predictions.csv").read_text(encoding="utf-8-sig").splitlines()) == 31
    assert len((tmp_path / "image_only_predictions.csv").read_text(encoding="utf-8-sig").splitlines()) == 4


def test_gui_replay_seek_freeze_and_all_elapsed_points(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    from deployment.deployment_viewer import create_window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args = SimpleNamespace(model="model_contact_gated.py", hz=150., render_fps=60,
        data_root=Path("test"), output=tmp_path, screenshot=None)
    window = create_window(args, QtCore, QtGui, QtWidgets, pg)
    window.show()
    window.loaded(fixture_data())
    window.pause()
    window.seek_to(10)
    assert len(window.wrench_lines[0][0].getData()[0]) == 11
    assert len(window.xy_pred.scatter.points()) == 11
    assert len(window.xy_true.scatter.points()) == 11
    assert window.xy_pred.opts["symbol"] == "o"
    assert window.xy_true.opts["symbol"] == "x"
    assert len(window.xy_image_only.getData()[0]) == 2
    assert len(window.wrench_image_only[0].getData()[0]) == 2
    window.seek_to(9)
    assert len(window.xy_image_only.getData()[0]) == 1  # No future 15 Hz samples shown.
    window.seek_to(10)
    # User zoom must survive the next playback/seek update.
    view = window.xy.getViewBox()
    view.scaleBy((.5, .5))
    zoomed = np.asarray(view.viewRange())
    window.seek_to(11)
    np.testing.assert_allclose(view.viewRange(), zoomed)
    assert len(window.xy_pred.scatter.points()) == 12
    window.fit_xy.click()
    assert np.ptp(view.viewRange()[0]) < np.ptp(zoomed[0])
    window.clear_trail()
    assert len(window.xy_pred.getData()[0]) == 1
    window.seek_to(2)
    assert window.trail_start == 2
    window.freeze.setChecked(True)
    assert not window.playing
    window.restart()
    window.anchor_time -= 1.
    window.tick()
    assert window.current == 29
    assert len(window.wrench_lines[0][0].getData()[0]) == 30
    assert len(window.xy_image_only.getData()[0]) == 3
    window.show_image_only.setChecked(False)
    assert not window.xy_image_only.isVisible()
    window.show_image_only.setChecked(True)
    assert window.xy_image_only.isVisible()
    assert not window.playing
    app.processEvents()
    assert window.grab().save(str(tmp_path / "ui.png"))
    window.close()


@pytest.mark.parametrize("image_shape", [(224, 224), (160, 320)])
def test_camera_resizes_while_paused_and_after_playback(tmp_path, image_shape):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
    from deployment.deployment_viewer import create_window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args = SimpleNamespace(model="model_triple_head", hz=150., render_fps=60,
                           data_root=Path("test"), output=tmp_path, screenshot=None)
    data = fixture_data()
    data.images["a"] = np.full((*image_shape, 3), 150, np.uint8)
    window = create_window(args, QtCore, QtGui, QtWidgets, pg)
    window.show()
    window.loaded(data)
    window.pause()
    app.processEvents()
    original_key = window.camera_pixmap.cacheKey()
    initial_height = window.camera.pixmap().height()

    def check_fit():
        app.processEvents()
        area, shown = window.camera.contentsRect(), window.camera.pixmap()
        height, width = image_shape
        scale = min(area.width() / width, area.height() / height)
        assert abs(shown.width() - width * scale) <= 1
        assert abs(shown.height() - height * scale) <= 1
        assert window.camera_pixmap.cacheKey() == original_key
        assert not window.playing

    window.resize(2200, 1400)
    check_fit()
    assert window.current == 0 and window.camera.pixmap().height() > initial_height
    enlarged_height = window.camera.pixmap().height()
    window.resize(1200, 800)
    check_fit()
    assert window.camera.pixmap().height() < enlarged_height
    window.seek_to(len(data.metadata) - 1)
    window.resize(1800, 1200)
    check_fit()
    assert window.current == len(data.metadata) - 1
    window.close()


def test_gui_switch_data_directory_and_recover_after_failure(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
    import pyqtgraph as pg
    import deployment.deployment_viewer as viewer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = ROOT / "deployment/test_data/deployment_test_150hz"
    second = ROOT / "deployment/test_data/deployment_test_150hz_t0232"
    calls, saved, errors = [], [], []

    def precompute(run_args, progress):
        calls.append((run_args.data_root, run_args.sequence, run_args.output))
        if len(calls) == 3:
            raise ValueError("test inference failure")
        data = fixture_data()
        data.images["a"][:] = 30 * len(calls)
        return data

    monkeypatch.setattr(viewer, "precompute", precompute)
    monkeypatch.setattr(viewer, "save_results", lambda data, path: saved.append(path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(second))
    args = SimpleNamespace(model="model_triple_head", hz=150., render_fps=60,
        data_root=first, sequence="slide_t0242", output=tmp_path, screenshot=None)
    window = viewer.create_window(args, QtCore, QtGui, QtWidgets, pg)

    def finish():
        assert window.worker.wait(5000)
        for _ in range(5):
            app.processEvents()
            QtTest.QTest.qWait(1)
        assert window.load_data.isEnabled()
        assert window.play.isEnabled()
        window.pause()

    window.start_precompute()
    assert not window.load_data.isEnabled()
    assert not window.play.isEnabled()
    finish()
    assert calls[0] == (first.resolve(), "slide_t0242", tmp_path)
    assert window.camera_pixmap.toImage().pixelColor(0, 0).red() == 30
    window.browse_data.click()
    window.load_data.click()
    finish()
    assert calls[1][0:2] == (second.resolve(), "slide_t0232")
    assert saved[1].parent == tmp_path and saved[1] != saved[0]
    assert window.camera_pixmap.toImage().pixelColor(0, 0).red() == 60
    window.load_data.click()
    finish()
    assert "test inference failure" in errors[-1]
    window.data_root_edit.setText(str(tmp_path / "missing"))
    window.load_data.click()
    assert len(calls) == 3
    assert "manifest.json" in errors[-1]
    window.close()


@pytest.fixture
def prepared_truth_cache(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    Image.new("RGB", (224, 224), (120, 90, 60)).save(root / "image.png")
    records = []
    for number, sequence in enumerate(("sourceA/t0001", "sourceB/t0001", "sourceA/t0002")):
        for order in (2, 4):  # A gap must not become an invented continuous GT segment.
            target = np.zeros((10, 11))
            target[:, 0] = np.arange(10) * .1 + number + order
            target[:, 1] = number
            target[:, 5] = 1.
            target[:, 9] = .1
            file = f"{number}_{order}.npz"
            times = np.arange(10, dtype=np.int64) * 6_666_667 + order * 100_000_000
            np.savez(root / file, audio=np.full((10, 6, 64, 28), -6., np.float32), target=target,
                     event_pc_ns=times, target_valid=np.ones((10, 11), dtype=bool))
            records.append({"kind": "slide", "sequence": sequence, "order": order, "events": 10,
                            "file": file, "image": "image.png", "target": target.tolist()})
    (root / "manifest.json").write_text(json.dumps({"version": 5, "records": records}), encoding="utf-8")
    return root


def test_ground_truth_parser_and_sequence_disambiguation(prepared_truth_cache):
    from deployment.deployment_viewer import parser
    args = parser().parse_args(["--ground-truth-only", "--data-root", str(prepared_truth_cache)])
    assert args.model is None and args.ground_truth_only and args.output is None
    with pytest.raises(SystemExit):
        parser().parse_args(["--ground-truth-only", "--model", "model_triple_head"])
    assert list_sequences(prepared_truth_cache) == ["sourceA/t0001", "sourceB/t0001", "sourceA/t0002"]
    with pytest.raises(ValueError, match="Ambiguous trajectory ID"):
        select_sequence(prepared_truth_cache, "t0001")
    rows = select_sequence(prepared_truth_cache, "t0002")
    assert len(rows) == 20 and rows[0]["sequence"] == "sourceA/t0002"
    assert [i for i, row in enumerate(rows) if row["reset_state"]] == [0, 10]


@pytest.fixture
def prepared_trim_cache(prepared_truth_cache):
    root = prepared_truth_cache
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    added = []
    for record in manifest["records"]:
        with np.load(root / record["file"]) as z:
            arrays = {key: z[key].copy() for key in z.files}
        if record["order"] == 2:
            # Returning motion must not be mistaken for a stationary window.
            arrays["target"][:, 0] = [0., .05, .1, .05, 0., 0., 0., 0., 0., 0.]
            if record["sequence"] == "sourceA/t0002":
                arrays["target"][:, 0] *= .01
        else:
            # Exact equality stays; a smaller window cuts off a later rebound too.
            diameter = .03 if record["sequence"] == "sourceA/t0001" else .01
            arrays["target"][:, 0] = 0.
            arrays["target"][1, 0] = diameter
            later = {key: value.copy() for key, value in arrays.items()}
            later["target"][:, 0] = np.linspace(0., .5, 10)
            later["event_pc_ns"] += 200_000_000
            filename = record["file"].replace("_4.npz", "_6.npz")
            np.savez(root / filename, **later)
            added.append({**record, "file": filename, "order": 6, "target": later["target"].tolist()})
        record["target"] = arrays["target"].tolist()
        np.savez(root / record["file"], **arrays)
    manifest["records"].extend(added)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_truth_trim_matches_training_and_preserves_cache(prepared_trim_cache):
    from mvts.prepared_selection import trim_prepared_records
    root = prepared_trim_cache
    original = (root / "manifest.json").read_bytes()
    manifest = json.loads(original)
    for sequence, expected_events in (("sourceA/t0001", 30), ("sourceB/t0001", 10)):
        records = [{**r, "id": f"{r['sequence']}/{r['order']}"} for r in manifest["records"]
                   if r["sequence"] == sequence]
        selected, report = trim_prepared_records(records, .03)
        data = precompute(SimpleNamespace(data_root=root, sequence=sequence, hz=150.,
                                         ground_truth_only=True, trim_stationary_xy_mm=.03))
        assert len(data.metadata) == expected_events == report["retained_events"]
        assert data.prediction is None
        assert data.summary["trim_selection"]["removed_events"] == report["removed_events"]
        expected = np.concatenate([r["target"] for r in sorted(selected, key=lambda r: r["order"])])
        np.testing.assert_array_equal(data.target, expected)
        assert all(row["sequence_index"] == i for i, row in enumerate(data.metadata))
    assert (root / "manifest.json").read_bytes() == original
    assert len(list(root.glob("*.npz"))) == 9
    with pytest.raises(ValueError, match="No ground-truth windows remain"):
        precompute(SimpleNamespace(data_root=root, sequence="sourceA/t0002", hz=150.,
                                   ground_truth_only=True, trim_stationary_xy_mm=.03))
    full = precompute(SimpleNamespace(data_root=root, sequence="sourceB/t0001", hz=150.,
                                     ground_truth_only=True, trim_stationary_xy_mm=0.))
    assert len(full.metadata) == 30 and not full.summary["trim_selection"]["enabled"]


@pytest.fixture
def inference_trim_package(tmp_path):
    root = tmp_path / "exported"
    root.mkdir()
    Image.new("RGB", (224, 224), "white").save(root / "image.png")
    rows = []
    for window in range(4):
        # Returning motion, exact boundary, stationary, then a moving rebound.
        x = ([0., .1, .2, .1, 0., 0., 0., 0., 0., 0.] if window == 0 else
             [0., (.029, .01, .5)[window - 1]] + [0.] * 8)
        for event in range(10):
            index = window * 10 + event
            target = np.zeros(11, np.float64)
            target[0] = x[event]
            target[2] = event * 100.  # Z motion must not affect the cutoff.
            filename = f"{index}.npz"
            np.savez(root / filename, audio=np.full((6, 64, 28), index, np.float32), target=target)
            rows.append(dict(kind="slide", sequence="test", sequence_index=index,
                             event_pc_ns=index * 6666667, data=filename, image="image.png",
                             reset_state=index == 0, image_update=event == 0, source_window_order=window))
    (root / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    return root


@pytest.mark.parametrize("threshold, expected", [(0., 40), (.029, 20), (.03, 10)])
def test_inference_trims_whole_windows_before_both_streams(inference_trim_package, monkeypatch, threshold, expected):
    import deployment.replay as replay
    from deployment.deployment_viewer import parser
    root = inference_trim_package
    original = {p.name: p.read_bytes() for p in root.iterdir()}
    args = parser().parse_args(["--model", "model_triple_head", "--data-root", str(root),
                                "--trim-stationary-xy-mm", str(threshold)])
    stats = {"audio_mean": [0.] * 6, "audio_std": [1.] * 6}
    model = SimpleNamespace(config=SimpleNamespace(disabled_mic_channels=[]))
    monkeypatch.setattr(replay, "prepare_model", lambda *a: (model, stats, torch.device("cpu"), {}))
    streams = []

    def infer(model, stats, device, rows, audio, inputs, progress):
        streams.append("main")
        assert len(rows) == len(audio) == expected
        assert sum(r["image_update"] for r in rows) == expected // 10
        np.testing.assert_array_equal([mel[0, 0, 0] for mel in audio], np.arange(expected))
        return (np.zeros((expected, 11)), np.ones(expected, bool), np.ones(expected, bool),
                np.full(expected, np.nan), np.ones(expected))

    def comparison(args, device, rows, inputs, target, prediction, valid, progress):
        streams.append("image-only")
        assert len(rows) == len(target) == len(prediction) == expected
        return None, {"enabled": True}

    monkeypatch.setattr(replay, "infer_events", infer)
    monkeypatch.setattr(replay, "precompute_image_only", comparison)
    data = precompute(args)
    assert streams == ["main", "image-only"]
    assert len(data.metadata) == len(data.mel) == len(data.target) == expected
    assert data.summary["trim_selection"]["source_windows"] == 4
    assert data.summary["trim_selection"]["removed_events"] == 40 - expected
    assert data.summary["trim_selection"]["enabled"] == bool(threshold)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == original


def test_inference_all_windows_trimmed_does_not_load_model(inference_trim_package, monkeypatch):
    import deployment.replay as replay
    from deployment.deployment_viewer import parser
    args = parser().parse_args(["--model", "model_triple_head", "--data-root", str(inference_trim_package),
                                "--trim-stationary-xy-mm", "1"])
    monkeypatch.setattr(replay, "prepare_model", lambda *a: pytest.fail("Empty selection must not load a model"))
    with pytest.raises(ValueError, match="No inference windows remain"):
        precompute(args)


@pytest.fixture
def prepared_xy_cache(prepared_trim_cache):
    root = prepared_trim_cache
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in manifest["records"]:
        if record["sequence"] == "sourceB/t0001" and record["order"] == 6:
            with np.load(root / record["file"]) as z:
                arrays = {key: z[key].copy() for key in z.files}
            arrays["target"][-1, 1] = 10.0001
            record["target"] = arrays["target"].tolist()
            np.savez(root / record["file"], **arrays)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_viewer_range_filters_complete_trajectory_before_tail(prepared_xy_cache, monkeypatch):
    import deployment.replay as replay
    from deployment.deployment_viewer import parser
    args = parser().parse_args(["--ground-truth-only", "--filter-xy-range"])
    assert args.filter_xy_range
    root = prepared_xy_cache
    before = (root / "manifest.json").read_bytes()
    assert len(list_sequences(root)) == 3
    assert list_sequences(root, True) == ["sourceA/t0001", "sourceA/t0002"]
    monkeypatch.setattr(replay, "prepare_model", lambda *a: pytest.fail("No inference in truth mode"))
    args = SimpleNamespace(data_root=root, sequence="sourceB/t0001", ground_truth_only=True,
                           hz=150., filter_xy_range=True, trim_stationary_xy_mm=.03)
    # The only outside point is in the tail that would otherwise be trimmed away.
    with pytest.raises(ValueError, match="Entire trajectory.*excluded"):
        precompute(args)
    args.filter_xy_range = False
    assert len(precompute(args).metadata) == 10
    args.filter_xy_range = True
    args.sequence = None
    data = precompute(args)
    assert data.summary["sequence"] == "sourceA/t0001" and data.summary["xy_range_selection"]["enabled"]
    assert data.prediction is None and (root / "manifest.json").read_bytes() == before


def test_viewer_range_catalog_and_switching_ids(prepared_xy_cache):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
    import pyqtgraph as pg
    from deployment.deployment_viewer import create_window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args = SimpleNamespace(model=None, ground_truth_only=True, hz=150., render_fps=60,
                           data_root=prepared_xy_cache, sequence=None, output=None, screenshot=None,
                           filter_xy_range=True, trim_stationary_xy_mm=0.)
    window = create_window(args, QtCore, QtGui, QtWidgets, pg)

    def finish():
        assert window.worker.wait(5000)
        for _ in range(5):
            app.processEvents()
            QtTest.QTest.qWait(1)
        window.pause()

    window.start_precompute()
    finish()
    assert window.sequence_box.count() == 2
    assert "sourceB/t0001" not in [window.sequence_box.itemText(i) for i in range(2)]
    assert window.data.summary["sequence"] == "sourceA/t0001"
    window.sequence_box.setCurrentIndex(1)
    window.sequence_box.activated.emit(1)
    finish()
    assert window.data.summary["sequence"] == "sourceA/t0002"
    assert window.data.summary["xy_range_selection"]["enabled"]
    window.close()


def test_ground_truth_reads_physical_labels_without_any_model(prepared_truth_cache, tmp_path, monkeypatch):
    import deployment.replay as replay

    def forbidden(*args, **kwargs):
        pytest.fail("Truth viewing must never build/load a model or run inference")

    for name in ("prepare_model", "load_checkpoint", "build_model", "infer_events", "precompute_image_only"):
        monkeypatch.setattr(replay, name, forbidden)
    args = SimpleNamespace(data_root=prepared_truth_cache, sequence="sourceB/t0001", ground_truth_only=True, hz=150.)
    data = precompute(args)
    assert data.prediction is None and data.image_only is None and data.latency_ms.size == 0
    assert data.summary["model"] is None and data.summary["checkpoint"] is None
    assert data.target_valid.all()
    assert [r["sequence_index"] for r in data.metadata] == list(range(20))
    for index, order in ((0, 2), (10, 4)):
        with np.load(prepared_truth_cache / f"1_{order}.npz") as z:
            np.testing.assert_array_equal(data.target[index:index + 10], z["target"])
            np.testing.assert_array_equal(data.mel[index:index + 10], z["audio"])
            assert data.metadata[index]["event_pc_ns"] == z["event_pc_ns"][0]
    output = tmp_path / "export"
    save_results(data, output)
    assert not (output / "predictions.npz").exists() and not (output / "predictions.csv").exists()
    with np.load(output / "ground_truth.npz") as z:
        np.testing.assert_array_equal(z["target"], data.target)
        assert "prediction" not in z.files


@pytest.mark.parametrize("trim_threshold", [0., .03])
def test_ground_truth_gui_switch_trajectory_without_export(prepared_truth_cache, tmp_path, monkeypatch, trim_threshold):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
    import pyqtgraph as pg
    from deployment.deployment_viewer import create_window
    import deployment.replay as replay

    def forbidden(*args, **kwargs):
        pytest.fail("GUI truth mode must not prepare a model")

    monkeypatch.setattr(replay, "prepare_model", forbidden)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args = SimpleNamespace(model=None, ground_truth_only=True, hz=150., render_fps=60,
                           data_root=prepared_truth_cache, sequence=None, output=None, screenshot=None,
                           trim_stationary_xy_mm=trim_threshold)
    window = create_window(args, QtCore, QtGui, QtWidgets, pg)
    window.show()

    def finish():
        assert window.worker.wait(5000)
        for _ in range(5):
            app.processEvents()
            QtTest.QTest.qWait(1)
        assert window.load_data.isEnabled() and window.sequence_box.isEnabled()
        window.pause()

    window.start_precompute()
    assert not window.sequence_box.isEnabled()
    finish()
    assert window.sequence_box.count() == 3
    assert window.data.summary["sequence"] == "sourceA/t0001"
    window.seek_to(19)
    assert len(window.xy_true.getData()[0]) == 20
    assert not window.xy_true.opts["connect"][9]  # Do not draw through the gap.
    assert not window.xy_pred.isVisible()
    assert not window.position_lines[0][0].isVisible()
    assert not window.wrench_lines[0][0].isVisible()
    assert "RMSE" not in window.metrics.text() and "step" not in window.status.text()
    window.sequence_box.setCurrentIndex(1)
    window.sequence_box.activated.emit(1)
    finish()
    assert window.data.summary["sequence"] == "sourceB/t0001"
    assert window.data.target[0, 1] == 1.
    assert window.data.summary["trim_selection"]["threshold_mm"] == trim_threshold
    assert args.output is None and not (prepared_truth_cache / "predictions.csv").exists()
    window.seek_to(19)
    app.processEvents()
    assert window.grab().save(str(tmp_path / "truth_ui.png"))
    window.close()


def test_truth_gui_applies_trimming_when_switching_ids(prepared_trim_cache, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
    import pyqtgraph as pg
    from deployment.deployment_viewer import create_window
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    args = SimpleNamespace(model=None, ground_truth_only=True, hz=150., render_fps=60,
                           data_root=prepared_trim_cache, sequence=None, output=None, screenshot=None,
                           trim_stationary_xy_mm=.03)
    window = create_window(args, QtCore, QtGui, QtWidgets, pg)

    def finish():
        assert window.worker.wait(5000)
        for _ in range(5):
            app.processEvents()
            QtTest.QTest.qWait(1)
        window.pause()

    window.start_precompute()
    finish()
    assert len(window.data.metadata) == 30
    window.sequence_box.setCurrentText("sourceB/t0001")
    window.sequence_box.activated.emit(1)
    finish()
    assert len(window.data.metadata) == 10
    assert window.seek.maximum() == 9
    assert "1/3" in window.details.text() and "20" in window.details.text()
    window.seek_to(9)
    assert len(window.xy_true.getData()[0]) == 10
    window.close()
