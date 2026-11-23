"""Regression checks for causal inference, selection and playback (no trained weights needed)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deployment.replay import (ROOT, ImageOnlyReplay, ReplayData, default_checkpoint, infer_events,
                               infer_image_only, model_name, save_results, select_sequence)


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
