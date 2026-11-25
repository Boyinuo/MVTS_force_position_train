"""Automatic external tests must use best.pt, preserve complete shapes and label masks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from deployment import demo_evaluation as demo
from mvts.dual_rate_data import feature_config


def stats():
    return {"target_mean": [0.] * 11, "target_std": [1.] * 11,
            "audio_mean": [0.] * 6, "audio_std": [1.] * 6, "contact_class_weights": [2., 3.]}


def test_auxiliary_loss_masks_and_training_weights():
    normalization = stats()
    normalization["target_mean"][3:9] = [2.] * 6
    normalization["target_std"][3:9] = [4.] * 6
    target = np.asarray([normalization["target_mean"]] * 2)
    target[0] = 1e6  # Background measurements must not enter regression losses.
    valid = np.ones((2, 11), bool)
    valid[0] = False
    raw = {"position": np.zeros((2, 3)), "wrench": np.zeros((2, 6)),
           "slip": np.zeros((2, 2)), "slip_prior": np.full((2, 2), 2.)}
    raw["wrench"][0] = -.5  # Denormalizes to physical zero.
    incoming = {"target": target, "valid": valid, "contact": np.array([0, 1]),
                "contact_valid": np.ones(2, bool)}
    result = demo.compute_losses(raw, incoming, normalization, {"slip_prior_weight": .2})
    assert result["position_loss"] == result["wrench_loss"] == result["slip_loss"] == 0
    assert result["zero_wrench_loss"] == result["main_loss"] == 0
    assert result["slip_prior_loss"] == pytest.approx(1.5)
    assert result["loss"] == pytest.approx(.3)
    raw["contact_logits"] = np.zeros(2)
    contact = demo.compute_losses(raw, incoming, normalization,
                                  {"slip_prior_weight": .2, "contact_weight": 2.})
    assert contact["contact_loss"] == pytest.approx(np.log(2.) * 2.5)
    assert contact["loss"] == pytest.approx(.3 + np.log(2.) * 5.)


def make_packages(root):
    for name in demo.DATASET_NAMES:
        folder = root / name
        (folder / "events").mkdir(parents=True)
        (folder / "images").mkdir()
        image = "images/frame.png"
        Image.new("RGB", (40, 40), (60, 90, 120)).save(folder / image)
        records, rows, sequences = [], [], []
        for seq_index in range(2):
            sequence = f"external/{name}/t{seq_index+1:04d}"
            target = np.zeros((2, 11))
            target[:, :2] = [[0., -8.2], [1., -7.]]  # Full shape survives default policy.
            files = []
            for t in range(2):
                filename = f"events/{seq_index}_{t}.npz"
                stamp = 1_000_000_000 + (seq_index * 10 + t) * 6_000_000
                np.savez(folder / filename, audio=np.zeros((6, 64, 28), np.float32), target=target[t],
                         target_valid=np.ones(11, bool), event_pc_ns=np.int64(stamp),
                         previous_pc_ns=np.int64(stamp - 6_000_000), contact=np.int8(1), contact_valid=True)
                files.append(filename)
                rows.append({"sequence": sequence, "source_window_id": sequence + "/0002",
                    "source_sub_event": t, "sequence_index": t, "call_index": len(rows),
                    "data": filename, "image": image, "event_pc_ns": stamp,
                    "image_update": t == 0, "reset_state": t == 0})
            sequences.append({"sequence": sequence})
            records.append({"id": sequence + "/0002", "sequence": sequence, "kind": "slide",
                            "order": 2, "events": 2, "target": target.tolist(), "image": image, "event_files": files})
        (folder / "manifest.json").write_text(json.dumps({"shape": name.removeprefix("demo_"),
            "features": feature_config().to_dict(), "records": records, "sequences": sequences}), encoding="utf-8")
        (folder / "events.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


@pytest.mark.parametrize("image_only", [False, True])
def test_external_suite_exports_all_three_and_never_feeds_labels(tmp_path, monkeypatch, capsys, image_only):
    root, output = tmp_path / "inputs", tmp_path / "run"
    make_packages(root)
    output.mkdir()
    checkpoint = output / "best.pt"
    checkpoint.write_bytes(b"best-checkpoint-fixture")
    calls = []

    class Estimator:
        model_name = "model_triple_head"
        config = SimpleNamespace(image_only=image_only, disabled_mic_channels=())

        def eval(self):
            return self

        def reset_state(self):
            self.cached = False

        def step(self, audio=None, *, image, frame_id, reset):
            if reset:
                self.reset_state()
            if image is not None:
                self.cached = True
            assert self.cached and (audio is None) == image_only
            calls.append({"reset": reset, "image_update": image is not None})
            return {"position": torch.zeros(1, 3), "wrench": torch.zeros(1, 6), "slip": torch.zeros(1, 2)}

    saved = {"stats": stats(), "args": {}, "epoch": 7, "split_ids": {"train": []}}
    monkeypatch.setattr(demo, "load_checkpoint", lambda *args: (Estimator(), saved, feature_config()))
    report = demo.evaluate_demo_suite(checkpoint, output, data_root=root, device="cpu")
    assert report["status"] == "complete" and set(report["datasets"]) == set(demo.DATASET_NAMES)
    assert len(calls) == 12 and sum(c["reset"] for c in calls) == 6
    for name in demo.DATASET_NAMES:
        assert (output / f"{name}.png").stat().st_size > 1000
        svg = (output / f"{name}.svg").read_text(encoding="utf-8")
        assert "<text" in svg
        metrics = json.loads((output / "demo_tests" / name / "metrics.json").read_text())
        assert metrics["events"] == 4 and metrics["best_epoch"] == 7
        assert metrics["mae"]["y_mm"] == pytest.approx(7.6)
        assert metrics["loss"] == pytest.approx(metrics["main_loss"])
        with np.load(output / "demo_tests" / name / "predictions.npz") as z:
            assert z["prediction"].shape == (4, 11)
    terminal = capsys.readouterr().out
    assert all(f"[{name}] best_epoch=7" in terminal for name in demo.DATASET_NAMES)
    assert "loss=" in terminal and "position_mae" in terminal and "slip_mae" in terminal
    assert checkpoint.read_bytes() == b"best-checkpoint-fixture"
    with pytest.raises(ValueError, match="selection is empty"):
        demo.preflight(root, 0., True)


@pytest.mark.parametrize("options,expected", [([], True), (["--no-demo-tests"], False), (["--audit-only"], False)])
def test_train_calls_suite_after_success_using_best_pt(tmp_path, monkeypatch, options, expected):
    import train
    import mvts.dual_rate_engine
    events = []
    monkeypatch.setattr(demo, "preflight", lambda *args: events.append("preflight"))
    monkeypatch.setattr(mvts.dual_rate_engine, "train_prepared", lambda args: events.append("trained"))
    def evaluate(checkpoint, output, **kwargs):
        assert Path(checkpoint) == tmp_path / "best.pt"
        assert kwargs["trim_mm"] == 0 and not kwargs["filter_xy"]
        events.append("external")
    monkeypatch.setattr(demo, "evaluate_demo_suite", evaluate)
    monkeypatch.setattr("sys.argv", ["train.py", "--model", "model_slip_gated", "--output", str(tmp_path), *options])
    train.main()
    assert events == (["preflight", "trained", "external"] if expected else ["trained"])
