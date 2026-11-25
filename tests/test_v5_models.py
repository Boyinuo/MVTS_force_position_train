"""Both estimators share v5 inputs, masks and physical background supervision."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from mvts.contact_data import (CONTACT_POLICY, CONTACT_SCHEMA, ContactPreparedDataset,
                               add_contact_supervision, fit_contact_statistics,
                               read_contact_prepared)
from mvts.dual_rate_data import (PreparedDataset, feature_config, fit_prepared_statistics,
                                 read_prepared, validate_prepared_partition)
from mvts.dual_rate_engine import run_sequence_epoch
from mvts.engine import save_predictions
from mvts.prepared_selection import (load_prepared_caches, prepare_training_data,
                                      trim_prepared_records)


@pytest.fixture
def v5_cache(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    Image.new("RGB", (64, 64), (100, 120, 140)).save(root / "image.png")
    records = []
    for i in range(3):
        for kind, event_kind, order in (("press", "background", 0), ("press", "target", 0),
                                        ("slide", None, 0), ("slide", None, 1)):
            n = 10 if kind == "slide" else 1
            target = np.full((n, 11), float(i + 1))
            if event_kind == "background":
                target[:] = -1000.  # Measurements must never become position/slip supervision.
            arrays, metadata = add_contact_supervision(
                {"target": target, "audio": np.full((n, 6, 64, 28), float(i), dtype=np.float32)},
                kind, event_kind, previous_available=order > 0)
            identifier = f"{kind}_{i}_{event_kind}_{order}"
            np.savez(root / f"{identifier}.npz", **arrays)
            records.append({"id": identifier, "file": f"{identifier}.npz", "image": "image.png",
                            "kind": kind, "events": n, "target": target.tolist(), "order": order,
                            "sequence": f"slide_{i}" if kind == "slide" else identifier,
                            "group": f"{kind}_{i}", **metadata})
    manifest = {"version": 5, "schema": CONTACT_SCHEMA, "contact_policy": CONTACT_POLICY,
                "features": feature_config().to_dict(), "smoke_test": True, "records": records}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "report.json").write_text("{}", encoding="utf-8")
    return root


def test_v5_inputs_and_normalization_match_between_models(v5_cache):
    triple, manifest = read_prepared(v5_cache)
    contact, _ = read_contact_prepared(v5_cache)
    assert manifest["version"] == 5 and triple == contact
    stats = fit_prepared_statistics(triple)
    reference = fit_contact_statistics(contact)
    for key in ("target_mean", "target_std", "audio_mean", "audio_std", "target_fit_counts", "fit_sample_count"):
        np.testing.assert_allclose(stats[key], reference[key])
    assert stats["target_fit_counts"] == [63] * 11
    assert stats["fit_sample_count"] == 66
    assert stats["target_mean"] == [2.] * 11
    for i in (0, 1, 2, 3):  # Background, loaded press, and two consecutive slide windows.
        a, b = PreparedDataset(triple, stats)[i], ContactPreparedDataset(contact, stats)[i]
        for key in ("image", "audio", "target", "target_valid", "contact", "contact_valid"):
            torch.testing.assert_close(a[key], b[key])
    background = [r for r in triple if r["contact_state"] == 0]
    with pytest.raises(ValueError, match="valid regression labels"):
        fit_prepared_statistics(background)


def test_v5_schema_and_array_supervision_are_validated(v5_cache):
    path = v5_cache / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**manifest, "schema": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a v5 contact cache"):
        read_prepared(v5_cache)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    records, _ = read_prepared(v5_cache)
    record = records[0]
    with np.load(v5_cache / record["file"]) as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["target_valid"][:] = True
    np.savez(v5_cache / record["file"], **arrays)
    with pytest.raises(ValueError, match="supervision changed"):
        PreparedDataset(records, {"audio_mean": [0.] * 6, "audio_std": [1.] * 6,
                                  "target_mean": [0.] * 11, "target_std": [1.] * 11})[0]


def test_triple_background_gradients_metrics_and_csv(tmp_path):
    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=1, image_only=False)
            self.prediction = torch.nn.Parameter(torch.zeros(1, 1, 11))

        def forward_sequence(self, image, audio, hidden, *, sequence_length):
            return {"position": self.prediction[..., :3], "wrench": self.prediction[..., 3:9],
                    "slip": self.prediction[..., 9:], "hidden": hidden}

    model = ConstantModel()
    stats = {"target_mean": list(range(1, 12)), "target_std": [1.] * 11}
    batch = {"image": torch.zeros(1, 3, 32, 32), "audio": torch.zeros(1, 1, 6, 64, 28),
             "target": torch.full((1, 1, 11), -1000.), "target_valid": torch.zeros(1, 1, 11, dtype=torch.bool),
             "contact": torch.zeros(1, 1, dtype=torch.int8), "contact_valid": torch.ones(1, 1, dtype=torch.bool),
             "sequence": ["background"], "id": ["background"], "order": torch.tensor([0])}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.)
    metrics, values = run_sequence_epoch(model, [batch], torch.device("cpu"), stats, optimizer,
                                         zero_wrench_weight=2.5)
    grad = model.prediction.grad[0, 0]
    assert not grad[:3].any() and not grad[9:].any()
    assert (grad[3:9] > 0).all()  # Descent moves physical force toward zero, not measured -1000.
    assert metrics["position_loss"] == metrics["wrench_loss"] == metrics["slip_loss"] == 0.
    assert metrics["zero_wrench_loss"] > 0
    assert metrics["loss"] == pytest.approx(2.5 * metrics["zero_wrench_loss"])
    assert metrics["no_contact_samples"] == 1
    assert metrics["mae"]["x_mm"] is None and metrics["mae"]["slip_x_mm"] is None
    assert metrics["mae"]["Fx"] == 4. and metrics["raw_mae"]["Fx"] is None
    assert metrics["no_contact_wrench_mae"]["Fx"] == 4.
    path = tmp_path / "predictions.csv"
    save_predictions(path, values, target_names=list(metrics["mae"]))
    with path.open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["label_valid_x_mm"] == row["label_valid_Fx"] == "False"
    assert float(row["true_x_mm"]) == -999.  # Raw measurement remains available, flagged invalid.


def test_checkpoint_partition_checks_v5_masks_and_legacy_v4(v5_cache):
    records, _ = read_prepared(v5_cache)
    checkpoint = {"split_ids": {"test": [r["id"] for r in records]},
                  "split_labels": {"test": [r["target"] for r in records]},
                  "split_supervision": {"test": [r["supervision_sha256"] for r in records]}}
    validate_prepared_partition(records, checkpoint, "test")
    changed = [{**r, "supervision_sha256": "changed"} for r in records]
    with pytest.raises(ValueError, match="supervision changed"):
        validate_prepared_partition(changed, checkpoint, "test")
    checkpoint.pop("split_supervision")
    with pytest.raises(ValueError, match="supervision changed"):
        validate_prepared_partition(records, checkpoint, "test")
    legacy = [{k: v for k, v in r.items() if k != "supervision_sha256"} for r in records]
    validate_prepared_partition(legacy, checkpoint, "test")


@pytest.mark.parametrize("model_name", ["model_triple_head", "model_contact_gated"])
def test_v5_train_resume_evaluate_and_predict(v5_cache, tmp_path, model_name):
    root = Path(__file__).resolve().parents[1]

    def run(*args, error=None):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=root, text=True,
                                capture_output=True, timeout=120)
        if error:
            assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr

    output = tmp_path / "run"
    base = ["train.py", "--no-demo-tests", "--model", model_name, "--prepared-root", v5_cache, "--output", output,
            "--device", "cpu", "--threads", 2, "--batch-size", 2, "--max-steps", 4,
            "--max-eval-steps", 4, "--zero-wrench-weight", 2.5]
    run(*base, "--epochs", 1)
    run(*base, "--resume", output / "last.pt", "--epochs", 2)
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["stats"]["target_fit_counts"] == [21] * 11
    assert all(checkpoint["split_supervision"]["train"])
    test = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert test["samples"] == 22 and test["zero_wrench_loss"] > 0
    evaluation = tmp_path / "evaluation"
    # The saved v5 path is used when --prepared-root is omitted.
    run("evaluate.py", "--checkpoint", output / "last.pt", "--device", "cpu", "--output", evaluation)
    evaluated = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert evaluated["samples"] == 22
    run("predict.py", "--checkpoint", output / "best.pt", "--image", v5_cache / "image.png",
        "--prepared-window", v5_cache / "slide_0_None_0.npz", "--device", "cpu",
        "--output", tmp_path / "prediction.json")
    predicted = json.loads((tmp_path / "prediction.json").read_text(encoding="utf-8"))
    assert len(predicted["predictions"]) == 10
    assert np.isfinite(list(predicted["predictions"][0]["prediction"].values())).all()
    # Manifest changes must be caught before resuming or evaluating saved weights.
    manifest_path = v5_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for r in manifest["records"]:
        r["supervision_sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run(*base, "--resume", output / "last.pt", "--epochs", 3, error="supervision changed")
    run("evaluate.py", "--checkpoint", output / "last.pt", "--device", "cpu",
        "--output", evaluation, error="supervision")


@pytest.fixture
def separate_caches(v5_cache, tmp_path):
    source = json.loads((v5_cache / "manifest.json").read_text(encoding="utf-8"))
    roots = []
    for name in ("press", "slide", "polyline"):
        root = tmp_path / name
        root.mkdir()
        shutil.copy2(v5_cache / "image.png", root / "image.png")
        records = []
        for original in source["records"]:
            if (original["kind"] == "press") != (name == "press"):
                continue
            r = dict(original)
            with np.load(v5_cache / r["file"]) as z:
                arrays = {k: z[k].copy() for k in z.files}
            if name != "press":
                arrays["target"][:, 0] += np.linspace(0., .1 if r["order"] == 0 else .001, r["events"])
                r["target"] = arrays["target"].tolist()
            elif r["contact_state"] == 0:
                # Background regression is masked, but its measured robot XY
                # remains usable for the same spatial selection as real caches.
                arrays["target"][:, :2] = 0.
                r["target"] = arrays["target"].tolist()
            if name == "polyline":
                for key in ("id", "sequence", "group"):
                    r[key] = "polyline/" + r[key]
            np.savez(root / r["file"], **arrays)
            records.append(r)
        (root / "manifest.json").write_text(json.dumps({**source, "records": records}), encoding="utf-8")
        (root / "report.json").write_text("{}", encoding="utf-8")
        roots.append(root)
    return roots


def test_multiple_caches_trim_after_split_and_fit_only_retained_training(separate_caches):
    all_records, _, _ = load_prepared_caches(separate_caches)
    assert len(all_records) == 18 and len({r["id"] for r in all_records}) == 18
    args = SimpleNamespace(prepared_root=separate_caches, trim_stationary_xy_mm=0.,
                           seed=42, val_fraction=.15, test_fraction=.15, max_groups=None)
    original, _, _ = prepare_training_data(args, "unused")
    args.trim_stationary_xy_mm = .03
    selected, _, report = prepare_training_data(args, "unused")
    assert sum(r["removed_windows"] for r in report["training_selection"].values()) == 6
    assert sum(r["removed_events"] for r in report["training_selection"].values()) == 60
    for key in original:
        assert {r["group"] for r in selected[key]} <= {r["group"] for r in original[key]}
        assert all(r["kind"] == "press" or r["order"] == 0 for r in selected[key])
    a, b = fit_prepared_statistics(selected["train"]), fit_contact_statistics(selected["train"])
    assert a["fit_sample_count"] == 42 and a["target_fit_counts"] == [41] * 11
    for key in ("target_mean", "target_std", "audio_mean", "audio_std"):
        np.testing.assert_array_equal(a[key], b[key])
    # Selection must never rewrite the complete cached trajectories.
    for root in separate_caches[1:]:
        records, _ = read_contact_prepared(root)
        assert len(records) == 6 and any(r["order"] == 1 for r in records)


@pytest.mark.parametrize("threshold", [-.01, float("nan"), float("inf")])
def test_invalid_trim_thresholds(threshold):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        trim_prepared_records([], threshold)


def test_multicache_overlap_and_duplicate_directories_are_rejected(separate_caches):
    with pytest.raises(ValueError, match="distinct prepared cache"):
        load_prepared_caches([separate_caches[0], separate_caches[0]])
    destination = separate_caches[0].parent / "overlap"
    shutil.copytree(separate_caches[0], destination)
    with pytest.raises(ValueError, match="Overlapping sample ID"):
        load_prepared_caches([separate_caches[0], destination])


def test_single_class_contact_cache_uses_finite_weights(separate_caches):
    records, _ = read_contact_prepared(separate_caches[1])
    with pytest.warns(UserWarning, match="only one contact class"):
        stats = fit_contact_statistics(records)
    assert stats["contact_class_weights"] == [0., 1.]
    assert stats["contact_classes_present"] == [1]


def test_mixed_v4_v5_batches_keep_uniform_fields_without_fake_contact(separate_caches):
    path = separate_caches[0] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["version"] = 4
    manifest["records"] = [{k: v for k, v in r.items()
                            if k not in ("contact_state", "supervision_sha256", "contact_source")}
                           for r in manifest["records"] if r["contact_state"] == 1]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    records, _, _ = load_prepared_caches(separate_caches[:2])
    stats = fit_prepared_statistics(records)
    data = PreparedDataset(records, stats)
    assert not data[0]["contact_valid"].any() and data[0]["target_valid"].all()
    assert data[-1]["contact_valid"].all()
    assert data[0].keys() == data[-1].keys()
    with pytest.raises(ValueError, match="requires a v5 contact cache"):
        load_prepared_caches(separate_caches[:2], contact_only=True)


@pytest.mark.parametrize("model_name", ["model_triple_head", "model_contact_gated"])
def test_multicache_trim_train_resume_and_evaluate(separate_caches, tmp_path, model_name):
    root = Path(__file__).resolve().parents[1]

    def run(*args, error=None):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=root, text=True,
                                capture_output=True, timeout=120)
        if error:
            assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr

    output = tmp_path / "combined_run"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common = ["--no-demo-tests", "--model", model_name, "--device", device, "--threads", 2, "--batch-size", 2,
              "--max-steps", 4, "--max-eval-steps", 4, "--output", output]
    run("train.py", *common, "--prepared-root", separate_caches[0], "--prepared-root", *separate_caches[1:],
        "--trim-stationary-xy-mm", .03, "--filter-xy-range", "--epochs", 1)
    # Both the cache list and threshold are inherited when resuming.
    run("train.py", *common, "--resume", output / "last.pt", "--epochs", 2)
    checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 2 and checkpoint["args"]["trim_stationary_xy_mm"] == .03
    assert checkpoint["args"]["filter_xy_range"] is True
    assert len(checkpoint["args"]["prepared_root"]) == 3
    assert checkpoint["stats"]["fit_sample_count"] == 42
    assert checkpoint["stats"]["target_fit_counts"] == [41] * 11
    report = json.loads((output / "data_report.json").read_text(encoding="utf-8"))
    assert report["training_selection"]["test"]["removed_events"] == 10
    evaluation = tmp_path / "evaluation"
    run("evaluate.py", "--checkpoint", output / "last.pt", "--device", device, "--output", evaluation)
    metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["samples"] == 12
    run("evaluate.py", "--checkpoint", output / "last.pt", "--prepared-root", *separate_caches,
        "--device", device, "--output", evaluation)
    run("train.py", *common, "--resume", output / "last.pt", "--epochs", 3,
        "--trim-stationary-xy-mm", 0, error="trim-stationary-xy-mm differs")
    run("train.py", *common, "--resume", output / "last.pt", "--epochs", 3,
        "--no-filter-xy-range", error="filter-xy-range differs")
