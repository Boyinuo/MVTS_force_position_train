import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from mvts.data import (TactileDataset, discover_records, fit_statistics,
                       limit_split_groups, location_group, split_records)
from mvts.features import FeatureConfig, LogMel
from mvts.architectures import available_models, build_model


def test_logmel_preserves_time_and_channel_amplitude():
    extractor = LogMel()
    time = torch.arange(2000) / 50000
    signal = 2000 + 100 * torch.sin(2 * torch.pi * 3000 * time)
    adc = signal[:, None].repeat(1, 6)
    adc[:, 1] = 2000 + 2 * (adc[:, 1] - 2000)
    mel = extractor(adc)
    assert mel.shape == (6, 64, 12)
    assert torch.isfinite(mel).all()
    active = mel[0] > -8
    assert active.any()
    # Doubling waveform amplitude quadruples power (log10(4)).
    assert torch.allclose((mel[1] - mel[0])[active], torch.full_like(mel[0][active], np.log10(4)), atol=0.01)
    assert extractor(adc.unsqueeze(0)).shape == (1, 6, 64, 12)
    assert torch.isfinite(extractor(torch.zeros_like(adc))).all()
    with pytest.raises(ValueError, match="ADC"):
        extractor(adc[:1999])


@pytest.mark.parametrize("model_name,feature_shape,token_count", [
    ("model", (1, 512, 8, 12), 13),
    ("model_standard_audio", (1, 512, 2, 1), 2),
])
def test_real_architecture_shapes_gradients_and_contact(model_name, feature_shape, token_count):
    model = build_model(model_name, {"dropout": 0})
    image, audio = torch.randn(1, 3, 224, 224), torch.randn(1, 6, 64, 12)
    observed = []
    tokens = []
    hook = model.audio_backbone.register_forward_hook(lambda _m, _i, out: observed.append(tuple(out.shape)))
    token_hook = model.transformer.register_forward_pre_hook(lambda _m, args: tokens.append(tuple(args[0].shape)))
    output = model(image, audio)
    hook.remove()
    token_hook.remove()
    assert observed == [feature_shape]
    assert tokens == [(1, token_count, 256)]
    assert output["position"].shape == (1, 3)
    assert output["wrench"].shape == (1, 6)
    loss = output["position"].square().mean() + output["wrench"].square().mean()
    loss.backward()
    for weight in (model.vision_backbone.stem[0].weight, model.audio_backbone.stem[0].weight,
                   model.position_embedding, model.attention_score[-1].weight):
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    del model
    contact_model = build_model(model_name, {"use_contact": True, "dropout": 0}).eval()
    with pytest.raises(ValueError, match="starts_in_contact"):
        contact_model.encode_tokens(image, audio)
    with torch.no_grad():
        a = contact_model.encode_tokens(image, audio, torch.tensor([0]))
        b = contact_model.encode_tokens(image, audio, torch.tensor([1]))
    assert a.shape == (1, token_count, 256) and not torch.allclose(a, b)


def test_model_module_selection():
    assert {"model", "model_standard_audio"}.issubset(available_models())
    for name in ("../model", "missing_model", "model.py"):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(name)


def test_audit_split_and_train_only_statistics(acquisition_data):
    records, report = discover_records(acquisition_data)
    assert len(records) == 16 and report["groups"] == 8
    assert report["rejected"] == []
    assert report["audio_marker_outside_window"] == 0
    assert report["sessions"][0]["warnings"]
    split = split_records(records)
    group_sets = [{r["group"] for r in split[k]} for k in ("train", "val", "test")]
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    for group in {r["group"] for r in records}:
        assert sum(group in groups for groups in group_sets) == 1
    config = FeatureConfig()
    stats = fit_statistics(split["train"], config)
    expected = np.asarray([r["target"] for r in split["train"]]).mean(axis=0)
    np.testing.assert_allclose(stats["target_mean"], expected)
    # Poison held-out labels; training normalization must remain unchanged.
    for record in split["val"]:
        record["target"] = [1e9] * 9
    assert fit_statistics(split["train"], config) == stats
    sample = TactileDataset(split["train"], config, stats)[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["audio"].shape == (6, 64, 12)
    assert sample["target"].shape == (9,)
    for subset in limit_split_groups(split, 1).values():
        assert len({r["group"] for r in subset}) == 1
    with pytest.raises(ValueError, match="Strict timing"):
        discover_records(acquisition_data, strict_timing=True)


def test_corrupt_labels_are_reported(acquisition_data):
    path = acquisition_data / "session_0/samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["force"]["wrench"] = [123] * 6
    rows[1]["valid"] = False
    rows[2]["kind"] = "background"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    records, report = discover_records(acquisition_data)
    assert len(records) == 13
    assert len(report["rejected"]) == 1 and "force mismatch" in report["rejected"][0]["reason"]
    assert report["excluded"]["source_invalid"] == 1
    assert report["excluded"]["staircase/background"] == 1


def test_grouping_ignores_measurement_jitter_and_mode():
    a = {"x_mm": 1.25, "y_mm": 2.5, "actual_xy_mm": [1.24, 2.51], "mode": "dynamic"}
    b = {**a, "actual_xy_mm": [1.26, 2.49], "mode": "staircase"}
    assert location_group(a) == location_group(b)
    with pytest.raises(ValueError, match="three"):
        split_records([{"group": "one"}])


@pytest.mark.parametrize("dataset,session", [(None, "session_0"), ("data_sensor_other", "session_1")])
def test_dataset_selection_audits_only_selected_folder(acquisition_data, monkeypatch, dataset, session):
    import train

    for name, source in (("data_sensor_normal", "session_0"), ("data_sensor_other", "session_1")):
        shutil.copytree(acquisition_data / source, acquisition_data / "data" / name / source)
    monkeypatch.chdir(acquisition_data)
    args = ["train.py", "--audit-only", "--output", "audit", "--device", "cpu", "--model", "model"]
    if dataset is not None:
        args += ["--dataset", dataset]
    monkeypatch.setattr(sys, "argv", args)
    train.main()
    report = json.loads((acquisition_data / "audit/data_report.json").read_text())
    assert report["total_events"] == report["included"] == 8
    assert [item["session"] for item in report["sessions"]] == [session]


@pytest.mark.parametrize("options", [
    ["--dataset", ""], ["--dataset", ".."], ["--dataset", "../outside"],
    ["--dataset", "data/nested"], ["--dataset", "data\\nested"],
    ["--dataset", "C:outside"], ["--dataset", "missing"],
    ["--dataset", "data_sensor_normal", "--data-root", "data"],
])
def test_invalid_dataset_arguments_fail_early(tmp_path, monkeypatch, options):
    import train

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", *options])
    with pytest.raises(SystemExit) as error:
        train.parse_args()
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device not available"))])
@pytest.mark.parametrize("model_name", ["model", "model_standard_audio"])
def test_cli_train_resume_evaluate_predict(acquisition_data, tmp_path, device, model_name):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"

    def run(script, *args):
        result = subprocess.run([sys.executable, str(root / script), *map(str, args)],
                                cwd=root, text=True, capture_output=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr

    base = ["--data-root", acquisition_data, "--output", output, "--batch-size", 2,
            "--threads", 2, "--max-groups", 1, "--max-steps", 1, "--max-eval-steps", 1,
            "--device", device]
    run("train.py", *base, "--epochs", 1, "--model", model_name)
    for name in ("best.pt", "last.pt", "normalization.json", "data_report.json", "test_predictions.csv"):
        assert (output / name).is_file()
    config = json.loads((output / "config.json").read_text())
    assert config["model_name"] == model_name
    environment = config["environment"]
    assert environment["device"] == device
    if device == "cuda":
        assert environment["cuda_available"] and environment["gpu_name"] and environment["cuda_runtime"]
    # Emulate old checkpoints without model identity: they must still load model.py.
    if model_name == "model" and device == "cpu":
        for name in ("best.pt", "last.pt"):
            checkpoint = torch.load(output / name, map_location="cpu", weights_only=True)
            checkpoint.pop("model_name")
            checkpoint["args"].pop("model")
            torch.save(checkpoint, output / name)
    # Omit --model on resume: recover its identity from the checkpoint.
    run("train.py", *base, "--epochs", 2, "--resume", output / "last.pt")
    assert json.loads((output / "config.json").read_text())["model_name"] == model_name
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in history] == [1, 2]
    assert all(np.isfinite(h["train"]["loss"]) for h in history)
    if model_name == "model_standard_audio" and device == "cpu":
        wrong_resume = subprocess.run(
            [sys.executable, str(root / "train.py"), *map(str, base), "--epochs", "3",
             "--resume", str(output / "last.pt"), "--model", "model"],
            cwd=root, text=True, capture_output=True, timeout=120)
        assert wrong_resume.returncode != 0 and "Resume model mismatch" in wrong_resume.stderr
    run("evaluate.py", "--checkpoint", output / "best.pt", "--data-root", acquisition_data,
        "--output", tmp_path / "evaluation", "--batch-size", 2, "--device", device)
    run("predict.py", "--checkpoint", output / "best.pt", "--image", acquisition_data / "session_0/frames/0.png",
        "--npz", acquisition_data / "session_0/samples/0.npz", "--output", tmp_path / "prediction.json",
        "--device", device)
    prediction = json.loads((tmp_path / "prediction.json").read_text())
    assert prediction["model_name"] == model_name
    assert len(prediction["prediction"]) == 9 and all(np.isfinite(list(prediction["prediction"].values())))
