import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from mvts.data import TactileDataset, discover_records, fit_statistics, split_records
from mvts.features import FeatureConfig, LogMel, extract_event_audio


def test_event_audio_preserves_legacy_and_crops_exported_boundary():
    adc = np.arange(12000, dtype=np.uint16).reshape(2000, 6)
    np.testing.assert_array_equal(extract_event_audio({"adc": adc}), adc)
    long_adc = np.pad(adc, ((3000, 3000), (0, 0)), constant_values=65535)
    for metadata in ({}, {"audio_before": 4000, "audio_after": 4000}):
        result = extract_event_audio({"adc": long_adc, **metadata})
        np.testing.assert_array_equal(result, adc)
        assert LogMel()(torch.from_numpy(result)).shape == (6, 64, 12)
    # Use the exported boundary even when it differs from the array midpoint.
    shifted = {"adc": long_adc, "audio_before": 3500, "audio_after": 4500}
    np.testing.assert_array_equal(extract_event_audio(shifted), long_adc[2500:4500])
    np.testing.assert_array_equal(long_adc[:3000], np.full((3000, 6), 65535))


@pytest.mark.parametrize("shape,metadata", [
    ((1999, 6), {}), ((2000, 5), {}), ((1000, 6), {}),
    ((8000, 6), {"audio_before": 4000, "audio_after": 3999}),
    ((8000, 6), {"audio_before": 500, "audio_after": 7500}),
    ((8000, 6), {"audio_before": 4000.5}),
    ((8000, 6), {"audio_before": np.array([4000])}),
])
def test_invalid_event_audio_is_rejected(shape, metadata):
    with pytest.raises(ValueError):
        extract_event_audio({"adc": np.zeros(shape), **metadata})


@pytest.mark.parametrize("mode,before,after,start,stop", [
    ("before_after", 700, 301, 2800, 3801),
    ("before_only", 900, 9999, 2600, 3500),
])
def test_configured_audio_windows(mode, before, after, start, stop):
    adc = np.arange(48000, dtype=np.uint16).reshape(8000, 6)
    data = {"adc": adc, "audio_before": 3500, "audio_after": 4500}
    config = FeatureConfig(alignment_mode=mode, before_samples=before, after_samples=after)
    np.testing.assert_array_equal(extract_event_audio(data, config=config), adc[start:stop])
    assert config.samples == stop - start
    assert config.after_samples == (after if mode == "before_after" else 0)
    assert FeatureConfig(**config.to_dict()) == config
    assert LogMel(config)(torch.from_numpy(adc[start:stop])).shape == (6, 64, config.audio_frames)


def test_before_only_accepts_window_ending_at_event():
    adc = np.arange(6000, dtype=np.uint16).reshape(1000, 6)
    config = FeatureConfig(alignment_mode="before_only", before_samples=700)
    data = {"adc": adc, "audio_before": 1000, "audio_after": 0}
    np.testing.assert_array_equal(extract_event_audio(data, config=config), adc[-700:])
    # Legacy exports still locate the event at the midpoint.
    np.testing.assert_array_equal(extract_event_audio({"adc": np.tile(adc, (2, 1))}, config=config),
                                  adc[-700:])


@pytest.mark.parametrize("options", [
    {"alignment_mode": "unknown", "before_samples": 1000, "after_samples": 1000},
    {"alignment_mode": "before_only", "before_samples": -1},
    {"alignment_mode": "before_only", "before_samples": 0},
    {"alignment_mode": "before_only", "before_samples": 1000.5},
    {"alignment_mode": "before_after", "before_samples": 1000, "after_samples": -1},
    {"alignment_mode": "before_after", "before_samples": 1000, "after_samples": 1000,
     "samples": 1000},
])
def test_invalid_alignment_configuration(options):
    with pytest.raises(ValueError):
        FeatureConfig(**options)


def test_configured_window_rejects_missing_samples_and_short_stft():
    data = {"adc": np.zeros((2000, 6)), "audio_before": 1000, "audio_after": 1000}
    config = FeatureConfig(alignment_mode="before_only", before_samples=1001)
    with pytest.raises(ValueError, match="1001 samples before"):
        extract_event_audio(data, config=config)
    config = FeatureConfig(alignment_mode="before_only", before_samples=100)
    with pytest.raises(ValueError, match="n_fft"):
        LogMel(config)
    # Old checkpoint dictionaries and the positional samples argument retain their window.
    legacy = FeatureConfig(sample_rate=50000, n_fft=512, hop_length=128,
                           n_mels=64, samples=2000, adc_scale=65535)
    assert legacy.before_samples == legacy.after_samples == 1000
    np.testing.assert_array_equal(extract_event_audio(data, 2000), data["adc"])


def expand_session(session):
    """Turn a legacy fixture into a portable 8000-point export, preserving its center."""
    for path in (session / "samples").glob("*.npz"):
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        data["adc"] = np.pad(data["adc"], ((3000, 3000), (0, 0)), constant_values=65535)
        for key, increment in (("device_us", 20), ("estimated_pc_ns", 20000)):
            data[key] = data[key][0].astype(np.float64) + (np.arange(8000) - 3000) * increment
        data["audio_before"] = np.int64(4000)
        data["audio_after"] = np.int64(4000)
        np.savez(path, **data)
    (session / "audio/session.json").unlink()
    (session / "session.json").unlink()
    (session / "dataset_summary.json").write_text(json.dumps({
        "sample_rate_hz": 50000,
        "sources": [{"audio_timing_quality_passed": True, "whole_session_completed": True}],
    }), encoding="utf-8")


def test_mixed_windows_use_identical_features_and_statistics(acquisition_data):
    config = FeatureConfig()
    records, _ = discover_records(acquisition_data)
    split = split_records(records)
    expected_stats = fit_statistics(split["train"], config)
    original = TactileDataset(records, config, expected_stats)
    expected_audio = [original[i]["audio"] for i in range(len(records))]
    expand_session(acquisition_data / "session_1")
    records, report = discover_records(acquisition_data)
    assert report["rejected"] == []
    assert report["audio_source_samples"] == {"2000": 8, "8000": 8}
    assert report["audio_input_window"] == {"before": 1000, "after": 1000, "samples": 2000}
    assert report["sample_rate_hz"] == 50000
    assert report["audio_marker_outside_window"] == 0
    assert report["sessions"][1]["warnings"] == []
    actual_stats = fit_statistics(split_records(records)["train"], config)
    assert actual_stats == expected_stats
    dataset = TactileDataset(records, config, actual_stats)
    for index, expected in enumerate(expected_audio):
        torch.testing.assert_close(dataset[index]["audio"], expected, rtol=0, atol=0)
    # The old incomplete session is still rejected in strict mode.
    with pytest.raises(ValueError, match="Strict timing"):
        discover_records(acquisition_data, strict_timing=True)
    _, strict_report = discover_records(acquisition_data / "session_1", strict_timing=True)
    assert strict_report["included"] == 8


def test_audit_validates_full_original_timestamps(acquisition_data):
    expand_session(acquisition_data / "session_1")
    path = acquisition_data / "session_1/samples/0.npz"
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    data["device_us"] = data["device_us"][3000:5000]
    np.savez(path, **data)
    records, report = discover_records(acquisition_data)
    assert len(records) == 15
    assert len(report["rejected"]) == 1
    assert report["rejected"][0]["reason"] == "Invalid audio timestamp shape"


def test_cli_mixed_window_train_evaluate_predict(acquisition_data, tmp_path):
    expand_session(acquisition_data / "session_1")
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "run"

    def run(script, *args):
        result = subprocess.run([sys.executable, str(root / script), *map(str, args)],
                                cwd=root, text=True, capture_output=True, timeout=240)
        assert result.returncode == 0, result.stdout + result.stderr

    # This fixture has 1000 pre-event points, matching the legacy model window.
    run("train.py", "--data-root", acquisition_data, "--model", "model",
        "--output", output, "--device", "cpu", "--epochs", 1, "--batch-size", 2,
        "--threads", 2, "--max-groups", 1, "--max-steps", 1, "--max-eval-steps", 1)
    config = json.loads((output / "config.json").read_text())
    assert config["features"]["samples"] == 2000
    assert config["model"]["audio_frames"] == 12
    run("evaluate.py", "--checkpoint", output / "best.pt", "--data-root", acquisition_data,
        "--device", "cpu", "--output", tmp_path / "evaluation")
    image = acquisition_data / "session_1/frames/0.png"
    path = acquisition_data / "session_1/samples/0.npz"
    prediction_path = tmp_path / "prediction.json"
    run("predict.py", "--checkpoint", output / "best.pt", "--image", image,
        "--npz", path, "--device", "cpu", "--output", prediction_path)
    prediction = json.loads(prediction_path.read_text())["prediction"]
    # Supplying the equivalent legacy window must produce the same prediction.
    with np.load(path, allow_pickle=False) as data:
        cropped_path = tmp_path / "cropped.npz"
        np.savez(cropped_path, adc=extract_event_audio(data))
    run("predict.py", "--checkpoint", output / "best.pt", "--image", image,
        "--npz", cropped_path, "--device", "cpu", "--output", prediction_path)
    assert json.loads(prediction_path.read_text())["prediction"] == prediction


@pytest.mark.parametrize("mode,before,after", [
    ("before_after", 700, 300),
    ("before_only", 1000, 9999),
])
def test_model_alignment_train_evaluate_predict(acquisition_data, tmp_path, monkeypatch, mode, before, after):
    from dataclasses import replace

    import evaluate
    import predict
    import train
    from mvts import model_sensor_final
    from mvts.engine import load_checkpoint

    # Emulate model header settings through ModelConfig without rewriting source files.
    model_config = model_sensor_final.ModelConfig(audio_alignment_mode=mode,
                                                  audio_samples_before=before,
                                                  audio_samples_after=after)
    monkeypatch.setattr(model_sensor_final, "ModelConfig", lambda **kwargs: replace(model_config, **kwargs))
    output = tmp_path / "configured_run"
    train_args = ["train.py", "--data-root", str(acquisition_data), "--model", "model_sensor_final",
                  "--output", str(output), "--device", "cpu", "--epochs", "1", "--batch-size", "2",
                  "--threads", "2", "--max-groups", "1", "--max-steps", "1", "--max-eval-steps", "1"]
    monkeypatch.setattr(sys, "argv", train_args)
    train.main()
    effective_after = after if mode == "before_after" else 0
    window = {"before": before, "after": effective_after, "samples": before + effective_after}
    report = json.loads((output / "data_report.json").read_text())
    assert report["audio_input_window"] == window
    assert report["audio_alignment_mode"] == mode
    model, checkpoint, config = load_checkpoint(output / "best.pt")
    assert config.alignment_mode == mode
    assert config.samples == window["samples"]
    assert model.config.audio_frames == config.audio_frames == 4
    records, _ = discover_records(acquisition_data, feature_config=config)
    record = records[0]
    sample = TactileDataset([record], config, checkpoint["stats"])[0]
    with np.load(record["npz"], allow_pickle=False) as data:
        # The fixture's event boundary is 1000.
        adc = data["adc"][1000 - before:1000 + effective_after].astype(np.float32)
    stats = checkpoint["stats"]
    expected = (LogMel(config)(torch.from_numpy(adc)) - torch.tensor(stats["audio_mean"])[:, None, None]) / (
        torch.tensor(stats["audio_std"])[:, None, None])
    torch.testing.assert_close(sample["audio"], expected)

    # Later model defaults may request unavailable samples. Evaluation/prediction
    # must recover the trained window from the checkpoint.
    model_config = replace(model_config, audio_samples_before=9999, audio_samples_after=9999)
    evaluation = tmp_path / "configured_evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(output / "best.pt"),
                                      "--data-root", str(acquisition_data), "--device", "cpu",
                                      "--output", str(evaluation)])
    evaluate.main()
    assert json.loads((evaluation / "data_report.json").read_text())["audio_input_window"] == window
    prediction = tmp_path / "configured_prediction.json"
    monkeypatch.setattr(sys, "argv", ["predict.py", "--checkpoint", str(output / "best.pt"),
                                      "--image", record["image"], "--npz", record["npz"],
                                      "--device", "cpu", "--output", str(prediction)])
    predict.main()
    assert len(json.loads(prediction.read_text())["prediction"]) == 9

    monkeypatch.setattr(sys, "argv", [*train_args[:], "--epochs", "2", "--resume", str(output / "last.pt")])
    train.main()
    assert json.loads((output / "config.json").read_text())["features"] == config.to_dict()
