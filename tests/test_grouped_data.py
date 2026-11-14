import json
import sys

import numpy as np
import pytest
import torch

from mvts.data import (TactileDataset, discover_records, extract_sample_audio,
                       fit_statistics, split_records)
from mvts.engine import load_checkpoint
from mvts.features import FeatureConfig, LogMel, extract_event_audio


@pytest.fixture
def mixed_grouped_data(acquisition_data):
    """One legacy session and one grouped session at the same eight locations."""
    session = acquisition_data / "session_1"
    rows = [json.loads(line) for line in (session / "samples.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(29)
    for row in rows:
        path = session / row["data"]
        with np.load(path, allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        force_times = row["pc_ns"] - 1_000_000 - np.arange(9, -1, -1) * 2_500_000
        times = force_times[:, None] + (np.arange(4000) - 4000) * 20000 + 10000
        wrench = data["force_wrench"][None, :] + np.arange(-9, 1)[:, None] * 0.01
        data.update(adc=rng.integers(1000, 2000, (10, 4000, 6), dtype=np.uint16),
                    estimated_pc_ns=times.astype(np.float64),
                    device_us=(times // 1000).astype(np.uint64),
                    audio_next_pc_ns=times[:, -1] + 20000,
                    force_pc_ns=force_times, force_wrench=wrench,
                    audio_before=np.int64(4000), audio_after=np.int64(0))
        np.savez(path, **data)
        row.update(sample_schema="image_force_audio_v2", audio_before=4000, audio_after=0,
                   force_sequence=[{"pc_est_ns": int(time), "wrench": force.tolist()}
                                   for time, force in zip(force_times, wrench)])
    (session / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return acquisition_data


def test_grouped_pairs_share_image_and_keep_legacy_samples(mixed_grouped_data):
    config = FeatureConfig(alignment_mode="before_only", before_samples=700)
    records, report = discover_records(mixed_grouped_data, feature_config=config)
    assert len(records) == 88
    assert report["included_events"] == 16 and report["grouped_events"] == 8
    assert report["rejected"] == [] and report["audio_marker_outside_window"] == 0
    assert report["audio_source_samples"] == {"2000": 8, "4000": 80}
    legacy = records[0]
    assert legacy["id"] == "session_0/0" and "force_index" not in legacy
    with np.load(legacy["npz"]) as source:
        np.testing.assert_array_equal(extract_sample_audio(source, config),
                                      extract_event_audio(source, config=config))
    children = [row for row in records if row["session"] == "session_1" and row["sample_id"] == 0]
    assert [row["force_index"] for row in children] == list(range(10))
    assert len({row["id"] for row in records}) == 88
    assert len({row["image"] for row in children}) == 1
    assert len({tuple(row["target"][:3]) for row in children}) == 1
    assert len({row["starts_in_contact"] for row in children}) == 1
    with np.load(children[0]["npz"]) as source:
        np.testing.assert_array_equal(extract_sample_audio(source, config), source["adc"][:, -700:, :])
        np.testing.assert_allclose([row["target"][3:] for row in children], source["force_wrench"])
        for index in range(10):
            np.testing.assert_array_equal(extract_sample_audio(source, config, index),
                                          source["adc"][index, -700:, :])
        with pytest.raises(ValueError, match="force_index"):
            extract_sample_audio(source, config, -1)
        with pytest.raises(ValueError, match="samples after"):
            extract_sample_audio(source, FeatureConfig())
    split = split_records(records)
    groups = [{row["group"] for row in split[key]} for key in ("train", "val", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    stats = fit_statistics(split["train"], config)
    np.testing.assert_allclose(stats["target_mean"], np.mean([r["target"] for r in split["train"]], axis=0))
    # Independently compute every mel feature to catch wrong pair selection in statistics.
    expected = []
    for record in split["train"]:
        with np.load(record["npz"]) as source:
            adc = source["adc"]
            adc = adc[record["force_index"], -700:] if "force_index" in record else adc[300:1000]
            expected.append(LogMel(config)(torch.from_numpy(adc.astype(np.float32))))
    mel = torch.stack(expected).double().numpy()
    np.testing.assert_allclose(stats["audio_mean"], mel.mean(axis=(0, 2, 3)), atol=1e-7)
    np.testing.assert_allclose(stats["audio_std"], mel.std(axis=(0, 2, 3)), atol=1e-6)
    dataset = TactileDataset(children, config, stats)
    with np.load(children[0]["npz"]) as source:
        expected_audio = LogMel(config)(torch.from_numpy(source["adc"][:, -700:].astype(np.float32)))
    expected_audio = ((expected_audio - torch.tensor(stats["audio_mean"], dtype=torch.float32)[:, None, None])
                      / torch.tensor(stats["audio_std"], dtype=torch.float32)[:, None, None])
    for index, row in enumerate(children):
        sample = dataset[index]
        assert sample["audio"].shape == (6, 64, config.audio_frames)
        torch.testing.assert_close(sample["audio"], expected_audio[index])
        restored = sample["target"].numpy() * stats["target_std"] + stats["target_mean"]
        np.testing.assert_allclose(restored, row["target"], atol=1e-6)


@pytest.mark.parametrize("corruption", ["force", "audio_alignment", "timestamp_shape"])
def test_bad_grouped_event_is_rejected_as_a_whole(mixed_grouped_data, corruption):
    path = mixed_grouped_data / "session_1/samples/0.npz"
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    if corruption == "force":
        data["force_wrench"][0, 0] += 1
    elif corruption == "audio_alignment":
        data["estimated_pc_ns"][0] += 1_000_000
    else:
        data["device_us"] = data["device_us"][-1]
    np.savez(path, **data)
    config = FeatureConfig(alignment_mode="before_only", before_samples=700)
    records, report = discover_records(mixed_grouped_data, feature_config=config)
    assert len(records) == 78 and len(report["rejected"]) == 1
    assert report["included_events"] == 15 and report["grouped_events"] == 7
    assert not any(row["session"] == "session_1" and row["sample_id"] == 0 for row in records)


def test_grouped_train_resume_evaluate_and_predict(mixed_grouped_data, tmp_path, monkeypatch):
    import evaluate
    import predict
    import train

    # The grouped session has 4000 pre-force points, matching the production model defaults.
    root = mixed_grouped_data / "session_1"
    output = tmp_path / "run"
    args = ["train.py", "--data-root", str(root), "--model", "model_contact_concat_sensor_final",
            "--output", str(output), "--device", "cpu", "--epochs", "1", "--batch-size", "2",
            "--threads", "2", "--max-groups", "1", "--max-steps", "1", "--max-eval-steps", "1"]
    monkeypatch.setattr(sys, "argv", args)
    train.main()
    _, checkpoint, config = load_checkpoint(output / "best.pt")
    assert config.alignment_mode == "before_only" and config.before_samples == 2000
    assert len(checkpoint["split_ids"]["train"]) == 10
    assert len(set(checkpoint["split_ids"]["train"])) == 10
    monkeypatch.setattr(sys, "argv", [*args, "--epochs", "2", "--resume", str(output / "last.pt")])
    train.main()
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(output / "best.pt"),
                                      "--data-root", str(root), "--device", "cpu",
                                      "--output", str(tmp_path / "evaluation")])
    evaluate.main()
    metrics = json.loads((tmp_path / "evaluation/metrics.json").read_text())
    assert metrics["samples"] == 10
    prediction_path = tmp_path / "prediction.json"
    predict_args = ["predict.py", "--checkpoint", str(output / "best.pt"), "--device", "cpu",
                    "--image", str(root / "frames/0.png"), "--npz", str(root / "samples/0.npz"),
                    "--output", str(prediction_path)]
    monkeypatch.setattr(sys, "argv", predict_args)
    predict.main()
    predictions = json.loads(prediction_path.read_text())["predictions"]
    assert [p["force_index"] for p in predictions] == list(range(10))
    assert all(len(p["prediction"]) == 9 for p in predictions)
    monkeypatch.setattr(sys, "argv", [*predict_args, "--force-index", "4"])
    predict.main()
    selected = json.loads(prediction_path.read_text())
    assert selected["force_index"] == 4
    np.testing.assert_allclose(list(selected["prediction"].values()),
                               list(predictions[4]["prediction"].values()), atol=1e-5, rtol=1e-5)
    # Equivalent single-window input retains the legacy single-prediction output.
    path = tmp_path / "single.npz"
    with np.load(root / "samples/0.npz") as data:
        np.savez(path, adc=data["adc"][4], audio_before=4000, audio_after=0, starts_in_contact=1)
    monkeypatch.setattr(sys, "argv", [*predict_args, "--npz", str(path)])
    predict.main()
    single = json.loads(prediction_path.read_text())
    assert "predictions" not in single
    np.testing.assert_allclose(list(single["prediction"].values()), list(selected["prediction"].values()),
                               atol=1e-6, rtol=1e-6)
