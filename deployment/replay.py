"""Precompute one exported deployment trajectory; no GUI or playback clock here."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from mvts.architectures import build_model, get_model_module
from mvts.engine import choose_device, load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = ("x_mm", "y_mm", "delta_z_mm", "Fx", "Fy", "Fz", "Mx", "My", "Mz",
                "slip_x_mm", "slip_y_mm")


@dataclass
class ImageOnlyReplay:
    event_indices: np.ndarray
    times: np.ndarray
    prediction: np.ndarray
    position_valid: np.ndarray
    latency_ms: np.ndarray


@dataclass
class ReplayData:
    metadata: list
    images: dict
    mel: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    position_valid: np.ndarray
    slip_valid: np.ndarray
    contact_probability: np.ndarray
    latency_ms: np.ndarray
    summary: dict
    image_only: ImageOnlyReplay | None = None

    def times(self, hz):
        # Playback is nominal deployment time; original acquisition times remain in metadata.
        return np.arange(len(self.metadata), dtype=np.float64) / hz


def model_name(value):
    return Path(value).stem


def select_sequence(root, sequence=None):
    rows = [json.loads(line) for line in (Path(root) / "events.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    slides = list(dict.fromkeys(row["sequence"] for row in rows if row["kind"] == "slide"))
    if not slides:
        raise ValueError("Dataset contains no slide sequence")
    sequence = sequence or slides[0]
    selected = [row for row in rows if row["kind"] == "slide" and row["sequence"] == sequence]
    if not selected:
        raise ValueError(f"Unknown slide sequence {sequence!r}; available: {slides}")
    if not selected[0]["reset_state"] or not selected[0]["image_update"]:
        raise ValueError("The trajectory must start with a state reset and a new image")
    indices = [row["sequence_index"] for row in selected]
    if indices != list(range(len(selected))):
        raise ValueError("Slide events must be complete and ordered from sequence_index=0")
    return selected


def default_checkpoint(name):
    module = get_model_module(name)
    path = Path(module.__file__).resolve().parent / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint for {name} at {path}. "
                                "Place best.pt beside the selected model module.")
    return path


def prepare_model(args, manifest, root):
    name = model_name(args.model)
    device = choose_device(args.device)
    if args.random_weights:
        torch.manual_seed(0)
        stats = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
        model = build_model(name, manifest["model_config"])
        if hasattr(model, "set_normalization"):
            model.set_normalization(stats)
        model = model.to(device)
        checkpoint_path, epoch = None, None
    else:
        checkpoint_path = default_checkpoint(name)
        model, checkpoint, features = load_checkpoint(checkpoint_path, device)
        if model.model_name != name:
            raise ValueError(f"--model {name} does not match checkpoint model {model.model_name}")
        stats, epoch = checkpoint["stats"], checkpoint.get("epoch")
        for key, value in features.to_dict().items():
            if key in manifest["features"] and value != manifest["features"][key]:
                raise ValueError(f"Checkpoint/dataset feature mismatch: {key}")
    if not callable(getattr(model, "step", None)):
        raise ValueError(f"{name} has no streaming step interface; choose a causal streaming model")
    for key, size in (("audio_mean", 6), ("audio_std", 6), ("target_mean", 11), ("target_std", 11)):
        value = np.asarray(stats[key])
        if value.shape != (size,) or not np.isfinite(value).all() or (key.endswith("std") and (value <= 0).any()):
            raise ValueError(f"Invalid checkpoint normalization: {key}")
    return model.eval(), stats, device, {"model": name, "checkpoint": str(checkpoint_path),
        "epoch": epoch, "random_weights": args.random_weights, "device": str(device)}


def infer_events(model, stats, device, rows, audio, image_inputs, progress=lambda *_: None):
    """Ordered batch-one step calls; labels are deliberately absent from this API."""
    predictions, position_valid, slip_valid, probabilities, latency = [], [], [], [], []
    mean = np.asarray(stats["target_mean"], dtype=np.float32)
    std = np.asarray(stats["target_std"], dtype=np.float32)
    model.reset_state()
    with torch.inference_mode():
        for index, (row, mel) in enumerate(zip(rows, audio)):
            image = image_inputs[row["image"]] if row["image_update"] else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model.step(torch.from_numpy(mel[None]).to(device),
                image=None if image is None else torch.from_numpy(image).to(device),
                frame_id=row["image"], reset=row["reset_state"])
            if "physical_prediction" in result:
                pred = result["physical_prediction"][0].cpu().numpy()
                valid = bool(result["position_valid"][0].item())
                slip_ok = bool(result["slip_valid"][0].item())
                probability = float(result["contact_probability"][0].item())
            else:
                pred = torch.cat([result[k] for k in ("position", "wrench", "slip")], -1)[0].cpu().numpy()
                pred = pred * std + mean
                valid, slip_ok, probability = True, not row["reset_state"], np.nan
            if pred.shape != (11,) or not np.isfinite(pred).all():
                raise FloatingPointError(f"Invalid prediction at sequence event {index}")
            latency.append((time.perf_counter() - started) * 1000)
            predictions.append(pred.copy())
            position_valid.append(valid)
            slip_valid.append(slip_ok)
            probabilities.append(probability)
            if index % 10 == 0 or index + 1 == len(rows):
                progress("推理", index + 1, len(rows))
    return (np.asarray(predictions), np.asarray(position_valid), np.asarray(slip_valid),
            np.asarray(probabilities), np.asarray(latency))


def infer_image_only(model, stats, device, rows, image_inputs, hz, progress=lambda *_: None):
    """Independent 15 Hz stream. No audio or ground-truth inputs, no future frames."""
    if not getattr(model.config, "image_only", False) or getattr(model, "audio_backbone", None) is not None:
        raise ValueError("best_image_only.pt must contain an image_only=True model without an audio backbone")
    count = int(np.floor((len(rows) - 1) / hz * 15 + 1e-9)) + 1
    times = np.arange(count, dtype=np.float64) / 15
    indices = np.floor(times * hz + 1e-9).astype(np.int64)
    mean, std = [np.asarray(stats[k], np.float32) for k in ("target_mean", "target_std")]
    if mean.shape != (11,) or std.shape != (11,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Image-only checkpoint requires 11 finite target means and positive standard deviations")
    model.eval()
    model.reset_state()
    predictions, valid, latency = [], [], []
    previous = -1
    with torch.inference_mode():
        for sample, index in enumerate(indices):
            row = rows[index]
            reset = any(r["reset_state"] for r in rows[previous + 1:index + 1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = model.step(image=torch.from_numpy(image_inputs[row["image"]]).to(device),
                frame_id=row["image"], reset=reset)
            if "physical_prediction" in result:
                prediction = result["physical_prediction"][0].cpu().numpy()
                ok = bool(result["position_valid"][0].item())
            else:
                prediction = torch.cat([result[k] for k in ("position", "wrench", "slip")], -1)[0].cpu().numpy()
                prediction = prediction * std + mean
                ok = True
            latency.append((time.perf_counter() - started) * 1000)
            if prediction.shape != (11,) or not np.isfinite(prediction).all():
                raise FloatingPointError(f"Invalid image-only prediction at sample {sample}")
            predictions.append(prediction.copy())
            valid.append(ok)
            previous = index
            progress("纯视觉 15 Hz 推理", sample + 1, len(indices))
    return ImageOnlyReplay(indices, times, np.asarray(predictions), np.asarray(valid), np.asarray(latency))


def precompute_image_only(args, device, rows, inputs, target, main_prediction, main_valid, progress):
    path = default_checkpoint(model_name(args.model)).with_name("best_image_only.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Missing 15 Hz image-only checkpoint: {path}. "
                                "Place best_image_only.pt beside best.pt, or use --no-image-only-comparison.")
    model, checkpoint, _ = load_checkpoint(path, device)
    comparison = infer_image_only(model, checkpoint["stats"], device, rows, inputs, args.hz, progress)
    indices, valid = comparison.event_indices, comparison.position_valid
    errors = np.linalg.norm(comparison.prediction[valid, :2] - target[indices[valid], :2], axis=1)
    main_ok = main_valid[indices]
    main_errors = np.linalg.norm(main_prediction[indices[main_ok], :2] - target[indices[main_ok], :2], axis=1)
    info = {"enabled": True, "model": model.model_name, "checkpoint": str(path),
        "epoch": checkpoint.get("epoch"), "hz": 15, "samples": len(indices),
        "step_mean_ms": float(comparison.latency_ms.mean()),
        "step_p95_ms": float(np.percentile(comparison.latency_ms, 95)),
        "position_coverage": float(valid.mean()),
        "xy_rmse_mm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "main_xy_rmse_at_15hz_mm": float(np.sqrt(np.mean(main_errors**2))) if len(main_errors) else None,
        "wrench_mae": np.abs(comparison.prediction[:, 3:9] - target[indices, 3:9]).mean(axis=0).tolist(),
        "input_note": "Image-only calls at t=k/15 s with latest available image, independent GRU; "
            "no audio, ground truth, extra hidden updates or interpolated predictions. "
            "GT is taken from the latest 150 Hz event at/before each call. "
            "15 Hz recurrent cadence may differ from the checkpoint training cadence."}
    return comparison, info


def precompute(args, progress=lambda *_: None):
    started = time.perf_counter()
    root = Path(args.data_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = select_sequence(root, args.sequence)
    torch.set_num_threads(args.threads)
    model, stats, device, info = prepare_model(args, manifest, root)
    mean = np.asarray(stats["audio_mean"], dtype=np.float64)[:, None, None]
    std = np.asarray(stats["audio_std"], dtype=np.float64)[:, None, None]
    images, inputs, audio, raw_mel, targets = {}, {}, [], [], []
    for index, row in enumerate(rows):
        with np.load(root / row["data"], allow_pickle=False) as z:
            mel, target = z["audio"].astype(np.float32), z["target"].copy()
        if mel.shape != (6, 64, 28) or target.shape != (11,) or not np.isfinite(mel).all() or not np.isfinite(target).all():
            raise ValueError(f"Invalid event data: {row['data']}")
        audio.append(np.ascontiguousarray(((mel.astype(np.float64) - mean) / std).astype(np.float32)))
        raw_mel.append(mel)
        targets.append(target)
        path = row["image"]
        if path not in images:
            with Image.open(root / path) as im:
                pixels = np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)).copy()
            images[path] = pixels
            chw = pixels.astype(np.float32).transpose(2, 0, 1) / np.float32(255)
            chw = (chw - np.asarray([.485, .456, .406], np.float32)[:, None, None]) / np.asarray(
                [.229, .224, .225], np.float32)[:, None, None]
            inputs[path] = np.ascontiguousarray(chw[None])
        progress("读取", index + 1, len(rows))
    pred, valid, slip_ok, probability, latency = infer_events(model, stats, device, rows, audio, inputs, progress)
    target = np.asarray(targets)
    comparison = None
    comparison_info = {"enabled": False}
    if getattr(args, "image_only_comparison", True) and not args.random_weights:
        # Drop the main model before loading the independent comparison to bound GPU memory.
        disabled_channels = list(model.config.disabled_mic_channels)
        del model
        comparison, comparison_info = precompute_image_only(
            args, device, rows, inputs, target, pred, valid, progress)
    else:
        disabled_channels = list(model.config.disabled_mic_channels)
    errors = np.linalg.norm(pred[valid, :2] - target[valid, :2], axis=1)
    info.update({"data_root": str(root), "sequence": rows[0]["sequence"], "events": len(rows),
        "image_updates": sum(bool(r["image_update"]) for r in rows), "replay_hz": args.hz,
        "precompute_seconds": time.perf_counter() - started,
        "step_mean_ms": float(latency.mean()), "step_p95_ms": float(np.percentile(latency, 95)),
        "step_over_budget": int((latency > 1000 / args.hz).sum()),
        "position_coverage": float(valid.mean()),
        "xy_rmse_mm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "xy_p95_mm": float(np.percentile(errors, 95)) if len(errors) else None,
        "wrench_mae": np.abs(pred[:, 3:9] - target[:, 3:9]).mean(axis=0).tolist(),
        "disabled_mic_channels": disabled_channels, "image_only": comparison_info,
        "timing_note": "Ordered batch-one cached-image inference, including input transfer/output CPU copy; "
            "no warmup, startup first call retained. Playback/render FPS is separate from inference latency.",
        "contact_truth_note": "All exported slide events are in contact by acquisition protocol; no Fz relabeling.",
        "spectrogram_note": "Unnormalised training log10 mel power, fixed display range [-10,-2.5]."})
    return ReplayData(rows, images, np.asarray(raw_mel), target, pred, valid, slip_ok, probability, latency, info, comparison)


def save_results(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(data.summary, ensure_ascii=False, indent=2), encoding="utf-8")
    extra = {}
    if data.image_only is not None:
        baseline = data.image_only
        extra = {"image_only_event_indices": baseline.event_indices, "image_only_times_s": baseline.times,
            "image_only_prediction": baseline.prediction, "image_only_position_valid": baseline.position_valid,
            "image_only_latency_ms": baseline.latency_ms}
        with (output / "image_only_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_index", "time_s", "sequence_index", "image", "position_valid", "latency_ms"]
                + [f"pred_{k}" for k in TARGET_NAMES] + [f"true_{k}" for k in TARGET_NAMES])
            for j, event in enumerate(baseline.event_indices):
                writer.writerow([j, baseline.times[j], data.metadata[event]["sequence_index"],
                    data.metadata[event]["image"], baseline.position_valid[j], baseline.latency_ms[j],
                    *baseline.prediction[j], *data.target[event]])
    np.savez_compressed(output / "predictions.npz", prediction=data.prediction, target=data.target,
        position_valid=data.position_valid, slip_valid=data.slip_valid,
        contact_probability=data.contact_probability, latency_ms=data.latency_ms,
        event_pc_ns=np.asarray([r["event_pc_ns"] for r in data.metadata], dtype=np.int64), **extra)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_index", "event_pc_ns", "image", "position_valid", "slip_valid",
            "contact_probability", "latency_ms"] + [f"pred_{k}" for k in TARGET_NAMES] + [f"true_{k}" for k in TARGET_NAMES])
        for i, row in enumerate(data.metadata):
            writer.writerow([row["sequence_index"], row["event_pc_ns"], row["image"],
                data.position_valid[i], data.slip_valid[i], data.contact_probability[i], data.latency_ms[i],
                *data.prediction[i], *data.target[i]])
