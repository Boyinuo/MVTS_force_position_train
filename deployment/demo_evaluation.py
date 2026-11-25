"""Post-training evaluation of demo_circle, demo_line and demo_triangle.

Uses the run's best.pt and its training statistics. External results never enter
training, scheduler decisions or best-checkpoint selection. Can also be run on
an existing checkpoint without retraining: python -m deployment.demo_evaluation.
"""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .compare_models import digest, load_inputs, select_windows, verify_package
from .trajectory_plot import save_trajectory_sheet
from mvts.dual_rate_data import TARGET_NAMES
from mvts.engine import choose_device, load_checkpoint
from mvts.regression import dimension_metrics, masked_huber, zero_wrench_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "deployment/test_data"
DATASET_NAMES = ("demo_circle", "demo_line", "demo_triangle")
SUPPORTED_MODELS = frozenset(("model_triple_head", "model_slip_gated", "model_contact_gated"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def preflight(data_root=DEFAULT_DATA_ROOT, trim_mm=0., filter_xy=False):
    if not math.isfinite(trim_mm) or trim_mm < 0:
        raise ValueError("demo-trim-stationary-xy-mm must be finite and nonnegative")
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Automatic demo plots require matplotlib; run uv sync, or use --no-demo-tests")
    root = Path(data_root).resolve()
    for name in DATASET_NAMES:
        package = root / name
        for filename in ("manifest.json", "events.jsonl"):
            if not (package / filename).is_file():
                raise FileNotFoundError(f"Missing external demo input: {package/filename}; "
                                        "set --demo-test-root or use --no-demo-tests")
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        selected, _ = select_windows(manifest, trim_mm, filter_xy)
        if not selected:
            raise ValueError(f"{name}: external test selection is empty; adjust --demo-filter-xy-range "
                             "or --demo-trim-stationary-xy-mm")


def compute_losses(raw, inputs, stats, training):
    """Same normalized masked objectives as training, aggregated over all events."""
    mean, std = [torch.tensor(stats[k], dtype=torch.float32) for k in ("target_mean", "target_std")]
    target = torch.tensor((inputs["target"] - np.asarray(stats["target_mean"])) /
                          np.asarray(stats["target_std"]), dtype=torch.float32)
    valid = torch.from_numpy(inputs["valid"])
    result = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in raw.items()}
    cv = torch.as_tensor(inputs["contact_valid"], dtype=torch.bool)
    contact = torch.as_tensor(inputs["contact"], dtype=torch.long)
    terms = {key: masked_huber(result[key], target[:, sl], valid[:, sl]) for key, sl in
             (("position", slice(0, 3)), ("wrench", slice(3, 9)), ("slip", slice(9, 11)))}
    terms["zero_wrench"] = zero_wrench_loss(result["wrench"], mean[3:9], std[3:9], cv & (contact == 0))
    if "slip_prior" in result:
        terms["slip_prior"] = masked_huber(result["slip_prior"], target[:, 9:], valid[:, 9:])
    if "contact_logits" in result:
        bce = F.binary_cross_entropy_with_logits(result["contact_logits"], contact.float(), reduction="none")
        weights = torch.tensor(stats["contact_class_weights"], dtype=torch.float32)[contact]
        terms["contact"] = torch.where(cv, bce * weights, 0.).sum() / cv.sum().clamp_min(1)
    weights = {k: training.get(k + "_weight", 1.) for k in terms}
    weights = {k: 1. if v is None else float(v) for k, v in weights.items()}
    main = sum(weights[k] * terms[k] for k in ("position", "wrench", "slip", "zero_wrench"))
    total = sum(weights[k] * value for k, value in terms.items())
    return {"loss": float(total), "main_loss": float(main),
            **{k + "_loss": float(v) for k, v in terms.items()}, "loss_weights": weights}


def infer(model, checkpoint, inputs, device, name):
    stats = checkpoint["stats"]
    image_only = bool(model.config.image_only)
    contact_aware = model.model_name == "model_contact_gated"
    slip_guided = bool(getattr(model, "requires_timing", False))
    audio = None
    if not image_only:
        normalized = ((inputs["audio"].astype(np.float64) - np.asarray(stats["audio_mean"])[None, :, None, None]) /
                      np.asarray(stats["audio_std"])[None, :, None, None]).astype(np.float32)
        audio = torch.tensor(normalized, device=device)
    images = {key: value.to(device) for key, value in inputs["images"].items()}
    raw = {k: [] for k in ("position", "wrench", "slip")}
    if slip_guided:
        raw["slip_prior"] = []
    if contact_aware:
        raw["contact_logits"] = []
    predicted, position_valid, slip_valid, alpha = [], [], [], []
    mean, std = np.asarray(stats["target_mean"]), np.asarray(stats["target_std"])
    model.eval()
    model.reset_state()
    with torch.inference_mode():
        for i, row in enumerate(inputs["rows"]):
            timing = {"delta_t": float(inputs["delta_t"][i]),
                      "image_age": float(inputs["image_age"][i])} if slip_guided else {}
            kwargs = {"image": images[row["image"]] if row["image_update"] else None,
                      "frame_id": row["image"], "reset": row["reset_state"], **timing}
            current = model.step(**kwargs) if image_only else model.step(audio[i:i+1], **kwargs)
            for key in raw:
                raw[key].append(current[key][0].detach().cpu().numpy())
            raw_vector = np.concatenate([raw[k][-1] for k in ("position", "wrench", "slip")])
            physical = raw_vector.astype(np.float64) * std + mean
            predicted.append(current["physical_prediction"][0].cpu().numpy() if contact_aware else physical)
            position_valid.append(bool(current["position_valid"][0].cpu()) if contact_aware else True)
            slip_valid.append(bool(current["slip_valid"][0].cpu()) if contact_aware else True)
            alpha.append(float(current["visual_weight"][0].cpu()) if slip_guided else np.nan)
            if i + 1 == len(inputs["rows"]) or inputs["rows"][i+1]["sequence"] != row["sequence"]:
                print(f"[{name}] {row['sequence'].rsplit('/',1)[-1]}: {i+1}/{len(inputs['rows'])} events", flush=True)
    raw = {k: np.stack(values) for k, values in raw.items()}
    raw_prediction = np.concatenate([raw[k] for k in ("position", "wrench", "slip")], -1).astype(np.float64) * std + mean
    prediction = np.stack(predicted)
    if not np.isfinite(prediction).all() or not np.isfinite(raw_prediction).all():
        raise FloatingPointError("Nonfinite external demo predictions")
    target, valid = inputs["target"].copy(), inputs["valid"].copy()
    unloaded = inputs["contact_valid"] & (inputs["contact"] == 0)
    target[unloaded, 3:9] = 0.
    valid[unloaded, 3:9] = True
    error = prediction - target
    xy_mask = valid[:, :2].all(-1)
    xy = np.linalg.norm(error[:, :2], axis=-1)[xy_mask]
    metrics = {**compute_losses(raw, inputs, stats, checkpoint["args"]),
        "events": len(prediction), "trajectories": len({r['sequence'] for r in inputs['rows']}),
        "mae": dimension_metrics(prediction, target, valid),
        "rmse": dimension_metrics(prediction, target, valid, True),
        "raw_mae": dimension_metrics(raw_prediction, inputs["target"], inputs["valid"]),
        "xy_mean_euclidean_mm": float(xy.mean()) if len(xy) else None,
        "xy_rmse_mm": float(np.sqrt((xy * xy).mean())) if len(xy) else None,
        "position_output_coverage": float(np.asarray(position_valid)[xy_mask].mean()) if xy_mask.any() else None}
    if slip_guided:
        metrics["visual_weight_mean"] = float(np.mean(alpha))
    return metrics, {"prediction": prediction, "raw_prediction": raw_prediction, "raw": raw,
                     "position_valid": np.asarray(position_valid), "slip_valid": np.asarray(slip_valid),
                     "visual_weight": np.asarray(alpha)}


def print_metrics(name, metrics, epoch):
    mae = metrics["mae"]
    value = lambda k: f"{mae[k]:.6f}" if mae[k] is not None else "N/A"
    print(f"[{name}] best_epoch={epoch} trajectories={metrics['trajectories']} events={metrics['events']} "
          f"loss={metrics['loss']:.6f} main_loss={metrics['main_loss']:.6f}", flush=True)
    print(f"[{name}] position_mae (mm): x={value('x_mm')}, y={value('y_mm')}, delta_z={value('delta_z_mm')}", flush=True)
    print(f"[{name}] wrench_mae: " + ", ".join(f"{k}={value(k)} {'N' if k.startswith('F') else 'N*m'}"
          for k in TARGET_NAMES[3:9]), flush=True)
    print(f"[{name}] slip_mae (mm): x={value('slip_x_mm')}, y={value('slip_y_mm')}", flush=True)


def evaluate_demo_suite(checkpoint_path, output, *, data_root=DEFAULT_DATA_ROOT,
                        device="auto", trim_mm=0., filter_xy=False):
    preflight(data_root, trim_mm, filter_xy)
    output, data_root, checkpoint_path = Path(output).resolve(), Path(data_root).resolve(), Path(checkpoint_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    details = output / "demo_tests"
    details.mkdir(exist_ok=True)
    device = choose_device(device)
    model, checkpoint, features = load_checkpoint(checkpoint_path, device)
    if model.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"External demo evaluation requires a dual-rate model, got {model.model_name}")
    info = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": digest(checkpoint_path),
            "best_epoch": checkpoint["epoch"], "model_name": model.model_name,
            "disabled_mic_channels": list(model.config.disabled_mic_channels),
            "image_only": bool(model.config.image_only), "device": str(device),
            "filter_xy_range": bool(filter_xy), "trim_stationary_xy_mm": trim_mm,
            "normalization_policy": "Saved training statistics; no test-set fitting",
            "inference_batch_size": 1, "state_cadence": "one update per actual source event",
            "tf32_enabled": False,
            "timing": "actual event_pc_ns/previous_pc_ns; never nominal replay offsets",
            "postprocessing": "none; no smoothing, alignment or slip integration",
            "datasets": {}, "status": "running"}
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    errors = []
    try:
        for name in DATASET_NAMES:
            folder = details / name
            folder.mkdir(exist_ok=True)
            try:
                root = data_root / name
                manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
                if manifest["features"] != features.to_dict():
                    raise ValueError(f"{name}: preprocessing differs from checkpoint")
                selected, selection = select_windows(manifest, trim_mm, filter_xy)
                write_json(folder/"selection.json", selection)
                count = verify_package(root) if (root/"checksums.sha256").is_file() else None
                inputs = load_inputs(root, selected)
                external_ids = {r["source_window_id"] for r in inputs["rows"]}
                if external_ids & set(checkpoint.get("split_ids", {}).get("train", [])):
                    raise ValueError("External demo window IDs overlap the checkpoint training IDs")
                metrics, predictions = infer(model, checkpoint, inputs, device, name)
                metrics.update(best_epoch=checkpoint["epoch"], model_name=model.model_name,
                               data_root=str(root), package_files_verified=count)
                figures = save_trajectory_sheet(output, name, inputs, predictions["prediction"], metrics,
                    model.model_name, checkpoint["epoch"], trim_mm, filter_xy,
                    position_valid=predictions["position_valid"], image_only=model.config.image_only)
                write_json(folder/"metrics.json", metrics)
                np.savez_compressed(folder/"predictions.npz", target=inputs["target"], target_valid=inputs["valid"],
                    prediction=predictions["prediction"], raw_prediction=predictions["raw_prediction"],
                    sequence=np.asarray([r["sequence"] for r in inputs["rows"]]),
                    event_pc_ns=np.asarray([r["event_pc_ns"] for r in inputs["rows"]], dtype=np.int64),
                    reset_state=np.asarray([r["reset_state"] for r in inputs["rows"]]),
                    position_valid=predictions["position_valid"], slip_valid=predictions["slip_valid"])
                with (folder/"predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["sequence", "sequence_index", "event_pc_ns", "position_valid", "slip_valid"]
                        + [prefix + k for prefix in ("true_", "pred_") for k in TARGET_NAMES])
                    for i,row in enumerate(inputs["rows"]):
                        writer.writerow([row["sequence"], row["sequence_index"], row["event_pc_ns"],
                            predictions["position_valid"][i], predictions["slip_valid"][i],
                            *inputs["target"][i], *predictions["prediction"][i]])
                info["datasets"][name] = {"status": "complete", **metrics, "figures": figures}
                print_metrics(name, metrics, checkpoint["epoch"])
            except Exception as exc:
                info["datasets"][name] = {"status": "failed", "error": str(exc)}
                errors.append(f"{name}: {exc}")
                print(f"[{name}] external evaluation failed: {exc}", flush=True)
            write_json(details/"summary.json", info)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous_tf32
    info["status"] = "failed" if errors else "complete"
    write_json(details/"summary.json", info)
    if errors:
        raise RuntimeError("Training checkpoint is preserved; external demo evaluation failed: " + "; ".join(errors))
    return info


def after_training(args):
    if not getattr(args, "demo_tests", True) or args.audit_only:
        return
    if args.model not in SUPPORTED_MODELS:
        print(f"External demo tests skipped: {args.model} does not use the dual-rate 11-D interface", flush=True)
        return
    return evaluate_demo_suite(Path(args.output)/"best.pt", args.output,
        data_root=args.demo_test_root, device=args.device,
        trim_mm=args.demo_trim_stationary_xy_mm, filter_xy=args.demo_filter_xy_range)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to the checkpoint directory")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trim-stationary-xy-mm", type=float, default=0.)
    parser.add_argument("--filter-xy-range", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    evaluate_demo_suite(args.checkpoint, args.output or args.checkpoint.parent, data_root=args.data_root,
        device=args.device, trim_mm=args.trim_stationary_xy_mm, filter_xy=args.filter_xy_range)


if __name__ == "__main__":
    main()
