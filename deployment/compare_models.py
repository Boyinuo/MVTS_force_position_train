"""Compare two saved best.pt models on the same external deployment package.

Uses physical-unit, unshifted predictions. Selection happens once before either
model runs. Each stream sees identical images, audio, timestamps and reset flags.
No training, test-set normalization fitting, smoothing or trajectory registration.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mvts.dual_rate_data import TARGET_NAMES
from mvts.engine import choose_device, load_checkpoint, seed_everything
from mvts.features import load_image
from mvts.prepared_selection import filter_xy_range, trim_prepared_records


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Package path escapes its data directory: {relative}")
    return path


def select_windows(manifest, trim_mm, use_range):
    records = [{**r, "file": r["event_files"][0]} for r in manifest["records"]]
    ranged, range_report = filter_xy_range(records, use_range)
    selected, trim_report = trim_prepared_records(ranged, trim_mm)
    # Retain the strict-rule audit even when the explicitly selected test policy
    # disables the XY filter. It documents why all ten circles would be removed.
    _, strict_report = filter_xy_range(records, True)
    trajectories = []
    for seq in manifest["sequences"]:
        source = [r for r in records if r["sequence"] == seq["sequence"]]
        retained = [r for r in selected if r["sequence"] == seq["sequence"]]
        xy = np.concatenate([np.asarray(r["target"])[:, :2] for r in source])
        trajectories.append({"sequence": seq["sequence"], "trajectory": seq["sequence"].rsplit("/", 1)[-1],
            "source_windows": len(source), "source_events": sum(r["events"] for r in source),
            "retained_windows": len(retained), "retained_events": sum(r["events"] for r in retained),
            "x_range_mm": [float(xy[:, 0].min()), float(xy[:, 0].max())],
            "y_range_mm": [float(xy[:, 1].min()), float(xy[:, 1].max())],
            "cutoff_window_order": trim_report["trajectory_cutoffs"].get(seq["sequence"])})
    return selected, {"actual_xy_selection": range_report, "trim_selection": trim_report,
                      "strict_xy_filter_audit": strict_report, "trajectories": trajectories}


def verify_package(root):
    n = 0
    for line in (root / "checksums.sha256").read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split(maxsplit=1)
        if digest(checked_path(root, relative.strip())) != expected:
            raise ValueError(f"Dataset checksum mismatch: {relative}")
        n += 1
    return n


def load_inputs(root, selected):
    all_rows = [json.loads(line) for line in (root / "events.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    by_file = {r["data"]: r for r in all_rows}
    if len(by_file) != len(all_rows):
        raise ValueError("Duplicate event files in deployment index")
    rows, audio, targets, valid, images, dt, age = [], [], [], [], {}, [], []
    contacts, contact_validity = [], []
    previous_order = {}
    for record in selected:
        expected = np.asarray(record["target"], dtype=np.float64)
        anchor = None
        for t, filename in enumerate(record["event_files"]):
            row = dict(by_file[filename])
            if (row["sequence"] != record["sequence"] or row["source_window_id"] != record["id"]
                    or row["source_sub_event"] != t or row["image"] != record["image"]):
                raise ValueError("Manifest and event index disagree")
            with np.load(checked_path(root, filename), allow_pickle=False) as z:
                mel, target, mask = z["audio"].copy(), z["target"].copy(), z["target_valid"].copy()
                stamp, previous = int(z["event_pc_ns"]), int(z["previous_pc_ns"])
                contact = int(z["contact"]) if "contact" in z else 0
                contact_valid = bool(z["contact_valid"]) if "contact_valid" in z else False
            if mel.shape != (6, 64, 28) or target.shape != (11,) or mask.shape != (11,):
                raise ValueError(f"Invalid event array shape: {filename}")
            np.testing.assert_array_equal(target, expected[t])
            if not np.isfinite(mel).all() or not np.isfinite(target).all() or stamp <= previous:
                raise ValueError(f"Invalid numerical input or event interval: {filename}")
            if row["event_pc_ns"] != stamp:
                raise ValueError("Event timestamp differs from the replay index")
            if t == 0:
                anchor = stamp  # Exported subevent zero is exactly the left image anchor.
                prior_order = previous_order.get(record["sequence"])
                row["reset_state"] = bool(row["reset_state"] or prior_order is None
                                           or prior_order + 1 != record["order"])
            if stamp < anchor:
                raise ValueError("Negative image age")
            if row["image"] not in images:
                images[row["image"]] = load_image(checked_path(root, row["image"]))[None]
            rows.append(row)
            audio.append(mel)
            targets.append(target)
            valid.append(mask.astype(bool))
            contacts.append(contact)
            contact_validity.append(contact_valid)
            dt.append((stamp - previous) * 1e-9)
            age.append((stamp - anchor) * 1e-9)
        previous_order[record["sequence"]] = record["order"]
    return {"rows": rows, "audio": np.stack(audio), "target": np.stack(targets),
            "valid": np.stack(valid), "images": images,
            "contact": np.asarray(contacts), "contact_valid": np.asarray(contact_validity),
            "delta_t": np.asarray(dt, dtype=np.float32), "image_age": np.asarray(age, dtype=np.float32)}


def checkpoint_info(path):
    model, c, features = load_checkpoint(path, "cpu")
    if model.model_name not in ("model_triple_head", "model_slip_gated") or model.config.image_only:
        raise ValueError("This comparison requires the two visual-acoustic triple-head models")
    if tuple(model.config.disabled_mic_channels) != (5,):
        raise ValueError("Both requested checkpoints must disable microphone 5")
    info = {"checkpoint": str(path), "checkpoint_sha256": digest(path), "epoch": c["epoch"],
            "model_name": model.model_name, "model_config": c["model_config"],
            "feature_config": features.to_dict(), "stats": c["stats"], "training": c["args"]}
    return model, c, info


def check_comparable(a, b, manifest):
    if a["feature_config"] != b["feature_config"] or a["feature_config"] != manifest["features"]:
        raise ValueError("Feature preprocessing must be identical across data and checkpoints")
    config_keys = ("d_model", "hidden_size", "nhead", "dropout", "audio_frames", "use_contact",
                   "audio_alignment_mode", "audio_samples_before", "audio_samples_after",
                   "disabled_mic_channels", "image_only")
    for key in config_keys:
        if a["model_config"][key] != b["model_config"][key]:
            raise ValueError(f"Shared model configuration differs: {key}")
    training_keys = ("epochs", "batch_size", "lr", "weight_decay", "seed", "val_fraction", "test_fraction",
                     "disable_mic_channels", "filter_xy_range", "trim_stationary_xy_mm",
                     "position_weight", "wrench_weight", "slip_weight", "zero_wrench_weight")
    for key in training_keys:
        if a["training"].get(key) != b["training"].get(key):
            raise ValueError(f"Shared training configuration differs: {key}")
    stats_equal = {}
    for key in ("audio_mean", "audio_std", "target_mean", "target_std"):
        stats_equal[key] = bool(np.array_equal(a["stats"][key], b["stats"][key]))
        if not stats_equal[key]:
            raise ValueError(f"This matched comparison expects identical saved training statistics: {key}")
    return {"shared_model_parameters": {k:a["model_config"][k] for k in config_keys},
            "shared_training_parameters": {k:a["training"].get(k) for k in training_keys},
            "normalization_arrays_exactly_equal": stats_equal,
            "normalization_policy": "Each checkpoint's own saved training statistics; never fit on demo data",
            "inference_batch_size": 1, "inference_dtype": "float32", "streaming": True,
            "timing": "actual event_pc_ns / previous_pc_ns; image age from source image-window anchor",
            "state_resets": "identical resets at new trajectories or acquisition gaps",
            "postprocessing": "none: no smoothing, coordinate registration, clipping or slip integration"}


def infer(model, checkpoint, inputs, device, label):
    model.to(device).eval()
    model.reset_state()
    stats = checkpoint["stats"]
    audio = torch.tensor(((inputs["audio"].astype(np.float64)
        - np.asarray(stats["audio_mean"])[None, :, None, None])
        / np.asarray(stats["audio_std"])[None, :, None, None]).astype(np.float32), device=device)
    images = {k:v.to(device) for k,v in inputs["images"].items()}
    mean, std = np.asarray(stats["target_mean"]), np.asarray(stats["target_std"])
    results, visual_weights, prior_speeds = [], [], []
    slip_guided = getattr(model, "requires_timing", False)
    # Check the single-event streaming API against ordered sequence inference on
    # the first complete image window. This does not change the replay state.
    first_window = inputs["rows"][0]["source_window_id"]
    length = sum(r["source_window_id"] == first_window for r in inputs["rows"])
    with torch.inference_mode():
        timing = {"delta_t": torch.tensor(inputs["delta_t"][:length][None], device=device),
                  "image_age": torch.tensor(inputs["image_age"][:length][None], device=device)} if slip_guided else {}
        reference = model.forward_sequence(images[inputs["rows"][0]["image"]], audio[:length][None], **timing)
        for i, row in enumerate(inputs["rows"]):
            kwargs = {"delta_t": float(inputs["delta_t"][i]),
                      "image_age": float(inputs["image_age"][i])} if slip_guided else {}
            current = model.step(audio[i:i+1], image=images[row["image"]] if row["image_update"] else None,
                                 frame_id=row["image"], reset=row["reset_state"], **kwargs)
            if i < length:
                for key in ("position", "wrench", "slip"):
                    torch.testing.assert_close(current[key], reference[key][:, i], rtol=1e-4, atol=1e-5)
            normalized = torch.cat([current[k] for k in ("position", "wrench", "slip")], -1)[0].cpu().numpy()
            results.append(normalized.astype(np.float64) * std + mean)
            visual_weights.append(float(current["visual_weight"][0].cpu()) if slip_guided else np.nan)
            prior_speeds.append(float(current["prior_speed_mm_s"][0].cpu()) if slip_guided else np.nan)
            if i + 1 == len(inputs["rows"]) or inputs["rows"][i+1]["sequence"] != row["sequence"]:
                print(f"{label}: {row['sequence'].rsplit('/',1)[-1]} done ({i+1}/{len(inputs['rows'])} events)", flush=True)
    prediction = np.stack(results)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite physical-unit prediction")
    return {"prediction": prediction, "visual_weight": np.asarray(visual_weights),
            "prior_speed_mm_s": np.asarray(prior_speeds), "streaming_sequence_check_passed": True}


def metrics(prediction, target, valid):
    err = prediction - target
    n = valid.sum(0)
    mae = np.where(valid, np.abs(err), 0.).sum(0) / np.maximum(n, 1)
    rmse = np.sqrt(np.where(valid, err**2, 0.).sum(0) / np.maximum(n, 1))
    xy = np.linalg.norm(err[:, :2], axis=1)[valid[:, :2].all(1)]
    slip = np.linalg.norm(err[:, 9:], axis=1)[valid[:, 9:].all(1)]
    return {"events": len(target), "mae": dict(zip(TARGET_NAMES, mae.tolist())),
            "rmse": dict(zip(TARGET_NAMES, rmse.tolist())),
            "valid_counts": dict(zip(TARGET_NAMES, n.tolist())),
            "xy_mean_euclidean_mm": float(xy.mean()), "xy_rmse_mm": float(np.sqrt((xy**2).mean())),
            "xy_p95_mm": float(np.quantile(xy, .95)),
            "slip_mean_euclidean_mm": float(slip.mean())}


def save_predictions(path, inputs, result):
    columns = ["sequence", "sequence_index", "call_index", "event_pc_ns", "delta_t_s", "image_age_s",
               "image_update", "reset_state", "image", "visual_weight", "prior_speed_mm_s"]
    columns += [f"true_{name}" for name in TARGET_NAMES] + [f"pred_{name}" for name in TARGET_NAMES]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, row in enumerate(inputs["rows"]):
            writer.writerow([row["sequence"], row["sequence_index"], row["call_index"], row["event_pc_ns"],
                inputs["delta_t"][i], inputs["image_age"][i], row["image_update"], row["reset_state"], row["image"],
                result["visual_weight"][i] if np.isfinite(result["visual_weight"][i]) else "",
                result["prior_speed_mm_s"][i] if np.isfinite(result["prior_speed_mm_s"][i]) else "",
                *inputs["target"][i], *result["prediction"][i]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "deployment/test_data/demo_circle")
    parser.add_argument("--baseline-checkpoint", type=Path, default=ROOT / "runs/v6_triple_head_slide_polyline_press_no_mic5_trim0p029_xyfilter/best.pt")
    parser.add_argument("--candidate-checkpoint", type=Path, default=ROOT / "runs/new_model_v6_slip_gated/best.pt")
    parser.add_argument("--trim-stationary-xy-mm", type=float, default=.029)
    parser.add_argument("--filter-xy-range", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/v6_demo_circle_compare_trim0p029")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    out, root = args.output.resolve(), args.data_root.resolve()
    if (out / "comparison.json").exists() or (out / "baseline_predictions.csv").exists():
        raise FileExistsError(f"Comparison output already exists: {out}; choose another --output")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    seed_everything(42)
    # Match both estimators under IEEE float32 rather than permitting TF32
    # contractions to differ between cuDNN's sequence and single-step GRU paths.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = choose_device(args.device)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["shape"] != "circle" or len(manifest["sequences"]) != 10:
        raise ValueError("Expected the external ten-trajectory demo_circle package")
    selected, selection = select_windows(manifest, args.trim_stationary_xy_mm, args.filter_xy_range)
    write_json(out / "selection.json", selection)
    if not selected:
        write_json(out / "comparison.json", {"status": "empty_selection", "events": 0,
            "selection": selection, "note": "No inference was run; the strict XY rule removes all ten trajectories."})
        raise ValueError("No trajectories survive selection; inspect selection.json")
    checked = verify_package(root)
    print(f"Dataset checksums verified: {checked} files; selected {len(selected)} windows", flush=True)
    inputs = load_inputs(root, selected)
    baseline, ca, ia = checkpoint_info(args.baseline_checkpoint.resolve())
    candidate, cb, ib = checkpoint_info(args.candidate_checkpoint.resolve())
    if baseline.model_name != "model_triple_head" or candidate.model_name != "model_slip_gated":
        raise ValueError("Baseline/candidate checkpoint architecture order is incorrect")
    comparable = check_comparable(ia, ib, manifest)
    comparable["tf32_enabled"] = False
    if ca["split_ids"] != cb["split_ids"] or ca["split_labels"] != cb["split_labels"]:
        raise ValueError("The two checkpoints do not share identical saved training partitions")
    external_ids = {r["source_window_id"] for r in inputs["rows"]}
    overlap = {name: len(external_ids & set(c["split_ids"]["train"]))
               for name,c in (("baseline", ca), ("candidate", cb))}
    if any(overlap.values()):
        raise ValueError("External test window IDs overlap checkpoint training IDs")
    a = infer(baseline, ca, inputs, device, "triple_head")
    b = infer(candidate, cb, inputs, device, "slip_gated")
    target, valid = inputs["target"], inputs["valid"]
    overall_a, overall_b = metrics(a["prediction"], target, valid), metrics(b["prediction"], target, valid)
    sequence_names = list(dict.fromkeys(r["sequence"] for r in inputs["rows"]))
    per_trajectory = []
    for seq in sequence_names:
        mask = np.asarray([r["sequence"] == seq for r in inputs["rows"]])
        ma, mb = [metrics(result["prediction"][mask], target[mask], valid[mask]) for result in (a,b)]
        per_trajectory.append({"sequence": seq, "trajectory": seq.rsplit("/",1)[-1],
            "events": int(mask.sum()), "baseline": ma, "candidate": mb,
            "xy_error_change_percent": 100. * (mb["xy_mean_euclidean_mm"] / ma["xy_mean_euclidean_mm"] - 1.)})
    comparison = {"status": "complete", "data_root": str(root), "device": str(device),
        "trajectories": len(sequence_names), "windows": len(selected), "events": len(inputs["rows"]),
        "selection": {"filter_xy_range": args.filter_xy_range, "trim_stationary_xy_mm": args.trim_stationary_xy_mm},
        "dataset_files_checksum_verified": checked, "original_manifest_sha256": digest(root/"manifest.json"),
        "actual_timing_used": True, "normalization_fitted_on_test": False, "training_id_overlap": overlap,
        "configuration_audit": comparable, "checkpoints": {"baseline": ia, "candidate": ib},
        "baseline": overall_a, "candidate": overall_b,
        "mae_change_percent": {k:100.*(overall_b["mae"][k]/overall_a["mae"][k]-1.) for k in TARGET_NAMES},
        "xy_error_change_percent": 100.*(overall_b["xy_mean_euclidean_mm"]/overall_a["xy_mean_euclidean_mm"]-1.),
        "candidate_xy_trajectory_wins": sum(x["candidate"]["xy_mean_euclidean_mm"] < x["baseline"]["xy_mean_euclidean_mm"] for x in per_trajectory),
        "macro_xy_error_mm": {name:float(np.mean([x[name]["xy_mean_euclidean_mm"] for x in per_trajectory])) for name in ("baseline", "candidate")},
        "streaming_sequence_check_passed": a["streaming_sequence_check_passed"] and b["streaming_sequence_check_passed"],
        "per_trajectory": per_trajectory}
    write_json(out/"comparison.json", comparison)
    save_predictions(out/"baseline_predictions.csv", inputs, a)
    save_predictions(out/"candidate_predictions.csv", inputs, b)
    np.savez_compressed(out/"predictions.npz", target=target, target_valid=valid,
        baseline=a["prediction"], candidate=b["prediction"],
        sequence=np.asarray([r["sequence"] for r in inputs["rows"]]),
        event_pc_ns=np.asarray([r["event_pc_ns"] for r in inputs["rows"]], dtype=np.int64),
        source_window_id=np.asarray([r["source_window_id"] for r in inputs["rows"]]),
        reset_state=np.asarray([r["reset_state"] for r in inputs["rows"]]),
        image_update=np.asarray([r["image_update"] for r in inputs["rows"]]),
        candidate_visual_weight=b["visual_weight"], candidate_prior_speed_mm_s=b["prior_speed_mm_s"])
    with (out/"trajectory_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory", "events", "baseline_xy_mean_mm", "candidate_xy_mean_mm", "xy_change_percent"]
            + [f"{model}_{name}_mae" for name in TARGET_NAMES for model in ("baseline", "candidate")])
        for x in per_trajectory:
            writer.writerow([x["trajectory"], x["events"], x["baseline"]["xy_mean_euclidean_mm"],
                x["candidate"]["xy_mean_euclidean_mm"], x["xy_error_change_percent"]]
                + [x[model]["mae"][name] for name in TARGET_NAMES for model in ("baseline", "candidate")])
    print(json.dumps({"events":comparison["events"],"baseline":overall_a,"candidate":overall_b,
        "xy_error_change_percent":comparison["xy_error_change_percent"],
        "candidate_xy_trajectory_wins":comparison["candidate_xy_trajectory_wins"]},ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
