"""Read event-export data, audit alignment, split by planned physical location."""
from collections import Counter
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .features import FeatureConfig, LogMel, extract_event_audio, load_image


TARGET_NAMES = ["x_mm", "y_mm", "delta_z_mm", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def location_group(record):
    # Use planned x/y so actual-position measurement jitter cannot separate stages.
    # Intentionally independent of session/mode: resumed sessions and staircase /
    # dynamic observations at the same position must stay in the same partition.
    return f"{float(record['x_mm']):.6f},{float(record['y_mm']):.6f}"


def _safe_file(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside session: {relative}")
    if not candidate.is_file():
        raise ValueError(f"Missing file: {candidate}")
    return candidate


def extract_sample_audio(data, feature_config, force_index=None):
    """Adapt grouped force/audio exports to the existing event-window extractor.

    Legacy exports return [N,6]. Grouped exports return [K,N,6], or [N,6]
    when selecting one force_index. Window metadata refers to each force.
    """
    adc = data["adc"]
    if adc.ndim == 2:
        if force_index is not None:
            raise ValueError("force_index requires grouped ADC")
        return extract_event_audio(data, config=feature_config)
    if adc.ndim != 3 or adc.shape[0] == 0 or adc.shape[-1] != 6:
        raise ValueError(f"Invalid ADC: {adc.shape}")
    if "audio_before" not in data or "audio_after" not in data:
        raise ValueError("Grouped ADC requires audio_before and audio_after metadata")
    window = {key: data[key] for key in ("audio_before", "audio_after")}
    if force_index is not None:
        if type(force_index) is not int or not 0 <= force_index < len(adc):
            raise ValueError("Invalid force_index for grouped ADC")
        return extract_event_audio({**window, "adc": adc[force_index]}, config=feature_config)
    return np.stack([extract_event_audio({**window, "adc": row}, config=feature_config)
                     for row in adc])


def _audit_grouped_forces(data, record, wrench, pc_times):
    """Check every force/audio pairing before an event is expanded atomically."""
    count = pc_times.shape[0]
    force_times = np.asarray(data["force_pc_ns"])
    if (force_times.shape != (count,) or force_times.dtype.kind not in "iu"
            or (np.diff(force_times.astype(np.float64)) <= 0).any()
            or (force_times > record["pc_ns"]).any()):
        raise ValueError("Invalid grouped force timestamps")
    sequence = record["force_sequence"]
    if not isinstance(sequence, list) or len(sequence) != count:
        raise ValueError("NPZ/JSON force sequence length mismatch")
    json_wrench = np.asarray([force["wrench"] for force in sequence], dtype=np.float64)
    if json_wrench.shape != wrench.shape or not np.allclose(wrench, json_wrench, atol=1e-6, rtol=0):
        raise ValueError("NPZ/JSON force sequence mismatch")
    if not np.array_equal(force_times, [force["pc_est_ns"] for force in sequence]):
        raise ValueError("NPZ/JSON force timestamp mismatch")
    # Pre-force exports contain samples strictly preceding the force. The next
    # sample brackets that force timestamp, including a possible acquisition gap.
    next_times = np.asarray(data["audio_next_pc_ns"], dtype=np.float64)
    if (next_times.shape != (count,) or not np.isfinite(next_times).all()
            or (pc_times[:, -1] >= force_times).any()
            or (next_times < force_times).any()):
        raise ValueError("Grouped audio does not bracket its corresponding force")
    return force_times


def discover_records(data_root, kinds=("target", "limit"), modes=("staircase", "dynamic"),
                     strict_timing=False, feature_config=None):
    """Audit every included sample; bad records are reported, never fabricated.

    Source-valid false / excluded kinds are counted separately. Any malformed
    included record is reported in rejected, and train.py refuses unless explicitly
    passed --skip-bad-records. Original data are never modified.
    """
    feature_config = feature_config or FeatureConfig()
    root = Path(data_root).resolve()
    manifests = sorted(root.rglob("samples.jsonl"))
    if not manifests:
        raise ValueError(f"No samples.jsonl under {root}")
    records, rejected, sessions = [], [], []
    exclusions, counts = Counter(), Counter()
    camera_ages, force_ages, rates, adc_ranges = [], [], [], []
    nonuniform_audio, marker_outside, samples_seen, included_events = 0, 0, 0, 0
    grouped_events = 0
    source_audio_samples = Counter()
    for manifest in manifests:
        session = manifest.parent
        audio_meta_path, meta_path = session / "audio/session.json", session / "session.json"
        audio_meta = read_json(audio_meta_path) if audio_meta_path.exists() else {}
        meta = read_json(meta_path) if meta_path.exists() else {}
        summary_path = session / "dataset_summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        rate = audio_meta.get("sample_rate_observed_hz", audio_meta.get("sample_rate_requested_hz"))
        if rate is None:
            rate = summary.get("sample_rate_hz")
        sources = summary.get("sources", [])
        timing_passed = audio_meta.get("timing_quality_passed")
        if not audio_meta and sources:
            timing_passed = all(s.get("audio_timing_quality_passed") is True for s in sources)
        completed = meta.get("completed")
        if completed is None and sources:
            completed = all(s.get("whole_session_completed") is True for s in sources)
        warnings = []
        if audio_meta.get("valid_for_time_matching") is False:
            warnings.append("audio.valid_for_time_matching=false; inspect acquisition clock/alignment")
        if meta.get("microphone_timing_valid") is False:
            warnings.append("session.microphone_timing_valid=false")
        if completed is not True:
            warnings.append("acquisition session incomplete; use exported valid events only")
        sessions.append({"session": session.name, "sample_rate_hz": rate,
                         "timing_quality_passed": timing_passed,
                         "valid_for_time_matching": audio_meta.get("valid_for_time_matching"),
                         "warnings": warnings})
        if strict_timing and (warnings or timing_passed is not True):
            raise ValueError(f"Strict timing rejected {session.name}: {warnings}")
        for lineno, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            samples_seen += 1
            try:
                r = json.loads(line)
                if r.get("valid") is not True:
                    exclusions["source_invalid"] += 1
                    continue
                if r.get("kind") not in kinds or r.get("mode") not in modes:
                    exclusions[f"{r.get('mode')}/{r.get('kind')}"] += 1
                    continue
                image_path = _safe_file(session, r["image"])
                npz_path = _safe_file(session, r["data"])
                if rate is None or not np.isfinite(rate) or rate <= 0:
                    raise ValueError("Missing/invalid acquisition sample rate")
                with Image.open(image_path) as im:
                    im.verify()
                with np.load(npz_path, allow_pickle=False) as data:
                    adc = data["adc"]
                    # Audit the original window and require the configured event window.
                    event_adc = extract_sample_audio(data, feature_config)
                    grouped = adc.ndim == 3
                    sample_count = len(adc) if grouped else 1
                    xy = np.asarray(data["actual_xy_mm"], dtype=np.float64)
                    wrench = np.asarray(data["force_wrench"], dtype=np.float64)
                    contact = int(data["starts_in_contact"].item())
                    wrench_shape = (sample_count, 6) if grouped else (6,)
                    if xy.shape != (2,) or wrench.shape != wrench_shape or contact not in (0, 1):
                        raise ValueError("Invalid label shape or contact flag")
                    position = np.concatenate((xy, [float(r["actual_depth_mm"])]))
                    if not np.isfinite(position).all() or not np.isfinite(wrench).all():
                        raise ValueError("Non-finite measured labels")
                    if not np.allclose(xy, r["actual_xy_mm"], atol=1e-6, rtol=0):
                        raise ValueError("NPZ/JSON actual_xy mismatch")
                    latest_wrench = wrench[-1] if grouped else wrench
                    if not np.allclose(latest_wrench, r["force"]["wrench"], atol=1e-6, rtol=0):
                        raise ValueError("NPZ/JSON force mismatch")
                    if contact != r["starts_in_contact"]:
                        raise ValueError("NPZ/JSON contact mismatch")
                    if int(data["marker_pc_ns"].item()) != int(r["pc_ns"]):
                        raise ValueError("NPZ/JSON marker mismatch")
                    times = np.asarray(data["device_us"], dtype=np.float64)
                    pc_times = np.asarray(data["estimated_pc_ns"], dtype=np.float64)
                    if times.shape != adc.shape[:-1] or pc_times.shape != adc.shape[:-1]:
                        raise ValueError("Invalid audio timestamp shape")
                    dt = np.diff(times)
                    if not np.isfinite(times).all() or not np.isfinite(pc_times).all():
                        raise ValueError("Non-finite audio timestamps")
                    if (dt <= 0).any() or (np.diff(pc_times) <= 0).any():
                        raise ValueError("Non-monotonic audio timestamps")
                    measured_rate = 1e6 / np.median(dt, axis=-1)
                    if (np.abs(measured_rate / rate - 1) > 0.02).any():
                        raise ValueError(f"Audio timestamp rate {measured_rate} != metadata {rate}")
                    gaps = np.count_nonzero(np.any(
                        dt > 1.5 * np.median(dt, axis=-1, keepdims=True), axis=-1))
                    force_times = None
                    outside = int(not pc_times[0] <= r["pc_ns"] <= pc_times[-1]) if not grouped else 0
                    if grouped:
                        if int(data["audio_after"].item()) != 0:
                            raise ValueError("Grouped force/audio exports require audio_after=0")
                        force_times = _audit_grouped_forces(data, r, wrench, pc_times)
                for key, values in (("camera_age_ms", camera_ages), ("force_age_ms", force_ages)):
                    if key in r:
                        age = float(r[key])
                        if not np.isfinite(age) or age < 0:
                            raise ValueError(f"Invalid {key}: {age}")
                        values.append(age)
                # Build all children before updating the audit or exposing records.
                base = {"id": f"{session.name}/{r['sample_id']}",
                        "session": session.name, "sample_id": r["sample_id"],
                        "image": str(image_path), "npz": str(npz_path),
                        "group": location_group(r), "mode": r["mode"], "kind": r["kind"],
                        "starts_in_contact": contact, "sample_rate": float(rate)}
                children = []
                for index, force in enumerate(np.atleast_2d(wrench)):
                    child = {**base, "target": np.concatenate((position, force)).tolist()}
                    if grouped:
                        child.update(id=f"{base['id']}/force_{index:02d}", force_index=index,
                                     force_pc_ns=int(force_times[index]))
                    children.append(child)
                records.extend(children)
                included_events += 1
                grouped_events += int(grouped)
                counts[f"{session.name}/{r['mode']}/{r['kind']}"] += sample_count
                rates.append(float(rate))
                source_audio_samples[str(adc.shape[-2])] += sample_count
                nonuniform_audio += int(gaps)
                marker_outside += outside
                adc_ranges.append([float(event_adc.min()), float(event_adc.max())])
            except (ValueError, KeyError, OSError, TypeError, EOFError, BadZipFile) as exc:
                rejected.append({"manifest": str(manifest), "line": lineno, "reason": str(exc)})
    if not records:
        raise ValueError(f"No usable training samples. Rejections: {rejected[:3]}")
    if max(rates) - min(rates) > 1:
        raise ValueError("Mixed sample rates: resample explicitly before combining sessions")

    def distribution(values):
        return {"min": float(np.min(values)), "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "max": float(np.max(values))} if values else {}

    report = {"total_events": samples_seen, "included": len(records),
              "included_events": included_events, "grouped_events": grouped_events,
              "groups": len({r['group'] for r in records}), "counts": dict(counts),
              "excluded": dict(exclusions), "rejected": rejected, "sessions": sessions,
              "sample_rate_hz": float(np.median(rates)), "camera_age_ms": distribution(camera_ages),
              "audio_source_samples": dict(source_audio_samples),
              "audio_input_window": {"before": feature_config.before_samples,
                                     "after": feature_config.after_samples,
                                     "samples": feature_config.samples},
              "audio_alignment_mode": feature_config.alignment_mode,
              "force_age_ms": distribution(force_ages), "audio_with_timestamp_gaps": nonuniform_audio,
              "audio_marker_outside_window": marker_outside,
              "adc_min": min(x[0] for x in adc_ranges), "adc_max": max(x[1] for x in adc_ranges),
              "target_names": TARGET_NAMES,
              "label_min": np.min([r["target"] for r in records], axis=0).tolist(),
              "label_max": np.max([r["target"] for r in records], axis=0).tolist(),
              "depth_definition": "actual_depth_mm: measured local-contact-relative indentation; no second subtraction",
              "wrench_definition": "force_wrench from NPZ; Fx/Fy/Fz: N, Mx/My/Mz: N*m; acquisition zero/sign/frame retained",
              "grouped_sample_definition": "one sample per force/audio pair; image, XY, depth and contact shared within event"}
    return records, report


def split_records(records, seed=42, val_fraction=0.15, test_fraction=0.15):
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    groups = sorted({r["group"] for r in records})
    if len(groups) < 3:
        raise ValueError("Need at least three distinct locations for independent train/val/test")
    np.random.default_rng(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    n_test = max(1, round(len(groups) * test_fraction))
    if n_val + n_test >= len(groups):
        raise ValueError("Too few groups for requested split fractions")
    val, test = set(groups[:n_val]), set(groups[n_val:n_val + n_test])
    split = {"train": [], "val": [], "test": []}
    for record in records:
        key = "val" if record["group"] in val else "test" if record["group"] in test else "train"
        split[key].append(record)
    return split


def limit_split_groups(split, max_groups, seed=42):
    """For smoke tests, retain whole randomly selected groups in every partition."""
    if max_groups is None:
        return split
    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    rng = np.random.default_rng(seed)
    result = {}
    for key, records in split.items():
        groups = sorted({r["group"] for r in records})
        rng.shuffle(groups)
        keep = set(groups[:max_groups])
        result[key] = [r for r in records if r["group"] in keep]
    return result


def fit_statistics(records, feature_config):
    """Fit targets per dimension and log-mel per ADC channel on training data only."""
    target = np.asarray([r["target"] for r in records], dtype=np.float64)
    extractor = LogMel(feature_config)
    total = np.zeros(6, dtype=np.float64)
    squares = np.zeros(6, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for start in range(0, len(records), 32):
            arrays = []
            audio_by_path = {}
            for record in records[start:start + 32]:
                path = record["npz"]
                if path not in audio_by_path:
                    with np.load(path, allow_pickle=False) as data:
                        audio_by_path[path] = extract_sample_audio(data, feature_config)
                adc = audio_by_path[path]
                if "force_index" in record:
                    adc = adc[record["force_index"]]
                arrays.append(adc.astype(np.float32))
            mel = extractor(torch.from_numpy(np.stack(arrays))).numpy().astype(np.float64)
            total += mel.sum(axis=(0, 2, 3))
            squares += np.square(mel).sum(axis=(0, 2, 3))
            count += mel.shape[0] * mel.shape[2] * mel.shape[3]
    mean = total / count
    return {"target_mean": target.mean(axis=0).tolist(),
            "target_std": np.maximum(target.std(axis=0), 1e-4).tolist(),
            "audio_mean": mean.tolist(),
            "audio_std": np.sqrt(np.maximum(squares / count - mean ** 2, 1e-8)).tolist(),
            "fit_sample_count": len(records)}


class TactileDataset(Dataset):
    def __init__(self, records, feature_config, stats):
        self.records = records
        self.feature_config = feature_config
        self.extractor = LogMel(feature_config)
        self.target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32)
        self.target_std = torch.tensor(stats["target_std"], dtype=torch.float32)
        self.audio_mean = torch.tensor(stats["audio_mean"], dtype=torch.float32)[:, None, None]
        self.audio_std = torch.tensor(stats["audio_std"], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with np.load(record["npz"], allow_pickle=False) as data:
            adc = torch.from_numpy(extract_sample_audio(
                data, self.feature_config, record.get("force_index")).astype(np.float32))
        audio = (self.extractor(adc) - self.audio_mean) / self.audio_std
        target = torch.tensor(record["target"], dtype=torch.float32)
        return {"image": load_image(record["image"]), "audio": audio,
                "target": (target - self.target_mean) / self.target_std,
                "starts_in_contact": record["starts_in_contact"], "id": record["id"]}
