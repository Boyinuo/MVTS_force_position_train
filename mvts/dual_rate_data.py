"""Versioned, model-independent preparation of ordered press/slide events.

Cache stores unnormalised log-mel, measured targets and original time indices.
No learned visual features are cached: changing model weights remains safe.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .data import read_json, write_json, _safe_file, extract_sample_audio
from .features import FeatureConfig, LogMel, load_image

TARGET_NAMES = ["x_mm", "y_mm", "delta_z_mm", "Fx", "Fy", "Fz", "Mx", "My", "Mz",
                "slip_x_mm", "slip_y_mm"]
CACHE_VERSION = 4


def feature_config():
    return FeatureConfig(alignment_mode="before_only", before_samples=4000, after_samples=0,
                         stft_right_align=True)


def json_lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def interpolate(times, values, query):
    if (np.diff(times) <= 0).any() or not np.isfinite(values).all():
        raise ValueError("Nonmonotonic timestamps or nonfinite labels")
    if np.min(query) < times[0] or np.max(query) > times[-1]:
        raise ValueError("Position interpolation would extrapolate")
    return np.stack([np.interp(query, times, values[:, i]) for i in range(values.shape[1])], -1)


def align_slide_interval(start, end, force_times, wrench, position, audio_times, adc,
                         previous_time=None, sample_index=None):
    """Ten events in [image_j,image_j+1): j plus nine interior times.

    Select the latest measured force at/before each evenly spaced event (400Hz
    quantisation is recorded, never relabelled as a measured 150Hz stream).
    Interpolate measured positions on the event grid for displacement labels.
    Audio strictly precedes each event and may span several exported intervals.
    """
    if end <= start:
        raise ValueError("Invalid camera interval")
    times = np.rint(start + np.arange(10, dtype=np.float64) * ((end - start) / 10)).astype(np.int64)
    previous_time = start - (end - start) / 10 if previous_time is None else previous_time
    previous = np.r_[int(round(previous_time)), times[:-1]]
    fi = np.searchsorted(force_times, times, side="right") - 1
    if (fi < 0).any() or ((times - force_times[fi]) > 3_000_000).any():
        raise ValueError("Missing causal force sample within 3ms")
    if len(np.unique(fi)) != 10:
        raise ValueError("Camera interval contains fewer than ten distinct force samples")
    pos = interpolate(force_times, position, times)
    slip = pos[:, :2] - interpolate(force_times, position, previous)[:, :2]
    stops = np.searchsorted(audio_times, times, side="left")
    if (stops < 4000).any() or (stops >= len(audio_times)).any():
        raise ValueError("Insufficient 4000-point causal audio history/bracketing")
    windows = []
    for t, stop in zip(times, stops):
        ts = audio_times[stop - 4000:stop]
        if (np.diff(ts) <= 0).any() or (np.diff(ts) > 30_000).any() or t - ts[-1] > 30_000:
            raise ValueError("Audio clock gap in causal window")
        if abs(float(np.median(np.diff(ts))) / 20_000 - 1) > .02:
            raise ValueError("Microphone timestamps do not match 50kHz")
        if sample_index is not None and (np.diff(sample_index[stop - 4000:stop]) != 1).any():
            raise ValueError("Missing microphone samples in causal window")
        windows.append(adc[stop - 4000:stop])
    return {"adc": np.stack(windows), "target": np.c_[pos, wrench[fi], slip],
            "event_pc_ns": times.astype(np.int64), "force_pc_ns": force_times[fi],
            "force_age_ns": times-force_times[fi],
            "audio_last_pc_ns": audio_times[stops - 1], "audio_stop": stops,
            "previous_pc_ns": previous.astype(np.int64)}


class CacheWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.extractor = LogMel(feature_config())
        self.records = []
        self.images = {}

    def add(self, identifier, image, arrays, **metadata):
        key = hashlib.sha256(identifier.encode()).hexdigest()[:24]
        image = Path(image)
        image_key = hashlib.sha256(str(image.resolve()).encode()).hexdigest()[:24]
        image_file = f"images/{image_key}.png"
        if image_key not in self.images:
            (self.root / "images").mkdir(exist_ok=True)
            with Image.open(image) as im:
                im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR).save(self.root / image_file)
            self.images[image_key] = True
        target = np.asarray(arrays["target"], dtype=np.float64)
        if target.ndim != 2 or target.shape[1] != 11 or not np.isfinite(target).all():
            raise ValueError("Invalid measured target")
        with torch.no_grad():
            mel = self.extractor(torch.from_numpy(arrays["adc"].astype(np.float32))).numpy()
        if not np.isfinite(mel).all():
            raise ValueError("Nonfinite acoustic features")
        (self.root / "events").mkdir(exist_ok=True)
        filename = f"events/{key}.npz"
        temp = self.root / f"events/{key}.tmp.npz"
        np.savez(temp, audio=mel, **{k: v for k, v in arrays.items() if k != "adc"})
        temp.replace(self.root / filename)
        self.records.append({"id": identifier, "file": filename, "image": image_file,
                             "events": len(target), "target": target.tolist(), **metadata})


def prepare_data(data_root, output, *, max_units_per_source=None, skip_unsupported=False, slide_sources=None,
                 contact_aware=False, press_sources=None):
    root = Path(data_root).resolve()
    if contact_aware:
        from .contact_data import (CONTACT_CACHE_VERSION, CONTACT_SCHEMA, CONTACT_POLICY,
                                   add_contact_supervision, contact_summary)
        if Path(output).exists():
            raise ValueError("Contact cache output must be a new directory; original caches are never overwritten")
    # Raw captures use attempts/, curated full-frame exports use trajectories/.
    slide_manifests = sorted(set(root.rglob("attempts/*/groups.jsonl")) |
                             set(root.rglob("trajectories/*/groups.jsonl")))
    available_slides = {p.parent.parent.parent.name for p in slide_manifests}
    selected_slides = None if slide_sources is None else set(slide_sources)
    if selected_slides is not None and (not selected_slides or selected_slides - available_slides):
        raise ValueError(f"Unknown/empty slide source selection: {sorted(selected_slides - available_slides)}")
    if (Path(output) / "manifest.json").exists():
        raise ValueError("Prepared cache already exists; reuse it for training, or choose a new output to regenerate")
    writer = CacheWriter(output)
    report = {"version": CACHE_VERSION, "source_root": str(root), "features": feature_config().to_dict(),
              "target_names": TARGET_NAMES, "excluded": [], "rejected": [], "sources": {},
              "selection": {"press_sources": "all available press sources",
                            "slide_sources": sorted(selected_slides) if selected_slides is not None else "all available slide sources",
                            "excluded_slide_sources": sorted(available_slides - selected_slides) if selected_slides is not None else []},
              "smoke_test": max_units_per_source is not None,
              "alignment": "10 event times per actual camera interval, half-open [start,end); latest available image",
              "warnings": ["Camera times are host read completion times, not measured exposure times.",
                           "Ten updates per actual frame interval approximate 150Hz; camera timestamp jitter is retained.",
                           "Press uses only the latest image-associated force/audio pair; the preceding nine forces are excluded."]}
    if contact_aware:
        report.update(version=CONTACT_CACHE_VERSION, schema=CONTACT_SCHEMA, contact_policy=CONTACT_POLICY)
    if press_sources is not None:
        report["selection"]["press_sources"] = sorted(press_sources)
    for manifest in sorted(root.rglob("samples.jsonl")):
        rows = json_lines(manifest)
        if not rows:
            continue
        if "image_start" in rows[0]:
            if selected_slides is not None and manifest.parent.name not in selected_slides:
                continue
            summary_path = manifest.parent / "dataset_summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {}
            if summary.get("schema") == "planar_slide_intervals_v2":
                # Its full force/audio streams are handled below via groups.jsonl.
                continue
            report["excluded"].append({"source": str(manifest.relative_to(root)), "reason":
                "planar_slide_v1 has no measured wrench stream and only every third camera frame"})
            continue
        session = manifest.parent
        if press_sources is not None and session.name not in press_sources:
            continue
        kinds = ("background", "target", "limit") if contact_aware else ("target", "limit")
        selected = [r for r in rows if r.get("valid") is True and r.get("kind") in kinds
                    and r.get("mode") in ("staircase", "dynamic")]
        if max_units_per_source:
            # Include several complete physical locations in a representative smoke cache.
            points = list(dict.fromkeys(r["point"] for r in selected))[:max_units_per_source]
            selected = [r for r in selected if r["point"] in points]
        for number, r in enumerate(selected, 1):
            identifier = f"{session.name}/{r['sample_id']}"
            try:
                path = _safe_file(session, r["data"])
                with np.load(path, allow_pickle=False) as z:
                    adc = extract_sample_audio(z, feature_config())
                    forces = np.atleast_2d(z["force_wrench"])
                    times = np.asarray(z["force_pc_ns"], dtype=np.int64)
                    audio_times = z["estimated_pc_ns"]
                    if adc.ndim != 3 or len(adc) != len(forces) or len(times) != len(forces):
                        raise ValueError("Expected grouped measured press force/audio pairs")
                    if (np.diff(times) <= 0).any() or (audio_times[:, -1] >= times).any():
                        raise ValueError("Noncausal or unordered press audio/force")
                    if not np.allclose(forces, [f["wrench"] for f in r["force_sequence"]], atol=1e-6, rtol=0):
                        raise ValueError("Press NPZ/JSON wrench mismatch")
                    if not np.array_equal(times, [f["pc_est_ns"] for f in r["force_sequence"]]):
                        raise ValueError("Press NPZ/JSON force time mismatch")
                    if (np.diff(audio_times, axis=1) <= 0).any() or (np.diff(audio_times, axis=1) > 30_000).any():
                        raise ValueError("Press audio clock gap")
                    if np.any(np.abs(np.median(np.diff(audio_times, axis=1), axis=1)/20_000-1) > .02):
                        raise ValueError("Press audio timestamps do not match 50kHz")
                    pos = np.r_[z["actual_xy_mm"], float(r["actual_depth_mm"])]
                    if not np.allclose(z["actual_xy_mm"], r["actual_xy_mm"], atol=1e-6, rtol=0):
                        raise ValueError("Press position mismatch")
                    # A press export contains ten past force windows for one image.
                    # Only its CURRENT (last) pair is a requested training event.
                    targets = np.c_[pos[None], forces[-1:], np.zeros((1, 2))]
                    arrays = {"adc": adc[-1:], "target": targets, "event_pc_ns": times[-1:],
                              "force_pc_ns": times[-1:], "audio_last_pc_ns": audio_times[-1:, -1]}
                contact_metadata = {}
                if contact_aware:
                    arrays, contact_metadata = add_contact_supervision(arrays, "press", r["kind"])
                writer.add(identifier, _safe_file(session, r["image"]), arrays, kind="press",
                           sequence=identifier, order=0,
                           force_index=len(forces)-1, image_pc_ns=r.get("camera",{}).get("pc_ns",r.get("pc_ns")),
                           group=f"press/{float(r['x_mm']):.6f},{float(r['y_mm']):.6f}",
                           source=str(path.relative_to(root)), source_size=path.stat().st_size,
                           **contact_metadata)
            except (ValueError, KeyError, OSError) as exc:
                report["rejected"].append({"id": identifier, "reason": str(exc)})
            if number % 200 == 0:
                print(f"Press {session.name}: {number}/{len(selected)}", flush=True)
        report["sources"][session.name] = len(selected)
        print(f"Prepared press {session.name}: {len(selected)} events", flush=True)

    attempt_dirs = defaultdict(list)
    for p in slide_manifests:
        if selected_slides is not None and p.parent.parent.parent.name not in selected_slides:
            continue
        with p.open(encoding="utf-8-sig") as handle:
            first = next((json.loads(line) for line in handle if line.strip()), {})
        if first.get("schema") != "planar_slide_intervals_v2":
            # Unsupported v1 exports were reported by their top-level manifest.
            continue
        attempt_dirs[p.parent.parent.parent].append(p.parent)
    for session, attempts in attempt_dirs.items():
        if max_units_per_source:
            attempts = attempts[:max_units_per_source]
        for number, attempt in enumerate(attempts, 1):
            result = read_json(attempt / "result.json")
            if result.get("status") != "complete":
                continue
            sequence = f"{session.name}/{attempt.name}"
            groups = json_lines(attempt / "groups.jsonl")
            try:
                paths = [_safe_file(attempt, g["data_file"]) for g in groups]
                keys = ("adc", "audio_pc_est_ns", "sample_index", "force_pc_est_ns", "force_wrench",
                        "force_local_xyz_mm", "force_depth_mm")
                chunks = defaultdict(list)
                for path in paths:
                    with np.load(path, allow_pickle=False) as z:
                        for k in keys:
                            chunks[k].append(z[k])
                merged = {k: np.concatenate(v) for k, v in chunks.items()}
                ft, at = merged["force_pc_est_ns"], merged["audio_pc_est_ns"]
                if (np.diff(ft) <= 0).any() or (np.diff(at) <= 0).any():
                    raise ValueError("Slide streams overlap or are not monotonic")
                positions = np.c_[merged["force_local_xyz_mm"][:, :2], merged["force_depth_mm"]]
                previous_time = None
                for i, g in enumerate(groups):
                    start, end = g["first_frame"]["pc_ns"], g["last_frame"]["pc_ns"]
                    if g.get("schema") != "planar_slide_intervals_v2" or g.get("frame_intervals") != 1:
                        raise ValueError("Slide requires actual adjacent camera frames")
                    if not g.get("valid") or g.get("clipped_head") or g.get("clipped_tail"):
                        report["excluded"].append({"id": f"{sequence}/{i}", "reason": "invalid/clipped camera interval"})
                        previous_time = None
                        continue
                    try:
                        arrays = align_slide_interval(start, end, ft, merged["force_wrench"], positions,
                            at, merged["adc"], previous_time, merged["sample_index"])
                        contact_metadata = {}
                        if contact_aware:
                            continuous = bool(writer.records and writer.records[-1]["sequence"] == sequence
                                              and writer.records[-1]["order"] == i-1)
                            arrays, contact_metadata = add_contact_supervision(
                                arrays, "slide", previous_available=continuous)
                        writer.add(f"{sequence}/{i:04d}", _safe_file(attempt, g["first_frame"]["file"]), arrays,
                                   kind="slide", sequence=sequence, order=i,
                                   group=f"slide/{session.name}/trial_{result['trial']}",
                                   source=str(paths[i].relative_to(root)), source_size=paths[i].stat().st_size,
                                   image_pc_ns=start, next_image_pc_ns=end, **contact_metadata)
                    except ValueError as exc:
                        report["excluded"].append({"id": f"{sequence}/{i}", "reason": str(exc)})
                    previous_time = start + 9 * (end - start) / 10
            except (ValueError, KeyError, OSError) as exc:
                report["rejected"].append({"id": sequence, "reason": str(exc)})
            if number % 25 == 0:
                print(f"Slide {session.name}: {number}/{len(attempts)}", flush=True)
        report["sources"][session.name] = len(attempts)
        print(f"Prepared slide {session.name}: {len(attempts)} trajectories", flush=True)
    report["windows"] = len(writer.records)
    report["events"] = sum(r["events"] for r in writer.records)
    report["counts"] = dict(Counter(r["kind"] for r in writer.records))
    if contact_aware:
        report["contact_counts"] = contact_summary(writer.records)
    write_json(writer.root / "report.json", report)
    unsupported = [x for x in report["excluded"] if "source" in x]
    if report["rejected"] or (unsupported and not skip_unsupported):
        raise ValueError("Preparation incomplete; inspect report.json (use --skip-unsupported for missing v1 supervision)")
    write_json(writer.root / "manifest.json", {"version": report["version"], "features": feature_config().to_dict(),
               **({"schema": CONTACT_SCHEMA, "contact_policy": CONTACT_POLICY} if contact_aware else {}),
               "smoke_test": report["smoke_test"], "records": writer.records})
    return writer.records, report


def read_prepared(root):
    root = Path(root)
    if not (root / "manifest.json").is_file():
        raise ValueError(f"No completed preparation cache at {root}; run python preprocess.py --output {root}")
    manifest = read_json(root / "manifest.json")
    if manifest["version"] != CACHE_VERSION or manifest["features"] != feature_config().to_dict():
        raise ValueError("Incompatible preparation cache; regenerate it")
    return [{**r, "cache_root": str(root.resolve())} for r in manifest["records"]], manifest


def subset_prepared(source_root, output, press_sources):
    """Copy all slide windows and selected press sessions to an independent cache."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    _, manifest = read_prepared(source_root)
    keep = set(press_sources)
    available = {r["id"].split("/", 1)[0] for r in manifest["records"] if r["kind"] == "press"}
    if not keep or keep - available:
        raise ValueError(f"Unknown/empty press source selection: {sorted(keep - available)}")
    if output.exists() or output.is_relative_to(source_root):
        raise ValueError("Subset output must be a new directory outside the source cache")
    records = [r for r in manifest["records"]
               if r["kind"] != "press" or r["id"].split("/", 1)[0] in keep]
    files = sorted({r[k] for r in records for k in ("file", "image")})
    # Validate every source before creating the new output. Copy rather than link:
    # later edits/removal of either cache cannot change the other's files.
    paths = [_safe_file(source_root, relative) for relative in files]
    output.mkdir(parents=True)
    copied_bytes = 0
    for number, path in enumerate(paths, 1):
        destination = output / path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if destination.stat().st_size != path.stat().st_size:
            raise OSError(f"Incomplete copy: {destination}")
        copied_bytes += destination.stat().st_size
        if number % 2000 == 0 or number == len(paths):
            print(f"Copied {number}/{len(paths)} cache files", flush=True)
    selection = {"source_cache": str(source_root), "press_sources": sorted(keep),
                 "slide_selection": "all source slide records, unchanged",
                 "excluded_press_sources": sorted(available - keep)}
    report = read_json(source_root / "report.json")
    report.update(windows=len(records), events=sum(r["events"] for r in records),
                  counts=dict(Counter(r["kind"] for r in records)), selection=selection,
                  copied_files=len(paths), copied_bytes=copied_bytes)
    report["sources"] = {k: v for k, v in report["sources"].items() if k not in available or k in keep}
    write_json(output / "report.json", report)
    write_json(output / "manifest.json", {**manifest, "selection": selection, "records": records})
    return records, report


def split_prepared(records, seed=42, val_fraction=.15, test_fraction=.15, max_groups=None):
    from .data import split_records, limit_split_groups
    # Stratify by acquisition kind; keep entire slide trajectories and press locations.
    output = {k: [] for k in ("train", "val", "test")}
    for kind in sorted({r["kind"] for r in records}):
        part = split_records([r for r in records if r["kind"] == kind], seed, val_fraction, test_fraction)
        part = limit_split_groups(part, max_groups, seed)
        for k in output:
            output[k].extend(part[k])
    return output


def fit_prepared_statistics(records):
    count, audio_count = 0, 0
    total, square = np.zeros(11), np.zeros(11)
    audio_total, audio_square = np.zeros(6), np.zeros(6)
    for r in records:
        with np.load(Path(r["cache_root"]) / r["file"]) as z:
            y, a = z["target"].astype(np.float64), z["audio"].astype(np.float64)
        total += y.sum(0)
        square += (y*y).sum(0)
        count += len(y)
        audio_total += a.sum((0, 2, 3))
        audio_square += (a*a).sum((0, 2, 3))
        audio_count += a.shape[0] * a.shape[2] * a.shape[3]
    mean, am = total/count, audio_total/audio_count
    std = np.sqrt(np.maximum(square/count - mean*mean, 1e-8))
    # A realistic mm floor prevents near-zero press-only displacement variance
    # from producing extreme normalised slip gradients.
    std[9:] = np.maximum(std[9:], .01)
    return {"target_mean": mean.tolist(), "target_std": std.tolist(),
            "audio_mean": am.tolist(), "audio_std": np.sqrt(np.maximum(audio_square/audio_count-am*am,1e-8)).tolist(),
            "fit_sample_count": count, "target_names": TARGET_NAMES}


class PreparedDataset(Dataset):
    def __init__(self, records, stats):
        self.records, self.stats = records, stats

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        root = Path(r["cache_root"])
        with np.load(root / r["file"], allow_pickle=False) as z:
            audio, target = z["audio"].copy(), z["target"].copy()
        s = self.stats
        return {"image": load_image(root/r["image"]),
                "audio": torch.tensor((audio-np.array(s["audio_mean"])[None,:,None,None])/
                                       np.array(s["audio_std"])[None,:,None,None], dtype=torch.float32),
                "target": torch.tensor((target-np.array(s["target_mean"]))/np.array(s["target_std"]), dtype=torch.float32),
                "id": r["id"], "sequence": r["sequence"], "order": r["order"]}


class OrderedBatchSampler(Sampler):
    """Interleave length-homogeneous batches, keeping every GRU stream ordered.

    Press has one event per image, slide has ten. Separate batch shapes avoid
    padding fake events or accidentally training the nine historical press forces.
    """
    def __init__(self, records, batch_size, shuffle=False, seed=42):
        sequences = defaultdict(list)
        for i, r in enumerate(records):
            sequences[(r.get("events",10), r["sequence"])].append((r["order"], i))
        self.streams = defaultdict(list)
        for (length, _), v in sequences.items():
            self.streams[length].append([i for _,i in sorted(v)])
        self.batch_size, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        streams_by_length = [list(v) for _,v in sorted(self.streams.items())]
        if self.shuffle:
            for streams in streams_by_length:
                rng.shuffle(streams)
            rng.shuffle(streams_by_length)
        self.epoch += 1
        iterators = [iter(self._batches(streams)) for streams in streams_by_length]
        while iterators:
            remaining = []
            for iterator in iterators:
                batch = next(iterator, None)
                if batch is not None:
                    yield batch
                    remaining.append(iterator)
            iterators = remaining

    def _batches(self, streams):
        queue = iter(streams)
        active = [iter(s) for s in list(next(queue, None) for _ in range(self.batch_size)) if s is not None]
        while active:
            batch, remaining = [], []
            for stream in active:
                index = next(stream, None)
                if index is None:
                    replacement = next(queue, None)
                    if replacement is None:
                        continue
                    stream = iter(replacement)
                    index = next(stream)
                batch.append(index)
                remaining.append(stream)
            active = remaining
            if batch:
                yield batch

    def __len__(self):
        # Exact batch count for this epoch's stream order, without advancing RNG.
        epoch = self.epoch
        n = sum(1 for _ in self)
        self.epoch = epoch
        return n
