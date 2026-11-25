"""Export the complete saved v6 test partitions as three portable packages.

Run: python -m deployment.prepare_v6_test_data
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image
import torch

from deployment.prepare_cached_slide import sha256, write_json
from deployment.replay import ROOT, TARGET_NAMES, select_sequence
from mvts.contact_data import supervision_digest
from mvts.data import extract_sample_audio
from mvts.features import FeatureConfig, LogMel
from mvts.prepared_selection import filter_xy_range, trim_prepared_records


DEFAULT_CHECKPOINT = ROOT / "runs/v6_triple_head_slide_polyline_press_no_mic5_trim0p029_xyfilter/best.pt"


class RawInputs:
    """Keep at most one raw slide trajectory in memory."""

    def __init__(self, root, features):
        self.root, self.features = root, features
        self.trajectory = None
        self.checked = {}
        self.press_sessions = {}

    def check(self, path, expected):
        key = str(path)
        if key not in self.checked:
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Raw checksum mismatch: {path}")
            self.checked[key] = actual

    def load(self, record, cached):
        path = self.root / record["source"].replace("\\", "/")
        if record["kind"] == "press":
            session = path.parent.parent
            if session not in self.press_sessions:
                samples = [json.loads(x) for x in (session / "samples.jsonl").read_text("utf-8-sig").splitlines() if x.strip()]
                checksums = json.loads((session / "manifest_sha256.json").read_text("utf-8-sig"))
                self.press_sessions[session] = ({r["sample_id"]: r for r in samples},
                                                {r["path"]: r["sha256"] for r in checksums})
            samples, checksums = self.press_sessions[session]
            image = session / samples[record["sequence"].rsplit("/", 1)[-1]]["image"]
            for p in (path, image):
                self.check(p, checksums[p.relative_to(session).as_posix()])
            with np.load(path, allow_pickle=False) as z:
                adc = extract_sample_audio(z, self.features, force_index=record["force_index"])[None]
                np.testing.assert_array_equal(cached["event_pc_ns"], z["force_pc_ns"][-1:])
                np.testing.assert_array_equal(cached["audio_last_pc_ns"], z["estimated_pc_ns"][-1:, -1])
            return adc, image
        trajectory = path.parent
        if trajectory != self.trajectory:
            self.trajectory = trajectory
            self.groups = [json.loads(x) for x in (trajectory / "groups.jsonl").read_text("utf-8-sig").splitlines() if x.strip()]
            self.checksums = json.loads((trajectory / "result.json").read_text("utf-8"))["files"]
            raw = []
            for group in self.groups:
                p = trajectory / group["data_file"]
                self.check(p, self.checksums[group["data_file"]])
                with np.load(p, allow_pickle=False) as z:
                    raw.append({k: z[k] for k in ("adc", "audio_pc_est_ns", "sample_index")})
            self.adc, self.times, indices = [np.concatenate([r[k] for r in raw]) for k in ("adc", "audio_pc_est_ns", "sample_index")]
            if np.any(np.diff(self.times) <= 0) or np.any(np.diff(indices) != 1):
                raise ValueError(f"Discontinuous raw audio: {trajectory}")
        group = self.groups[record["order"]]
        if path != trajectory / group["data_file"]:
            raise ValueError("Source interval/order mismatch")
        image = trajectory / group["first_frame"]["file"]
        self.check(image, self.checksums[image.relative_to(trajectory).as_posix()])
        stops = np.searchsorted(self.times, cached["event_pc_ns"], side="left")
        np.testing.assert_array_equal(stops, cached["audio_stop"])
        if np.any(stops < 4000) or np.any(stops >= len(self.adc)):
            raise ValueError("Insufficient causal ADC history")
        np.testing.assert_array_equal(self.times[stops - 1], cached["audio_last_pc_ns"])
        windows = []
        for stamp, stop in zip(cached["event_pc_ns"], stops):
            ts = self.times[stop - 4000:stop]
            if np.any(np.diff(ts) > 30000) or not ts[-1] < stamp <= self.times[stop] or stamp - ts[-1] > 30000:
                raise ValueError("Noncausal or gapped ADC")
            windows.append(self.adc[stop - 4000:stop])
        return np.stack(windows), image


def export_dataset(name, checkpoint, ckpt, output_root):
    cache = ROOT / f".cache/data_final_v6_{name}_contact"
    data_root = ROOT / f"data/data_final/data_final_{name}"
    output = output_root / f"v6_{name}"
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    manifest = json.loads((cache / "manifest.json").read_text("utf-8"))
    if manifest["features"] != ckpt["feature_config"]:
        raise ValueError("Feature configuration mismatch")
    records, range_report = filter_xy_range(manifest["records"], True)
    records, trim_report = trim_prepared_records(records, ckpt["args"].get("trim_stationary_xy_mm", 0.) or 0.)
    test_ids = set(ckpt["split_ids"]["test"])
    expected = {r["id"] for r in manifest["records"]} & test_ids
    records = sorted((r for r in records if r["id"] in test_ids), key=lambda r: (r["sequence"], r["order"]))
    selected = {r["id"] for r in records}
    if not selected or selected != expected:
        raise ValueError("Saved test partition missing after range/tail selection")
    saved_labels = dict(zip(ckpt["split_ids"]["test"], ckpt["split_labels"]["test"]))
    saved_supervision = dict(zip(ckpt["split_ids"]["test"], ckpt["split_supervision"]["test"]))
    for record in records:
        if record["target"] != saved_labels[record["id"]] or record["supervision_sha256"] != saved_supervision[record["id"]]:
            raise ValueError(f"Labels changed since training: {record['id']}")
    for split in ("train", "val"):
        other = set(ckpt["split_ids"][split])
        if selected & other:
            raise ValueError(f"Test IDs overlap {split}")
        other_groups = {r["group"] for r in manifest["records"] if r["id"] in other}
        if {r["group"] for r in records} & other_groups:
            raise ValueError(f"Test physical groups overlap {split}")
    features = FeatureConfig(**manifest["features"])
    extractor = LogMel(features)
    stats = ckpt["stats"]
    mean, std = [np.asarray(stats[k], np.float64)[:, None, None] for k in ("audio_mean", "audio_std")]
    raw = RawInputs(data_root, features)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".prepare_v6_{name}_", dir=output_root) as temporary:
        stage = Path(temporary) / "package"
        (stage / "events").mkdir(parents=True)
        (stage / "images").mkdir()
        rows, exported, source_hashes = [], [], {}
        previous, previous_order, sequence_index, last_time = None, None, 0, None
        max_difference = 0.
        with (stage / "labels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["call_index", "kind", "sequence", "sequence_index", "event_pc_ns", *TARGET_NAMES])
            for number, r in enumerate(records, 1):
                with np.load(cache / r["file"], allow_pickle=False) as z:
                    arrays = {k: z[k].copy() for k in z.files}
                np.testing.assert_array_equal(arrays["target"], r["target"])
                if supervision_digest(arrays) != r["supervision_sha256"]:
                    raise ValueError(f"Supervision changed: {r['id']}")
                xy = arrays["target"][:, :2]
                if not (np.isfinite(xy).all() and ((xy >= [-10, -8]) & (xy <= [10, 10])).all()):
                    raise ValueError("Exported XY outside --filter-xy-range")
                adc, raw_image = raw.load(r, arrays)
                if adc.dtype != np.uint16 or adc.shape != (r["events"], 4000, 6):
                    raise ValueError("Unexpected ADC shape/type")
                with torch.inference_mode():
                    recomputed = extractor(torch.from_numpy(adc.astype(np.float32))).numpy()
                max_difference = max(max_difference, float(np.max(np.abs(recomputed - arrays["audio"]))))
                np.testing.assert_allclose(recomputed, arrays["audio"], atol=2e-4, rtol=0)
                image_name = r["image"]
                shutil.copyfile(cache / image_name, stage / image_name)
                with Image.open(raw_image) as image:
                    expected_image = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
                with Image.open(stage / image_name) as image:
                    np.testing.assert_array_equal(np.asarray(image), expected_image)
                for key in ("file", "image"):
                    source_hashes[r[key]] = sha256(cache / r[key])
                new_sequence = previous != r["sequence"]
                if new_sequence:
                    sequence_index, last_time = 0, None
                gap = new_sequence or r["order"] != previous_order + 1
                for event in range(r["events"]):
                    call = len(rows)
                    stamp = int(arrays["event_pc_ns"][event])
                    if last_time is not None and stamp <= last_time:
                        raise ValueError("Nonmonotonic timestamps within a sequence")
                    last_time = stamp
                    filename = f"events/{call:06d}.npz"
                    payload = {k: v[event] for k, v in arrays.items()}
                    payload.update(adc=adc[event], audio_input=((arrays["audio"][event].astype(np.float64) - mean) / std).astype(np.float32))
                    np.savez_compressed(stage / filename, **payload)
                    # Read every exported array back; includes validity/contact labels.
                    with np.load(stage / filename, allow_pickle=False) as saved:
                        for key, value in payload.items():
                            np.testing.assert_array_equal(saved[key], value)
                    row = dict(call_index=call, kind=r["kind"], sequence=r["sequence"], sequence_index=sequence_index,
                               schedule_offset_ns=round(call * 1e9 / 150), sequence_offset_ns=round(sequence_index * 1e9 / 150),
                               event_pc_ns=stamp, data=filename, image=image_name,
                               reset_state=event == 0 and (r["kind"] == "press" or gap), image_update=event == 0,
                               source_window_id=r["id"], source_sub_event=event, source_window_order=r["order"])
                    rows.append(row)
                    writer.writerow([call, r["kind"], r["sequence"], sequence_index, stamp, *arrays["target"][event]])
                    sequence_index += 1
                exported.append({**r, "source_cache_file": r["file"], "original_split": "test",
                                 "event_files": [row["data"] for row in rows[-r["events"]:]]})
                previous, previous_order = r["sequence"], r["order"]
                if number % 100 == 0 or number == len(records):
                    print(f"{name}: {number}/{len(records)} windows, {len(rows)} events", flush=True)
        (stage / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(stage / "normalization.json", stats)
        reference = ROOT / "deployment/test_data/deployment_test_150hz_t0232"
        shutil.copyfile(reference / "reader.py", stage / "reader.py")
        shutil.copyfile(ROOT / "mvts/features.py", stage / "preprocessing_reference.py")
        template = json.loads((reference / "manifest.json").read_text("utf-8"))
        counts = dict(windows=len(records), sequences=len({r["sequence"] for r in records}),
                      press=sum(r["kind"] == "press" for r in rows), slide_windows=sum(r["kind"] == "slide" for r in records),
                      slide_events=sum(r["kind"] == "slide" for r in rows), total_calls=len(rows))
        write_json(stage / "manifest.json", {
            **{k: template[k] for k in ("format", "replay_hz", "target_names", "target_units", "timing")},
            "dataset": name, "features": manifest["features"], "model_config": ckpt["model_config"], "counts": counts,
            "source_cache": cache.relative_to(ROOT).as_posix(), "source_manifest_sha256": sha256(cache / "manifest.json"),
            "checkpoint_split": "test", "checkpoint_partition_window_counts": {"train": 0, "val": 0, "test": len(records)},
            "normalization_source": {"checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
                                     "model": ckpt["model_name"], "epoch": ckpt["epoch"]},
            "checkpoint_data_selection": {"filter_xy_range": True, "trim_stationary_xy_mm": ckpt["args"].get("trim_stationary_xy_mm", 0.)},
            "xy_range_mm": {"x": [-10, 10], "y": [-8, 10], "inclusive": True},
            "range_selection": range_report, "tail_selection": trim_report,
            "records": exported, "source_cache_file_sha256": source_hashes,
            "normalization_note": "Checkpoint training statistics; use each selected model's own statistics. All six ADC channels retained."})
        if name != "press":
            for sequence in dict.fromkeys(row["sequence"] for row in rows):
                select_sequence(stage, sequence)
        write_json(stage / "verification.json", {**counts, "all_saved_test_records_included": True,
            "saved_checkpoint_labels_and_supervision_exact": True,
            "test_train_val_id_and_group_disjoint": True, "all_exported_xy_within_inclusive_range": True,
            "raw_files_sha256_verified": len(raw.checked), "raw_adc_logmel_recomputed": True,
            "max_logmel_absolute_difference": max_difference, "logmel_absolute_tolerance": 2e-4,
            "every_exported_array_readback_exact": True, "images_match_raw_training_preprocessing": True,
            "all_contact_and_validity_labels_retained": True, "state_resets": sum(r["reset_state"] for r in rows)})
        usage = (f".venv/Scripts/python.exe deployment/test_data/v6_{name}/reader.py --kind press" if name == "press" else
                 f".venv/Scripts/python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/v6_{name}")
        (stage / "README.md").write_text(f"""# v6 {name} 完整测试集

共 {counts['windows']} 个窗口/独立样本、{counts['total_calls']} 个事件、{counts['sequences']} 条序列。
来源：`.cache/data_final_v6_{name}_contact`；test 划分和标准化参数来自 `{checkpoint.relative_to(ROOT).as_posix()}`。
仅包含 checkpoint 保存的全部 test 记录，无 train/val 样本或物理分组重叠。

已启用 `--filter-xy-range`：x∈[-10,10] mm、y∈[-8,10] mm，包含边界。
滑移/折线在截尾前检查整条轨迹，任一点越界即剔除整条；按压逐样本检查（含背景）。
沿用 checkpoint 的 `{ckpt['args'].get('trim_stationary_xy_mm', 0.):g} mm` 末端截取规则，不补回被训练筛选剔除的窗口。

```powershell
{usage}
```

滑移/折线可在部署界面的轨迹下拉框中切换完整来源/轨迹 ID。
按压是独立单事件；使用 `reader.py --kind press` 检查并通过 `load_events(root, kind='press')` 调用网络，
每个样本均须重置状态和更新图像。现有滑移查看器仅支持滑移/折线。

`events.jsonl` 每行一次调用；`events/*.npz` 含六路原始 ADC uint16[4000,6]、
未标准化 Log-Mel float32[6,64,28]、标准化 audio_input、target[11]、接触/有效性标签与采集时间。
`images/` 保存原始缓存的 224×224 RGB 图像；`labels.csv` 为物理单位真值。
`reader.py` 仅依赖 NumPy/Pillow。测试其他模型时需使用其自身的标准化参数。
保留六路原始数据；禁用第5路由模型配置执行。名义调度150Hz，不修改采集时间或重新采样。
`manifest.json` 保存来源与划分，`verification.json` 保存核验结果，`checksums.sha256` 保存包内校验和。

重新导出：`python -m deployment.prepare_v6_test_data --checkpoint {checkpoint.relative_to(ROOT).as_posix()} --output NEW_DIRECTORY`。
""", encoding="utf-8")
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        (stage / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files), encoding="utf-8")
        stage.rename(output)
    return counts, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/test_data")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not ckpt["args"].get("filter_xy_range"):
        raise ValueError("A checkpoint trained with --filter-xy-range is required")
    for name in ("slide", "polyline", "press"):
        if (args.output / f"v6_{name}").exists():
            raise FileExistsError(args.output / f"v6_{name}")
    torch.set_num_threads(4)
    counts, selected = {}, set()
    for name in ("slide", "polyline", "press"):
        counts[name], ids = export_dataset(name, checkpoint, ckpt, args.output.resolve())
        selected.update(ids)
    if selected != set(ckpt["split_ids"]["test"]):
        raise ValueError("Combined exported records differ from saved test partition")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
