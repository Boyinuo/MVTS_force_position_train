"""Convert the 30-trial raw test ZIP into three portable deployment packages.

python -m deployment.prepare_demo_test_data --source PATH_TO_ZIP
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch

from deployment.prepare_cached_slide import write_json
from deployment.replay import ROOT, TARGET_NAMES, select_sequence
from mvts.contact_data import add_contact_supervision
from mvts.dual_rate_data import align_slide_interval, feature_config
from mvts.features import LogMel


SHAPES = {"line": "直线", "circle": "圆形", "triangle": "三角形"}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_arrays(archive, name):
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def export_shape(archive, prefix, trials, shape, checkpoint, ckpt, source_info, output):
    if output.exists():
        raise FileExistsError(output)
    config = feature_config()
    if config.to_dict() != ckpt["feature_config"]:
        raise ValueError("Checkpoint feature configuration differs from training preprocessing")
    stats = ckpt["stats"]
    mean, std = [np.asarray(stats[k], np.float64)[:, None, None] for k in ("audio_mean", "audio_std")]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid checkpoint normalization")
    extractor = LogMel(config)
    with tempfile.TemporaryDirectory(prefix=f".prepare_demo_{shape}_", dir=output.parent) as temp:
        stage = Path(temp) / "package"
        (stage / "events").mkdir(parents=True)
        (stage / "images").mkdir()
        rows, records, sequences, exclusions = [], [], [], []
        for trial in trials:
            result, trial_prefix = trial["result"], trial["prefix"]
            short_id = trial_prefix.rstrip("/").rsplit("/", 1)[-1]
            sequence = f"{prefix.rstrip('/')}/{short_id}"
            groups = [json.loads(line) for line in archive.read(trial_prefix + "groups.jsonl").decode("utf-8-sig").splitlines() if line.strip()]
            keys = ("adc", "audio_pc_est_ns", "sample_index", "force_pc_est_ns", "force_wrench",
                    "force_local_xyz_mm", "force_depth_mm")
            chunks = [read_arrays(archive, trial_prefix + g["data_file"]) for g in groups]
            merged = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in keys}
            del chunks
            ft, at = merged["force_pc_est_ns"], merged["audio_pc_est_ns"]
            if (np.diff(ft) <= 0).any() or (np.diff(at) <= 0).any() or (np.diff(merged["sample_index"]) != 1).any():
                raise ValueError(f"Discontinuous force/audio streams: {short_id}")
            positions = np.c_[merged["force_local_xyz_mm"][:, :2], merged["force_depth_mm"]]
            previous_time, previous_order, sequence_index, last_stamp = None, None, 0, None
            first_call = len(rows)
            for order, group in enumerate(groups):
                identifier = f"{sequence}/{order:04d}"
                start, end = group["first_frame"]["pc_ns"], group["last_frame"]["pc_ns"]
                if group.get("schema") != "planar_slide_intervals_v2" or group.get("frame_intervals") != 1:
                    raise ValueError(f"Expected adjacent actual camera frames: {identifier}")
                if not group.get("valid") or group.get("clipped_head") or group.get("clipped_tail"):
                    exclusions.append(dict(id=identifier, reason="invalid/clipped camera interval"))
                    previous_time = None
                    continue
                try:
                    arrays = align_slide_interval(start, end, ft, merged["force_wrench"], positions,
                        at, merged["adc"], previous_time, merged["sample_index"])
                except ValueError as exc:
                    exclusions.append(dict(id=identifier, reason=str(exc)))
                    previous_time = start + 9 * (end - start) / 10
                    continue
                previous_time = start + 9 * (end - start) / 10
                continuous = previous_order is not None and order == previous_order + 1
                arrays, contact_metadata = add_contact_supervision(arrays, "slide", previous_available=continuous)
                adc = arrays.pop("adc")
                if adc.dtype != np.uint16 or adc.shape != (10, 4000, 6):
                    raise ValueError(f"Unexpected ADC layout: {identifier}")
                with torch.inference_mode():
                    mel = extractor(torch.from_numpy(adc.astype(np.float32))).numpy()
                normalized = ((mel.astype(np.float64) - mean) / std).astype(np.float32)
                if not np.isfinite(mel).all() or not np.isfinite(normalized).all():
                    raise ValueError(f"Nonfinite acoustic features: {identifier}")
                image_name = f"images/{short_id}_{order:04d}.png"
                image_source = trial_prefix + group["first_frame"]["file"]
                with Image.open(io.BytesIO(archive.read(image_source))) as image:
                    expected_image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    expected_image.save(stage / image_name)
                with Image.open(stage / image_name) as image:
                    np.testing.assert_array_equal(np.asarray(image), np.asarray(expected_image))
                event_files = []
                for event in range(10):
                    call = len(rows)
                    filename = f"events/{call:06d}.npz"
                    payload = {k: v[event] for k, v in arrays.items()}
                    payload.update(adc=adc[event], audio=mel[event], audio_input=normalized[event])
                    np.savez_compressed(stage / filename, **payload)
                    with np.load(stage / filename, allow_pickle=False) as saved:
                        for k, value in payload.items():
                            np.testing.assert_array_equal(saved[k], value)
                    stamp = int(arrays["event_pc_ns"][event])
                    if last_stamp is not None and stamp <= last_stamp:
                        raise ValueError(f"Nonmonotonic event times: {identifier}")
                    last_stamp = stamp
                    rows.append(dict(call_index=call, kind="slide", shape=shape, sequence=sequence,
                        sequence_index=sequence_index, schedule_offset_ns=round(call * 1e9 / 150),
                        sequence_offset_ns=round(sequence_index * 1e9 / 150), event_pc_ns=stamp,
                        data=filename, image=image_name, reset_state=event == 0 and not continuous,
                        image_update=event == 0, source_window_id=identifier, source_sub_event=event,
                        source_window_order=order))
                    event_files.append(filename)
                    sequence_index += 1
                records.append(dict(id=identifier, kind="slide", shape=shape, sequence=sequence,
                    order=order, events=10, image=image_name, target=arrays["target"].tolist(),
                    original_source=trial_prefix + group["data_file"], source_image=image_source,
                    original_split="test", event_files=event_files, **contact_metadata))
                previous_order = order
            if sequence_index == 0:
                raise ValueError(f"No valid windows for {short_id}; cannot retain all ten trajectories")
            sequences.append(dict(sequence=sequence, trial=result["trial"], shape=shape,
                first_call_index=first_call, events=sequence_index, windows=sequence_index // 10,
                raw_intervals=len(groups), item=result["item"], path_motion=result["path_motion"],
                source_result=trial_prefix + "result.json"))
            print(f"{shape}: {short_id}, {sequence_index // 10}/{len(groups)} windows, {sequence_index} events", flush=True)
        if len(sequences) != 10:
            raise ValueError(f"Expected ten {shape} trajectories")
        (stage / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with (stage / "labels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["call_index", "kind", "shape", "sequence", "sequence_index", "event_pc_ns", *TARGET_NAMES])
            for row in rows:
                record = records[row["call_index"] // 10]
                writer.writerow([row[k] for k in ("call_index", "kind", "shape", "sequence", "sequence_index", "event_pc_ns")]
                                + record["target"][row["source_sub_event"]])
        write_json(stage / "normalization.json", stats)
        reference = ROOT / "deployment/test_data/v6_slide"
        shutil.copyfile(reference / "reader.py", stage / "reader.py")
        shutil.copyfile(ROOT / "mvts/features.py", stage / "preprocessing_reference.py")
        template = json.loads((reference / "manifest.json").read_text("utf-8"))
        counts = dict(sequences=len(sequences), press=0, slide_windows=len(records), slide_events=len(rows), total_calls=len(rows))
        manifest = {**{k: template[k] for k in ("format", "replay_hz", "target_names", "target_units", "timing")},
            "shape": shape, "shape_chinese": SHAPES[shape], "features": config.to_dict(),
            "model_config": ckpt["model_config"], "counts": counts, "source_archive": source_info,
            "source_split": "test", "checkpoint_split": None,
            "split_note": "External acquisition test package, not a subset of the training checkpoint's saved train/val/test IDs.",
            "data_selection": {"filter_xy_range": False, "trim_stationary_xy_mm": 0.,
                               "rule": "Keep all valid aligned windows in every trajectory, including stationary windows and polygon corners."},
            "normalization_source": {"checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_sha256": file_hash(checkpoint), "epoch": ckpt["epoch"], "model": ckpt["model_name"]},
            "normalization_note": "Use checkpoint training statistics; no fitting on demo test data. Viewer uses its selected model's own statistics.",
            "sequences": sequences, "records": records, "excluded_intervals": exclusions}
        write_json(stage / "manifest.json", manifest)
        for sequence in sequences:
            selected = select_sequence(stage, sequence["sequence"])
            assert len(selected) == sequence["events"]
            assert sum(row["image_update"] for row in selected) == sequence["windows"]
        write_json(stage / "verification.json", {**counts, "all_ten_source_trajectories_retained": True,
            "shape_verified_from_result_item_and_path_motion": True,
            "source_archive_all_file_checksums_verified": source_info["verified_files"],
            "training_alignment_and_preprocessing_used": True, "six_raw_adc_channels_retained": True,
            "every_exported_array_readback_exact": True, "images_match_training_preprocessing": True,
            "all_sequence_selection_checks_passed": True, "normalization_fitted_on_demo": False,
            "state_resets": sum(r["reset_state"] for r in rows), "excluded_intervals": len(exclusions)})
        (stage / "README.md").write_text(f"""# {SHAPES[shape]}滑移部署演示数据

`{output.name}` 包含 {len(sequences)} 条{SHAPES[shape]}轨迹、{len(records)} 个图像窗口、{len(rows)} 个事件。
来源：`{source_info['name']}`；类别由逐条 `result.json` 的 `item.shape` 和 `path_motion.shape` 核对。
本包是外部采集测试集，未声称属于训练 checkpoint 保存的测试划分。

```powershell
uv run deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/{output.name}
# 只检查读取，不运行模型
.venv/Scripts/python.exe deployment/test_data/{output.name}/reader.py --kind slide
```

在查看器的轨迹下拉框中切换十条数据，也可用 `--sequence {sequences[0]['sequence'].rsplit('/', 1)[-1]}` 指定。
保留全部有效窗口，不默认启用 `--filter-xy-range` 或静止末端截取。
无效/裁剪区间、因果音频历史不足等沿用训练预处理的排除规则，详情见 `manifest.json`。
多边形拐点的停顿也保留；`--trim-stationary-xy-mm` 会从首个低于阈值的窗口截去全部后续窗口，可能截断完整图形。

每窗十个事件，按原始相邻图像、400 Hz 力采样和50 kHz音频对齐，未用150 Hz重新插值标签。
`events.jsonl` 每行一次调用，150 Hz仅为名义播放调度；保留原始采集时间。
`events/*.npz` 包含 uint16 ADC[4000,6]、未标准化 Log-Mel[6,64,28]、标准化 audio_input、
物理单位 target[11]、接触/有效性标签和原始时间戳。`images/` 为224×224 RGB输入图像。
各轨迹首事件及缺失窗口后的首事件重置状态，每个窗口首事件更新图像。
`reader.py` 仅依赖 NumPy/Pillow；`load_events(root, kind='slide')` 提供网络调用输入。
模型只接收图像和音频，target用于对照；六路原始ADC全部保留，禁用通道由模型配置执行。
标准化参数来自 `{checkpoint.relative_to(ROOT).as_posix()}` 的训练集，没有在本测试集上拟合。
`labels.csv` 保存真值；`manifest.json` 保存分类与来源；`verification.json` 和 `checksums.sha256` 用于核验。

重建：`python -m deployment.prepare_demo_test_data --source PATH_TO_ZIP --output NEW_DIRECTORY`。
""", encoding="utf-8")
        files = sorted(p for p in stage.rglob("*") if p.is_file())
        (stage / "checksums.sha256").write_text("".join(f"{file_hash(p)}  {p.relative_to(stage).as_posix()}\n" for p in files), encoding="utf-8")
        stage.rename(output)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/test_data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "mvts/model_triple_head/best.pt")
    args = parser.parse_args()
    output = args.output.resolve()
    for shape in SHAPES:
        if (output / f"demo_{shape}").exists():
            raise FileExistsError(output / f"demo_{shape}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.set_num_threads(4)
    source = args.source.resolve()
    print("Hashing source ZIP...", flush=True)
    source_info = dict(name=source.name, sha256=file_hash(source), bytes=source.stat().st_size)
    with zipfile.ZipFile(source) as archive:
        summaries = [n for n in archive.namelist() if n.endswith("/dataset_summary.json")]
        if len(summaries) != 1:
            raise ValueError("Expected one dataset root")
        prefix = summaries[0].removesuffix("dataset_summary.json")
        summary = json.loads(archive.read(summaries[0]))
        if summary.get("schema") != "planar_slide_intervals_v2" or summary.get("package_role") != "test":
            raise ValueError("Expected raw slide intervals in an external test package")
        checksums = json.loads(archive.read(prefix + "manifest_sha256.json"))
        for number, item in enumerate(checksums, 1):
            content = archive.read(prefix + item["path"])
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError(f"Source checksum mismatch: {item['path']}")
            if number % 500 == 0:
                print(f"Source SHA256: {number}/{len(checksums)}", flush=True)
        source_info["verified_files"] = len(checksums)
        trials = []
        for name in sorted(archive.namelist()):
            if not name.endswith("/result.json"):
                continue
            result = json.loads(archive.read(name))
            shape = result["item"]["shape"]
            if result.get("status") != "complete" or shape not in SHAPES or result["path_motion"]["shape"] != shape:
                raise ValueError(f"Unexpected trial metadata: {name}")
            trials.append(dict(prefix=name.removesuffix("result.json"), result=result, shape=shape))
        if dict(Counter(t["shape"] for t in trials)) != {s: 10 for s in SHAPES}:
            raise ValueError("Expected ten line, circle and triangle trajectories")
        counts = {shape: export_shape(archive, prefix, [t for t in trials if t["shape"] == shape], shape,
                    checkpoint, ckpt, source_info, output / f"demo_{shape}") for shape in SHAPES}
        print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
