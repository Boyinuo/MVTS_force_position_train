"""Export one retained cached slide as portable single-event deployment inputs.

python -m deployment.prepare_cached_slide --trajectory t0128
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import ModuleType

import numpy as np
from PIL import Image
import torch

from deployment.replay import ROOT, TARGET_NAMES, select_sequence
from mvts.features import FeatureConfig, LogMel, load_image


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert(cache, data_root, trajectory, checkpoint, output, *, split=None):
    cache, data_root, checkpoint, output = [Path(p).resolve() for p in (cache, data_root, checkpoint, output)]
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    records = sorted([r for r in manifest["records"] if r["kind"] == "slide"
                      and (r["sequence"] == trajectory or r["sequence"].rsplit("/", 1)[-1] == trajectory)],
                     key=lambda r: r["order"])
    if not records or len({r["sequence"] for r in records}) != 1:
        raise ValueError("Expected exactly one matching trajectory")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if split is not None:
        from mvts.prepared_selection import filter_xy_range, trim_prepared_records
        records, _ = filter_xy_range(records, ckpt["args"].get("filter_xy_range", False))
        records, _ = trim_prepared_records(records, ckpt["args"].get("trim_stationary_xy_mm", 0.) or 0.)
        split_ids = set(ckpt["split_ids"][split])
        if not records or any(r["id"] not in split_ids for r in records):
            raise ValueError(f"Selected trajectory is not entirely in checkpoint {split!r} split after selection")
    trajectory = records[0]["sequence"].rsplit("/", 1)[-1]
    orders = [r["order"] for r in records]
    if orders != list(range(orders[0], orders[0] + len(orders))):
        raise ValueError("Expected contiguous source windows")
    if ckpt["feature_config"] != manifest["features"]:
        raise ValueError("Checkpoint and cache feature configurations differ")
    stats, config = ckpt["stats"], ckpt["model_config"]
    selected_ids = {r["id"] for r in records}
    partition = {k: len(selected_ids.intersection(ids)) for k, ids in ckpt.get("split_ids", {}).items()}
    unrecorded = len(selected_ids) - sum(partition.values())
    if unrecorded:
        partition["not_recorded"] = unrecorded
    partition_note = ", ".join(f"{k}={count}" for k, count in partition.items()) or "not recorded"
    raw_root = data_root / records[0]["source"].replace("\\", "/")
    raw_root = raw_root.parent
    groups = [json.loads(line) for line in (raw_root / "groups.jsonl").read_text("utf-8-sig").splitlines() if line.strip()]
    raw_checksums = json.loads((raw_root / "result.json").read_text("utf-8"))["files"]
    raw_arrays, checked_files = [], set()

    def check_raw(path):
        relative = path.relative_to(raw_root).as_posix()
        if relative not in checked_files:
            if sha256(path) != raw_checksums[relative]:
                raise ValueError(f"Raw file checksum mismatch: {path}")
            checked_files.add(relative)

    for group in groups:
        path = raw_root / group["data_file"]
        check_raw(path)
        with np.load(path, allow_pickle=False) as z:
            raw_arrays.append({k: z[k] for k in ("adc", "audio_pc_est_ns", "sample_index")})
    adc, audio_times, sample_index = [np.concatenate([r[k] for r in raw_arrays])
                                    for k in ("adc", "audio_pc_est_ns", "sample_index")]
    if adc.dtype != np.uint16 or adc.shape[1] != 6 or np.any(np.diff(audio_times) <= 0) or np.any(np.diff(sample_index) != 1):
        raise ValueError("Raw audio must contain six continuous uint16 channels")
    torch.set_num_threads(4)
    extractor = LogMel(FeatureConfig(**manifest["features"]))
    mean, std = [np.asarray(stats[k], dtype=np.float64)[:, None, None] for k in ("audio_mean", "audio_std")]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid audio normalization")
    sequence = f"slide_{trajectory}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".prepare_{trajectory}_", dir=output.parent) as temp:
        stage = Path(temp) / "package"
        (stage / "events").mkdir(parents=True)
        (stage / "images").mkdir()
        rows, labels, exported, source_hashes = [], [], [], {}
        max_difference = 0.
        for r in records:
            with np.load(cache / r["file"], allow_pickle=False) as z:
                cached = {k: z[k].copy() for k in z.files}
            np.testing.assert_array_equal(cached["target"], r["target"])
            times = cached["event_pc_ns"]
            stops = np.searchsorted(audio_times, times, side="left")
            np.testing.assert_array_equal(stops, cached["audio_stop"])
            if np.any(stops < 4000) or np.any(stops >= len(adc)):
                raise ValueError("Insufficient causal ADC history/bracketing")
            windows = np.stack([adc[stop-4000:stop] for stop in stops])
            for stamp, stop in zip(times, stops):
                ts = audio_times[stop-4000:stop]
                if np.any(np.diff(ts) > 30000) or not ts[-1] < stamp <= audio_times[stop] or stamp-ts[-1] > 30000:
                    raise ValueError("Noncausal or gapped ADC history")
            np.testing.assert_array_equal(audio_times[stops-1], cached["audio_last_pc_ns"])
            with torch.inference_mode():
                recomputed = extractor(torch.from_numpy(windows)).numpy()
            max_difference = max(max_difference, float(np.max(np.abs(recomputed-cached["audio"]))))
            np.testing.assert_allclose(recomputed, cached["audio"], atol=2e-4, rtol=0)
            mel = cached["audio"]
            normalized = ((mel.astype(np.float64)-mean)/std).astype(np.float32)
            image_name = f"images/{sequence}_{r['order']:04d}.png"
            shutil.copyfile(cache / r["image"], stage / image_name)
            group = groups[r["order"]]
            if (data_root / r["source"].replace("\\", "/")).resolve() != (raw_root / group["data_file"]).resolve():
                raise ValueError("Cache source and raw interval order disagree")
            raw_image = raw_root / group["first_frame"]["file"]
            check_raw(raw_image)
            with Image.open(raw_image) as im:
                expected = np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
            with Image.open(stage / image_name) as im:
                np.testing.assert_array_equal(np.asarray(im.convert("RGB")), expected)
            for key in ("file", "image"):
                source_hashes[r[key]] = sha256(cache / r[key])
            for index in range(r["events"]):
                call = len(rows)
                data_name = f"events/{call:06d}.npz"
                arrays = {k: cached[k][index] for k in ("target", "event_pc_ns", "force_pc_ns", "force_age_ns", "audio_last_pc_ns", "previous_pc_ns", "audio_stop",
                          "contact", "contact_valid", "previous_contact", "transition_valid", "target_valid") if k in cached}
                np.savez_compressed(stage / data_name, adc=windows[index], audio=mel[index],
                                    audio_input=normalized[index], **arrays)
                rows.append(dict(call_index=call, kind="slide", sequence=sequence, sequence_index=call,
                    schedule_offset_ns=round(call*1e9/150), sequence_offset_ns=round(call*1e9/150),
                    event_pc_ns=int(times[index]), data=data_name, image=image_name,
                    reset_state=call == 0, image_update=index == 0, source_window_id=r["id"],
                    source_sub_event=index, source_window_order=r["order"]))
                labels.append([call, "slide", sequence, call, int(times[index]), *cached["target"][index]])
            exported.append(dict(id=r["id"], kind="slide", order=r["order"], events=r["events"],
                                 image=image_name, source_cache_file=r["file"], original_source=r["source"]))
        if any(a["event_pc_ns"] >= b["event_pc_ns"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Event timestamps are not strictly increasing")
        (stage / "events.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        with (stage / "labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["call_index", "kind", "sequence", "sequence_index", "event_pc_ns", *TARGET_NAMES])
            writer.writerows(labels)
        write_json(stage / "normalization.json", stats)
        shutil.copyfile(ROOT / "deployment/test_data/deployment_test_150hz_t0232/reader.py", stage / "reader.py")
        shutil.copyfile(ROOT / "mvts/features.py", stage / "preprocessing_reference.py")
        template = json.loads((ROOT / "deployment/test_data/deployment_test_150hz_t0232/manifest.json").read_text("utf-8"))
        result = {k: template[k] for k in ("format", "replay_hz", "target_names", "target_units", "timing")}
        result.update(features=manifest["features"], model_config=config,
            source_cache=cache.relative_to(ROOT).as_posix(), source_manifest_sha256=sha256(cache / "manifest.json"),
            counts=dict(press=0, slide_windows=len(records), slide_events=len(rows), total_calls=len(rows)),
            slide_sequence=records[0]["sequence"], replay_sequence=sequence, slide_source_orders=orders,
            slide_raw_intervals=len(groups), records=exported, source_cache_file_sha256=source_hashes,
            excluded_source_intervals=[dict(order=i, reason="Excluded by source preprocessing or checkpoint data selection")
                                       for i in range(len(groups)) if i not in orders],
            normalization_source=dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha256(checkpoint), model=ckpt["model_name"], epoch=ckpt["epoch"]),
            normalization_note="Copied checkpoint training statistics. Other checkpoints must use their own statistics. Viewer automatically uses its selected checkpoint statistics.",
            checkpoint_partition_window_counts=partition,
            checkpoint_split=split,
            checkpoint_data_selection={k: ckpt.get("args", {}).get(k) for k in ("filter_xy_range", "trim_stationary_xy_mm")},
            selection_note=(f"All retained {trajectory} windows after checkpoint data selection; every window belongs to the saved {split} split. "
                            f"Partition window counts: {partition_note}." if split else
                            f"All retained {trajectory} windows; no additional time or image cropping. Selected-checkpoint partition window counts: {partition_note}. Directory name test_data describes deployment inputs, not a held-out split."))
        write_json(stage / "manifest.json", result)
        reader = ModuleType("portable_cached_slide_reader")
        reader.__file__ = str(stage / "reader.py")
        exec(compile((stage / "reader.py").read_text("utf-8"), reader.__file__, "exec"), reader.__dict__)
        events = reader.load_events(stage, kind="slide")
        assert len(events) == len(rows)
        assert select_sequence(stage, sequence) == rows
        for row, event in zip(rows, events):
            r, i = records[row["source_window_order"]-orders[0]], row["source_sub_event"]
            with np.load(cache / r["file"], allow_pickle=False) as z, np.load(stage / row["data"], allow_pickle=False) as out:
                for key in ("target", "event_pc_ns", "force_pc_ns", "force_age_ns", "audio_last_pc_ns", "previous_pc_ns", "audio_stop", "audio"):
                    np.testing.assert_array_equal(out[key], z[key][i])
                np.testing.assert_array_equal(event["audio_input"][0], ((z["audio"][i].astype(np.float64)-mean)/std).astype(np.float32))
                stop = int(np.searchsorted(audio_times, row["event_pc_ns"], side="left"))
                np.testing.assert_array_equal(out["adc"], adc[stop-4000:stop])
            if event["image_input"] is not None:
                np.testing.assert_array_equal(event["image_input"][0], load_image(stage / row["image"]).numpy())
            assert event["adc"].shape == (4000, 6) and event["adc"].dtype == np.uint16
            assert event["audio_input"].shape == (1, 6, 64, 28) and np.isfinite(event["audio_input"]).all()
        write_json(stage / "verification.json", dict(calls=len(rows), image_updates=sum(r["image_update"] for r in rows),
            state_resets=sum(r["reset_state"] for r in rows), slide_orders_contiguous=True,
            raw_files_sha256_verified=len(checked_files), raw_adc_logmel_recomputed=True,
            max_logmel_absolute_difference=max_difference, logmel_verification_absolute_tolerance=2e-4,
            exported_logmel_exact_match_source_cache=True, all_targets_and_timestamps_match_source=True,
            normalized_audio_exact_match=True, image_tensors_exact_match_training_preprocessing=True,
            portable_reader_calls=len(events), replay_sequence_selection_verified=True, six_microphone_channels_retained=True))
        split_note = (f"已按 checkpoint 的范围筛选与末端截取规则选择，全部窗口属于保存的 `{split}` 划分。"
                      if split else "目录名 `test_data` 表示部署输入目录，并不代表独立测试划分。")
        rebuild_options = f" --split {split} --checkpoint {result['normalization_source']['checkpoint']}" if split else ""
        (stage / "README.md").write_text(f"""# {trajectory} 部署调用数据

来源：`{result['source_cache']}` 中 `{result['slide_sequence']}` 的连续窗口，order {orders[0]}–{orders[-1]}，共 {len(rows)} 个事件、{len(records)} 张原始缓存图像。
保留所选窗口的原始标签、时间戳和六路音频；不按红框裁剪网络输入图像，不补零。
导出时所选 checkpoint 的划分窗口数：{partition_note}。{split_note}

## 在项目根目录调用

```powershell
.venv/Scripts/python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/{output.name} --sequence {sequence}
# 只预测并保存结果，不打开界面：
.venv/Scripts/python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/{output.name} --sequence {sequence} --headless --output deployment/results/{output.name}
# 检查可移植读取接口：
.venv/Scripts/python.exe deployment/test_data/{output.name}/reader.py --kind slide
```

已有界面中也可选择本目录后点击“加载并推理”。纯视觉对照默认使用现有 `best_image_only.pt`；可用 `--no-image-only-comparison` 关闭。

## 文件和输入接口

- `events.jsonl`：每行一次调用，顺序索引 0–{len(rows)-1}。第一步重置状态，每 10 步更新图像。
- `events/*.npz`：原始 ADC `uint16[4000,6]`、未标准化 Log-Mel `float32[6,64,28]`、标准化 `audio_input`、物理量 `target[11]` 及采集时间。
- `images/`：224×224 RGB 输入图像，未叠加红框或轨迹。
- `reader.py`：只依赖 NumPy/Pillow。`load_events(root)` 返回 `audio_input[1,6,64,28]`、`image_input[1,3,224,224]`（未更新时为 None）、`adc`、`target`、`meta`。
- `normalization.json`：导出时 `{result['normalization_source']['checkpoint']}` 的训练标准化参数。换模型时请使用新 checkpoint 的参数；部署界面会自动使用所选 checkpoint 的参数。
- `labels.csv`、`manifest.json`、`verification.json`、`checksums.sha256`：标签、来源、检查和校验和。

调用网络时只传图像和音频；`target` 是对照真值。按事件顺序连续调用 `model.step()`，遵循 `meta.reset_state` 和 `meta.image_update`。
名义回放为 150 Hz；实际采集事件跨度 {(rows[-1]['event_pc_ns']-rows[0]['event_pc_ns'])/1e9:.9f} 秒，原时间戳不变。

重新导出：`python -m deployment.prepare_cached_slide --cache {result['source_cache']} --data-root {data_root.relative_to(ROOT).as_posix()} --trajectory {result['slide_sequence']}{rebuild_options} --output NEW_DIRECTORY`。
""", encoding="utf-8")
        files = sorted(p for p in stage.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        (stage / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files), encoding="utf-8")
        stage.rename(output)
    return result["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/data_final_v4_faster_trim_xy_0p05mm")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/data_final")
    parser.add_argument("--trajectory", default="t0128")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "mvts/model_triple_head/best.pt")
    parser.add_argument("--split", choices=("train", "val", "test"), help="Apply checkpoint data selection and require this saved partition")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"deployment/test_data/deployment_test_150hz_{args.trajectory}"
    print(json.dumps(convert(args.cache, args.data_root, args.trajectory, args.checkpoint, output, split=args.split), indent=2))


if __name__ == "__main__":
    main()
