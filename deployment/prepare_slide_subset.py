"""Convert the t0232 raw/test archive to the portable 150 Hz deployment format.

Run from the project root: python -m deployment.prepare_slide_subset
Only the supplied prepared test subset is exported; raw-only intervals are not
silently added to the evaluation set. Source files are read directly from ZIP.
"""
import argparse
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

from mvts.features import FeatureConfig, LogMel, load_image
from deployment.replay import ROOT, TARGET_NAMES


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_npz(archive, name):
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def convert(source, output, reference):
    source, output, reference = map(lambda p: Path(p).resolve(), (source, output, reference))
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    template = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((reference / "normalization.json").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix=".prepare_t0232_", dir=output.parent) as temp, zipfile.ZipFile(source) as archive:
        stage = Path(temp) / "package"
        (stage / "events").mkdir(parents=True)
        (stage / "images").mkdir()
        manifests = [n for n in archive.namelist() if n.endswith("/prepared_test_subset/manifest.json")]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one prepared_test_subset/manifest.json")
        prefix = manifests[0].removesuffix("prepared_test_subset/manifest.json")
        checksums = json.loads(archive.read(prefix + "checksums_sha256.json"))
        for name, expected in checksums.items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
        manifest = json.loads(archive.read(manifests[0]))
        if manifest["features"] != template["features"]:
            raise ValueError("Source/reference feature configurations differ")
        if manifest["extraction"]["partition"] != "test":
            raise ValueError("Archive subset must be from the test partition")
        records = manifest["records"]
        orders = [r["order"] for r in records]
        if not records or orders != list(range(orders[0], orders[0] + len(records))):
            raise ValueError("Expected contiguous ordered test windows")
        if len({r["sequence"] for r in records}) != 1:
            raise ValueError("Expected a single slide trajectory")
        trajectory = records[0]["sequence"].rsplit("/", 1)[-1]
        sequence = f"slide_{trajectory}"
        raw_prefix = prefix + f"raw/{trajectory}/"
        groups = [json.loads(line) for line in archive.read(raw_prefix + "groups.jsonl").decode("utf-8-sig").splitlines() if line.strip()]
        raw = [read_npz(archive, raw_prefix + group["data_file"]) for group in groups]
        adc = np.concatenate([r["adc"] for r in raw])
        audio_times = np.concatenate([r["audio_pc_est_ns"] for r in raw])
        sample_indices = np.concatenate([r["sample_index"] for r in raw])
        if (np.diff(audio_times) <= 0).any() or (np.diff(sample_indices) != 1).any():
            raise ValueError("Raw audio must be monotonic and complete")
        extractor = LogMel(FeatureConfig(**manifest["features"]))
        mean = np.asarray(stats["audio_mean"], np.float64)[:, None, None]
        std = np.asarray(stats["audio_std"], np.float64)[:, None, None]
        rows, labels, exported_records = [], [], []
        max_difference = 0.
        for record in records:
            if record["kind"] != "slide":
                raise ValueError("Only slide windows are supported")
            cached = read_npz(archive, prefix + "prepared_test_subset/" + record["file"])
            np.testing.assert_array_equal(cached["target"], record["target"])
            times = cached["event_pc_ns"]
            stops = np.searchsorted(audio_times, times, side="left")
            np.testing.assert_array_equal(stops, cached["audio_stop"])
            if (stops < 4000).any() or (stops >= len(adc)).any():
                raise ValueError("Insufficient causal ADC history/bracketing")
            windows = np.stack([adc[stop - 4000:stop] for stop in stops])
            for event_time, stop in zip(times, stops):
                ts = audio_times[stop - 4000:stop]
                if (np.diff(ts) > 30_000).any() or not ts[-1] < event_time <= audio_times[stop] or event_time - ts[-1] > 30_000:
                    raise ValueError("Noncausal or gapped ADC history")
            np.testing.assert_array_equal(audio_times[stops - 1], cached["audio_last_pc_ns"])
            with torch.inference_mode():
                mel = extractor(torch.from_numpy(windows)).numpy()
            difference = float(np.max(np.abs(mel - cached["audio"])))
            max_difference = max(max_difference, difference)
            # Float32 FFT/DC reductions vary across runtimes; log amplifies the
            # difference in quiet bins. Retain exact source features after QA.
            np.testing.assert_allclose(mel, cached["audio"], atol=2e-4, rtol=0)
            mel = cached["audio"]
            image_name = f"images/{sequence}_{record['order']:04d}.png"
            (stage / image_name).write_bytes(archive.read(prefix + "prepared_test_subset/" + record["image"]))
            # Verify the prepared image against the actual source camera frame.
            group = groups[record["order"]]
            with Image.open(io.BytesIO(archive.read(raw_prefix + group["first_frame"]["file"]))) as image:
                expected_image = np.asarray(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
            with Image.open(stage / image_name) as image:
                np.testing.assert_array_equal(np.asarray(image.convert("RGB")), expected_image)
            normalized = ((mel.astype(np.float64) - mean) / std).astype(np.float32)
            for index in range(record["events"]):
                call = len(rows)
                data_name = f"events/{call:06d}.npz"
                arrays = {key: cached[key][index] for key in
                          ("target", "event_pc_ns", "force_pc_ns", "force_age_ns", "audio_last_pc_ns", "previous_pc_ns")}
                np.savez_compressed(stage / data_name, adc=windows[index], audio=mel[index],
                                    audio_input=normalized[index], **arrays)
                row = dict(call_index=call, kind="slide", sequence=sequence, sequence_index=call,
                           schedule_offset_ns=round(call * 1e9 / 150), sequence_offset_ns=round(call * 1e9 / 150),
                           event_pc_ns=int(times[index]), data=data_name, image=image_name,
                           reset_state=call == 0, image_update=index == 0, source_window_id=record["id"],
                           source_sub_event=index, source_window_order=record["order"])
                rows.append(row)
                labels.append([call, "slide", sequence, call, int(times[index]), *cached["target"][index]])
            exported_records.append(dict(id=record["id"], kind="slide", order=record["order"],
                events=record["events"], image=image_name, source_cache_file=record["file"],
                original_source=record["source"], original_split="test"))
        (stage / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with (stage / "labels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["call_index", "kind", "sequence", "sequence_index", "event_pc_ns", *TARGET_NAMES])
            writer.writerows(labels)
        for name in ("normalization.json", "reader.py", "preprocessing_reference.py"):
            shutil.copyfile(reference / name, stage / name)
        result = {key: template[key] for key in ("format", "replay_hz", "features", "target_names", "target_units", "model_config")}
        result.update(source_archive=source.name, source_archive_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_cache=manifest["extraction"]["source_cache"], selection_split=manifest["extraction"]["source_split_files"],
            counts=dict(press=0, slide_windows=len(records), slide_events=len(rows), total_calls=len(rows)),
            slide_sequence=records[0]["sequence"], slide_source_orders=orders, slide_raw_intervals=len(groups),
            excluded_source_intervals=[dict(id=f"{records[0]['sequence']}/{i}",
                reason="Not present in the supplied prepared test subset; raw data retained in source ZIP")
                for i in range(len(groups)) if i not in orders],
            normalization_source="deployment_test_150hz/normalization.json (reference package)",
            normalization_note="Copied reference statistics, not fitted on this test subset. Viewer uses the selected checkpoint statistics.",
            timing=template["timing"], records=exported_records)
        write_json(stage / "manifest.json", result)
        (stage / "README.md").write_text(f"""# t0232 — 150 Hz 部署测试数据

来源：`{source.name}` 的 `prepared_test_subset/`，共 {len(records)} 个连续测试窗口、{len(rows)} 个事件、{len(records)} 张图像。
窗口 order 为 {orders[0]}–{orders[-1]}；这里只导出压缩包提供的测试子集，不扩充到整条原始轨迹。

格式与 `deployment_test_150hz` 相同：`events.jsonl` 每行对应一次调用，`events/*.npz` 保存
uint16 ADC `[4000,6]`、未标准化 Log-Mel `[6,64,28]`、标准化 `audio_input`、物理单位 `target[11]` 与原始时间戳。
`labels.csv` 保存全部标签；`images/` 保存训练预处理后的 224×224 RGB 图像。
首事件重置 GRU，此后连续传递状态；每 10 个事件更新一次图像。名义调度频率 150 Hz，事件跨度 {(len(rows)-1)/150:.6f} s。
原始采集时间、标签（包括首事件的滑移增量）均保持原样，没有按 150 Hz 重采样或补零。

`normalization.json` 与 `model_config` 沿用参考包，并非由本测试集拟合。
部署界面会使用所选 checkpoint 自身的标准化参数；独立 reader 使用其他模型时请传 `--normalization`。
输入始终保留全部 6 路麦克风，模型负责禁用通道。

```powershell
python reader.py --kind slide
python reader.py --kind slide --realtime
# 在项目根目录：
.venv/Scripts/python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/{output.name}
```

`verification.json` 记录源 ZIP 校验、逐事件特征重算与标签核对结果；`checksums.sha256` 覆盖包内其余文件。
""", encoding="utf-8")
        # Exercise the copied portable reader as well as checking exported arrays.
        from types import ModuleType
        reader = ModuleType("portable_t0232_reader")
        reader.__file__ = str(stage / "reader.py")
        exec(compile((stage / "reader.py").read_text(encoding="utf-8"), reader.__file__, "exec"), reader.__dict__)
        events = reader.load_events(stage, kind="slide")
        if len(events) != len(rows):
            raise ValueError("Portable reader event count mismatch")
        for row, event in zip(rows, events):
            with np.load(stage / row["data"], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(event["audio_input"][0], ((arrays["audio"].astype(np.float64)-mean)/std).astype(np.float32))
            if event["image_input"] is not None:
                np.testing.assert_array_equal(event["image_input"][0], load_image(stage / row["image"]).numpy())
        write_json(stage / "verification.json", dict(calls=len(rows), source_checksums_verified=len(checksums),
            all_selected_from_supplied_test_subset=True, all_supplied_test_windows_included=True,
            slide_orders_contiguous=True, raw_adc_logmel_recomputed=True,
            max_logmel_absolute_difference=max_difference, six_microphone_channels_retained=True,
            logmel_verification_absolute_tolerance=2e-4, exported_logmel_exact_match_source_cache=True,
            logmel_verification_note="Float32 FFT/DC runtime differences are amplified in quiet log-power bins; exported features preserve exact source-cache values.",
            all_targets_and_timestamps_match_source=True, image_tensors_exact_match_training_preprocessing=True,
            normalized_audio_exact_match=True, portable_reader_calls=len(events),
            image_updates=sum(r["image_update"] for r in rows), state_resets=sum(r["reset_state"] for r in rows)))
        files = sorted(path for path in stage.rglob("*") if path.is_file())
        (stage / "checksums.sha256").write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(stage).as_posix()}\n" for path in files), encoding="utf-8")
        stage.rename(output)
    return result["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "deployment/test_data/slide_t0232_raw_and_test_subset.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "deployment/test_data/deployment_test_150hz_t0232")
    parser.add_argument("--reference", type=Path, default=ROOT / "deployment/test_data/deployment_test_150hz")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output, args.reference), indent=2))


if __name__ == "__main__":
    main()
