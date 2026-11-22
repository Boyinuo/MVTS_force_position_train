"""Measure single-event streaming latency, separately from startup and file I/O.

Example: python benchmark_inference.py --device cuda --output runs/latency_cuda.json
Stop concurrent GPU training first. No network weights or training files are changed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from mvts.data import extract_sample_audio, read_json, write_json
from mvts.dual_rate_data import read_prepared
from mvts.engine import choose_device, load_checkpoint
from mvts.features import LogMel, load_image


def gpu_status():
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/triple_head_small/best.pt")
    parser.add_argument("--prepared-root", default=".cache/data_final_v4_small")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="runs/inference_latency.json")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("iterations/threads must be positive, warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    before = gpu_status()
    if args.device == "cuda" and before:
        for gpu in before.splitlines():
            if float(gpu.split(",")[-3]) > 20:
                raise RuntimeError("GPU is busy; stop training before measuring isolated inference latency")
    device = choose_device(args.device)
    model, checkpoint, config = load_checkpoint(args.checkpoint, device)
    contact_aware = model.model_name == "model_contact_gated"
    if model.model_name not in ("model_triple_head", "model_contact_gated"):
        raise ValueError("This benchmark requires a dual-rate triple-head/contact-gated model")
    model.eval()
    # Match the current training backend settings; no AMP or torch.compile.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if contact_aware:
        from mvts.contact_data import read_contact_prepared
        records,_ = read_contact_prepared(args.prepared_root)
    else:
        records, _ = read_prepared(args.prepared_root)
    record = next(r for r in records if r["kind"] == "press")
    root = Path(args.prepared_root)
    raw_root = Path(read_json(root / "report.json")["source_root"])
    raw_path = raw_root / record["source"]
    with np.load(raw_path, allow_pickle=False) as z:
        index = record.get("force_index", len(z["adc"]) - 1)
        raw = torch.from_numpy(extract_sample_audio(z, config, index).astype(np.float32)).unsqueeze(0)
    # Locate the actual camera-resolution image. Decode only once, outside timing;
    # the new-frame pipeline below still includes RGB resize and normalisation.
    manifest_path = raw_path.parent.parent / "samples.jsonl"
    sample_id = record["id"].split("/", 1)[1]
    source_record = next(json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines()
                         if line.strip() and str(json.loads(line).get("sample_id")) == sample_id)
    with Image.open(manifest_path.parent / source_record["image"]) as im:
        rgb = im.convert("RGB").copy()
    image_mean = torch.tensor([.485, .456, .406])[:, None, None]
    image_std = torch.tensor([.229, .224, .225])[:, None, None]

    def prepare_image():
        pixels = np.array(rgb.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32, copy=True)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1) / 255.
        return ((tensor - image_mean) / image_std).unsqueeze(0)

    image_cpu = prepare_image()
    torch.testing.assert_close(image_cpu[0], load_image(root / record["image"]))
    stats = checkpoint["stats"]
    audio_mean = torch.tensor(stats["audio_mean"], dtype=torch.float32)[:, None, None]
    audio_std = torch.tensor(stats["audio_std"], dtype=torch.float32)[:, None, None]
    target_mean = torch.tensor(stats["target_mean"], dtype=torch.float32)
    target_std = torch.tensor(stats["target_std"], dtype=torch.float32)
    extractor = LogMel(config).eval()  # CPU features, matching the current predict.py.
    with torch.inference_mode():
        mel_cpu = (extractor(raw) - audio_mean) / audio_std
    mel, image = mel_cpu.to(device), image_cpu.to(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def prime():
        model.reset_state()
        model.step(mel, image=image)
        sync()

    def network_cached():
        return model.step(mel)

    def network_new_image():
        return model.step(mel, image=image)

    def pipeline(new_image):
        audio = ((extractor(raw) - audio_mean) / audio_std).to(device)
        current_image = prepare_image().to(device) if new_image else None
        result = model.step(audio, image=current_image)
        if contact_aware:
            return result["physical_prediction"].cpu()
        prediction = torch.cat([result[k] for k in ("position", "wrench", "slip")], -1).cpu()
        return prediction * target_std + target_mean

    def measure(name, fn):
        with torch.inference_mode():
            prime()
            for _ in range(args.warmup):
                fn()
            sync()
            values = []
            for _ in range(args.iterations):
                sync()
                start = time.perf_counter_ns()
                result = fn()
                sync()
                values.append((time.perf_counter_ns() - start) / 1e6)
            tensor = result["position"] if isinstance(result, dict) else result
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Nonfinite benchmark prediction")
        values = np.asarray(values)
        summary = {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
                   "p95_ms": float(np.percentile(values, 95)), "min_ms": float(values.min()),
                   "max_ms": float(values.max()), "over_150hz_budget_fraction": float((values > 1000/150).mean()),
                   "samples_ms": values.tolist()}
        print(f"{name}: mean={summary['mean_ms']:.3f}ms, p95={summary['p95_ms']:.3f}ms", flush=True)
        return summary

    report = {"device": str(device), "gpu_before": before, "torch": str(torch.__version__),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "cpu_threads": args.threads, "checkpoint": str(Path(args.checkpoint).resolve()),
              "checkpoint_epoch": checkpoint["epoch"], "batch_size": 1, "dtype": "float32",
              "warmup": args.warmup, "iterations": args.iterations, "original_image_size": list(rgb.size),
              "input_audio_samples_per_channel": config.samples, "feature_config": config.to_dict(),
              "timing": "synchronised wall latency, single event, persistent model and GRU state",
              "excluded": ["model loading", "file reading/image decoding", "sensor acquisition and communication"],
              "network_inputs": "normalised image/log-mel already on inference device",
              "pipeline": "in-memory raw ADC -> CPU STFT/log-mel/normalisation -> device transfer -> network -> CPU 11-D output; new-image case also resizes/normalises original RGB",
              "results": {}}
    for name, fn in (("network_cached_visual", network_cached), ("network_new_visual", network_new_image),
                     ("pipeline_cached_visual", lambda: pipeline(False)), ("pipeline_new_visual", lambda: pipeline(True))):
        report["results"][name] = measure(name, fn)
    report["weighted_15hz_camera_150hz_mean_ms"] = {
        prefix: (9*report["results"][f"{prefix}_cached_visual"]["mean_ms"] +
                 report["results"][f"{prefix}_new_visual"]["mean_ms"])/10
        for prefix in ("network", "pipeline")}
    report["gpu_after"] = gpu_status()
    write_json(args.output, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
