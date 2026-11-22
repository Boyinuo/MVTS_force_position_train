"""Train the diagram's visual-acoustic tactile regression architecture.

Example: python train.py --prepared-root .cache/data_final_v4 --epochs 1 --max-groups 3 --max-steps 2
"""
import argparse
import json
from pathlib import Path
import platform
import time

import torch
from torch.utils.data import DataLoader

from mvts.data import (TactileDataset, discover_records, fit_statistics, limit_split_groups,
                       split_records, write_json)
from mvts.engine import (choose_device, load_checkpoint, run_epoch, save_checkpoint,
                         save_predictions, seed_everything, validate_positive_arguments)
from mvts.features import FeatureConfig
from mvts.architectures import DEFAULT_MODEL, available_models, build_model, get_model_module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--dataset", metavar="NAME",
                             help="Subfolder name under data/ (triple-head default: data_final)")
    data_source.add_argument("--data-root", metavar="PATH",
                             help="Read a data directory directly instead of selecting --dataset")
    parser.add_argument("--output", default="runs/train")
    parser.add_argument("--prepared-root", help="Triple-head prepared cache from preprocess.py")
    parser.add_argument("--image-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Triple-head: use camera images only, bypassing audio. Resume defaults to checkpoint setting")
    parser.add_argument("--disable-mic-channels", type=int, nargs="+", choices=range(1, 7),
                        help="Triple-head: exclude microphone numbers 1..6; e.g. 5. Resume defaults to checkpoint setting")
    parser.add_argument("--model", choices=available_models(),
                        help=f"Model module filename without .py; default: {DEFAULT_MODEL}, or checkpoint model on resume")
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=0,
                        help="Data-loading processes; 0 loads in the main process (default)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use-contact", action="store_true", help="Optional contact embedding; diagram default is off")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--wrench-weight", type=float, default=1.0)
    parser.add_argument("--slip-weight", type=float, default=1.0)
    parser.add_argument("--contact-weight", type=float, default=1.0, help="Contact-aware BCE loss weight")
    parser.add_argument("--zero-wrench-weight", type=float, default=1.0, help="No-contact physical zero wrench loss weight")
    parser.add_argument("--contact-off-threshold", type=float, help="Contact-aware hysteresis release threshold (default .4)")
    parser.add_argument("--contact-on-threshold", type=float, help="Contact-aware hysteresis onset threshold (default .6)")
    parser.add_argument("--max-groups", type=int, help="Smoke: at most this many locations per split")
    parser.add_argument("--max-steps", type=int, help="Smoke: limit training batches per epoch")
    parser.add_argument("--max-eval-steps", type=int, help="Smoke: limit validation/test batches")
    parser.add_argument("--strict-timing", action="store_true")
    parser.add_argument("--skip-bad-records", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", help="Path to last.pt; retain the original training/split options")
    args = parser.parse_args()
    if args.disable_mic_channels is not None:
        if args.image_only:
            parser.error("--image-only cannot be combined with --disable-mic-channels")
        if len(set(args.disable_mic_channels)) != len(args.disable_mic_channels) or len(args.disable_mic_channels) == 6:
            parser.error("--disable-mic-channels must be distinct and leave at least one microphone enabled")
    if args.data_root is None:
        if args.dataset is None:
            args.dataset = "data_final" if args.model in (None, "model_triple_head", "model_contact_gated") else "data_sensor_normal"
        if (not args.dataset or args.dataset in (".", "..")
                or any(char in args.dataset for char in ("/", "\\", ":"))):
            parser.error("--dataset must be a single subfolder name under data/")
        args.data_root = str(Path("data") / args.dataset)
    if not args.prepared_root and not Path(args.data_root).is_dir():
        parser.error(f"Data directory does not exist: {args.data_root}")
    return args


def main():
    args = parse_args()
    validate_positive_arguments(args)
    if args.image_only is not None:
        selected_model = args.model or (torch.load(args.resume, map_location="cpu", weights_only=True).get(
            "model_name", "model") if args.resume else DEFAULT_MODEL)
        if selected_model != "model_triple_head":
            raise ValueError("--image-only/--no-image-only is supported only by model_triple_head")
    if args.model == "model_contact_gated":
        from mvts.contact_engine import train_contact_prepared
        return train_contact_prepared(args)
    if args.model == "model_triple_head" or (args.model is None and not args.resume):
        from mvts.dual_rate_engine import train_prepared
        args.model = "model_triple_head"
        return train_prepared(args)
    if args.resume and args.model is None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved.get("model_name") == "model_contact_gated":
            from mvts.contact_engine import train_contact_prepared
            args.model = "model_contact_gated"
            return train_contact_prepared(args)
        if saved.get("model_name") == "model_triple_head":
            from mvts.dual_rate_engine import train_prepared
            args.model = "model_triple_head"
            return train_prepared(args)
    if args.disable_mic_channels is not None:
        raise ValueError("--disable-mic-channels is supported only by model_triple_head")
    start = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep previous runs reviewable; use --resume or a fresh directory.
    if not args.resume and not args.audit_only and (output / "last.pt").exists():
        raise ValueError(f"{output}/last.pt exists; use --resume or a new --output")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Data directory: {Path(args.data_root).resolve()}", flush=True)
    checkpoint = None
    if args.resume:
        model, checkpoint, feature_config = load_checkpoint(args.resume, device)
        if args.model is not None and args.model != model.model_name:
            raise ValueError(f"Resume model mismatch: --model {args.model}, checkpoint uses {model.model_name}")
        args.model = model.model_name
        args.use_contact = args.use_contact or model.config.use_contact
    else:
        args.model = args.model or DEFAULT_MODEL
        model_config = get_model_module(args.model).ModelConfig()
        args.use_contact = args.use_contact or model_config.use_contact
        feature_config = FeatureConfig(alignment_mode=model_config.audio_alignment_mode,
                                       before_samples=model_config.audio_samples_before,
                                       after_samples=model_config.audio_samples_after)
    records, report = discover_records(args.data_root, strict_timing=args.strict_timing,
                                      feature_config=feature_config)
    write_json(output / "data_report.json", report)
    print(f"Audited {report['total_events']} events; included {len(records)}; "
          f"locations {report['groups']}; rejected {len(report['rejected'])}", flush=True)
    for session in report["sessions"]:
        for warning in session["warnings"]:
            print(f"DATA WARNING [{session['session']}]: {warning}", flush=True)
    if report["rejected"] and not args.skip_bad_records:
        raise ValueError("Rejected data found; review data_report.json, then optionally --skip-bad-records")
    if args.audit_only:
        return
    split = limit_split_groups(split_records(records, args.seed, args.val_fraction, args.test_fraction),
                               args.max_groups, args.seed)
    if checkpoint is None:
        feature_config.sample_rate = report["sample_rate_hz"]
    elif abs(feature_config.sample_rate - report["sample_rate_hz"]) > 1:
        raise ValueError("Resume sample rate differs from current data")
    if args.resume:
        previous_config = output / "config.json"
        if previous_config.exists():
            previous = json.loads(previous_config.read_text(encoding="utf-8"))
            if previous.get("model_name", DEFAULT_MODEL) != args.model:
                raise ValueError("Output directory belongs to a different model; choose a new --output")
        for name in ("seed", "batch_size", "lr", "weight_decay", "val_fraction", "test_fraction",
                     "max_groups", "max_steps", "max_eval_steps", "use_contact", "position_weight", "wrench_weight"):
            if getattr(args, name) != checkpoint["args"][name]:
                raise ValueError(f"Resume option mismatch: --{name.replace('_', '-')}")
        for key, partition in split.items():
            if [r["id"] for r in partition] != checkpoint["split_ids"][key]:
                raise ValueError("Dataset/split changed; start a new run for new samples")
            if [r["target"] for r in partition] != checkpoint["split_labels"][key]:
                raise ValueError("Dataset labels changed; start a new run")
        stats = checkpoint["stats"]
    else:
        print(f"Fitting normalization on {len(split['train'])} training samples", flush=True)
        statistics_start = time.perf_counter()
        stats = fit_statistics(split["train"], feature_config)
        print(f"Normalization ready in {time.perf_counter() - statistics_start:.1f}s", flush=True)
        model_config.use_contact = args.use_contact
        model_config.audio_frames = feature_config.audio_frames
        model = build_model(args.model, model_config).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {key: DataLoader(TactileDataset(partition, feature_config, stats),
                               batch_size=args.batch_size, shuffle=(key == "train"),
                               generator=generator if key == "train" else None,
                               num_workers=args.workers, pin_memory=(device.type == "cuda"))
               for key, partition in split.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Plateau schedule can be restored even when the total number of epochs changes.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    first_epoch, best_loss = 1, float("inf")
    best_path = output / "best.pt"
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        first_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_loss"]
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        if not best_path.exists():
            # A moved last.pt cannot stand in for the best validation weights.
            old_best = Path(args.resume).resolve().parent / "best.pt"
            if not old_best.exists():
                raise ValueError("Resume needs sibling best.pt for final best-model evaluation")
            import shutil
            shutil.copy2(old_best, best_path)
    if first_epoch > args.epochs:
        raise ValueError(f"Checkpoint already completed {first_epoch - 1} epochs; increase --epochs")
    write_json(output / "splits.json", split)
    write_json(output / "normalization.json", stats)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "device": str(device), "cuda_available": torch.cuda.is_available(),
                   "cuda_runtime": torch.version.cuda,
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "parameters": sum(p.numel() for p in model.parameters())}
    write_json(output / "config.json", {"training": vars(args), "model_name": args.model,
                                        "model": model.config.to_dict(),
                                        "features": feature_config.to_dict(), "environment": environment})
    print(json.dumps(environment), flush=True)
    print(f"Model: {args.model} (mvts/{args.model}.py)", flush=True)
    print("Split samples: " + str({k: len(v) for k, v in split.items()}), flush=True)
    print("Split groups: " + str({k: len({r['group'] for r in v}) for k, v in split.items()}), flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics, _ = run_epoch(model, loaders["train"], device, stats, optimizer, args.max_steps,
                                      args.position_weight, args.wrench_weight,
                                      progress_label=f"Epoch {epoch}/{args.epochs} train")
        val_metrics, _ = run_epoch(model, loaders["val"], device, stats, max_steps=args.max_eval_steps,
                                   position_weight=args.position_weight, wrench_weight=args.wrench_weight,
                                   progress_label=f"Epoch {epoch}/{args.epochs} val")
        scheduler.step(val_metrics["loss"])
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss = val_metrics["loss"]
        log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - epoch_start, "train": train_metrics, "val": val_metrics}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, ensure_ascii=False, allow_nan=False) + "\n")
        if improved:
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_loss, stats,
                            feature_config, args, split, generator, best_path)
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, epoch, best_loss, stats,
                        feature_config, args, split, generator, best_path)
        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_metrics['loss']:.6f}, "
              f"val_loss={val_metrics['loss']:.6f}, {log['seconds']:.1f}s", flush=True)
    best_model, best_checkpoint, _ = load_checkpoint(best_path, device)
    test_metrics, predictions = run_epoch(best_model, loaders["test"], device, stats,
                                          max_steps=args.max_eval_steps,
                                          position_weight=args.position_weight, wrench_weight=args.wrench_weight,
                                          progress_label="Test")
    write_json(output / "test_metrics.json", test_metrics)
    save_predictions(output / "test_predictions.csv", predictions)
    summary = {"environment": environment, "model_name": args.model, "total_events": report["total_events"],
               "eligible_samples": len(records), "split_samples": {k: len(v) for k, v in split.items()},
               "smoke_test": any(v is not None for v in (args.max_groups, args.max_steps, args.max_eval_steps)),
               "best_epoch": best_checkpoint["epoch"], "completed_epoch": args.epochs,
               "test": test_metrics, "seconds": time.perf_counter() - start,
               "note": "Pipeline verification on incomplete data does not establish model accuracy."}
    write_json(output / "summary.json", summary)
    mae = test_metrics["mae"]
    print(f"test_loss={test_metrics['loss']:.6f}", flush=True)
    print("test_position_mae (mm): " + ", ".join(
        f"{label}={mae[key]:.6f}" for label, key in
        (("x", "x_mm"), ("y", "y_mm"), ("delta_z", "delta_z_mm"))), flush=True)
    print("test_wrench_mae: " + ", ".join(
        f"{name}={mae[name]:.6f} {unit}" for name, unit in
        (("Fx", "N"), ("Fy", "N"), ("Fz", "N"),
         ("Mx", "N*m"), ("My", "N*m"), ("Mz", "N*m"))), flush=True)
    print(f"saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
