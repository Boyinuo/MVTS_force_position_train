"""Ordered truncated BPTT, physical-unit metrics, and triple-head training."""
import json
from itertools import islice
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .architectures import DUAL_RATE_MODELS, build_model
from .data import write_json
from .dual_rate_data import (TARGET_NAMES, PreparedDataset, OrderedBatchSampler, feature_config,
                             fit_prepared_statistics,
                             validate_prepared_partition)
from .engine import choose_device, seed_everything, save_checkpoint, load_checkpoint, save_predictions
from .regression import dimension_metrics, masked_huber, zero_wrench_loss
from .prepared_selection import prepare_training_data


def make_loader(records, stats, batch_size, workers=0, training=False, seed=42, *, include_timing=False):
    return DataLoader(PreparedDataset(records, stats, include_timing=include_timing),
                      batch_sampler=OrderedBatchSampler(records, batch_size, training, seed),
                      num_workers=workers)


def run_sequence_epoch(model, loader, device, stats, optimizer=None, max_steps=None,
                       position_weight=1., wrench_weight=1., slip_weight=1., progress_label="Run",
                       zero_wrench_weight=1., slip_prior_weight=1.):
    training = optimizer is not None
    model.train(training)
    # Detached state crosses windows, never epochs, partitions or independent trials.
    state = {}
    slip_guided = getattr(model, "requires_timing", False)
    if slip_guided:
        model.set_normalization(stats)
    sums, seen = np.zeros(6 if slip_guided else 5), 0
    gate_weights = []
    predictions, targets, ids = [], [], []
    validity, unloaded_masks, contact_masks = [], [], []
    contact_supervised = False
    mean, std = np.array(stats["target_mean"]), np.array(stats["target_std"])
    wrench_mean = torch.tensor(mean[3:9], device=device, dtype=torch.float32)
    wrench_std = torch.tensor(std[3:9], device=device, dtype=torch.float32)
    started = last_log = time.perf_counter()
    total = min(len(loader), max_steps) if max_steps else len(loader)
    print(f"{progress_label}: 0/{total} batches", flush=True)
    iterator = loader if max_steps is None else islice(loader, max_steps)
    for step, batch in enumerate(iterator, 1):
        image, target = [batch[k].to(device) for k in ("image", "target")]
        audio = None if model.config.image_only else batch["audio"].to(device)
        valid = batch["target_valid"].to(device).bool()
        cv = batch.get("contact_valid", torch.zeros(target.shape[:2], dtype=torch.bool)).to(device).bool()
        unloaded = cv & (batch["contact"].to(device) == 0) if "contact" in batch else torch.zeros_like(cv)
        contact_supervised = contact_supervised or "contact" in batch
        h, history_flags = [], []
        for sequence, order in zip(batch["sequence"], batch["order"].tolist()):
            previous = state.get(sequence)
            has_history = previous is not None and previous[0]+1 == order
            hidden = previous[1] if has_history else torch.zeros(
                1, 1, model.config.hidden_size, device=device)
            h.append(hidden)
            history_flags.append(has_history)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            timing = {}
            if slip_guided:
                if not all(k in batch for k in ("delta_t", "image_age")):
                    raise ValueError("Slip-guided training requires a loader with include_timing=True")
                timing = {k: batch[k].to(device) for k in ("delta_t", "image_age")}
                timing["has_history"] = torch.tensor(history_flags, device=device, dtype=torch.bool)
            result = model.forward_sequence(image, audio, torch.cat(h, dim=1),
                                            sequence_length=target.shape[1], **timing)
            losses = [masked_huber(result[k], target[..., sl], valid[..., sl]) for k, sl in
                      (("position", slice(0,3)), ("wrench", slice(3,9)), ("slip", slice(9,11)))]
            losses.append(zero_wrench_loss(result["wrench"], wrench_mean, wrench_std, unloaded))
            weights = [position_weight, wrench_weight, slip_weight, zero_wrench_weight]
            if slip_guided:
                losses.append(masked_huber(result["slip_prior"], target[..., 9:], valid[..., 9:]))
                weights.append(slip_prior_weight)
                gate_weights.append(result["visual_weight"].detach().cpu().numpy().reshape(-1))
            loss = sum(w*v for w,v in zip(weights,losses))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite triple-head loss")
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite triple-head gradient")
                optimizer.step()
        for i, (sequence, order) in enumerate(zip(batch["sequence"], batch["order"].tolist())):
            state[sequence] = (order, result["hidden"][:,i:i+1].detach())
        pred = torch.cat([result[k] for k in ("position", "wrench", "slip")], dim=-1)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite predictions")
        n = target.shape[0]*target.shape[1]
        sums += n*np.array([loss.item()]+[v.item() for v in losses])
        seen += n
        predictions.append(pred.detach().cpu().numpy().reshape(-1,11)*std+mean)
        targets.append(target.detach().cpu().numpy().reshape(-1,11)*std+mean)
        validity.append(valid.cpu().numpy().reshape(-1,11))
        unloaded_masks.append(unloaded.cpu().numpy().reshape(-1))
        contact_masks.append(cv.cpu().numpy().reshape(-1))
        ids.extend(f"{identifier}/event_{j:02d}" for identifier in batch["id"] for j in range(target.shape[1]))
        now = time.perf_counter()
        if step == 1 or step == total or now-last_log > 10:
            print(f"{progress_label}: {step}/{total}, loss={sums[0]/seen:.6f}, events={seen}, seconds={now-started:.1f}", flush=True)
            last_log = now
    if not seen:
        raise ValueError("No events evaluated")
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    valid, unloaded, cv = np.concatenate(validity), np.concatenate(unloaded_masks), np.concatenate(contact_masks)
    physical_truth, physical_valid = targets.copy(), valid.copy()
    physical_truth[unloaded, 3:9] = 0.
    physical_valid[:, 3:9] |= cv[:, None]
    metrics = {"loss": float(sums[0]/seen), "position_loss": float(sums[1]/seen),
               "wrench_loss": float(sums[2]/seen), "slip_loss": float(sums[3]/seen),
               "zero_wrench_loss": float(sums[4]/seen),
               "samples": seen, "batches": step,
               "mae": dimension_metrics(predictions, physical_truth, physical_valid),
               "rmse": dimension_metrics(predictions, physical_truth, physical_valid, True),
               "raw_mae": dimension_metrics(predictions, targets, valid),
               "raw_rmse": dimension_metrics(predictions, targets, valid, True),
               "mean_predictor_mae": dimension_metrics(np.broadcast_to(mean, targets.shape), physical_truth, physical_valid),
               "target_valid_counts": dict(zip(TARGET_NAMES, valid.sum(0).tolist())),
               "no_contact_samples": int(unloaded.sum()),
               "no_contact_wrench_mae": dict(zip(TARGET_NAMES[3:9], np.abs(predictions[unloaded,3:9]).mean(0).tolist()))
                                        if unloaded.any() else None}
    values = (ids, predictions, targets)
    if slip_guided:
        gate = np.concatenate(gate_weights)
        metrics.update(slip_prior_loss=float(sums[5]/seen), visual_weight_mean=float(gate.mean()),
                       visual_weight_min=float(gate.min()), visual_weight_max=float(gate.max()),
                       audio_weight_mean=float(1. - gate.mean()))
    # Preserve the legacy CSV format for v4. V5 exports also mark invalid labels.
    return metrics, (*values, valid) if contact_supervised else values


def train_prepared(args):
    if args.model not in DUAL_RATE_MODELS:
        raise ValueError(f"Unsupported dual-rate regression model: {args.model}")
    slip_guided = args.model == "model_slip_gated"
    if args.use_contact:
        raise ValueError("Triple-head model does not consume contact labels")
    if args.strict_timing:
        raise ValueError("Strict timing cannot be certified: camera exposure timestamps are unavailable; see cache report")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = choose_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"last.pt").exists() and not args.resume:
        raise ValueError("Output contains last.pt; use --resume or a new --output")
    config, checkpoint = feature_config(), None
    if args.resume:
        model, checkpoint, config = load_checkpoint(args.resume, device)
        if model.model_name != args.model:
            raise ValueError("Resume model mismatch")
    default_roots = ([f".cache/data_final_v6_{kind}_contact" for kind in ("press", "slide", "polyline")]
                     if slip_guided else ".cache/data_final_v4")
    split, manifest, report = prepare_training_data(args, default_roots, checkpoint)
    write_json(output/"data_report.json", report)
    if args.audit_only:
        print({"windows":report["windows"], "splits": {k:len(v) for k,v in split.items()},
               "selection": {k: {field: value[field] for field in
                              ("threshold_mm", "retained_windows", "removed_windows", "removed_events")}
                             for k, value in report["training_selection"].items()},
               "xy_range_selection": report["xy_range_selection"]})
        return
    if args.resume:
        saved_channels = list(model.config.disabled_mic_channels)
        if args.disable_mic_channels is not None and sorted(args.disable_mic_channels) != saved_channels:
            raise ValueError("Resume microphone selection differs from checkpoint; start a new run to change channels")
        args.disable_mic_channels = saved_channels
        if args.image_only is not None and args.image_only != model.config.image_only:
            raise ValueError("Resume image_only differs from checkpoint; start a new run to change input modalities")
        args.image_only = model.config.image_only
        for name in ("seed","batch_size","lr","weight_decay","val_fraction","test_fraction", "max_groups",
                     "max_steps","max_eval_steps","position_weight","wrench_weight","slip_weight","zero_wrench_weight"):
            if getattr(args,name) != checkpoint["args"].get(name, 1. if name == "zero_wrench_weight" else None):
                raise ValueError(f"Resume option mismatch: {name}")
        for k in split:
            validate_prepared_partition(split[k], checkpoint, k)
        stats = checkpoint["stats"]
    else:
        args.disable_mic_channels = sorted(args.disable_mic_channels or [])
        args.image_only = bool(args.image_only)
        stats = fit_prepared_statistics(split["train"])
        model = build_model(args.model, {"disabled_mic_channels": args.disable_mic_channels,
                                                  "image_only": args.image_only}).to(device)
    if slip_guided:
        previous_weight = checkpoint["args"]["slip_prior_weight"] if checkpoint else 1.
        if args.slip_prior_weight is not None and checkpoint and args.slip_prior_weight != previous_weight:
            raise ValueError("Resume slip_prior_weight differs from checkpoint; start a new run")
        args.slip_prior_weight = previous_weight if args.slip_prior_weight is None else args.slip_prior_weight
        model.set_normalization(stats)
    print(f"Input mode: {'camera images only' if model.config.image_only else 'camera images + microphones'}", flush=True)
    print(f"Microphones enabled: {[i+1 for i in model.active_mic_indices]}; "
          f"disabled: {list(model.config.disabled_mic_channels)}", flush=True)
    loaders = {k:make_loader(v,stats,args.batch_size,args.workers,k=="train",args.seed,
                           include_timing=slip_guided) for k,v in split.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    generator = torch.Generator().manual_seed(args.seed)
    first, best = 1, float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        first, best = checkpoint["epoch"]+1, checkpoint["best_loss"]
        loaders["train"].batch_sampler.epoch = checkpoint["epoch"]
        if not (output/"best.pt").exists():
            import shutil
            shutil.copy2(Path(args.resume).parent/"best.pt",output/"best.pt")
    if first > args.epochs:
        raise ValueError("Increase --epochs beyond the completed checkpoint epoch")
    write_json(output/"normalization.json",stats)
    write_json(output/"splits.json",split)
    write_json(output/"config.json",{"training":vars(args), "model_name":model.model_name,
               "model":model.config.to_dict(),"features":config.to_dict(), "device":str(device)})
    for epoch in range(first,args.epochs+1):
        train,_ = run_sequence_epoch(model,loaders["train"],device,stats,optimizer,args.max_steps,
                                     args.position_weight,args.wrench_weight,args.slip_weight,f"Epoch {epoch} train",
                                     zero_wrench_weight=args.zero_wrench_weight,
                                     slip_prior_weight=args.slip_prior_weight if slip_guided else 1.)
        val,_ = run_sequence_epoch(model,loaders["val"],device,stats,max_steps=args.max_eval_steps,
            position_weight=args.position_weight,wrench_weight=args.wrench_weight,slip_weight=args.slip_weight,
            progress_label=f"Epoch {epoch} val",zero_wrench_weight=args.zero_wrench_weight,
            slip_prior_weight=args.slip_prior_weight if slip_guided else 1.)
        scheduler.step(val["loss"])
        improved = val["loss"] < best
        best = min(best,val["loss"])
        with (output/"history.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"epoch":epoch,"train":train,"val":val})+"\n")
        for path in ([output/"best.pt"] if improved else [])+[output/"last.pt"]:
            save_checkpoint(path,model,optimizer,scheduler,epoch,best,stats,config,args,split,generator,output/"best.pt")
    model,checkpoint,_ = load_checkpoint(output/"best.pt",device)
    test,values = run_sequence_epoch(model,loaders["test"],device,stats,max_steps=args.max_eval_steps,
        position_weight=args.position_weight,wrench_weight=args.wrench_weight,slip_weight=args.slip_weight,
        progress_label="Test",zero_wrench_weight=args.zero_wrench_weight,
        slip_prior_weight=args.slip_prior_weight if slip_guided else 1.)
    write_json(output/"test_metrics.json",test)
    save_predictions(output/"test_predictions.csv",values,target_names=TARGET_NAMES)
    write_json(output/"summary.json",{"model_name":model.model_name, "device":str(device),
        "best_epoch":checkpoint["epoch"], "completed_epoch":args.epochs,
        "smoke_test":manifest["smoke_test"] or bool(args.max_groups or args.max_steps or args.max_eval_steps),
        "split_windows":{k:len(v) for k,v in split.items()},"test":test})
    print(json.dumps(test,ensure_ascii=False),flush=True)
