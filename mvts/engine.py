"""Training, physical-unit evaluation, and self-contained checkpoints."""
import csv
from itertools import islice
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import TARGET_NAMES
from .features import FeatureConfig
from .architectures import DEFAULT_MODEL, build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in this PyTorch environment; use --device cpu")
    return torch.device(name)


def run_epoch(model, loader, device, stats, optimizer=None, max_steps=None,
              position_weight=1.0, wrench_weight=1.0, *, progress_label=None,
              progress_interval=10.0):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.SmoothL1Loss()
    mean = np.asarray(stats["target_mean"])
    std = np.asarray(stats["target_std"])
    sums = np.zeros(3, dtype=np.float64)
    predictions, targets, identifiers = [], [], []
    seen, batches = 0, 0
    total_batches = min(len(loader), max_steps) if max_steps is not None else len(loader)
    started = previous_end = last_log = time.perf_counter()
    data_seconds = compute_seconds = 0.0
    if progress_label:
        print(f"{progress_label}: 0/{total_batches} batches; loading first batch", flush=True)
    batch_iterator = loader if max_steps is None else islice(loader, max_steps)
    for batch in batch_iterator:
        batch_ready = time.perf_counter()
        data_seconds += batch_ready - previous_end
        image, audio, target, contact = [batch[k].to(device, non_blocking=device.type == "cuda") for k in
                                          ("image", "audio", "target", "starts_in_contact")]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            result = model(image, audio, contact)
            pos_loss = loss_fn(result["position"], target[:, :3])
            force_loss = loss_fn(result["wrench"], target[:, 3:])
            loss = position_weight * pos_loss + wrench_weight * force_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check data and normalization")
            if training:
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
        prediction = torch.cat((result["position"], result["wrench"]), dim=-1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite prediction")
        n = target.shape[0]
        sums += n * np.array([loss.item(), pos_loss.item(), force_loss.item()])
        seen += n
        batches += 1
        predictions.append(prediction.detach().cpu().numpy() * std + mean)
        targets.append(target.detach().cpu().numpy() * std + mean)
        identifiers.extend(batch["id"])
        now = time.perf_counter()
        compute_seconds += now - batch_ready
        if progress_label and (batches == 1 or batches == total_batches or now - last_log >= progress_interval):
            elapsed = now - started
            eta = elapsed / batches * (total_batches - batches)
            print(f"{progress_label}: {batches}/{total_batches} batches, "
                  f"loss={sums[0] / seen:.6f}, {seen / elapsed:.1f} samples/s, "
                  f"elapsed={elapsed:.1f}s, eta={eta:.1f}s, "
                  f"data={1000 * data_seconds / batches:.1f}ms/batch, "
                  f"compute={1000 * compute_seconds / batches:.1f}ms/batch", flush=True)
            last_log = now
        previous_end = time.perf_counter()
    if not seen:
        raise ValueError("Empty loader / zero evaluation steps")
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    error = predictions - targets
    metrics = {"loss": float(sums[0] / seen), "position_loss": float(sums[1] / seen),
               "wrench_loss": float(sums[2] / seen), "samples": seen, "batches": batches,
               "mae": dict(zip(TARGET_NAMES, np.abs(error).mean(axis=0).tolist())),
               "rmse": dict(zip(TARGET_NAMES, np.sqrt(np.square(error).mean(axis=0)).tolist())),
               "mean_predictor_mae": dict(zip(TARGET_NAMES, np.abs(targets - mean).mean(axis=0).tolist()))}
    return metrics, (identifiers, predictions, targets)


def save_predictions(path, values, target_names=None):
    target_names = target_names or TARGET_NAMES
    identifiers, predictions, targets = values
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id"] + [f"true_{k}" for k in target_names] + [f"pred_{k}" for k in target_names])
        for identifier, prediction, target in zip(identifiers, predictions, targets):
            writer.writerow([identifier] + target.tolist() + prediction.tolist())


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    # Checkpoints created before model selection use the original fusion architecture.
    model = build_model(checkpoint.get("model_name", "model"), checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    if getattr(model,"model_name",None) == "model_contact_gated" and "stats" in checkpoint:
        saved_mean = torch.tensor(checkpoint["stats"]["target_mean"],dtype=model.target_mean.dtype)
        saved_std = torch.tensor(checkpoint["stats"]["target_std"],dtype=model.target_std.dtype)
        if (not model.normalization_ready.item() or not torch.equal(model.target_mean,saved_mean)
                or not torch.equal(model.target_std,saved_std)):
            raise ValueError("Contact checkpoint physical normalization differs from saved stats")
    return model.to(device), checkpoint, FeatureConfig(**checkpoint["feature_config"])


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_loss, stats, feature_config,
                    args, split, generator, best_model_path):
    checkpoint = {
        "format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "best_loss": best_loss,
        "stats": stats, "model_config": model.config.to_dict(),
        "model_name": getattr(model, "model_name", DEFAULT_MODEL),
        "feature_config": feature_config.to_dict(), "args": vars(args),
        "split_ids": {k: [r["id"] for r in records] for k, records in split.items()},
        "split_labels": {k: [r["target"] for r in records] for k, records in split.items()},
        "split_supervision": {k: [r.get("supervision_sha256") for r in records] for k,records in split.items()},
        "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "best_model_path": str(best_model_path),
    }
    # Replace atomically; interrupted writes do not destroy the previous checkpoint.
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_positive_arguments(args):
    for name in ("epochs", "batch_size", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_steps", "max_eval_steps", "max_groups"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("lr", "position_weight", "wrench_weight", "slip_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("contact_weight", "zero_wrench_weight"):
        value = getattr(args,name,1.)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if args.workers < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("workers/weight_decay must be nonnegative")
