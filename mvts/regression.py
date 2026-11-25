"""Shared masked regression objectives and physical-unit error metrics."""
import numpy as np
import torch
from torch.nn import functional as F

from .dual_rate_data import TARGET_NAMES


def masked_huber(prediction, target, valid):
    mask = valid.expand_as(prediction)
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    return torch.where(mask, error, 0.).sum() / mask.sum().clamp_min(1)


def zero_wrench_loss(prediction, mean, std, unloaded):
    """Supervise physical zero before any gate, scaled per force/torque axis."""
    physical = prediction * std + mean
    return masked_huber(physical / std, torch.zeros_like(physical), unloaded[..., None])


def dimension_metrics(prediction, target, valid, squared=False):
    error = (prediction - target)**2 if squared else np.abs(prediction - target)
    count = valid.sum(0)
    values = np.where(valid, error, 0.).sum(0) / np.maximum(count, 1)
    if squared:
        values = np.sqrt(values)
    return {name: float(value) if n else None
            for name, value, n in zip(TARGET_NAMES, values, count)}
