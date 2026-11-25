"""Causal runtime for an independent-image anchor checkpoint (stage two).

The input tensor is the SAME normalized RGB tensor returned by
mvts.features.load_image. No target, force, audio or future image is accepted.
Outputs concern position only: the original wrench/slip heads were not refined.
"""
from pathlib import Path

import numpy as np
import torch

from mvts.engine import load_checkpoint
from mvts.features import load_image


class ImageExposurePredictor:
    def __init__(self, checkpoint, device='cuda'):
        self.model, payload, _ = load_checkpoint(checkpoint, device)
        if payload.get('experiment_type') != 'image_exposure_independent_anchor_v2':
            raise ValueError('Use original model.step for the baseline fallback; this runtime requires an anchor-v2 checkpoint')
        self.model.eval()
        self.device = device
        self.config = payload['anchor_config']
        self.mean = np.asarray(payload['stats']['target_mean'][:3])
        self.std = np.asarray(payload['stats']['target_std'][:3])
        self.reset()

    def reset(self):
        self.anchor = None
        self.image_ns = None
        self.velocity = np.zeros(3)
        self.frame_id = None

    @torch.inference_mode()
    def update(self, image, frame_pc_ns, *, frame_id=None, reset=False):
        """Call once when an image arrives; timestamps must share a monotonic clock.

        Explicit reset is required for independent trajectories or known missing
        intervals, matching the discontinuity resets in offline evaluation.
        """
        if reset:
            self.reset()
        timestamp = int(frame_pc_ns)
        if self.image_ns is not None and timestamp <= self.image_ns:
            raise ValueError('Frame timestamps must increase; reset at a new trajectory')
        if isinstance(image, (str, Path)):
            image = load_image(image)
        if image.ndim == 3:
            image = image[None]
        if image.shape != (1, 3, 224, 224):
            raise ValueError('Expected one normalized image [3,224,224] or [1,3,224,224]')
        result = self.model.forward_sequence(image.to(self.device), sequence_length=1)
        anchor = result['position'][0, 0].cpu().numpy() * self.std + self.mean
        if self.anchor is not None:
            observed = (anchor - self.anchor) / ((timestamp - self.image_ns) * 1e-9)
            speed = float(np.linalg.norm(observed))
            if speed > self.config['max_speed_mm_s']:
                observed *= self.config['max_speed_mm_s'] / speed
            weight = self.config['smoothing']
            self.velocity = weight * observed + (1 - weight) * self.velocity
        self.anchor, self.image_ns, self.frame_id = anchor, timestamp, frame_id
        return self.predict(timestamp)

    def predict(self, event_pc_ns):
        """Estimate position at event time using only the latest available image."""
        if self.anchor is None:
            raise ValueError('Update with the first available image before predicting')
        age_ns = int(event_pc_ns) - self.image_ns
        if age_ns < 0:
            raise ValueError('A frame cannot be used before its read-completion time')
        age = age_ns * 1e-9 + self.config['delay_ms'] * .001
        return {'position_mm': (self.anchor + self.velocity * age).copy(),
                'anchor_mm': self.anchor.copy(), 'velocity_mm_s': self.velocity.copy(),
                'image_age_ms': age * 1000., 'event_pc_ns': int(event_pc_ns),
                'estimated_image_midpoint_pc_ns': self.image_ns - int(round(self.config['delay_ms'] * 1e6))}
