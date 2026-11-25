"""Causal position refinement with an explicit image exposure midpoint.

This experimental module never changes the original fusion model. Its inputs
are frozen image features, host timing and (optionally) original predictions.
All positions returned here are in physical millimetres; velocities are mm/s.
Exposure duration supplies a midpoint assumption, not a measured camera delay.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class ExposurePositionModel(nn.Module):
    """Update once per image, then propagate its anchor to each event time.

    ``frame_dt`` is elapsed host-read time between adjacent images, in seconds.
    ``host_ages`` is event time minus the current image host-read timestamp.
    Under the explicit zero post-exposure-delay assumption, the image's physical
    midpoint precedes that timestamp by half ``exposure_ms``. Consequently::

        position = anchor + velocity * (host_age + exposure_ms / 2000
                                       + readout_delay_ms / 1000)

    ``base_anchor`` is an optional *prediction*, never a measured position.
    When provided it makes this a residual model. Targets belong only in losses.
    Calling this module on a padded batch is safe for earlier valid outputs;
    losses must mask padding, and padded terminal hidden states must not be reused.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        hidden_size: int = 128,
        exposure_ms: float = 15.6,
        readout_delay_ms: float = 0.0,
        target_mean: tuple[float, ...] = (0.0, 0.0, 0.0),
        target_std: tuple[float, ...] = (1.0, 1.0, 1.0),
        max_speed: float = 200.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_size < 2:
            raise ValueError("feature_dim and hidden_size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not torch.isfinite(torch.tensor(exposure_ms)) or exposure_ms < 0:
            raise ValueError("exposure_ms must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(readout_delay_ms)) or readout_delay_ms < 0:
            raise ValueError("readout_delay_ms must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(max_speed)) or max_speed <= 0:
            raise ValueError("max_speed must be finite and positive")
        mean, std = torch.as_tensor(target_mean, dtype=torch.float32), torch.as_tensor(target_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("target_mean/std must be finite three-vectors with positive std")
        self.config = {
            "feature_dim": int(feature_dim), "hidden_size": int(hidden_size),
            "exposure_ms": float(exposure_ms), "readout_delay_ms": float(readout_delay_ms),
            "target_mean": mean.tolist(),
            "target_std": std.tolist(), "max_speed": float(max_speed),
            "dropout": float(dropout),
        }
        self.feature_dim, self.hidden_size = feature_dim, hidden_size
        self.exposure_ms, self.max_speed = float(exposure_ms), float(max_speed)
        self.readout_delay_ms = float(readout_delay_ms)
        self.register_buffer("target_mean", mean)
        self.register_buffer("target_std", std)
        # The feature difference makes image-to-image change explicit. Actual dt
        # is supplied separately, so no nominal 15 Hz assumption enters motion.
        self.input_projection = nn.Sequential(
            nn.Linear(2 * feature_dim + 1, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.anchor_head = self._head(hidden_size, dropout)
        self.velocity_head = self._head(hidden_size, dropout)
        # A copied baseline starts unperturbed: no displacement or residual bias.
        for head in (self.anchor_head, self.velocity_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _head(hidden_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 3),
        )

    def forward(
        self,
        features: Tensor,
        frame_dt: Tensor,
        host_ages: Tensor,
        base_anchor: Tensor | None = None,
        *,
        hidden: Tensor | None = None,
        previous_feature: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim or features.shape[1] < 1:
            raise ValueError("features must have shape [B,F,feature_dim] with F >= 1")
        batch, frames, _ = features.shape
        if frame_dt.shape != (batch, frames) or host_ages.ndim != 3 or host_ages.shape[:2] != (batch, frames) or host_ages.shape[-1] < 1:
            raise ValueError("frame_dt must be [B,F] and host_ages [B,F,T]")
        if not torch.isfinite(features).all() or not torch.isfinite(frame_dt).all() or not torch.isfinite(host_ages).all():
            raise ValueError("features and timing must be finite")
        if (frame_dt <= 0).any() or (host_ages < 0).any():
            raise ValueError("frame_dt must be positive and host_ages nonnegative")
        if base_anchor is not None and (base_anchor.shape != (batch, frames, 3) or not torch.isfinite(base_anchor).all()):
            raise ValueError("base_anchor must be finite [B,F,3] physical predictions")
        if previous_feature is None:
            previous_feature = features[:, 0]
        if previous_feature.shape != (batch, self.feature_dim):
            raise ValueError("previous_feature must have shape [B,feature_dim]")
        predecessor = torch.cat((previous_feature[:, None], features[:, :-1]), dim=1)
        delta = features - predecessor
        # Scale time into a convenient range; actual dt is retained, without
        # dividing feature differences by tiny gaps or accessing future frames.
        inputs = torch.cat((features, delta, (frame_dt * 15.0).unsqueeze(-1)), dim=-1)
        states, hidden_out = self.gru(self.input_projection(inputs), hidden)
        residual = self.anchor_head(states) * self.target_std
        origin = self.target_mean if base_anchor is None else base_anchor
        anchor = origin + residual
        velocity = torch.tanh(self.velocity_head(states)) * self.max_speed
        age = host_ages + self.exposure_ms / 2000.0 + self.readout_delay_ms / 1000.0
        position = anchor.unsqueeze(-2) + velocity.unsqueeze(-2) * age.unsqueeze(-1)
        return {
            "position": position, "anchor": anchor, "velocity": velocity,
            "hidden": hidden_out, "last_feature": features[:, -1],
        }

    def export_checkpoint(self, **metadata: Any) -> dict[str, Any]:
        """Build a standalone payload; caller chooses its independent output path."""
        return {**metadata, "model_name": "experimental_exposure_position_v1",
                "model_config": dict(self.config), "model": self.state_dict()}


def from_checkpoint(checkpoint: dict[str, Any], device: str | torch.device = "cpu") -> ExposurePositionModel:
    """Restore this model without registering or changing baseline architectures."""
    if checkpoint.get("model_name") != "experimental_exposure_position_v1":
        raise ValueError("Not an exposure-position checkpoint")
    model = ExposurePositionModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device)
