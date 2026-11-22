"""Four-head causal estimator with a separate deterministic contact state machine.

  Cached image -> Visual ResNet18 -> 16x128 --+
  6x4000 audio -> per-mic ResNet18 -> 28x128 -+-> cross-attention -> GRU(128,256)
                                                                  |
                             +---------------+--------------------+-------------+
                             |               |                    |             |
                          position(3)    wrench(6)             slip(2)     contact(1)
                             +---------------+--------------------+          sigmoid
                                             |                                  |
                                      DE-NORMALIZE                   hysteresis <-> delayed c
                                             +-----------------------> physical gate

forward/forward_sequence return RAW normalized regressions plus contact logits.
gate_outputs returns physical-unit outputs and validity flags, using PREDICTED
contact only. step carries BOTH h and predicted c, and caches visual tokens.
Ground-truth labels and acquisition starts_in_contact are never model inputs.
Contact transitions do not reset GRU memory. No learned transition experts.
"""
from dataclasses import dataclass
import math

import torch

from .model_triple_head import ModelConfig as BaseConfig, TactileFusionNet as BaseNet


@dataclass
class ModelConfig(BaseConfig):
    contact_off_threshold: float = .4
    contact_on_threshold: float = .6
    contact_initial_threshold: float = .5

    def __post_init__(self):
        super().__post_init__()
        off, initial, on = self.contact_off_threshold, self.contact_initial_threshold, self.contact_on_threshold
        if not all(math.isfinite(x) for x in (off, initial, on)) or not 0 <= off < initial < on <= 1:
            raise ValueError("Contact thresholds must satisfy 0 <= off < initial < on <= 1")


class TactileFusionNet(BaseNet):
    def __init__(self, config=None):
        super().__init__(config or ModelConfig())
        self.contact_head = self._head(1)
        # Persist normalization with weights so standalone streaming gates in
        # physical units rather than mapping normalized zero to the label mean.
        self.register_buffer("target_mean", torch.zeros(11))
        self.register_buffer("target_std", torch.ones(11))
        self.register_buffer("normalization_ready", torch.tensor(False))

    def set_normalization(self, stats):
        mean = torch.as_tensor(stats["target_mean"], device=self.target_mean.device, dtype=self.target_mean.dtype)
        std = torch.as_tensor(stats["target_std"], device=self.target_std.device, dtype=self.target_std.dtype)
        if mean.shape != (11,) or std.shape != (11,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization must contain 11 finite means and positive standard deviations")
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)
        self.normalization_ready.fill_(True)

    def _predict(self, states):
        result = super()._predict(states)
        logits = self.contact_head(states).squeeze(-1)
        return {**result, "contact_logits": logits, "contact_probability": logits.sigmoid()}

    def forward(self, image, audio, starts_in_contact=None, *, hidden=None, visual_tokens=None):
        if starts_in_contact is not None:
            raise ValueError("Ground-truth/acquisition contact flags are not inputs; use predicted contact in gate_outputs")
        return super().forward(image, audio, hidden=hidden, visual_tokens=visual_tokens)

    def gate_outputs(self, result, previous_contact=None):
        """Gate RAW predictions after de-normalization, with a delayed predicted c.

        previous_contact is [B], values -1 (unknown), 0 or 1. An unknown first
        state is classified directly and has no fabricated transition or slip
        reference. Transition codes: -1 unknown, 0=00, 1=01, 2=10, 3=11.
        Invalid position/slip are represented by zero PLUS explicit validity.
        """
        if not self.normalization_ready.item():
            raise RuntimeError("Call set_normalization(stats) before physical output gating")
        probability = result["contact_probability"]
        raw = torch.cat([result[k] for k in ("position", "wrench", "slip")], dim=-1)
        if raw.shape[:-1] != probability.shape or raw.shape[-1] != 11:
            raise ValueError("Regression and contact prediction shapes do not match")
        if not torch.isfinite(probability).all() or not torch.isfinite(raw).all():
            raise FloatingPointError("Nonfinite raw/contact predictions cannot be hidden by output gating")
        single = probability.ndim == 1
        if probability.ndim not in (1, 2):
            raise ValueError("Expected contact probabilities [B] or [B,T]")
        probs = probability[:, None] if single else probability
        b, length = probs.shape
        previous = torch.full((b,), -1, device=probs.device, dtype=torch.long)
        if previous_contact is not None:
            supplied = torch.as_tensor(previous_contact, device=probs.device)
            if supplied.shape != (b,) or not ((supplied == -1) | (supplied == 0) | (supplied == 1)).all():
                raise ValueError("previous_contact must be [B] predicted states -1/0/1")
            previous = supplied.long()
        states, transitions, slip_valids = [], [], []
        for t in range(length):
            p = probs[:, t]
            current = torch.where(previous < 0, p >= self.config.contact_initial_threshold, previous.bool())
            current = torch.where(p >= self.config.contact_on_threshold, True, current)
            current = torch.where(p <= self.config.contact_off_threshold, False, current)
            states.append(current.long())
            transitions.append(torch.where(previous >= 0, previous*2+current.long(), -1))
            slip_valids.append(current & (previous == 1))
            previous = current.long()
        state, transition, slip_valid = [torch.stack(x, dim=1) for x in (states, transitions, slip_valids)]
        if single:
            state, transition, slip_valid = state[:, 0], transition[:, 0], slip_valid[:, 0]
        physical = raw*self.target_std + self.target_mean
        position_valid = state == 1
        position = torch.where(position_valid[..., None], physical[..., :3], 0.)
        wrench = torch.where(position_valid[..., None], physical[..., 3:9], 0.)
        slip = torch.where(slip_valid[..., None], physical[..., 9:], 0.)
        return {**result, "contact_state": state, "transition": transition,
                "transition_valid": transition >= 0, "position_valid": position_valid,
                "slip_valid": slip_valid, "last_contact": previous,
                "position_physical": position, "wrench_physical": wrench, "slip_physical": slip,
                "ungated_position_physical": physical[..., :3],
                "ungated_wrench_physical": physical[..., 3:9],
                "ungated_slip_physical": physical[..., 9:],
                "physical_prediction": torch.cat((position, wrench, slip), dim=-1)}

    def reset_state(self):
        super().reset_state()
        self._stream_contact = None

    @torch.no_grad()
    def step(self, audio, image=None, *, frame_id=None, reset=False):
        if not self.normalization_ready.item():
            raise RuntimeError("Call set_normalization(stats) before streaming inference")
        result = super().step(audio, image=image, frame_id=frame_id, reset=reset)
        result = self.gate_outputs(result, self._stream_contact)
        self._stream_contact = result["last_contact"].detach()
        return result
