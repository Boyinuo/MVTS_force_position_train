# =============================================================================
# Visual-tactile and acoustic fusion with binary contact concatenation
# Default dimensions (B = batch size)
#
# RGB [B,3,224,224]                 ADC [B,2000,6]
#        |                               |
# Visual ResNet-18                 Log-Mel [B,6,64,12]
#        |                               |
# [B,512,7,7]                      Modified Audio ResNet-18
#        |                               |
# Global AvgPool                   [B,512,8,12]
# -> Linear(512,256)               Frequency AvgPool -> Linear(512,256)
#        |                               |
# 1 visual token [B,1,256]          12 audio tokens [B,12,256]
#        +---------------+---------------+
#                        |
#          Modality and positional embeddings
#                        |
#          Transformer Encoder x 2 [B,13,256]
#                        |
#          Self-Attention Pooling [B,256]
#                        |                  Binary contact c_(t-1) [B,1]
#                        +-----------------------------+
#                                        |
#                              Concatenate [B,257]
#                                /              \
#                  Position MLP                   Wrench MLP
#                  257 -> 128 -> 3                 257 -> 128 -> 6
#                    (x,y,delta_z)                 (Fx,Fy,Fz,Mx,My,Mz)
#
# Each MLP uses LayerNorm(257), Linear, GELU, Dropout and Linear.
# Training uses the dataset's starts_in_contact as the supplied contact prior.
# For sequential inference, supply the previous binary contact prediction through
# the starts_in_contact argument, or predict.py --previous-contact 0/1.
# Audio alignment defaults are inherited from model_sensor_final.py.
# =============================================================================
"""Concatenate a binary contact prior after visual-acoustic attention pooling."""
from dataclasses import dataclass

import torch
from torch import nn

from .model_sensor_final import (
    AUDIO_ALIGNMENT_MODE,
    AUDIO_SAMPLES_AFTER,
    AUDIO_SAMPLES_BEFORE,
    ModelConfig as BaseModelConfig,
    TactileFusionNet as BaseTactileFusionNet,
)


@dataclass
class ModelConfig(BaseModelConfig):
    use_contact: bool = True
    audio_alignment_mode: str = AUDIO_ALIGNMENT_MODE
    audio_samples_before: int = AUDIO_SAMPLES_BEFORE
    audio_samples_after: int = AUDIO_SAMPLES_AFTER


class TactileFusionNet(BaseTactileFusionNet):
    def __init__(self, config=None):
        config = config or ModelConfig()
        if not config.use_contact:
            raise ValueError("Contact concatenation requires use_contact=True")
        super().__init__(config)

    def _make_contact_embedding(self):
        # Contact is concatenated with the pooled feature in forward().
        return None

    def _head(self, outputs):
        d = self.config.d_model
        return nn.Sequential(nn.LayerNorm(d + 1), nn.Linear(d + 1, d // 2), nn.GELU(),
                             nn.Dropout(self.config.dropout), nn.Linear(d // 2, outputs))

    def forward(self, image, audio, starts_in_contact=None):
        if starts_in_contact is None:
            raise ValueError("starts_in_contact is required for contact concatenation")
        contact = torch.as_tensor(starts_in_contact, device=image.device)
        batch_size = image.shape[0]
        if contact.shape not in ((batch_size,), (batch_size, 1)):
            raise ValueError("starts_in_contact must have shape [B] or [B,1]")
        if not ((contact == 0) | (contact == 1)).all():
            raise ValueError("starts_in_contact must contain only binary values 0 or 1")
        h = self.transformer(self.encode_tokens(image, audio))
        weights = self.attention_score(h).softmax(dim=1)
        pooled = (weights * h).sum(dim=1)
        fused = torch.cat((pooled, contact.reshape(batch_size, 1).to(dtype=pooled.dtype)), dim=-1)
        return {"position": self.position_head(fused), "wrench": self.force_head(fused)}
