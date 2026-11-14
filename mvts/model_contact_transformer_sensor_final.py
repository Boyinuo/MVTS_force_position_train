# =============================================================================
# Visual-acoustic fusion with binary contact added before the Transformer
# Default dimensions (B = batch size)
#
# RGB [B,3,224,224]                  ADC [B,2000,6]
#        |                                |
# Visual ResNet-18                  Log-Mel [B,6,64,12]
# -> Global AvgPool                 Modified Audio ResNet-18
# -> Linear(512,256)                -> Frequency AvgPool -> Linear(512,256)
#        |                                |
# 1 visual token [B,1,256]           12 audio tokens [B,12,256]
#        +----------------+---------------+
#                         |
#           Modality and positional embeddings
#                         |
# Tokens [B,13,256] + Embedding(2,256)(binary contact) [B,1,256]
#                         |
#           Transformer Encoder x 2 [B,13,256]
#                         |
#           Self-Attention Pooling [B,256]
#                    /            \
#          Position MLP            Wrench MLP
#          256 -> 128 -> 3          256 -> 128 -> 6
#
# Audio alignment uses ModelConfig from model_contact_concat_sensor_final.py,
# including its current before_only / 2000 pre-event sample configuration.
# Training supplies starts_in_contact. Sequential inference can supply the
# previous contact prediction through predict.py --previous-contact 0/1.
# =============================================================================
"""Add a binary contact embedding to visual and acoustic Transformer inputs."""
import torch

from .model_contact_concat_sensor_final import ModelConfig
from .model_sensor_final import TactileFusionNet as BaseTactileFusionNet


class TactileFusionNet(BaseTactileFusionNet):
    def __init__(self, config=None):
        config = config or ModelConfig()
        if not config.use_contact:
            raise ValueError("Contact Transformer inputs require use_contact=True")
        super().__init__(config)

    def encode_tokens(self, image, audio, starts_in_contact=None):
        if starts_in_contact is None:
            raise ValueError("starts_in_contact is required for contact Transformer inputs")
        contact = torch.as_tensor(starts_in_contact, device=image.device)
        batch_size = image.shape[0]
        if contact.shape not in ((batch_size,), (batch_size, 1)):
            raise ValueError("starts_in_contact must have shape [B] or [B,1]")
        if not ((contact == 0) | (contact == 1)).all():
            raise ValueError("starts_in_contact must contain only binary values 0 or 1")
        return super().encode_tokens(image, audio, contact.reshape(batch_size).long())
