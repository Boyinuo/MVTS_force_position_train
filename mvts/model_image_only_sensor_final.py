# =============================================================================
# Image-only ablation of model_sensor_final (B = batch size)
#
# RGB image [B,3,224,224]
#             |
# Visual ResNet-18 -> [B,512,7,7]
#             |
# Global average pool -> [B,512] -> Linear(512,256)
#             |
# Single vision token [B,1,256] + modality/positional embeddings
#             |
# Transformer Encoder x 2 (d_model=256, nhead=8, FFN=1024)
#             |
# Attention pooling -> [B,256]
#             |
#        +----+----+
#        |         |
# Position head   Wrench head
#   [B,3]           [B,6]
#
# The visual backbone, Transformer and regression heads follow model_sensor_final.
# No audio backbone, audio projection or audio tokens are instantiated.
# With one token, pooling has weight 1 and self-attention has no cross-token
# interactions; the Transformer and heads remain for the modality ablation.
# audio/starts_in_contact are accepted only for the existing engine interface and
# never read. use_contact=True is rejected to keep this a strictly image-only model.
# The existing data pipeline still reads/audits audio and computes Log-Mel features.
# Example:
# uv run python train.py --dataset data_sensor_final \
#     --model model_image_only_sensor_final --output runs/image_only_sensor_final
# =============================================================================
"""Image-only tactile regression, compatible with the shared training interface."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model_sensor_final import ResNet18Features


# 共享数据流程的音频窗口配置，由 train.py 传给 FeatureConfig。
# "before_after" 取事件前、后指定点数，"before_only" 仅取事件前指定点数。
# "before_only" 忽略 AUDIO_SAMPLES_AFTER。
AUDIO_ALIGNMENT_MODE = "before_only"
AUDIO_SAMPLES_BEFORE = 2000
AUDIO_SAMPLES_AFTER = 1000


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    layers: int = 2
    dropout: float = 0.1
    # Retained for shared configuration compatibility; unused by this network.
    audio_frames: int = 12
    use_contact: bool = False
    audio_alignment_mode: str = AUDIO_ALIGNMENT_MODE
    audio_samples_before: int = AUDIO_SAMPLES_BEFORE
    audio_samples_after: int = AUDIO_SAMPLES_AFTER

    def to_dict(self):
        return asdict(self)


class TactileFusionNet(nn.Module):
    """Predict position and wrench from images; optional audio is ignored."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.use_contact:
            raise ValueError("Image-only ablation requires use_contact=False; omit --use-contact")

        self.vision_backbone = ResNet18Features()
        self.vision_projection = nn.Linear(512, c.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, c.d_model))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, c.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.nhead, dim_feedforward=4 * c.d_model,
            dropout=c.dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=c.layers,
                                                 enable_nested_tensor=False)
        for module in self.transformer.layers:
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.attention_score = nn.Sequential(nn.Linear(c.d_model, c.d_model // 2),
                                              nn.Tanh(), nn.Linear(c.d_model // 2, 1))
        self.position_head = self._head(3)
        self.force_head = self._head(6)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    @property
    def audio_token_count(self):
        return 0

    def _head(self, outputs):
        d = self.config.d_model
        return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
                             nn.Dropout(self.config.dropout), nn.Linear(d // 2, outputs))

    def encode_tokens(self, image, audio=None, starts_in_contact=None):
        if image.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected image [B,3,224,224], got {tuple(image.shape)}")
        vision = self.vision_projection(self.vision_backbone(image).mean(dim=(-2, -1)))
        return vision.unsqueeze(1) + self.modality_embedding + self.position_embedding

    def forward(self, image, audio=None, starts_in_contact=None):
        h = self.transformer(self.encode_tokens(image))
        weights = self.attention_score(h).softmax(dim=1)
        z = (weights * h).sum(dim=1)
        return {"position": self.position_head(z), "wrench": self.force_head(z)}
