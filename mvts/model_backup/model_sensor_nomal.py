# =============================================================================
# 当前训练网络架构（默认 ModelConfig；B 表示 batch size）
# =============================================================================
#
#       视觉分支 Vision                           声学分支 Audio
#
#   RGB image [B, 3, 224, 224]              Log-Mel [B, 6, 64, 12]
#                |                                    |
#                v                                    v
#   +--------------------------+         +------------------------------+
#   | Visual ResNet-18         |         | Modified Audio ResNet-18     |
#   | 7x7 Conv, stride=(2,2)   |         | 3x3 Conv, stride=(1,1)       |
#   | -> BN -> ReLU -> MaxPool |         | -> BN -> ReLU (no MaxPool)   |
#   | -> 4 residual stages     |         | -> 4 residual stages         |
#   +--------------------------+         +------------------------------+
#                |                                    |
#         [B, 512, 7, 7]                       [B, 512, 8, 12]
#                |                                    |
#                v                                    v
#   Global AvgPool (height,width)         Freq AvgPool (frequency only)
#                |                                    |
#            [B, 512]                           [B, 512, 12]
#                |                                    |
#       Linear(512 -> 256)                     transpose(1,2)
#                |                                    |
#       unsqueeze(token axis)                Linear(512 -> 256)
#                |                                    |
#          [B, 1, 256]                           [B, 12, 256]
#       (1 vision token)                      (12 temporal tokens)
#                |                                    |
#       + vision modality embedding           + audio modality embedding
#                |                                    |
#                +-----------------+------------------+
#                                  |
#                                  v
#                    Concatenate along token axis
#                           [B, 13, 256]
#                                  |
#                    + learned positional embedding
#                                  |
#                    + optional contact embedding (*)
#                                  |
#                                  v
#                +-----------------------------------+
#                | Transformer Encoder x 2 layers    |
#                | d_model=256, nhead=8, FFN=1024    |
#                | activation=GELU, dropout=0.1      |
#                +-----------------------------------+
#                                  |
#                           H [B, 13, 256]
#                                  |
#                                  v
#                +-----------------------------------+
#                | Self-Attention Pooling            |
#                | score: Linear(256,128) -> Tanh    |
#                | -> Linear(128,1)                  |
#                | weights = softmax(score, tokens)  |
#                | z = sum(weights * H, tokens)      |
#                +-----------------------------------+
#                                  |
#                             z [B, 256]
#                                  |
#                 +----------------+----------------+
#                 |                                 |
#                 v                                 v
#   +----------------------------+    +-----------------------------+
#   | Position Head              |    | Force/Wrench Head           |
#   | LayerNorm(256)             |    | LayerNorm(256)              |
#   | -> Linear(256,128) -> GELU |    | -> Linear(256,128) -> GELU  |
#   | -> Dropout(0.1)            |    | -> Dropout(0.1)             |
#   | -> Linear(128,3)           |    | -> Linear(128,6)            |
#   +----------------------------+    +-----------------------------+
#                 |                                 |
#                 v                                 v
#       position [B, 3]                      wrench [B, 6]
#       (x, y, delta_z)                  (Fx, Fy, Fz, Mx, My, Mz)
#
# 补充约定：
#   1. 两个 ResNet 均为每个 stage 含 2 个 BasicBlock，通道数 64/128/256/512。
#      视觉各 stage 首块 stride: (1,1), (2,2), (2,2), (2,2)。
#      声学各 stage 首块 stride: (1,1), (2,1), (2,1), (2,1)，始终保留 12 帧。
#   2. (*) use_contact 默认 False；启用时 starts_in_contact [B]（0/1）经
#      Embedding(2,256) 得到 [B,256]，广播相加到全部 13 个 token，不新增 token。
#   3. Log-Mel 在 features.py 中提取：2000x6 ADC -> STFT(n_fft=512, hop=128,
#      center=False) -> 64 Mel -> log10(power)，默认实际采样率为 50 kHz。
#   4. 网络输出为标准化标签空间的预测；engine.py / predict.py 负责反归一化，
#      得到位置 mm、力 N、力矩 N*m。两路骨干默认从随机权重开始训练。
#   5. encode_tokens() 组织分支与 token 编码；forward() 组织 Transformer、
#      注意力池化和双输出头。图中的维度、层数等按默认配置标注。
# =============================================================================

"""ResNet-18 (vision + time-preserving audio), Transformer, attention pooling.

Implemented with torch only; no torchvision binary dependency or weight download.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn


# 新训练的音频窗口配置，由 train.py 传给 FeatureConfig。
# "before_after" 取事件前、后指定点数，允许前后点数不同。
# "before_only" 仅取事件前指定点数，忽略 AUDIO_SAMPLES_AFTER。
AUDIO_ALIGNMENT_MODE = "before_after"
AUDIO_SAMPLES_BEFORE = 1000
AUDIO_SAMPLES_AFTER = 1000


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    layers: int = 2
    dropout: float = 0.1
    audio_frames: int = 12
    use_contact: bool = False
    audio_alignment_mode: str = AUDIO_ALIGNMENT_MODE
    audio_samples_before: int = AUDIO_SAMPLES_BEFORE
    audio_samples_after: int = AUDIO_SAMPLES_AFTER

    def to_dict(self):
        return asdict(self)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                          nn.BatchNorm2d(out_channels))
            if in_channels != out_channels or stride != (1, 1) else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + identity)


class ResNet18Features(nn.Module):
    def __init__(self, audio=False, *, in_channels=3):
        super().__init__()
        if audio:
            self.stem = nn.Sequential(nn.Conv2d(6, 64, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.MaxPool2d(3, 2, 1))
        channels = 64
        stages = []
        for stage, width in enumerate((64, 128, 256, 512)):
            stride = (1, 1) if stage == 0 else ((2, 1) if audio else (2, 2))
            stages.append(nn.Sequential(BasicBlock(channels, width, stride),
                                        BasicBlock(width, width)))
            channels = width
        self.stages = nn.Sequential(*stages)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.stages(self.stem(x))


class TactileFusionNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.vision_backbone = ResNet18Features()
        self.audio_backbone = self._make_audio_backbone()
        self.vision_projection = nn.Linear(512, c.d_model)
        self.audio_projection = nn.Linear(512, c.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + self.audio_token_count, c.d_model))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 2, c.d_model))
        self.contact_embedding = nn.Embedding(2, c.d_model) if c.use_contact else None
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.nhead, dim_feedforward=4 * c.d_model,
            dropout=c.dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=c.layers,
                                                 enable_nested_tensor=False)
        # Encoder layers are cloned by torch; give each its own initial matrices.
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
        return self.config.audio_frames

    def _make_audio_backbone(self):
        return ResNet18Features(audio=True)

    def _encode_audio_tokens(self, audio):
        # [B,512,8,12] -> pool frequency ONLY -> [B,12,512] -> [B,12,256]
        return self.audio_projection(self.audio_backbone(audio).mean(dim=2).transpose(1, 2))

    def _head(self, outputs):
        d = self.config.d_model
        return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(),
                             nn.Dropout(self.config.dropout), nn.Linear(d // 2, outputs))

    def encode_tokens(self, image, audio, starts_in_contact=None):
        if image.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected image [B,3,224,224], got {tuple(image.shape)}")
        if audio.shape[1:] != (6, 64, self.config.audio_frames):
            raise ValueError(f"Expected audio [B,6,64,{self.config.audio_frames}], got {tuple(audio.shape)}")
        vision = self.vision_projection(self.vision_backbone(image).mean(dim=(-2, -1)))
        audio = self._encode_audio_tokens(audio)
        tokens = torch.cat((vision.unsqueeze(1) + self.modality_embedding[:, :1],
                            audio + self.modality_embedding[:, 1:]), dim=1)
        tokens = tokens + self.position_embedding
        if self.contact_embedding is not None:
            if starts_in_contact is None:
                raise ValueError("starts_in_contact is required for this model")
            tokens = tokens + self.contact_embedding(starts_in_contact.long()).unsqueeze(1)
        return tokens

    def forward(self, image, audio, starts_in_contact=None):
        h = self.transformer(self.encode_tokens(image, audio, starts_in_contact))
        weights = self.attention_score(h).softmax(dim=1)
        z = (weights * h).sum(dim=1)
        return {"position": self.position_head(z), "wrench": self.force_head(z)}
