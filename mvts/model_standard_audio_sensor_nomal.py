# =============================================================================
# model_standard_audio：标准下采样的六通道 Audio ResNet-18，输出 1 个音频 token
# B = batch size；下图尺寸按默认 ModelConfig 标注。
#
#   RGB image [B,3,224,224]                 Log-Mel [B,6,64,12]
#               |                                   |
#               v                                   v
#   +--------------------------+      +--------------------------------+
#   | Visual ResNet-18         |      | Standard Audio ResNet-18       |
#   | 7x7 Conv, stride=(2,2)   |      | 7x7 Conv, stride=(2,2)         |
#   | BN -> ReLU -> MaxPool    |      | BN -> ReLU -> MaxPool          |
#   | 4 stages, 2 blocks each  |      | 4 stages, 2 blocks each        |
#   +--------------------------+      +--------------------------------+
#               |                                   |
#         [B,512,7,7]                         [B,512,2,1]
#               |                                   |
#   Global AvgPool (H,W)                  Global AvgPool (F,T)
#               |                                   |
#           [B,512]                             [B,512]
#               |                                   |
#       Linear(512,256)                     Linear(512,256)
#               |                                   |
#        [B,1,256] (vision)                  [B,1,256] (audio)
#               |                                   |
#        + vision modality                   + audio modality
#               |                                   |
#               +-----------------+-----------------+
#                                 |
#                      Concatenate [B,2,256]
#                                 |
#                 + learned positional embedding
#                 + optional contact embedding (default off)
#                                 |
#              Transformer Encoder x 2 (256 dims, 8 heads, FFN=1024)
#                                 |
#                            [B,2,256]
#                                 |
#                     Self-Attention Pooling
#                                 |
#                             [B,256]
#                                 |
#                 +---------------+---------------+
#                 |                               |
#          Position Head                    Force/Wrench Head
#             [B,3]                             [B,6]
#         (x,y,delta_z)                  (Fx,Fy,Fz,Mx,My,Mz)
#
# 音频首层仅将标准 RGB ResNet 的输入通道数从 3 设为 6，不混合或丢弃麦克风通道。
# 音频 stem 输出 [B,64,16,3]；各 stage 输出依次为 [B,64,16,3]、[B,128,8,2]、
# [B,256,4,1]、[B,512,2,1]。每个 stage 首块 stride 为 (1,1)/(2,2)/(2,2)/(2,2)。
# 频率和时间两个维度一起平均池化，因此仅产生 1 个音频 token。
# ModelConfig.audio_frames=12 表示输入谱图帧数；audio_token_count=1 表示输出 token 数。
# 各输出头均为 LayerNorm -> Linear(256,128) -> GELU -> Dropout -> Linear。
# 输入预处理、标准化标签、反归一化、损失与训练划分沿用 model.py 对应流程。
# =============================================================================
"""Six-channel standard ResNet-18 audio branch with a single global token."""

from dataclasses import dataclass

from .model_sensor_nomal import (ModelConfig as BaseModelConfig, ResNet18Features,
                                 TactileFusionNet as BaseTactileFusionNet)


# 新训练的音频窗口配置，由 train.py 传给 FeatureConfig。
# "before_after" 取事件前、后指定点数，"before_only" 仅取事件前指定点数。
# "before_only" 忽略 AUDIO_SAMPLES_AFTER。
AUDIO_ALIGNMENT_MODE = "before_after"
AUDIO_SAMPLES_BEFORE = 1000
AUDIO_SAMPLES_AFTER = 1000


@dataclass
class ModelConfig(BaseModelConfig):
    audio_alignment_mode: str = AUDIO_ALIGNMENT_MODE
    audio_samples_before: int = AUDIO_SAMPLES_BEFORE
    audio_samples_after: int = AUDIO_SAMPLES_AFTER


class TactileFusionNet(BaseTactileFusionNet):
    @property
    def audio_token_count(self):
        return 1

    def _make_audio_backbone(self):
        return ResNet18Features(in_channels=6)

    def _encode_audio_tokens(self, audio):
        # [B,512,2,1] -> [B,512] -> [B,256] -> [B,1,256]
        features = self.audio_backbone(audio).mean(dim=(-2, -1))
        return self.audio_projection(features).unsqueeze(1)
