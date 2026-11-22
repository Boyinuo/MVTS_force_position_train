# =============================================================================
# 当前三预测头网络架构（默认 ModelConfig；B 表示 batch size）
# =============================================================================
#
#       慢速视觉分支 Vision                        快速声学分支 Audio
#       新图像到达时更新（约 15 Hz）                每个子事件更新（滑移约 150 Hz）
#
#   RGB image [B, 3, 224, 224]              Log-Mel [B, 6, 64, 28]
#                |                                    |
#                v                                    v
#   +--------------------------+         +----------------------------------+
#   | Visual ResNet-18         |         | Select enabled microphones      |
#   | 7x7 Conv, stride=(2,2)   |         | M=6 by default; disable #5: M=5  |
#   | -> BN -> ReLU -> MaxPool |         | [B,M,64,28] -> [B*M,1,64,28]     |
#   | -> 4 residual stages    |         +----------------------------------+
#   +--------------------------+                          |
#                |                                        v
#         [B, 512, 7, 7]                  +----------------------------------+
#                |                        | Shared Modified Audio ResNet-18  |
#                v                        | Same weights for every mic       |
#   Adaptive AvgPool2d(4,4)               | 3x3 Conv, stride=(1,1)           |
#                |                        | -> BN -> ReLU (no MaxPool)       |
#         [B, 512, 4, 4]                  | -> 4 stages, frequency strides   |
#                |                        +----------------------------------+
#       flatten(2).transpose(1,2)                         |
#                |                                [B*M,512,8,28]
#          [B, 16, 512]                                   |
#                |                                Freq AvgPool ONLY
#       Linear(512 -> 128)                                 |
#                |                                 [B*M,28,512]
#                |                                        |
#                |                                Linear(512 -> 128)
#                |                                        |
#                |                                  [B,M,28,128]
#                |                                        |
#                |                                Mean over M microphones
#                |                                        |
#       V_j [B, 16, 128]                           S_k [B, 28, 128]
#       (16 spatial tokens)                       (28 temporal tokens)
#                |                                        |
#       cache latest visual tokens                        |
#                |                                        |
#                +-------------------+--------------------+
#                                    |
#                                    v
#                   Add separate audio / visual modality embeddings
#                   Concat(S_k, V_j) + learned positional embedding
#                             memory [B, 44, 128]
#                                    |
#                   +----------------+--------------------+
#                   |                                     |
#                   v                                     v
#       Q = Linear(memory[:,27:28])               K = V = memory
#       (last audio token, after embeddings)      (28 audio + 16 visual)
#             [B, 1, 128]                            [B, 44, 128]
#                   |                                     |
#                   +----------------+--------------------+
#                                    |
#                                    v
#                  +--------------------------------------+
#                  | Joint Multi-Head Cross-Attention     |
#                  | embed_dim=128, heads=8, dropout=0.1 |
#                  | A = MHA(Q, K, V)  [B,1,128]         |
#                  | f_k = LayerNorm(A + Q)               |
#                  +--------------------------------------+
#                                    |
#                             f_k [B, 1, 128]
#                                    |
#                                    v
#                  +--------------------------------------+
#    h_(k-1) ----> | Stateful GRU(128,256), 1 layer      |
#    [1,B,256]     | Update temporal memory per event    |
#                  +--------------------------------------+
#                                    |
#                     output[:,0] = h_k [B, 256]
#                     hidden state     [1, B, 256] ----> next event
#                                    |
#                  +-----------------+------------------+
#                  |                 |                  |
#                  v                 v                  v
#       +-------------------+ +-------------------+ +-------------------+
#       | Position Head     | | Wrench Head       | | Slip Head         |
#       | LayerNorm(256)    | | LayerNorm(256)    | | LayerNorm(256)    |
#       | Linear(256,128)   | | Linear(256,128)   | | Linear(256,128)   |
#       | GELU, Dropout(.1) | | GELU, Dropout(.1) | | GELU, Dropout(.1) |
#       | Linear(128,3)     | | Linear(128,6)     | | Linear(128,2)     |
#       +-------------------+ +-------------------+ +-------------------+
#                  |                 |                  |
#                  v                 v                  v
#         position [B,3]       wrench [B,6]          slip [B,2]
#         (x,y,delta_z)        (Fx,Fy,Fz,           (delta_sx,delta_sy)
#                               Mx,My,Mz)
#
# 补充约定：
#   1. 两路骨干各含 4 个 stage，每个 stage 含 2 个 BasicBlock，通道数为
#      64/128/256/512；BasicBlock 为两层 3x3 卷积及残差连接。
#      视觉各 stage 首块 stride: (1,1), (2,2), (2,2), (2,2)。
#      声学各 stage 首块 stride: (1,1), (2,1), (2,1), (2,1)，保留 28 个时间步。
#      声学不是把 6 路当卷积输入通道：每路独立编码、共享权重，再按麦克风平均。
#   2. features.py / 预处理缓存负责固定声学前端：事件之前 4000x6 ADC（50 kHz，
#      80 ms）-> 除以固定 ADC 尺度、逐通道去直流 -> 取最右侧 3968 点 ->
#      Hann STFT(n_fft=512, hop=128, center=False) -> 64 Slaney Mel ->
#      log10(power)，得到 [6,64,28]。仅用训练集拟合的逐通道统计量标准化。
#      图像保留完整视场，缩放到 224x224 并做 ImageNet 固定均值/标准差归一化。
#   3. disabled_mic_channels 默认为 ()，使用全部 6 路；编号从 1 开始。
#      --disable-mic-channels 5 对应去掉数组索引 4，只编码其余 5 路并取均值。
#      缓存和模型外部输入仍为 6 路，不需要重建缓存；该设置保存在 checkpoint，
#      恢复训练、评估和推理沿用同一设置。模型内部融合 token 总数仍为 44。
#   4. 位置、六维力和滑移均为监督标签，不输入网络。use_contact 必须为 False；
#      starts_in_contact 只是兼容接口参数，不参与编码或融合。
#      按压每张图像只取最后一条当前力及其之前 4000 点音频，滑移标签为 (0,0)。
#      滑移在相邻图像的 [t_j,t_(j+1)) 内形成左端+9个内部等分子事件，复用 I_j；
#      力标签取子事件之前最近实测值，位置插值到子事件时刻，滑移取相邻位置 XY 差。
#   5. forward() 处理单事件，返回三个预测头、hidden 和 visual_tokens。
#      forward_sequence() 输入 image [B,3,224,224]、audio [B,T,6,64,28]；
#      图像只编码一次，各子事件的融合 token 组成 [B,T,128] 后送入 GRU，
#      三头输出依次为 [B,T,3]、[B,T,6]、[B,T,2]；按压 T=1，滑移 T=10。
#      dual_rate_engine.py 在同一连续轨迹的窗口间携带 detached GRU 状态，
#      不同轨迹/独立按压、窗口中断或新训练/评估阶段重置，梯度只跨当前窗口。
#   6. 流式推理先 eval()；step() 自动复用视觉 token 和 GRU 状态，只有新图像
#      到达时更新视觉特征，新轨迹调用 reset_state()。模型权重中不保存流式状态。
#   7. 默认 d_model=128、hidden_size=256、nhead=8、dropout=0.1；骨干随机初始化。
#      本实现为一次联合交叉注意力+GRU，没有额外 Transformer Encoder 堆叠。
#      三头输出位于标准化标签空间，dual_rate_engine.py / predict.py 反归一化：
#      位置/滑移单位 mm，力 N，力矩 N*m；delta_z 沿用采集端局部接触面相对压入量。
#      名义 150 Hz 子事件按真实相机间隔十等分；实际频率会随时间戳抖动变化。
# =============================================================================

"""Dual-rate causal fusion: 16 visual + 28 audio tokens -> attention -> GRU.

Input: RGB [B,3,H,W], log-mel [B,6,64,28] (4000 raw samples/ch).
Each microphone is encoded independently by the SAME ResNet-18, then averaged.
Q = last audio token; K,V = concatenate(audio, latest available visual tokens).
The 128-D fused token updates GRU(128,256); three heads predict 3/6/2 values.
Targets (including wrench) are never fed back into the estimator.
Use forward_sequence for ordered training and step for cached streaming inference.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbones import ResNet18Features


@dataclass
class ModelConfig:
    d_model: int = 128
    hidden_size: int = 256
    nhead: int = 8
    dropout: float = 0.1
    audio_frames: int = 28
    use_contact: bool = False
    audio_alignment_mode: str = "before_only"
    audio_samples_before: int = 4000
    audio_samples_after: int = 0
    disabled_mic_channels: tuple[int, ...] = ()  # Human numbering: 1 through 6.

    def __post_init__(self):
        channels = tuple(self.disabled_mic_channels)
        if (any(type(k) is not int or not 1 <= k <= 6 for k in channels)
                or len(set(channels)) != len(channels) or len(channels) == 6):
            raise ValueError("disabled_mic_channels must be distinct numbers 1..6, leaving at least one enabled")
        self.disabled_mic_channels = tuple(sorted(channels))

    def to_dict(self):
        return asdict(self)


class TactileFusionNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.active_mic_indices = tuple(i for i in range(6) if i + 1 not in c.disabled_mic_channels)
        if c.use_contact:
            raise ValueError("The diagram does not use a ground-truth contact input")
        if c.audio_frames != 28 or c.audio_samples_before != 4000 or c.audio_samples_after != 0 or c.audio_alignment_mode != "before_only":
            raise ValueError("Triple-head model requires 4000 causal samples and 28 STFT frames")
        self.vision_backbone = ResNet18Features()
        # A shared single-channel encoder, applied independently to six mics.
        self.audio_backbone = ResNet18Features(audio=True, in_channels=1)
        self.vision_projection = nn.Linear(512, c.d_model)
        self.audio_projection = nn.Linear(512, c.d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, 44, c.d_model) * .02)
        self.modality_embedding = nn.Parameter(torch.randn(1, 2, c.d_model) * .02)
        self.query = nn.Linear(c.d_model, c.d_model)
        self.attention = nn.MultiheadAttention(c.d_model, c.nhead, c.dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(c.d_model)
        self.gru = nn.GRU(c.d_model, c.hidden_size, batch_first=True)
        self.position_head = self._head(3)
        self.force_head = self._head(6)
        self.slip_head = self._head(2)
        self.reset_state()

    def _head(self, outputs):
        h = self.config.hidden_size
        return nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h // 2), nn.GELU(),
                             nn.Dropout(self.config.dropout), nn.Linear(h // 2, outputs))

    def encode_visual(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB [B,3,H,W]")
        features = F.adaptive_avg_pool2d(self.vision_backbone(image), (4, 4))
        return self.vision_projection(features.flatten(2).transpose(1, 2))

    def encode_audio(self, audio):
        if audio.ndim != 4 or audio.shape[1:] != (6, 64, 28):
            raise ValueError("Expected log-mel [B,6,64,28]")
        b = audio.shape[0]
        if self.config.disabled_mic_channels:
            audio = audio[:, self.active_mic_indices]
        microphones = len(self.active_mic_indices)
        features = self.audio_backbone(audio.reshape(b * microphones, 1, 64, 28))
        tokens = self.audio_projection(features.mean(2).transpose(1, 2))
        return tokens.reshape(b, microphones, 28, self.config.d_model).mean(1)

    def fuse(self, visual, audio):
        acoustic = self.encode_audio(audio)
        memory = torch.cat((acoustic + self.modality_embedding[:, :1],
                            visual + self.modality_embedding[:, 1:]), dim=1)
        memory = memory + self.position_embedding
        q = self.query(memory[:, 27:28])
        fused, _ = self.attention(q, memory, memory, need_weights=False)
        return self.fusion_norm(fused + q)

    def _predict(self, states):
        return {"position": self.position_head(states), "wrench": self.force_head(states),
                "slip": self.slip_head(states)}

    def forward(self, image, audio, starts_in_contact=None, *, hidden=None, visual_tokens=None):
        visual = self.encode_visual(image) if visual_tokens is None else visual_tokens
        states, hidden = self.gru(self.fuse(visual, audio), hidden)
        return {**self._predict(states[:, 0]), "hidden": hidden, "visual_tokens": visual}

    def forward_sequence(self, image, audio, hidden=None):
        """One image anchor per ordered interval; gradients span all its events."""
        if audio.ndim != 5:
            raise ValueError("Expected sequence audio [B,T,6,64,28]")
        visual = self.encode_visual(image)
        fused = torch.cat([self.fuse(visual, audio[:, t]) for t in range(audio.shape[1])], dim=1)
        states, hidden = self.gru(fused, hidden)
        return {**self._predict(states), "hidden": hidden}

    def reset_state(self):
        self._stream_hidden = None
        self._stream_visual = None
        self._stream_frame_id = None

    @torch.no_grad()
    def step(self, audio, image=None, *, frame_id=None, reset=False):
        """Inference only. Reset at a new trajectory/contact experiment.

        Supply an image on first use/new frame; omit it for intervening events.
        Stream state is deliberately excluded from state_dict/checkpoints.
        """
        if self.training:
            raise RuntimeError("Call eval() before streaming inference")
        if reset:
            self.reset_state()
        if image is not None and (frame_id is None or frame_id != self._stream_frame_id):
            self._stream_visual = self.encode_visual(image)
            self._stream_frame_id = frame_id
        elif frame_id is not None and frame_id != self._stream_frame_id:
            raise ValueError("A new frame_id requires its image")
        if self._stream_visual is None:
            raise ValueError("First streaming event requires an image")
        result = self.forward(None, audio, hidden=self._stream_hidden, visual_tokens=self._stream_visual)
        self._stream_hidden = result["hidden"].detach()
        return result
