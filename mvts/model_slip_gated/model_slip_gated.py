# =============================================================================
# 滑移先验门控网络架构（方案 II，默认 ModelConfig；B 为 batch size）
# =============================================================================
#
#       慢速视觉分支 Vision                       快速声学分支 Audio
#       新图像到达时更新并缓存                     每个因果子事件更新
#
#   RGB image [B,3,224,224]                  Log-Mel [B,6,64,28]
#                |                                    |
#                v                                    v
#   +---------------------------+         +----------------------------------+
#   | Visual ResNet-18          |         | Select M enabled microphones    |
#   | 7x7 Conv, stride=(2,2)    |         | M=6 by default                   |
#   | -> BN -> ReLU -> MaxPool  |         | [B,M,64,28] -> [B*M,1,64,28]     |
#   | -> 4 residual stages     |         +----------------------------------+
#   +---------------------------+                         |
#                |                                        v
#          [B,512,7,7]                    +----------------------------------+
#                |                        | Shared Modified Audio ResNet-18  |
#     Adaptive AvgPool2d(4,4)              | 3x3 Conv, stride=(1,1)           |
#                |                        | -> BN -> ReLU (no MaxPool)       |
#          [B,512,4,4]                    | -> 4 frequency-strided stages    |
#                |                        +----------------------------------+
#     flatten + transpose                                 |
#                |                                [B*M,512,8,28]
#          [B,16,512]                                      |
#                |                                Freq AvgPool ONLY
#     Linear(512 -> 128)                                   |
#                |                                 [B*M,28,512]
#                |                                        |
#                |                                Linear(512 -> 128)
#                |                                        |
#                |                                  [B,M,28,128]
#                |                                        |
#                |                                Mean over M microphones
#                |                                        |
#       V_j [B,16,128]                             S_k [B,28,128]
#       cache latest tokens                               |
#                |                           +------------+----------------+
#                |                           |                             |
#                |                           v                             v
#                |                    Temporal mean                 Audio memory
#                |                    a_k [B,128]                   [B,28,128]
#                |                           |                      + modality/pos
#                |                           v                             |
#                |     h_(k-1) ------> concat(a_k, h_(k-1))                 |
#                |     [B,256]            [B,384]                          |
#                |                           |                             |
#                |                  +-------------------------+            |
#                |                  | Coarse Slip Prior Head  |            |
#                |                  | LN(384)                 |            |
#                |                  | -> Linear(384,128)      |            |
#                |                  | GELU -> Dropout(.1)     |            |
#                |                  | -> Linear(128,2)        |            |
#                |                  +-------------------------+            |
#                |                           |                             |
#                |                  slip_prior [B,2]                       |
#                |                           |                             |
#                |                  de-normalize to mm                     |
#                |                           |                             |
#                |                  speed = ||prior_mm|| / delta_t         |
#                |                           |                             |
#                |                  detach speed (default)                 |
#                |                           |                             |
#                |                           v                             |
#                |     concat(LN(a_k), log1p(speed/10), image_age/(1/15))   |
#                |                        [B,130]                          |
#                |                           |                             |
#                |              Linear(130,64) -> GELU -> Linear(64,1)     |
#                |                           |                             |
#                |                        sigmoid                          |
#                |                           |                             |
#                |                      alpha [B,1]                        |
#                |                      visual weight                      |
#                |                                                         |
#                v                                                         |
#       Visual memory [B,16,128]                                           |
#       + modality/pos                                                     |
#                |                                                         |
#                |       h_(k-1) -> Linear(256,128) -> Q [B,1,128]         |
#                |       no history: learned initial_query [B,1,128]       |
#                |                           |                             |
#                |             +-------------+-------------+               |
#                |             |                           |               |
#                v             v                           v               v
#       Visual Cross-Attention(128,8 heads)    Audio Cross-Attention(128,8 heads)
#       K=V=visual memory; Q=history          K=V=audio memory; Q=history
#                       |                                  |
#                z_visual [B,1,128]                  z_audio [B,1,128]
#                       |                                  |
#                       +----------------+-----------------+
#                                        |
#               f_k = LayerNorm(alpha*z_visual + (1-alpha)*z_audio)
#                                [B,1,128]
#                                        |
#     h_(k-1) [1,B,256] ------> Stateful GRU(128,256), 1 layer
#                                        |
#                         output[:,0] = h_k [B,256]
#                         hidden [1,B,256] ----------------> next event
#                                        |
#                    +-------------------+-------------------+
#                    |                   |                   |
#                    v                   v                   v
#             Position Head         Wrench Head          Slip Head
#             LN -> 256/128/3       LN -> 256/128/6      LN -> 256/128/2
#                    |                   |                   |
#             position [B,3]        wrench [B,6]         slip [B,2]
#             (x,y,delta_z)         (Fx,Fy,Fz,           (delta_sx,delta_sy)
#                                    Mx,My,Mz)
#
# 补充约定：
#   1. 编码器继承 model_triple_head：4 个 stage，每个含 2 个 BasicBlock，通道为
#      64/128/256/512。视觉 stage 首块 stride 为 (1,1)/(2,2)/(2,2)/(2,2)，
#      声学为 (1,1)/(2,1)/(2,1)/(2,1)，逐麦共享权重、只池化频率再按麦克风平均。
#      输入为事件之前 4000 点/通道的 [6,64,28] Log-Mel；禁用麦克风仍由内部筛选，
#      外部保持 6 路输入。本模型要求视觉和声学同时存在，不支持 image_only。
#   2. a_k 是未加模态/位置嵌入的声学 token 均值；滑移先验只读取当前因果音频和
#      上一事件 hidden，不读取当前 GRU 输出或最终滑移预测。无历史时 hidden 为零，
#      Q 使用可学习 initial_query；只有最后的 GRU 更新 hidden，不循环调用预测头。
#   3. slip_prior 是标准化的二维滑移增量，使用 target_mean/std 的最后两维恢复 mm。
#      速度单位 mm/s；delta_t、image_age 单位 s，分别要求正数和非负数，均非标签。
#      默认 detach_slip_for_gate=True 仅截断“速度 -> 门控”的梯度；先验仍可由其
#      独立训练损失学习。先验、最终滑移及门控诊断均随结果返回。
#   4. alpha 是每个样本/事件一个标量；视觉权重 alpha，声学权重 1-alpha。
#      gate 最后一层 bias 初始化为 log(.8/.2)，实际初始输出还取决于输入与权重。
#      alpha 是学习权重，不保证速度越大视觉权重越小，也不是校准置信度。
#      两路注意力共享 Q，但各自拥有独立参数；不拼成 44-token 联合注意力，
#      融合后无额外 Q 或当前音频残差旁路，也没有额外 Transformer Encoder。
#   5. 最终三头均为 LN(256) -> Linear(256,128) -> GELU -> Dropout(.1) ->
#      Linear(128,out)，输出仍为标准化的 3/6/2 维回归；物理单位由外部反归一化。
#      前向前调用 set_normalization(training_stats)，统计量保存在 checkpoint。
#      真实接触、力、位置、滑移及 starts_in_contact 均不作为模型输入。
#   6. forward() 处理单事件，返回三头、先验、权重、时序诊断、hidden 和视觉 token。
#      forward_sequence() 输入一张图像及 [B,T,6,64,28] 音频，图像只编码一次；
#      按时间循环“先验 -> 门控融合 -> GRU”，下一事件使用刚更新的 hidden，
#      三头输出 [B,T,3/6/2]，先验 [B,T,2]，权重/速度/时间诊断 [B,T]。
#   7. 训练提供缓存的实际时间信息。直接调用省略时间时 delta_t 默认 1/150 s；
#      单事件 image_age 默认 0，序列默认事件零有新图像，后续年龄累加 delta_t。
#      流式先 eval()；step() 缓存视觉 token、hidden、图像年龄，新图像默认年龄归零，
#      复用图像时年龄增加 delta_t。显式 image_age 可覆盖默认值。
#      新轨迹调用 reset_state()；流式缓存不进入 checkpoint。
#   8. 默认 d_model=128、hidden_size=256、nhead=8、dropout=.1、gate_hidden_size=64，
#      speed_scale_mm_s=10、age_scale_s=1/15；骨干随机初始化。
# =============================================================================

"""Scheme II: causal slip prior -> gated dual attention -> GRU -> three heads.

New images provide 16 cached spatial tokens; each current causal audio window
provides 28 temporal tokens (shared per-microphone encoder, then mic average).
The current audio summary and PREVIOUS hidden state predict a coarse normalized
XY slip increment. Its de-normalized speed, image age and audio context produce
the visual weight alpha. A shared historical query independently reads both
modalities, followed by LayerNorm(alpha*z_visual + (1-alpha)*z_audio).
Only the final GRU updates hidden state. The coarse prior has its own training
loss; the final position/wrench/slip heads retain the baseline's 3/6/2 outputs.

delta_t and image_age are seconds, never targets. Training supplies cache timing.
Direct calls may omit timing for an explicit nominal-150-Hz fallback; sequence
calls then assume a fresh image at event zero, while step() tracks image age.
Call set_normalization(training_stats) before forwarding; buffers persist in
checkpoints. No contact/force/position/slip ground truth is an estimator input.
The scalar gate is learned, not a calibrated confidence or monotonic constraint.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from ..model_triple_head import ModelConfig as BaseConfig, TactileFusionNet as BaseNet


@dataclass
class ModelConfig(BaseConfig):
    gate_hidden_size: int = 64
    detach_slip_for_gate: bool = True
    nominal_delta_t_s: float = 1. / 150.
    speed_scale_mm_s: float = 10.
    age_scale_s: float = 1. / 15.
    initial_visual_weight: float = .8

    def __post_init__(self):
        super().__post_init__()
        if self.image_only:
            raise ValueError("model_slip_gated requires both image and audio; image_only is unsupported")
        if self.d_model < 1 or self.hidden_size < 2 or self.nhead < 1 or self.d_model % self.nhead:
            raise ValueError("Positive dimensions and d_model divisible by nhead are required")
        if type(self.gate_hidden_size) is not int or self.gate_hidden_size < 1:
            raise ValueError("gate_hidden_size must be a positive integer")
        for key in ("nominal_delta_t_s", "speed_scale_mm_s", "age_scale_s"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not math.isfinite(self.initial_visual_weight) or not 0 < self.initial_visual_weight < 1:
            raise ValueError("initial_visual_weight must be strictly between 0 and 1")
        if type(self.detach_slip_for_gate) is not bool:
            raise ValueError("detach_slip_for_gate must be boolean")


class TactileFusionNet(BaseNet):
    requires_timing = True
    requires_normalization = True

    def __init__(self, config=None):
        super().__init__(config or ModelConfig())
        c = self.config
        # Reuse the encoders, embeddings, GRU and final heads, but replace the
        # baseline audio query / joint attention entirely (no unused parameters).
        del self.attention
        self.query = nn.Linear(c.hidden_size, c.d_model)
        self.initial_query = nn.Parameter(torch.randn(1, 1, c.d_model) * .02)
        self.visual_attention = nn.MultiheadAttention(c.d_model, c.nhead, c.dropout, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(c.d_model, c.nhead, c.dropout, batch_first=True)
        self.slip_prior_head = nn.Sequential(
            nn.LayerNorm(c.d_model + c.hidden_size),
            nn.Linear(c.d_model + c.hidden_size, c.hidden_size // 2),
            nn.GELU(), nn.Dropout(c.dropout), nn.Linear(c.hidden_size // 2, 2))
        self.gate_audio_norm = nn.LayerNorm(c.d_model)
        self.modality_gate = nn.Sequential(
            nn.Linear(c.d_model + 2, c.gate_hidden_size), nn.GELU(),
            nn.Linear(c.gate_hidden_size, 1))
        nn.init.constant_(self.modality_gate[-1].bias,
                          math.log(c.initial_visual_weight / (1. - c.initial_visual_weight)))
        self.register_buffer("target_mean", torch.zeros(11))
        self.register_buffer("target_std", torch.ones(11))
        self.register_buffer("normalization_ready", torch.tensor(False))

    def set_normalization(self, stats):
        mean = torch.as_tensor(stats["target_mean"], device=self.target_mean.device, dtype=self.target_mean.dtype)
        std = torch.as_tensor(stats["target_std"], device=self.target_std.device, dtype=self.target_std.dtype)
        if (mean.shape != (11,) or std.shape != (11,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Normalization must contain 11 finite means and positive standard deviations")
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)
        self.normalization_ready.fill_(True)

    @staticmethod
    def _timing(value, reference, shape, default, name):
        value = torch.as_tensor(default if value is None else value,
                                device=reference.device, dtype=reference.dtype)
        try:
            value = torch.broadcast_to(value, shape)
        except RuntimeError as exc:
            raise ValueError(f"{name} must be scalar or broadcastable to {shape}") from exc
        if not torch.isfinite(value).all() or ((value <= 0).any() if name == "delta_t" else (value < 0).any()):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'delta_t' else 'nonnegative'}")
        return value

    def _history(self, visual, hidden, has_history):
        b = visual.shape[0]
        if hidden is not None and hidden.shape != (1, b, self.config.hidden_size):
            raise ValueError("hidden must have shape [1,B,hidden_size]")
        present = torch.full((b,), hidden is not None, device=visual.device, dtype=torch.bool)
        if has_history is not None:
            present = torch.as_tensor(has_history, device=visual.device)
            if present.shape != (b,) or present.dtype != torch.bool:
                raise ValueError("has_history must be a boolean [B] tensor")
            if hidden is None and present.any():
                raise ValueError("has_history=True requires a hidden state")
        hidden = visual.new_zeros(1, b, self.config.hidden_size) if hidden is None else hidden
        return hidden * present[None, :, None], present

    def fuse(self, visual, audio=None, *, hidden=None, has_history=None, delta_t=None, image_age=None):
        """Return one gated token and diagnostic predictions, without updating GRU."""
        if not self.normalization_ready.item():
            raise RuntimeError("Call set_normalization(stats) before slip-guided fusion")
        if visual.ndim != 3 or visual.shape[1:] != (16, self.config.d_model):
            raise ValueError("Expected visual tokens [B,16,d_model]")
        b = visual.shape[0]
        hidden, present = self._history(visual, hidden, has_history)
        dt = self._timing(delta_t, visual, (b,), self.config.nominal_delta_t_s, "delta_t")
        age = self._timing(image_age, visual, (b,), 0., "image_age")
        acoustic = self.encode_audio(audio)
        if acoustic.shape[0] != b:
            raise ValueError("Image and audio batch sizes must agree")
        summary = acoustic.mean(dim=1)
        prior = self.slip_prior_head(torch.cat((summary, hidden[0]), dim=-1))
        prior_mm = prior * self.target_std[9:] + self.target_mean[9:]
        speed = torch.linalg.vector_norm(prior_mm, dim=-1) / dt
        gate_speed = speed.detach() if self.config.detach_slip_for_gate else speed
        gate_input = torch.cat((self.gate_audio_norm(summary),
                                torch.log1p(gate_speed / self.config.speed_scale_mm_s)[:, None],
                                (age / self.config.age_scale_s)[:, None]), dim=-1)
        alpha = self.modality_gate(gate_input).sigmoid()
        query = torch.where(present[:, None, None], self.query(hidden[0])[:, None],
                            self.initial_query.expand(b, -1, -1))
        visual_memory = visual + self.modality_embedding[:, 1:] + self.position_embedding[:, 28:]
        audio_memory = acoustic + self.modality_embedding[:, :1] + self.position_embedding[:, :28]
        zv, _ = self.visual_attention(query, visual_memory, visual_memory, need_weights=False)
        za, _ = self.audio_attention(query, audio_memory, audio_memory, need_weights=False)
        # No unconditional current-audio residual bypasses the modality gate.
        fused = self.fusion_norm(alpha[:, None] * zv + (1. - alpha[:, None]) * za)
        return {"fused": fused, "slip_prior": prior, "slip_prior_mm": prior_mm,
                "prior_speed_mm_s": speed, "visual_weight": alpha[:, 0],
                "audio_weight": 1. - alpha[:, 0], "delta_t": dt, "image_age": age}

    def forward(self, image, audio=None, starts_in_contact=None, *, hidden=None,
                visual_tokens=None, delta_t=None, image_age=None, has_history=None):
        if starts_in_contact is not None:
            raise ValueError("Ground-truth/acquisition contact flags are not model inputs")
        visual = self.encode_visual(image) if visual_tokens is None else visual_tokens
        hidden, present = self._history(visual, hidden, has_history)
        result = self.fuse(visual, audio, hidden=hidden, has_history=present,
                           delta_t=delta_t, image_age=image_age)
        states, hidden = self.gru(result.pop("fused"), hidden)
        return {**self._predict(states[:, 0]), **result, "hidden": hidden, "visual_tokens": visual}

    def forward_sequence(self, image, audio=None, hidden=None, *, sequence_length=None,
                         delta_t=None, image_age=None, has_history=None):
        if audio is None or audio.ndim != 5 or audio.shape[1] < 1 or audio.shape[2:] != (6, 64, 28):
            raise ValueError("Expected sequence audio [B,T,6,64,28]")
        b, length = audio.shape[:2]
        if sequence_length is not None and (type(sequence_length) is not int or sequence_length != length):
            raise ValueError("sequence_length must match audio event count")
        visual = self.encode_visual(image)
        if visual.shape[0] != b:
            raise ValueError("Image and audio batch sizes must agree")
        hidden, present = self._history(visual, hidden, has_history)
        dt = self._timing(delta_t, visual, (b, length), self.config.nominal_delta_t_s, "delta_t")
        default_age = torch.cat((dt.new_zeros(b, 1), dt[:, 1:]), dim=1).cumsum(dim=1)
        age = self._timing(image_age, visual, (b, length), default_age, "image_age")
        states, diagnostics = [], []
        for t in range(length):
            result = self.fuse(visual, audio[:, t], hidden=hidden, has_history=present,
                               delta_t=dt[:, t], image_age=age[:, t])
            state, hidden = self.gru(result.pop("fused"), hidden)
            states.append(state)
            diagnostics.append(result)
            present = torch.ones(b, device=visual.device, dtype=torch.bool)
        return {**self._predict(torch.cat(states, dim=1)),
                **{k: torch.stack([r[k] for r in diagnostics], dim=1) for k in diagnostics[0]},
                "hidden": hidden}

    def reset_state(self):
        super().reset_state()
        self._stream_image_age = None

    @torch.no_grad()
    def step(self, audio=None, image=None, *, frame_id=None, reset=False, delta_t=None, image_age=None):
        """One causal event; pass measured seconds to override nominal timing.

        A new image resets age to zero unless an explicit capture age is supplied.
        Repeated events add delta_t to the cached age. Only final h is committed.
        """
        if self.training:
            raise RuntimeError("Call eval() before streaming inference")
        if reset:
            self.reset_state()
        refresh = image is not None and (frame_id is None or frame_id != self._stream_frame_id)
        if refresh:
            self._stream_visual = self.encode_visual(image)
            self._stream_frame_id = frame_id
        elif frame_id is not None and frame_id != self._stream_frame_id:
            raise ValueError("A new frame_id requires its image")
        if self._stream_visual is None:
            raise ValueError("First streaming event requires an image")
        visual = self._stream_visual
        dt = self._timing(delta_t, visual, (visual.shape[0],), self.config.nominal_delta_t_s, "delta_t")
        age = (torch.zeros_like(dt) if refresh or self._stream_image_age is None
               else self._stream_image_age + dt) if image_age is None else image_age
        result = self.forward(None, audio, hidden=self._stream_hidden, visual_tokens=visual,
                              delta_t=dt, image_age=age)
        self._stream_hidden = result["hidden"].detach()
        self._stream_image_age = result["image_age"].detach()
        return result
