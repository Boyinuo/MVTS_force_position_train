# =============================================================================
# 接触门控四预测头网络架构（默认 ModelConfig；B 为 batch size，M 为启用麦克风数）
# =============================================================================
#
#       慢速视觉分支 Vision                       快速声学分支 Audio
#       新图像到达时更新并缓存                     每个因果子事件更新
#
#   RGB image [B,3,224,224]                  Log-Mel [B,6,64,28]
#                |                                    |
#                v                                    v
#   +---------------------------+         +----------------------------------+
#   | Visual ResNet-18          |         | Select enabled microphones      |
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
#                |                                        |
#                +-------------------+--------------------+
#                                    |
#                   Add audio / visual modality embeddings
#                   Concat(S_k, V_j) + positional embedding
#                             memory [B,44,128]
#                                    |
#                   +----------------+--------------------+
#                   |                                     |
#       Q = Linear(memory[:,27:28])                K = V = memory
#       (last embedded audio token)                 [B,44,128]
#             [B,1,128]                                   |
#                   +----------------+--------------------+
#                                    |
#                   Joint Cross-Attention(128, heads=8)
#                   f_k = LayerNorm(MHA(Q,K,V) + Q)
#                             f_k [B,1,128]
#                                    |
#     h_(k-1) [1,B,256] ---> Stateful GRU(128,256), 1 layer
#                                    |
#                    output[:,0] = h_k [B,256]
#                    hidden [1,B,256] --------------------> next event
#                                    |
#          +----------------+--------+--------+----------------+
#          |                |                 |                |
#          v                v                 v                v
#    Position Head     Wrench Head        Slip Head       Contact Head
#    LN -> 256/128/3   LN -> 256/128/6    LN -> 256/128/2  LN -> 256/128/1
#          |                |                 |                |
#    position [B,3]    wrench [B,6]       slip [B,2]     contact_logits [B]
#          |                |                 |                |
#          +----------------+-----------------+             sigmoid
#                           |                                  |
#              concat -> raw [B,11]                   probability p_k [B]
#                           |                                  |
#           physical = raw * target_std + target_mean           |
#                           |                                  v
#                           |                  +-----------------------------+
#                           |                  | Deterministic hysteresis    |
#                           |     c_(k-1) ---> | unknown: p >= .5            |
#                           |     (-1/0/1)     | p >= .6: ON; p <= .4: OFF   |
#                           |                  | otherwise keep previous c   |
#                           |                  +-----------------------------+
#                           |                            |             |
#                           |                       c_k [B]       delay/store
#                           |                            |             |
#                           |                            |       next c_(k-1)
#                           +-------------+--------------+
#                                         |
#                              Physical output gating
#                                         |
#          +------------------------------+-------------------------------+
#          |                              |                               |
#          v                              v                               v
#   position_physical [B,3]      wrench_physical [B,6]         slip_physical [B,2]
#   keep iff c_k == 1           keep iff c_k == 1             keep iff previous=1
#   otherwise zero             otherwise zero               AND current=1
#   position_valid = c_k        unloaded wrench = 0          slip_valid = (1->1)
#
# 补充约定：
#   1. 两路编码器继承 model_triple_head：每个 stage 含 2 个 BasicBlock，通道为
#      64/128/256/512；视觉 stage 首块 stride 为 (1,1)/(2,2)/(2,2)/(2,2)，
#      声学为 (1,1)/(2,1)/(2,1)/(2,1)，保留 28 个时间 token。声学逐麦共享权重。
#      输入为事件之前 4000 点/通道生成的 [6,64,28] Log-Mel，不使用未来音频。
#      disabled_mic_channels 采用 1..6 编号，外部输入仍为 6 路，内部筛选后求均值。
#   2. 四个 head 均为 LayerNorm(256) -> Linear(256,128) -> GELU ->
#      Dropout(.1) -> Linear(128,out)。前三头预测标准化回归值，接触头输出 logit。
#      接触门控不参与前面的编码、注意力或 GRU 更新，也不因接触切换重置 GRU。
#   3. forward()/forward_sequence() 返回未门控的回归、接触 logit/probability
#      及 hidden；序列输出为 [B,T,3/6/2] 和 [B,T]，图像在窗口内只编码一次。
#      gate_outputs() 才做反归一化和物理门控；先调用 set_normalization(stats)。
#      归一化统计量保存在 checkpoint，流式视觉缓存、hidden 和接触状态不保存。
#   4. previous_contact 是上一次预测状态（-1=未知，0=离开，1=接触）。首次未知
#      状态按 initial_threshold=.5 分类；后续按 off=.4/on=.6 滞回，含阈值边界。
#      transition = 2*previous + current：0=00、1=01、2=10、3=11；首次为 -1，
#      transition_valid=False。首次即使预测接触，slip_valid 仍为 False。
#      无效位置/滑移用零值及显式 validity 标志表示，原始物理回归也保留在结果中。
#   5. 物理输出顺序为 (x,y,delta_z,Fx,Fy,Fz,Mx,My,Mz,delta_sx,delta_sy)，
#      位置/滑移单位 mm，力 N，力矩 N*m。必须先反归一化，再将无效输出置零。
#      接触状态只来自预测；真实接触、力、位置、滑移及 starts_in_contact 不输入模型。
#   6. 流式推理先 eval()；step() 复用视觉 token、GRU hidden 和预测接触状态，
#      自动调用 gate_outputs()。新轨迹调用 reset_state()，三类缓存一起重置。
#   7. 默认 d_model=128、hidden_size=256、nhead=8、dropout=.1，骨干随机初始化。
#      image_only=True 时继承基线的纯视觉路径：仅 16 个 token，Q 为其均值的线性
#      投影，不构造声学编码器/投影；四预测头及后处理接触门控保持相同定义。
# =============================================================================

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

from ..model_triple_head import ModelConfig as BaseConfig, TactileFusionNet as BaseNet


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
