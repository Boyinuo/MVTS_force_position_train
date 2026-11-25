# 方案二：滑移先验引导的双速率门控融合

模型名为 `model_slip_gated`。本目录包含模型实现、包入口以及对应的 PNG / SVG 架构图。

当前因果声学特征与上一时刻 GRU 状态先预测二维粗滑移增量。粗增量反归一化为 mm 后，除以真实事件间隔得到速度；速度、图像年龄和声学摘要共同生成视觉权重 `alpha`。两路 Attention 使用同一个历史 Query，分别读取视觉 / 声学 token，执行 `LayerNorm(alpha * z_visual + (1-alpha) * z_audio)`，再进行一次 GRU 更新。末端继续预测位置 3 维、六维力 6 维与最终滑移 2 维。

粗滑移先验另有掩码 Huber 损失。默认对送入门控的速度执行 `detach`，防止位置 / 力任务通过该数值路径扭曲滑移回归。门控仍能通过声学摘要与主任务损失学习。普通 MLP 不保证权重随速度单调变化；训练和预测会导出权重，便于检查是否学到预期行为。

## 训练

```powershell
.venv\Scripts\python.exe train.py --model model_slip_gated --prepared-root .cache/data_final_v6_press_contact .cache/data_final_v6_slide_contact .cache/data_final_v6_polyline_contact --device cuda --epochs 30 --batch-size 8 --workers 10 --slip-prior-weight 1 --output runs/v6_slip_gated
```

新模型未指定 `--prepared-root` 时，默认选择以上三个 v6 目录。v6 是数据集名称，其缓存格式仍是 v5 接触监督格式。现有 `--filter-xy-range`、`--trim-stationary-xy-mm`、`--disable-mic-channels` 可照常使用。该架构需要视觉与音频，不支持 `--image-only`。原默认模型仍是 `model_triple_head`。

训练/续训结束后默认用本次 `best.pt` 自动测试 `demo_circle`、`demo_line`、`demo_triangle`，在运行目录直接写入三个同名 PNG 和 SVG，并在终端分别打印 loss、main_loss 和各维 MAE。详情位于 `demo_tests/summary.json` 及三个子目录。外测默认完整保留轨迹，不继承训练的 XY 筛选或静止截取；可用 `--demo-trim-stationary-xy-mm`、`--demo-filter-xy-range` 独立设置，或用 `--no-demo-tests` 关闭。

输出目录统一采用 `runs/v6_xxx` 格式。新模型省略 `--output` 时使用 `runs/v6_slip_gated`；续训省略时使用传入 checkpoint 所在目录。短流程测试目录为 `runs/v6_slip_gated_smoke`。

`--slip-prior-weight` 默认 1，允许设为 0 做辅助损失消融；续训省略该选项会继承原值。切换架构或辅助损失权重应新开训练目录。位置、力、滑移以及接触真值仅用于监督或损失掩码，不输入估计器。

```powershell
.venv\Scripts\python.exe train.py --resume runs/v6_slip_gated/last.pt --device cuda --epochs 60 --batch-size 8 --workers 10 --output runs/v6_slip_gated
.venv\Scripts\python.exe evaluate.py --checkpoint runs/v6_slip_gated/best.pt --device cuda --output runs/v6_slip_gated/evaluation
```

续训时批量大小、学习率、划分种子及损失权重等选项需与原运行一致。评估自动使用权重记录的测试轨迹和训练统计量。

## 单事件与流式调用

```python
from mvts.architectures import build_model

model = build_model("model_slip_gated")
model.set_normalization(training_stats)  # target_mean / target_std，均为 11 维
model.eval()
first = model.step(audio, image=image, frame_id=0, delta_t=0.0067, image_age=0.0)
next_event = model.step(next_audio, frame_id=0, delta_t=0.0065, image_age=0.0065)
model.reset_state()  # 新轨迹开始时调用
```

图像 `[B,3,H,W]`、归一化 Log-Mel `[B,6,64,28]`；序列音频 `[B,T,6,64,28]`，时间参数为 `[B,T]`，单位秒。`forward_sequence()` 每个窗口仅编码一次图像，按事件顺序融合和更新状态，窗口间使用截断梯度。批内新轨迹通过 `has_history=False` 选择可学习初始 Query。

训练会从缓存读取 `event_pc_ns`、`previous_pc_ns` 和图像时间，先进行整数纳秒差分，再转为浮点秒，避免大时间戳丢失精度。连续滑移缺失时间信息时明确报错。独立按压没有连续历史，门控使用名义 `1/150 s` 间隔；相机主机读出时间略晚于标签时，主机时间计算的负图像年龄置零，不将其解释为曝光时间保证。直接 Python 调用省略时间参数时采用名义 150 Hz；实际部署建议始终提供实测间隔和图像年龄。

`position` / `wrench` / `slip` / `slip_prior` 输出位于标准化标签空间；`slip_prior_mm` 和 `prior_speed_mm_s` 是物理量。`visual_weight`、`audio_weight` 为标量系数，和为 1。归一化常量保存于 checkpoint；`predict.py` 还会输出这些诊断值。

`predict.py --prepared-window` 会从相邻缓存清单恢复图像时间，以便与训练使用同一图像年龄。若仅复制单独的 NPZ 而没有清单，则将图像视为窗口起点的新图像；需要精确图像年龄的部署应使用 Python 接口显式传入。

图中传感器缩略图和波形为示意，架构图本身不代表训练性能结果。
