# 150Hz 部署测试数据

整个文件夹或 ZIP 可直接复制到另一个项目。读取只依赖 NumPy、Pillow，不依赖训练项目的 `mvts`。每个 `events/*.npz` 对应一次网络调用，按 `events.jsonl` 的行序读取即可。

- 一条快速滑移轨迹 `t0242` 在 `data_final_v4_faster` 中的全部 29 个连续有效窗口，共 290 个子事件。第一个子事件重置状态，其后持续传递 GRU 隐状态。
- 共 290 次调用，仅保留滑移轨迹，不含独立按压样本，全部来自 `runs/triple_head_faster` 的 test 划分。此小包用于部署联调，整体指标请用完整测试集。

## 直接试读

```bash
pip install numpy pillow
python reader.py
python reader.py --kind slide --realtime
```

`reader.py` 会先把输入全部读到内存，再开始调用。`--realtime` 按 150Hz 的绝对截止时间回放，相邻调用目标间隔约 6.667ms；实际速度受模型耗时和系统调度影响，超时时不丢样本。默认不等待，适合检查输入与推理结果。

## 一次调用的输入

`reader.load_events()` 返回列表，每项包含：

| 字段 | 形状 / 类型 | 用途 |
| --- | --- | --- |
| `audio_input` | float32 `[1,6,64,28]` | 已按随包 normalization.json 标准化，可直接送入模型 |
| `image_input` | float32 `[1,3,224,224]` 或 None | RGB、CHW、ImageNet 标准化；仅新图像时提供 |
| `adc` | uint16 `[4000,6]` | 六通道原始 ADC，50kHz，事件之前 80ms，用于测试自己的音频预处理 |
| `target` | float64 `[11]` | 物理单位真值，用于对照，不作为网络输入 |
| `meta` | dict | 时间、图像更新和状态重置标记 |

NPZ 内 `audio` 是未标准化 Log-Mel `[6,64,28]`，`audio_input` 不含 batch 维；读取器负责加 batch 维。六路顺序沿用采集顺序，通道维不重排。所有数值文件均可 `np.load(path, allow_pickle=False)`。

`meta.image_update=True` 时更新视觉缓存；滑移每 10 次调用换一张图像，其余 9 次复用视觉缓存。`meta.reset_state=True` 时重置 GRU 隐状态和视觉缓存。若部署接口要求每次都传图像，可在外部保存最近的 `image_input` 并复用。

```python
from reader import load_events, replay

events = load_events(kind="slide")

def infer(event):
    meta = event["meta"]
    # 在这里调用自己的部署接口：
    # 1. meta["reset_state"] 为 True：重置 GRU 和视觉缓存。
    # 2. event["image_input"] 非 None：更新视觉缓存。
    # 3. 用 event["audio_input"] 推理一次，并保留新的 GRU 状态。
    # 4. 推理输出转回物理单位后，与 event["target"] 对照。
    pass

replay(events, infer, realtime=True)
```

采用本项目 PyTorch `model.step` 接口时，可将回调替换为：

```python
def infer(event):
    meta = event["meta"]
    image = event["image_input"]
    result = model.step(
        torch.from_numpy(event["audio_input"]).to(device),
        image=None if image is None else torch.from_numpy(image).to(device),
        frame_id=meta["image"], reset=meta["reset_state"],
    )
    # 使用前调用 model.eval()；输出按下文反标准化。
```

## 标准化与输出

随包标准化参数来自 `runs/triple_head_faster/normalization.json`。对应模型配置在 `manifest.json` 的 `model_config`：该次训练禁用了第 5 路麦克风（1 起始编号），模型内部处理，输入文件仍保留全部 6 路。部署时应保持自己的导出模型所要求的输入和通道选择。

若部署其他训练运行的权重，应使用其标准化参数：

```python
events = load_events(kind="slide", normalization="your_normalization.json")
```

`audio_input = (audio - audio_mean[:,None,None]) / audio_std[:,None,None]`。已经包含预处理的部署模型应输入 `adc` 或未标准化 `audio`（按其接口选择），避免重复标准化。

网络 11 维输出若仍为标准化值，按 `prediction_physical = prediction * target_std + target_mean` 转回物理单位后再比较。

顺序：`x_mm, y_mm, delta_z_mm, Fx, Fy, Fz, Mx, My, Mz, slip_x_mm, slip_y_mm`。位置/滑移单位 mm，力 N，力矩 N*m；滑移标签是相邻子事件 XY 增量。完整标签也在 `labels.csv`。

## 时序与原始数据边界

`events.jsonl` 的 `schedule_offset_ns` 是整个 290 次调用（从 0 开始）的 150Hz 回放时间；`sequence_offset_ns` 是各独立序列内的回放时间，均以 ns 为单位。`event_pc_ns` 是原始采集时间，保留真实帧间隔抖动。150Hz 仅规定回放调度，没有重新插值音频或真值，也没有修改原始采集时间。包内仅含一条连续滑移序列。

滑移的缓存窗口原始 order 为 2–30，全部导出。原始轨迹有 32 个图像区间：order 0 和 31 被原预处理排除为首尾裁剪区间；order 1 缺少完整 4000 点历史音频。该包严格包含缓存内全部有效事件，不补零、不补造首尾事件。详情见 `manifest.json.excluded_source_intervals`。

音频参考实现为独立文件 `preprocessing_reference.py`（需要 PyTorch）：Slaney Mel、Hann 窗、FFT=512、hop=128、64 Mel、`center=False`、右对齐、`log10(clamp(power,1e-10))`。先对全部 4000 点除以 65535 并去每通道 DC，再保留右侧 3968 点做 STFT，得到 28 帧。导出时已由原始 ADC 重算全部特征，与源缓存逐值一致，见 `verification.json`。

`checksums.sha256` 覆盖包内其他文件，可用于跨项目复制后的完整性检查。本包只含测试输入、标签和参考预处理，不含模型权重。
