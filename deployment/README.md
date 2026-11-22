# MVTS 滑移部署模拟界面

先加载选定滑移轨迹的全部事件，按原顺序执行单事件推理，完整保留视觉缓存、GRU 状态和模型自身的接触状态。推理与结果保存完成后，界面才开始播放。程序不读取相机或麦克风设备。

## 环境与启动

在项目根目录执行（使用项目虚拟环境）：

```powershell
uv sync --frozen
uv pip install -r deployment/requirements.txt

# 必须选择模型；自动加载 mvts/model_contact_gated/best.pt，默认 150 Hz
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_contact_gated

# 使用其他架构时，自动加载该模型目录中的 best.pt（需要先放入权重）
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_triple_head.py

# 指定模型、同格式数据包和滑移序列
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_contact_gated.py --data-root deployment/test_data/deployment_test_150hz --sequence slide_t0242 --device cuda
```

**`--model` 必填，不设默认模型。** 程序使用项目注册的架构，自动读取该模型模块所在目录的 `best.pt`，无需也不再支持 `--checkpoint`。例如 `--model model_contact_gated` 对应 `mvts/model_contact_gated/best.pt`。`--model` 接受模型模块名称、`.py` 文件名或 `mvts/<model>/<model>.py` 路径；模型需支持 `step`、视觉缓存和连续状态接口。缺少权重或 checkpoint 的 `model_name` 不匹配时会明确报错，不会忽略缺失参数或用随机权重补全。

本次演示使用已确认的 `mvts/model_contact_gated/best.pt`：`model_contact_gated`，第 24 轮，六路麦克风均启用。

默认数据目录为项目根目录下 `deployment/test_data/deployment_test_150hz`，默认选择其中第一条完整滑移序列 `slide_t0242`：290 个子事件、29 次图像更新，150 Hz 下回放时长约 1.93 秒。10 个独立按压样本不包含在本次演示中。

## 界面与时间含义

- 左上：相机原始 RGB 图像，224×224，按事件的 `image_update` 更新并保持比例。
- 右上：每个已播放事件的预测位置显示青色圆点，实测位置显示橙色叉点，并保留连线；当前点用更大标记突出显示。滚轮缩放、左键拖动平移、右键拖动缩放，播放更新时保留缩放范围。可点击“适应已播轨迹”或“重置视图”。标记大小固定，重叠点可放大检查；可切换 X、Y、Δz 时间曲线。
- 左下：Fx、Fy、Fz、Mx、My、Mz，使用反标准化后的物理量。
- 右下：当前事件之前 80 ms 的六路原始训练 Log-Mel，固定色阶 `[-10,-2.5]`，横轴 ms，纵轴 Mel bin。训练中禁用的麦克风会明确标记，但仍可查看该路数据。
- 底部：播放/暂停、重新回放、清空 XY 历史、冻结/解除冻结、0.25×–2× 速度、拖动进度。

`--hz 150` 定义事件的部署回放时间轴，不改变原始特征、标签或采集时间。默认 `--render-fps 60`，界面以 60 Hz 为目标刷新，实际帧率受绘制耗时和系统调度影响。每次刷新把所有已到期的事件加入历史曲线，不要求屏幕每秒绘制 150 帧。刷新延迟时会追上绝对时间轴，期间的曲线点不丢失；图像和频谱显示最近到期的事件。回放结束保留最后一帧。

预计算能把网络推理、文件读取和图像解码移出播放阶段，减少播放卡顿；总推理计算量不会减少，启动等待会增加。顶部的 step 均值/P95 是真实预计算阶段的单事件耗时（包含输入传输、网络和输出拷回 CPU；不含文件读取，不预热，保留第一步开销），与 UI 刷新 FPS 分开。该程序不能单凭回放达到 150 Hz 就证明在线部署能达到 150 Hz。

接触门控模型使用自身预测接触状态与有效标志；无效位置不绘制为原点。三头模型没有接触分类头，界面明确显示这一点，不用实测 Fz 构造预测接触状态。所选滑移数据的真实接触依据采集协议均为接触。位置误差按有效预测点统计，并报告覆盖率和漏判数；三头模型不具备接触漏判评估能力。XY 轨迹直接使用位置头输出，不把滑移增量累加冒充位置。

始终从 NPZ 的未标准化 `audio` 重做 checkpoint 对应的音频标准化，不直接使用数据包预存的其他模型 `audio_input`。标签仅供显示和对照，不进入模型调用。权重特征配置必须与数据包一致。

## 导出与检查

结果自动保存到 `deployment/results/日期_时间/`，也可通过 `--output` 指定。包含 `summary.json`、`predictions.csv`、`predictions.npz`：完整物理单位预测、实测值、有效标志、接触概率、逐事件耗时及原始采集时间。预测点对应 `events.jsonl` 的 `sequence_index`。

```powershell
# 只预计算并导出，不显示窗口
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_contact_gated --headless

# 最后一帧截图后退出（仍然先完整推理）
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_contact_gated --screenshot deployment/results/demo.png

# 回归检查：轨迹选择、连续推理/缓存、物理量、结果导出、回放/拖动/冻结
.venv\Scripts\python.exe -m pytest deployment/test_replay.py -q
```

`--random-weights` 仅用于明确要求的界面测试，仍须指定 `--model`，界面会显示红色“随机未训练权重”，不能用于评估模型。正常流程始终加载训练权重。
