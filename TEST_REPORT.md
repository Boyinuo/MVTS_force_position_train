# 训练 pipeline 本地测试记录

以下为 2026-11-06（Asia/Shanghai）的 CPU 测试记录。原始数据未修改。本次验证目标是完整运行所给网络的训练与推理流程，不是完成正式精度实验。2026-11-07 的 CUDA 环境迁移验证记录见本文末尾。

## 环境

项目环境：`E:\Github\MVTS_force_position_train\.venv`。Python 3.11.18、PyTorch 2.15.0+cpu、NumPy 2.4.6、Pillow 12.3.0、pytest 9.2.0。实际模型 24,294,090 个可训练参数，视觉输入 3×224×224，音频输入 6×64×12，融合 token 为 13×256。未使用预训练权重。

`uv sync --locked --link-mode copy` 已完成；`uv run --frozen python ...` 确认解释器进入项目 `.venv`；`uv lock --check --offline` 通过。该次测试使用 CPU 版环境，CUDA 执行路径由后续迁移验证覆盖。

## 数据检查

全量扫描两个 session，共 2103 个事件：

| 类别 | 数量 |
| --- | ---: |
| 阶梯按压 target | 1589 |
| 阶梯按压 limit | 6 |
| 动态按压 target | 54 |
| 排除的悬空背景 | 454 |
| 纳入训练/验证/测试划分 | 1649 |
| 唯一计划位置组 | 400 |
| 拒绝的坏记录 | 0 |

事件 NPZ 的 ADC 均为 2000×6，时间戳采样率与会话 50 kHz 一致。检查到的 ADC 范围 679–2251；没有音频时间戳大间隙、没有 marker 落在对应窗口之外。NPZ 的实测位置、六维力、接触标志和 marker 与 JSON 一致。

图像年龄中位数 39.02 ms，最大 122.79 ms；力年龄中位数 31.27 ms，最大 67.52 ms。这些是软件关联的年龄，不代表已测得硬件同步误差。两个 session 的 `timing_quality_passed=true`，同时存在 `valid_for_time_matching=false` 和 `microphone_timing_valid=false`，采集会话也未完成。默认调试流程已显式警告并使用导出的有效事件；正式精度实验前仍需核查这些标志及采集端对齐逻辑。

通过机器人位姿、探头长度与参考平面复算样本的压入量：参考平面深度约 1.78784 mm，减去本点接触面深度约 1.14564 mm，得到 0.64220 mm，与 `actual_depth_mm` 一致。因此程序直接使用该字段，不再次减接触面深度。

## 自动测试

执行 `.venv/Scripts/python.exe -m pytest -q`：**6 passed in 31.86s**。

测试涵盖：

- 2000 点 ADC 到 6×64×12 Log-Mel；零输入有限值；幅值加倍后的功率关系。
- 完整双路 ResNet 的输出形状；声学骨干为 512×8×12；两路骨干、位置编码和注意力池化均有有限且非零的梯度。
- 可选接触状态 embedding 及缺少状态时的报错。
- 跨 session、跨采集模式的同位置分组；训练、验证、测试组无交集；只用训练集计算归一化。
- 错误六维力标签被拒绝，背景和源端无效记录正确排除，严格同步模式拒绝未通过条件。
- 使用完整网络的 CLI 训练、checkpoint 保存、恢复到下一轮、独立评估和单样本预测。

另通过全部 Python 文件的语法编译检查和 `git diff --check`。

## 真实数据小规模试跑与恢复

执行：

```powershell
uv run --frozen python train.py --epochs 2 --batch-size 4 --max-groups 4 --max-steps 3 --max-eval-steps 2 --output runs/smoke
uv run --frozen python train.py --epochs 3 --batch-size 4 --max-groups 4 --max-steps 3 --max-eval-steps 2 --output runs/smoke --resume runs/smoke/last.pt
```

训练/验证/测试分别选择 4 个位置组，含 16/16/19 条记录。每轮训练实际处理 12 条，验证与训练结束时的测试各处理 8 条。

| 轮次 | 训练损失 | 验证损失 |
| --- | ---: | ---: |
| 1 | 1.002078 | 9.742703 |
| 2 | 0.890737 | 9.847137 |
| 3，断点恢复 | 0.844742 | 9.793953 |

验证最优为第 1 轮。独立 `evaluate.py` 成功读取 checkpoint 并评估整个 19 条测试分区；`predict.py` 成功输出反归一化的 9 维有限值。小规模试跑的归一化仅拟合 4 个训练位置，验证损失受位置分布差异影响很大，不能与全量实验直接比较。

## 全部有效数据完整运行一轮

执行：

```powershell
uv run --frozen python train.py --epochs 1 --batch-size 8 --output runs/pipeline_full1
```

未使用任何 `max-*` 限制，`summary.json` 中 `smoke_test=false`。这表示数据遍历完整，不表示正式训练已收敛。

| 分区 | 位置组 | 实际处理记录 | batch 数 | 标准化联合损失 |
| --- | ---: | ---: | ---: | ---: |
| 训练 | 280 | 1154 | 145 | 0.423247 |
| 验证 | 60 | 241 | 31 | 0.455251 |
| 测试 | 60 | 254 | 32 | 0.310752 |

一轮训练与验证约 124.7 秒；包含审查、归一化、模型保存和测试的总流程约 141.1 秒。测试使用验证损失选择的 `best.pt`，并非最后一个 batch 的临时状态。

| 测试维度 | 模型 MAE | 训练均值预测基线 MAE |
| --- | ---: | ---: |
| x，mm | 2.3159 | 4.6944 |
| y，mm | 1.6540 | 2.9831 |
| Δz，mm | 0.5273 | 0.9689 |
| Fx，N | 0.02860 | 0.04030 |
| Fy，N | 0.06218 | 0.09019 |
| Fz，N | 0.88278 | 1.50071 |
| Mx，N·m | 0.08958 | 0.14221 |
| My，N·m | 0.11554 | 0.17426 |
| Mz，N·m | 0.00691 | 0.00990 |

单位根据用户对原始标签的确认标注为 N 和 N·m，以上数值未作单位换算。这些值仅记录本次一次划分、一轮训练的调试结果。数据缺少计划中的完整探头/动态工况覆盖，时间匹配标志尚未解释清楚，未做多随机种子、跨 session/跨探头独立泛化或充分收敛实验。

结果保存在 `runs/pipeline_full1/`：`best.pt`、`last.pt`、`history.jsonl`、`test_metrics.json`、`test_predictions.csv`、`summary.json` 等。已额外用 `best.pt` 成功运行 `predict.py`，结果为 `example_prediction.json`。该目录与 `runs/smoke/` 均保留在本地，并由 `.gitignore` 排除。

## 2026-11-07 CUDA 环境迁移验证

项目 `.venv` 已从 `torch 2.15.0+cpu` 切换为 **torch 2.14.1+cu126**。官方 CUDA 12.6 索引所提供的兼容版本为 2.14.1，`pyproject.toml` 和 `uv.lock` 已同步更新；不需要单独安装完整 CUDA Toolkit。Python、NumPy、Pillow 版本保持前述记录。显卡为 NVIDIA GeForce RTX 3060 Laptop GPU（6 GB，compute capability 8.6），驱动 572.83。

CUDA wheel 下载期间原 R2 CDN 出现 TLS 中断，最终从另一个官方 CDN 分块续传，并与 `uv.lock` 的 SHA256 完整核对：`4cff57ddd28c8bdf8bf2fc473f149c1b5f0063b1a6bd57593288e0bf18459adb`。通过 uv 从本地 wheel 缓存安装后，`uv sync --locked --offline`、`uv lock --check --offline` 和正常的 `uv run --frozen` 均通过，后续运行不会切回 CPU 包。完整 wheel 保留在 `.cache/wheels/`，临时分块已清理；该缓存不纳入 Git。

### GPU 检测与自动测试

- `torch.__version__ = 2.14.1+cu126`，`torch.version.cuda = 12.6`，`torch.cuda.is_available() = True`。
- GPU 上矩阵前向、反向通过，梯度为有限值。
- 自动测试扩展为 CPU / CUDA 两条端到端路径，执行结果：**7 passed in 44.76s**。GPU 路径显式使用 `--device cuda`，检查训练配置实际记录的设备、GPU 名称和 CUDA runtime，并完成训练、恢复、评估及单样本预测。
- 训练和独立评估使用 pinned host memory，设备传输支持 non-blocking；GPU 名称及 CUDA runtime 记录在 `config.json` 和 `summary.json`。

### 真实数据全量一轮 GPU 训练

执行用户后续可直接使用的 uv 命令：

```powershell
uv run --frozen python train.py --device cuda --epochs 1 --batch-size 8 --output runs/gpu_pipeline1
```

全部 1649 条记录按原规则划分，未加 `max-*` 限制。配置与汇总均确认 `device=cuda`。

| 分区 | 记录数 | 标准化联合损失 |
| --- | ---: | ---: |
| 训练 | 1154 | 0.428612 |
| 验证 | 241 | 0.477672 |
| 测试 | 254 | 0.328849 |

一轮训练与验证耗时约 **18.5 秒**，含数据审查、归一化、模型保存及测试的整体流程约 **51.6 秒**。此前 CPU 一轮训练与验证约 124.7 秒；这只是两次实测记录，未做严格性能基准。GPU 测试 MAE 为 x=1.7959 mm、y=1.7726 mm、Δz=0.6355 mm，其他维度和逐样本预测见输出目录的 `test_metrics.json` 和 `test_predictions.csv`。

输出目录：`runs/gpu_pipeline1/`，保留 `best.pt`、`last.pt`、全部配置、划分、归一化参数和测试预测。模型保存后已重新加载最佳模型进行测试，验证了 CUDA checkpoint 的读写。

### 已有 CPU checkpoint 的兼容性

`runs/pipeline_full1/best.pt` 已通过 `predict.py --device cuda` 在 GPU 上完成单样本预测，9 个输出均为有限值，结果保存在 `runs/gpu_prediction_from_cpu.json`。

进一步从旧 CPU 试跑的第 3 轮 checkpoint 恢复，并在 GPU 上完成第 4 轮：

```powershell
uv run --frozen python train.py --device cuda --epochs 4 --batch-size 4 --max-groups 4 --max-steps 3 --max-eval-steps 2 --output runs/gpu_resume_cpu --resume runs/smoke/last.pt
```

该轮训练损失 0.675147，验证损失 9.727751；检查点与测试输出保存在 `runs/gpu_resume_cpu/`。验证覆盖了旧模型、优化器状态和归一化参数的加载。CPU / CUDA 后端及 PyTorch 版本存在差异，不保证训练数值逐位一致。

这次迁移解决的是 GPU 执行与环境复现。原始数据尚未采齐、软件时间关联标志仍需核查等限制继续适用，以上单轮结果不代表最终模型精度。

## 2026-11-07 多架构选择与单音频 token 验证

新增 `mvts/model_standard_audio.py`，通过 `train.py --model model_standard_audio` 选择；原架构通过 `--model model` 选择，未指定时仍默认原架构。新架构采用六通道输入的标准 ResNet-18 stem 和二维下采样，声学特征为 `[B,512,2,1]`，全局平均池化后生成 1 个音频 token，与 1 个视觉 token 合并为 `[B,2,256]`。新架构共 24,306,634 个参数。两种模型文件顶部都有 ASCII 架构图。

`mvts/architectures.py` 按文件名发现和创建模型，checkpoint 保存 `model_name`。评估、预测与恢复训练自动根据该字段选择架构；缺少该字段的旧检查点按 `model.py` 加载。显式选择不匹配的恢复架构会报错，输出目录也会检查已有架构记录。

自动测试结果：**11 passed in 98.31s**。覆盖两种网络的音频特征尺寸、融合 token 数、双分支及注意力参数的有效梯度、可选接触状态、非法模型名，以及两种模型在 CPU/CUDA 上的训练、checkpoint 保存和自动恢复、评估及预测。另模拟删除旧检查点中的架构字段，确认向后兼容，并检查恢复时选错架构会拒绝执行。

真实数据执行：

```powershell
uv run --frozen python train.py --model model_standard_audio --device cuda --epochs 1 --batch-size 8 --output runs/standard_audio_pipeline1
```

扫描 2103 个事件，纳入 1649 条记录、400 个位置组，未发现坏记录。训练/验证/测试分别为 1154/241/254 条，未设置 `max-*` 限制。

| 指标 | 数值 |
| --- | ---: |
| train_loss | 0.421370 |
| val_loss | 0.453651 |
| test_loss | 0.282149 |
| 测试 x MAE，mm | 1.766683 |
| 测试 y MAE，mm | 1.404313 |
| 测试 delta_z MAE，mm | 0.531488 |
| 测试 Fx MAE，N | 0.030830 |
| 测试 Fy MAE，N | 0.062027 |
| 测试 Fz MAE，N | 0.825287 |
| 测试 Mx MAE，N·m | 0.081433 |
| 测试 My MAE，N·m | 0.088947 |
| 测试 Mz MAE，N·m | 0.006180 |

训练与验证一轮约 18.1 秒，结果及所选架构名称保存在 `runs/standard_audio_pipeline1/`。这些结果证明新架构的完整训练流程可用；单轮、单次划分尚不足以判断新架构是否稳定优于原架构。
