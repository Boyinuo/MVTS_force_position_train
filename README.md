# MVTS 视觉声学触觉训练

根据提供的网络图、`阶梯式与动态法向按压采集方案.docx` 和 `data/` 事件导出，实现位置与六维力联合回归。当前用途是用尚未采齐的数据验证训练 pipeline。默认两个 ResNet 从随机权重开始，不自动下载预训练模型。

## 环境与迁移

项目独立环境为 `.venv/`，Python 版本记录在 `.python-version`，直接依赖在 `pyproject.toml`，精确版本和下载校验在 `uv.lock`。无需激活环境，统一通过 uv 运行：

```powershell
uv sync --locked --link-mode copy
uv run --frozen python train.py --help
uv run --frozen pytest -q
```

迁移时复制代码、`pyproject.toml`、`uv.lock`、`.python-version` 和 `data/`，在目标机器运行 `uv sync --locked`；不要直接搬运 `.venv`。部署时不需要测试依赖可用 `uv sync --locked --no-dev`。`--link-mode copy` 适用于 uv 缓存与项目不在同一盘符的 Windows 环境。

默认环境现已切换为官方 **CUDA 12.6 版 PyTorch**，由 `pyproject.toml` 的 `pytorch-cu126` 索引和 `uv.lock` 固定。目标平台为带兼容 NVIDIA 驱动的 Windows / Linux；本机 RTX 3060 Laptop GPU（6 GB）、驱动 572.83。安装的是 CUDA 12.6 runtime 的 wheel，不要求另外安装完整 CUDA Toolkit。版本选择依据 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/) 与 [NVIDIA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

默认 `--device auto` 会在 CUDA 可用时使用 GPU，迁移到无兼容 GPU 的机器时回退 CPU；希望保证此次训练确实使用 GPU，可显式指定 `--device cuda`，不可用时直接报错。CUDA wheel 也支持 `--device cpu`。训练配置和汇总会记录实际设备、GPU 名称与 CUDA runtime 版本。

```powershell
# 检查安装和 GPU 检测
uv run --frozen python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 强制使用 GPU，建议 6 GB 显存先从 batch-size 8 开始
uv run --frozen python train.py --device cuda --epochs 30 --batch-size 8 --output runs/gpu_train
```

若迁移目标需要纯 CPU wheel，修改索引 URL 为 `https://download.pytorch.org/whl/cpu`，执行 `uv lock --upgrade-package torch`、`uv sync --locked` 并保存新的锁文件。仅使用 `--device cpu` 不会替换已安装的 wheel。CUDA 版安装包较大，首次同步需要等待下载。

## 先跑通真实数据

在仓库根目录执行：

训练数据按数据集放在 `data/` 下的同名子文件夹中。现有两个采集会话已移至 `data/data_sensor_normal/`：

```text
data/
└── data_sensor_normal/
    ├── normal_20261104_180418_947105/
    └── normal_20261104_200613_851364/
```

`train.py --dataset data_sensor_normal` 只读取该子文件夹中的数据；不指定时默认使用 `data_sensor_normal`。新增数据集时，在 `data/` 下创建另一个文件夹（例如 `data_sensor_other`），将采集会话放入其中，然后使用 `--dataset data_sensor_other`。参数值必须与子文件夹名称一致。仍可用 `--data-root PATH` 直接指定数据目录，与 `--dataset` 互斥。选中的名称和实际数据路径会保存在 `config.json` 的 `training` 中。不同数据集建议使用不同的 `--output`。

```powershell
# 扫描所选数据集的全部样本并输出数据检查报告，不训练
uv run --frozen python train.py --dataset data_sensor_normal --audit-only --output runs/audit

# 使用真实数据的小规模试跑，保留完整网络和 224x224 输入
uv run --frozen python train.py --epochs 2 --batch-size 4 --max-groups 4 --max-steps 3 --max-eval-steps 2 --output runs/smoke

# 数据齐全后，全量训练；取消所有 max-* 限制
uv run --frozen python train.py --dataset data_sensor_normal --epochs 30 --batch-size 8 --output runs/full
```

`--max-groups` 在训练、验证、测试各自随机选取至多指定数量的完整位置组，不会把同一位置拆开。`--max-steps` 和 `--max-eval-steps` 分别限制每轮训练及评估的 batch 数；试跑指标仅针对实际处理的样本，不能用来判断泛化精度。`summary.json` 会记录 `smoke_test`、划分大小和实际评估样本数。

## 网络与预处理

现在支持通过 `--model` 选择 `mvts/` 中同名模型文件（参数不带 `.py`）。默认仍使用 `model.py`；每个架构建议使用独立的输出目录。

| 参数 | 模型文件 | 声学主干与池化 | 音频 token / 融合 token |
| --- | --- | --- | --- |
| `--model model` | `mvts/model.py` | Modified ResNet-18，仅频率维下采样/池化 | 12 / 13 |
| `--model model_standard_audio` | `mvts/model_standard_audio.py` | 标准 ResNet-18 下采样，频率和时间全局平均池化 | 1 / 2 |

```powershell
# 原模型：1 个视觉 token + 12 个音频 token
uv run --frozen python train.py --model model --device cuda --epochs 30 --output runs/baseline_temporal

# 新模型：1 个视觉 token + 1 个音频 token
uv run --frozen python train.py --model model_standard_audio --device cuda --epochs 30 --output runs/standard_audio

# 新模型的真实数据小规模 pipeline 检查
uv run --frozen python train.py --model model_standard_audio --device cuda --epochs 1 --batch-size 4 --max-groups 4 --max-steps 3 --max-eval-steps 2 --output runs/standard_audio_smoke
```

新模型保留六通道 Log-Mel 输入，标准 ResNet-18 首层输入通道数设为 6，使用 7×7、stride=(2,2) 的卷积和 MaxPool，各残差 stage 按标准方式同时下采样频率与时间。输入 `[B,6,64,12]` 得到 `[B,512,2,1]`，全局平均池化和 Linear(512,256) 后形成 `[B,1,256]` 音频 token。配置中的 `audio_frames=12` 表示输入谱图帧数，输出音频 token 数由模型的 `audio_token_count=1` 指定。视觉分支、2 层 Transformer、注意力池化和双回归头采用相同设计。每个模型文件顶部均附有字符架构图。

架构名记录在 `config.json`、`summary.json` 和 checkpoint 的 `model_name` 中。`evaluate.py`、`predict.py` 以及未显式指定 `--model` 的 `train.py --resume` 会根据 checkpoint 自动构建对应架构；旧 checkpoint 缺少架构名时按 `model.py` 加载。恢复训练时显式指定其他架构会报错。两种架构的权重形状不同，对比实验应各自启动新训练。

后续增加架构时，在 `mvts/` 下新增 `model_*.py`，暴露 `ModelConfig` 和 `TactileFusionNet`（可像新模型一样继承已有类）。模型配置需要支持 `use_contact`、`to_dict()`，模型提供 `config` 属性，接受 `forward(image, audio, starts_in_contact=None)`，返回 `{"position": [B,3], "wrench": [B,6]}`。`mvts/architectures.py` 自动发现文件，`train.py --help` 会列出可选文件名，训练入口无需逐一添加分支。保持相同的 seed、数据划分、训练轮数等设置有助于比较架构，模型优劣应通过验证集选择后再做最终测试。

以下表格描述默认的 `model.py` 架构：

| 分支 | 输入与计算 | 输出 |
| --- | --- | --- |
| 视觉 | PNG → RGB → 完整图像缩放至 224×224 → ImageNet 固定归一化 → 标准 ResNet-18 → 全局平均池化 → Linear | 1×256 |
| 声学 | uint16 ADC 2000×6 → 除以 65535 → 每通道去直流 → Hann STFT，n_fft=512，hop=128，center=False → 64 个 Slaney Mel 滤波器 → log10 功率 | 6×64×12 |
| 声学骨干 | 3×3 stem，stride=(1,1)，无 maxpool；四个 stage 各 2 个 BasicBlock，stride 依次 (1,1)、(2,1)、(2,1)、(2,1) | 512×8×12 |
| 声学 token | 只对频率维平均池化，再 Linear(512,256) | 12×256 |
| 融合 | 拼接视觉与声学 token，加入可学习位置/模态编码；2 层 Transformer，8 heads，FFN=1024；可学习注意力池化 | 256 |
| 输出 | 两个独立 MLP head | 位置 (x,y,Δz) 3 维；力/力矩 (Fx,Fy,Fz,Mx,My,Mz) 6 维 |

STFT 时间帧数严格为 `floor((2000-512)/128)+1=12`，不补成 13 帧。Mel 使用 Slaney 分段频率映射和面积归一化，避免 50 kHz / 512 点 FFT 下低频滤波器为空。ADC 除数是 uint16 容器的固定尺度，不代表已标定的电压；不对单条音频做峰值或方差归一化，以保留相对幅值信息。声学特征随后使用**训练集**每通道均值和标准差标准化。

实测采集元数据为 **50 kHz**，2000 点约 **40 ms**；图示是 48 kHz、约 41.67 ms。实现使用数据的真实采样率构建 Mel 频率轴，不虚构重采样；维度仍完全相同。若今后导入不同采样率会拒绝混用，需先明确重采样策略。

2000 点和 8000 点原始音频统一通过 `mvts.features.extract_event_audio()` 选取事件前后各 1000 点。优先使用 NPZ 的 `audio_before` / `audio_after` 定位并校验事件边界；旧文件没有这些字段时使用数组中点。原有 2000 点窗口保持完整，4000+4000 点窗口截取 `[3000:5000, :]`。训练集统计量、训练/验证/测试以及 `predict.py` 均调用同一函数，随后生成 `[6,64,12]` 的 Log-Mel；`model_sensor_final.py` 保留 `audio_frames=12`。原始数据文件保持完整，不足前后各 1000 点或窗口元数据不一致时直接报错。50 kHz 下仍需事件后 20 ms 的音频，在线采集到后 1000 点即可计算。

`data_sensor_final` 的采样率和会话质量信息可从 `dataset_summary.json` 读取；原始音频和完整时间戳仍接受审查。`data_report.json` 新增 `audio_source_samples`（纳入样本的原始窗口长度计数）和 `audio_input_window`（实际输入的 1000+1000 点窗口）。例如：

```powershell
uv run --frozen python train.py --dataset data_sensor_final --model model_sensor_final --epochs 30 --output runs/sensor_final
```

不随机裁剪、翻转或旋转图像，避免改变触觉位置与力方向的标签关系。ImageNet 固定归一化只是输入变换，默认视觉骨干没有 ImageNet 预训练权重。

`starts_in_contact` 保留在数据中。默认不输入网络以遵循示意图；希望利用采集方案中的先验状态时添加 `--use-contact`，其 embedding 会加入 13 个 token，不改变两路主干和输出形状。预测时须提供相同语义的上一时刻接触状态。

## 数据与标签约定

训练程序仅在所选的 `data/<dataset>/`（或显式指定的 `--data-root`）中递归发现 `samples.jsonl`，使用其中的相对图像路径和 NPZ 路径，而不是采集电脑上的旧绝对路径。NPZ 只需 2000 点事件窗口，不依赖大型 `microphone_stream.bin`，迁移训练数据时可只复制每个 session 的 `samples.jsonl`、`session.json`、`audio/session.json`、`samples/`、`frames/`。

- 默认仅使用 `valid=true`、`kind=target/limit`、`mode=staircase/dynamic`。悬空背景不参与位置回归，以免把悬空探头坐标视为接触位置。触发限值的数据使用当时的**实测标签**，不拿目标深度代替。
- `x,y` 使用 NPZ 的 `actual_xy_mm`，`Δz` 使用 JSON 的 `actual_depth_mm`。已通过机器人位姿与几何参数核对：该字段已减去 `local_contact_plane_depth_mm`，是相对本点接触面的压入量，不能再减一次。
- 六维标签使用 NPZ 的 `force_wrench`，与 JSON `force.wrench` 交叉核对。保留采集端的去零偏、符号和坐标系，不再翻转 Fz、不旋转力、不做力矩原点平移。用户已确认原始标签单位：位置为 mm，Fx/Fy/Fz 为 N，Mx/My/Mz 为 N·m；直接按这些单位使用数据。
- `image_force_audio_v2` 分组数据的 `adc` 为 `(K,N,6)`、`force_wrench` 为 `(K,6)`。每个事件展开为 K 个样本，共享图像、事件级实测 XY/深度及接触变量。第 i 个样本使用第 i 条力与对应音频，编号后缀为 `/force_00` 等，所有子样本跟随物理位置一起划分。逐条核对 JSON `force_sequence`、力时间戳和音频边界，重叠音频段按各自时间轴审查。
- `data_sensor3_press` 每个事件包含 10 条力，每条力配有前 4000 点音频。使用 `model_contact_concat_sensor_final` 时，各自截取末尾 2000 点，再复用现有 Log-Mel 特征。旧二维数据保留原来的裁窗规则，评估、预测和恢复训练继续使用 checkpoint 保存的特征配置。缺少所需前后采样点时会报错。
- 按**计划位置 (x_mm,y_mm)，保留 6 位小数**分组后，约 70%/15%/15% 分为训练/验证/测试。同一点的不同深度、速度、模式、续采 session 全部留在一个分区。实际测量坐标有噪声，不用它做分组。未来多探头数据没有探头身份元数据时，同坐标也会保守地合并；如需跨探头泛化实验，应另设按探头留出的划分。
- 音频与 9 维标签的归一化仅在训练集拟合。损失为 `position_weight * SmoothL1(position_normalized) + wrench_weight * SmoothL1(wrench_normalized)`，默认两组权重各 1。优化器 AdamW，验证损失平台期降低学习率，梯度裁剪范数 5。
- 测试集仅在训练结束后用验证集最佳 checkpoint 评估。输出逐维原单位 MAE、RMSE，以及始终预测训练均值的 MAE 基线。

`data_report.json` 检查文件、图像解码、ADC 形状、标签有限值、NPZ/JSON 一致性、时标单调性、采样率、同步年龄和音频窗口间隙。坏记录会写入 `rejected` 并阻止训练，检查后可显式使用 `--skip-bad-records`；不会用零值填充缺失模态或标签。

现有会话存在 `valid_for_time_matching=false` / `microphone_timing_valid=false` 等标志，且是软件关联，不是硬件同步。默认保留已导出的有效事件供 pipeline 调试，并记录警告，不能据此认定时间对齐已验证。`--strict-timing` 会对不完整或同步标志未通过的会话报错；用于正式训练前的严格检查。超出音频窗口的 marker 和时间戳间隙也会计数，正式精度实验前需要回查采集时钟与事件导出逻辑。

## 输出、恢复与推理

训练输出目录包含：`data_report.json`、`splits.json`、`config.json`、`normalization.json`、`history.jsonl`、`best.pt`、`last.pt`、`test_metrics.json`、`test_predictions.csv`、`summary.json`。`runs/` 和 `.venv/` 已加入 `.gitignore`；原始 `data/` 不作修改。

训练完成后，终端同时打印 `test_loss`、位置逐分量 `test_position_mae (mm)`（x、y、delta_z）和六维力/力矩逐分量 `test_wrench_mae`（Fx、Fy、Fz、Mx、My、Mz）。MAE 已反归一化，位置单位为 mm，力的每个分量标注 N，力矩的每个分量标注 N*m（即 N·m）；数值与 `test_metrics.json` 中的 `mae` 一致，终端显示到小数点后 6 位。

训练、验证和测试阶段在加载首批数据前打印状态，首批完成后及之后约每 10 秒输出批次进度、累计 loss、样本速度、已用时间和预计剩余时间。`data` 为平均每批数据等待时间，`compute` 为平均每批传输、计算及指标汇总时间。默认 `--workers 0` 在主进程中读取 PNG、NPZ 并提取特征，GPU 会等待数据；可使用 `--workers 2` 并行加载，并根据进度日志比较实际速度。

检查点包含网络配置、特征配置、归一化、优化器、学习率调度器、随机状态和划分 ID/标签。已有目录需通过 `--resume` 继续，或选择新的输出目录。恢复时保留原始参数，并选择相同数据集，`--epochs` 表示总轮数；新增数据应启动新的训练以重新划分和计算统计量。

旧模型名称 `model` 和 `model_standard_audio` 分别映射到 `model_sensor_nomal` 和 `model_standard_audio_sensor_nomal`。未保存模型名称的早期 checkpoint 使用 `model` 架构加载，保留原有权重及特征配置。

分组 NPZ 的 `predict.py` 默认输出全部配对的 `predictions` 列表，每项包含从 0 开始的 `force_index` 和九维 `prediction`。添加 `--force-index 4` 可只预测第 5 组，并返回单条 `prediction`。旧二维 NPZ 继续返回单条 `prediction`。预测只读取图像、音频、窗口元数据及可选接触变量。

```powershell
uv run --frozen python train.py --dataset data_sensor3_press --model model_contact_concat_sensor_final --epochs 30 --output runs/sensor3_contact_concat
```

```powershell
# 接上面 smoke 例子，从第 3 轮继续到第 4 轮
uv run --frozen python train.py --epochs 4 --batch-size 4 --max-groups 4 --max-steps 3 --max-eval-steps 2 --output runs/smoke --resume runs/smoke/last.pt

# 对 checkpoint 保存的整个测试分区重新评估（迁移后仍用当前 data-root 解析文件）
uv run --frozen python evaluate.py --checkpoint runs/smoke/best.pt --data-root data/data_sensor_normal --output runs/evaluation

# 单样本推理：不需要位置或六维力 ground truth，只读取图像、adc 和可选的接触状态
uv run --frozen python predict.py --checkpoint runs/smoke/best.pt --image data/data_sensor_normal/normal_20261104_180418_947105/frames/00000902.png --npz data/data_sensor_normal/normal_20261104_180418_947105/samples/00002_staircase_001_1_target.npz --output runs/prediction.json
```

`best.pt` 按验证损失选择，`last.pt` 用于恢复；恢复迁移时同时保留两个文件。`predict.py` 输出已反归一化的 9 维结果。未来在线部署需提供与训练一致的图像处理、采样率、2000 点窗口和时钟关联；当前窗口含事件后 1000 点，因此不是零延迟推理。

## 代码

- `mvts/model.py`：两路 ResNet-18、Transformer 和回归头。
- `mvts/model_standard_audio.py`：标准下采样的六通道音频 ResNet-18，生成单个音频 token。
- `mvts/architectures.py`：按模型文件名发现、创建和恢复架构。
- `mvts/features.py`：Log-Mel 和图像预处理。
- `mvts/data.py`：数据审查、位置分组、归一化和 Dataset。
- `mvts/engine.py`：训练/评估、指标、检查点读写。
- `train.py` / `evaluate.py` / `predict.py`：命令行入口。
- `tests/`：特征幅值关系、网络梯度、分组防泄漏、坏标签和端到端训练/恢复/预测测试。

参考实现约定：[PyTorch STFT](https://docs.pytorch.org/docs/stable/generated/torch.stft.html)、[ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。实际运行结果见 `TEST_REPORT.md`。
