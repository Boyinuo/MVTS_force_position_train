# MVTS 滑移部署模拟界面

## 训练结束后自动外测与绘图（2026-11-25）

`train.py` 对 `model_triple_head`（含纯视觉）、`model_slip_gated`、`model_contact_gated` 默认开启自动外测。正常训练/续训完成、保存并选定 `best.pt` 后，依次测试 `deployment/test_data/demo_circle`、`demo_line`、`demo_triangle`，终端分别输出 `loss`、`main_loss`、位置/六维力/滑移 MAE。

本次训练输出目录中直接保存六张图：

```text
runs/v6_xxx/
  best.pt
  demo_circle.png       demo_circle.svg
  demo_line.png         demo_line.svg
  demo_triangle.png     demo_triangle.svg
  demo_tests/
    summary.json
    demo_circle/        # metrics.json、predictions.csv/npz、selection.json、绘图 QA
    demo_line/
    demo_triangle/
```

轨迹图沿用双模型对比图的2×5面板、共同坐标范围、网格与配色风格，每套图显示十条真值及**本次 best.pt 的预测**，不额外加载另一个模型。外测沿用本次权重的麦克风选择、归一化、图像处理、因果音频和状态重置；不重新拟合、平滑、配准或积分滑移。

默认外测完整保留三种形状，**不继承训练的 XY 筛选和末端截取**。圆形略超出 Y=-8mm 会被训练筛选规则整条排除，而三角形中途停顿会使首个静止窗口截取只留下第一段。因此外测通过独立参数控制：

```powershell
# 普通训练命令会自动生成三套外测图与指标
uv run train.py --model model_slip_gated --device cuda --output runs/v6_slip_gated

# 如需沿用此前的外测截取口径，显式设置；默认值为0
uv run train.py --model model_slip_gated --demo-trim-stationary-xy-mm 0.029 --output runs/v6_slip_gated_trim_demo

# 可选：关闭外测，或更换包含三个demo子目录的根目录
uv run train.py --model model_triple_head --no-demo-tests --output runs/v6_no_demo
uv run train.py --model model_slip_gated --demo-test-root PATH_TO_TEST_DATA --output runs/v6_custom_demo

# 已有训练可直接补生成，无需重新训练；默认写到best.pt所在目录
uv run -m deployment.demo_evaluation --checkpoint runs/new_model_v6_slip_gated/best.pt --device cuda
```

`loss` 使用该模型训练时的加权标准化 Smooth L1 目标，并包含已启用的粗滑移辅助项/接触分类项；`main_loss` 只包含末端三项回归和无接触零力项，便于比较。MAE 为物理单位；接触门控模型同时保存原始 MAE 和预测有效率，失效预测不会从 MAE 中消失。纯视觉模型与训练一样每个子事件更新 GRU，缓存同帧图像，不改为15Hz递推。

三套外测只在全部训练结束后运行，绝不用于学习率调度、权重更新或最佳轮次选择。`--audit-only` 不执行外测。依赖已纳入 `pyproject.toml`；`uv run` 会准备 Matplotlib，无需安装绘图插件。

## 外部圆形测试集的双模型对比

直接比较 `runs` 中保存的两个 `best.pt`，不替换模型目录中的权重：

```powershell
uv run -m deployment.compare_models --data-root deployment/test_data/demo_circle --baseline-checkpoint runs/v6_triple_head_slide_polyline_press_no_mic5_trim0p029_xyfilter/best.pt --candidate-checkpoint runs/new_model_v6_slip_gated/best.pt --trim-stationary-xy-mm 0.029 --no-filter-xy-range --device cuda --output runs/v6_demo_circle_compare_trim0p029
```

两边使用完全相同的事件、图像、轨迹重置与原始采集时间，禁用第5路麦克风；沿用各自 checkpoint 的训练统计量，并核对训练划分、共同参数与归一化数组一致。输出包含总体/逐轨迹指标、物理量预测 CSV/NPZ、筛选记录和权重哈希。推理统一使用 float32 并关闭 TF32；不执行平滑、坐标配准、滑移积分或重新训练。

本次圆形测试仅按用户选择关闭**测试时**的 XY 筛选：十条原轨迹的最小 Y 均低于固定下界 −8mm，严格 `--filter-xy-range` 会删除全部轨迹。保留 `0.029mm` 末端截取后为115个窗口、1150个事件。程序会同时保存严格 XY 规则的空集审计，不修改训练筛选规则或原数据。

## 直接查看缓存中的轨迹真值（2026-11-24 新增）

使用同一个 `deployment_viewer.py`，增加 `--ground-truth-only` 即可直接打开 `.cache/data_final_v6_slide_contact`。此模式不要求 `--model`，不加载权重、不执行主模型或纯视觉推理，默认不生成结果文件，也不需要转换部署数据包或复制缓存。

```powershell
# 直接打开 v6 滑移缓存，在界面中选择轨迹
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact

# 启动时指定完整来源/轨迹 ID
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact --sequence sensor3_probe3mm_fast_slide_fullframes_training_20261121/t0128

# 与 train.py 使用相同的0.03mm末端截取规则
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact --trim-stationary-xy-mm 0.03

# 固定 XY 范围筛选，可与末端截取同时使用
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact --filter-xy-range --trim-stationary-xy-mm 0.029

# 无界面读取检查；不加 --output 时不导出文件
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact --sequence sensor3_probe3mm_fast_slide_fullframes_training_20261121/t0128 --headless
```

目录栏下面的“滑移轨迹 ID”下拉框列出当前缓存全部600条3mm轨迹，支持输入检索；选择条目后自动加载并从头回放，也可输入完整 ID 后回车。v6 缓存中的旧4mm来源已剔除。ID 保留来源名，以区分不同来源的同名 `t0128`；命令行只在短 ID 唯一时允许省略来源。加载期间禁用轨迹切换，切换数据目录会重建轨迹列表。此选择框也支持原有部署数据包的多轨迹回放。

`--trim-stationary-xy-mm MM` 在真值查看和模型推理模式均可使用，与训练共用截取实现：计算每个图像窗口内全部子事件位置真值的最大两点 XY 欧氏距离，第一次严格小于阈值时，从该窗口开始剔除整条轨迹的全部后续窗口；忽略 Z，等于阈值保留，后续抖动增大也不恢复。默认 `0` 关闭，省略参数显示数据包中保留的完整轨迹。切换轨迹 ID 后继续应用启动时指定的阈值；界面显示保留/原始窗口数和剔除子事件数。图像、位置、六维力、频谱和播放进度同步截取，不改写缓存。如果首个窗口即低于阈值，界面会提示没有保留数据，可降低阈值或设为 `0` 后重新启动。此参数也支持已有部署数据包，按原始图像更新边界整窗分组。

推理模式先按测试集 XY 真值选择保留窗口，再执行主模型及纯视觉对照推理和结果保存；被剔除的事件不进入网络，也不更新其状态。默认读取 `deployment/test_data/v6_slide`，该包已按 `0.029 mm` 截取过，再指定同一阈值通常不会进一步减少数据；提高阈值可继续截取，降低阈值不能恢复导出时已删除的窗口。

```powershell
uv run deployment/deployment_viewer.py --model model_triple_head --trim-stationary-xy-mm 0.029
```

`--filter-xy-range` 是独立开关，固定保留范围为 **x∈[-10,10] mm、y∈[-8,10] mm**，包含边界，不需要在命令行传入具体数值。它检查每条完整滑移/折线轨迹的全部 XY 真值，任意一点越界就剔除整条轨迹，不保留局部片段；判断发生在末端截取之前，因此即使越界点位于稍后会被截去的末端，也会剔除该轨迹。开启后 ID 下拉框仅列出通过筛选的轨迹，切换 ID 继续应用开关。当前直线滑移缓存保留206/600条轨迹，折线缓存保留154/597条；不改变缓存内容，也不推理。默认关闭，省略或使用 `--no-filter-xy-range` 可关闭；此查看器开关仅用于真值模式。

训练端使用同名开关和相同实现，同时对按压数据逐独立样本筛选，XY 越界就去除该按压样本（包括无接触背景）。当前按压缓存保留11,719/13,159个样本。`train.py` 先按完整分组划分，再进行整条轨迹/独立按压的范围筛选、末端截取，最后拟合训练集归一化。续训和 `evaluate.py` 自动继承 checkpoint 的范围开关；改变开关需要新训练。

真值模式显示缓存图像、全部子事件的 XY 轨迹、X/Y/Δz 时间曲线、六维力真值和六路未标准化 Log-Mel，沿用播放/暂停、倍速、拖动、冻结和缩放功能。滑移增量真值直接显示缓存标签，不通过累加增量构造位置；缺失窗口造成的轨迹中断不会被连成一条线。界面不显示模型预测、预测误差、接触概率或推理耗时。回放仍使用 `--hz` 指定的名义时间轴，原始采集时间戳保持不变。

该模式同样支持 v4/v5 格式的其他滑移/折线缓存和已有部署数据包。显式增加 `--output DIRECTORY` 时才导出 `ground_truth.csv`、`ground_truth.npz` 和 `summary.json`，不会生成预测文件。以下章节描述原有模型推理模式。

---

先加载选定滑移轨迹的全部事件，按原顺序执行单事件推理，完整保留视觉缓存、GRU 状态和模型自身的接触状态。同时默认运行独立的 15 Hz 纯视觉对照模型。两条流的推理与结果保存全部完成后，界面才开始播放。程序不读取相机或麦克风设备。

## 环境与启动

在项目根目录执行（使用项目虚拟环境）：

```powershell
uv sync --frozen
uv pip install -r deployment/requirements.txt

# 必须选择模型；自动加载所选目录内的 best.pt 和 best_image_only.pt
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_triple_head

# 仅运行原 150 Hz 模型（不要求 best_image_only.pt）
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_contact_gated --no-image-only-comparison

# 指定模型、同格式数据包和滑移序列
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_triple_head.py --data-root deployment/test_data/deployment_test_150hz --sequence slide_t0242 --device cuda
```

**`--model` 必填，不设默认模型。** 程序使用项目注册的架构，自动读取该模型模块所在目录的 `best.pt`，无需也不再支持 `--checkpoint`。例如 `--model model_contact_gated` 对应 `mvts/model_contact_gated/best.pt`。`--model` 接受模型模块名称、`.py` 文件名或 `mvts/<model>/<model>.py` 路径；模型需支持 `step`、视觉缓存和连续状态接口。缺少权重或 checkpoint 的 `model_name` 不匹配时会明确报错，不会忽略缺失参数或用随机权重补全。

本次演示使用已确认的 `mvts/model_contact_gated/best.pt`：`model_contact_gated`，第 24 轮，六路麦克风均启用。

新增纯视觉对照权重为 `mvts/model_triple_head/best_image_only.pt`（第 15 轮，`image_only=True`）。使用 `--model model_triple_head` 时自动读取，与同目录的主模型 `best.pt` 比较。对照模型采用权重里保存的架构和独立归一化参数；必须是真正的纯视觉网络，程序拒绝用融合模型权重冒充纯视觉模型。默认启用对照，缺少该文件会明确报错；可用 `--no-image-only-comparison` 关闭。

纯视觉对照只传入相机图像，每隔 `1/15 s` 更新一次独立 GRU，不接收音频、GT 或主模型状态，也不会在两次调用之间额外推进隐藏状态。标准数据的图像恰好每 10 个 150 Hz 事件更新一次，因此整条轨迹为主模型 290 次、纯视觉 29 次调用。若修改 `--hz`，纯视觉仍按独立 15 Hz 时间轴调用，采用当时最新可用图像。

紫色方点/连线表示纯视觉预测，在 XY、位置时间曲线和六维力图中同步显示；连线只连接真实调用的预测点，没有插值成 150 Hz 输出。右上显示最近一次纯视觉输出时刻和坐标，以及已播放 15 Hz 采样点对应 GT 的位置误差。可用“显示纯视觉 15 Hz”复选框隐藏/显示曲线，缩放、拖动和重播继续可用。GT 和主模型仍保留全部 150 Hz 数据，界面目标刷新率仍为 60 Hz。

注意：该纯视觉 checkpoint 沿用训练时的模型与权重，但本界面对照按要求把递归调用节奏设为 15 Hz。如果它训练时按 150 Hz 更新 GRU，这种调用节奏变化可能影响误差；对照结果应按本次 15 Hz 部署方式解释。

`deployment/test_data/deployment_test_150hz` 仅保存完整滑移序列 `slide_t0242`：290 个子事件、29 次图像更新，150 Hz 下回放时长约 1.93 秒。已移除 10 个独立按压样本及其图像，调用索引和回放调度从 0 开始；滑移输入、标签和原始采集时间保持原样。

窗口顶部新增“测试数据目录”选择栏：可直接输入目录或点击“浏览…”，再点击“加载并推理”。选择包含 `manifest.json` 和 `events.jsonl` 的部署数据包目录即可。切换目录时自动选择新包的第一条滑移序列，重新加载网络并重置序列状态，完成推理后开始播放。推理期间禁用目录切换和播放操作；后续加载的结果保存到初始输出目录下独立的 `reload_*` 子目录。

`slide_t0232_raw_and_test_subset.zip` 已转换为 `deployment/test_data/deployment_test_150hz_t0232`，包含包内提供的全部 8 个连续测试窗口（order 2–9）、80 个子事件和 8 张图像，按 150 Hz 调度。仅导出原有测试子集，未将 ZIP 中其他原始窗口加入测试集。ADC 从原始数据恢复并重算 Log-Mel 核对，保留源缓存特征、标签和原始采集时间。标准化参数沿用参考包，界面仍使用所选 checkpoint 的参数。

```powershell
# 从原 ZIP 重新转换到新目录（目标目录须不存在）
.venv\Scripts\python.exe -m deployment.prepare_slide_subset --output deployment/test_data/t0232_rebuilt

# 直接打开 t0232，或启动后在目录选择栏中切换
.venv\Scripts\python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/deployment_test_150hz_t0232
```

`deployment/test_data/deployment_test_150hz_t0128` 保存红框内最长完整轨迹的近似候选 `slide_t0128`，来源为 `.cache/data_final_v4_faster_trim_xy_0p05mm`：order 2–5，共 4 张图像、40 个事件，实际事件跨度约 0.259 秒、XY 路径长度约 12.746 mm。保留全部六路音频、原图、标签与时间戳，未按红框裁剪输入。ADC 从原始轨迹恢复，重算 Log-Mel 与缓存完全一致。包内标准化参数来自导出时的三头模型 checkpoint；界面仍自动使用所选 checkpoint 自身参数。该轨迹属于当前三头模型的训练划分，用于预测调用和演示，不作为独立测试集。

```powershell
# 打开 t0128；增加 --headless 可只预测并保存结果
.venv\Scripts\python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/deployment_test_150hz_t0128 --sequence slide_t0128

# 从截断缓存重新导出到新目录（目标目录须不存在）
.venv\Scripts\python.exe -m deployment.prepare_cached_slide --trajectory t0128 --output deployment/test_data/t0128_rebuilt
```

包内 `reader.py` 可独立读取网络输入，调用方法见该目录的 `README.md`。导出后已通过 40 次主模型推理和 4 次纯视觉对照推理。

同一条 `t0128` 在未截断缓存 `.cache/data_final_v4_faster` 中的全部有效窗口另存为 `deployment/test_data/deployment_test_150hz_t0128_faster`：order 2–11，共 10 张图像、100 个事件，实际事件跨度约 0.655 秒。前 40 个事件与上述截断包完全一致，另保留末尾 60 个事件；原始预处理已排除的区间没有补入。已核对源缓存、原始 ADC/Log-Mel、图像和时间戳，并完成 100 次主模型预测及 10 次纯视觉对照预测。

```powershell
# 打开未截断缓存中的 t0128
.venv\Scripts\python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/deployment_test_150hz_t0128_faster --sequence slide_t0128

# 从未截断缓存重新导出到新目录
.venv\Scripts\python.exe -m deployment.prepare_cached_slide --cache .cache/data_final_v4_faster --trajectory t0128 --output deployment/test_data/t0128_faster_rebuilt
```

## 界面与时间含义

- 左上：相机原始 RGB 图像，224×224，按事件的 `image_update` 换帧；窗口拉伸、缩小或最大化时自动按显示区域缩放，保持宽高比，暂停或回放结束后同样生效。
- 右上：每个已播放事件的预测位置显示青色圆点，实测位置显示橙色叉点，并保留连线；当前点用更大标记突出显示。滚轮缩放、左键拖动平移、右键拖动缩放，播放更新时保留缩放范围。可点击“适应已播轨迹”或“重置视图”。标记大小固定，重叠点可放大检查；可切换 X、Y、Δz 时间曲线。
- 左下：Fx、Fy、Fz、Mx、My、Mz，使用反标准化后的物理量。
- 右下：当前事件之前 80 ms 的六路原始训练 Log-Mel，固定色阶 `[-10,-2.5]`，横轴 ms，纵轴 Mel bin。训练中禁用的麦克风会明确标记，但仍可查看该路数据。
- 底部：播放/暂停、重新回放、清空 XY 历史、冻结/解除冻结、0.25×–2× 速度、拖动进度。

`--hz 150` 定义事件的部署回放时间轴，不改变原始特征、标签或采集时间。默认 `--render-fps 60`，界面以 60 Hz 为目标刷新，实际帧率受绘制耗时和系统调度影响。每次刷新把所有已到期的事件加入历史曲线，不要求屏幕每秒绘制 150 帧。刷新延迟时会追上绝对时间轴，期间的曲线点不丢失；图像和频谱显示最近到期的事件。回放结束保留最后一帧。

预计算能把网络推理、文件读取和图像解码移出播放阶段，减少播放卡顿；总推理计算量不会减少，启动等待会增加。顶部的 step 均值/P95 是真实预计算阶段的单事件耗时（包含输入传输、网络和输出拷回 CPU；不含文件读取，不预热，保留第一步开销），与 UI 刷新 FPS 分开。该程序不能单凭回放达到 150 Hz 就证明在线部署能达到 150 Hz。

接触门控模型使用自身预测接触状态与有效标志；无效位置不绘制为原点。三头模型没有接触分类头，界面明确显示这一点，不用实测 Fz 构造预测接触状态。所选滑移数据的真实接触依据采集协议均为接触。位置误差按有效预测点统计，并报告覆盖率和漏判数；三头模型不具备接触漏判评估能力。XY 轨迹直接使用位置头输出，不把滑移增量累加冒充位置。

始终从 NPZ 的未标准化 `audio` 重做 checkpoint 对应的音频标准化，不直接使用数据包预存的其他模型 `audio_input`。标签仅供显示和对照，不进入模型调用。权重特征配置必须与数据包一致。

## 导出与检查

结果自动保存到 `deployment/results/日期_时间/`，也可通过 `--output` 指定。包含 `summary.json`、`predictions.csv`、`predictions.npz`：完整物理单位预测、实测值、有效标志、接触概率、逐事件耗时及原始采集时间。预测点对应 `events.jsonl` 的 `sequence_index`。启用对照时新增 `image_only_predictions.csv`（每个 15 Hz 调用一行），NPZ 新增 `image_only_*` 数组，summary 的 `image_only` 字段记录独立耗时、误差及主模型在同一组 15 Hz 时刻上的位置误差。

```powershell
# 只预计算并导出，不显示窗口
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_triple_head --headless

# 最后一帧截图后退出（仍然先完整推理）
.venv\Scripts\python.exe deployment\deployment_viewer.py --model model_triple_head --screenshot deployment/results/demo.png

# 回归检查：轨迹选择、连续推理/缓存、物理量、结果导出、回放/拖动/冻结
.venv\Scripts\python.exe -m pytest deployment/test_replay.py -q
```

`--random-weights` 仅用于明确要求的界面测试，仍须指定 `--model`，界面会显示红色“随机未训练权重”，不能用于评估模型。正常流程始终加载训练权重。
