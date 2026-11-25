# 滑移轨迹真值查看（2026-11-24 更新）

现有部署界面增加 `--ground-truth-only` 模式，可直接打开 v6 缓存，不需要模型权重、不执行推理、默认不导出文件。界面的“滑移轨迹 ID”下拉框支持检索和切换全部600条轨迹，使用完整来源名区分重复短 ID。

```powershell
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact

# 按训练时相同的0.03mm规则截取，切换轨迹时继续应用阈值
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact --trim-stationary-xy-mm 0.03

# 固定区域筛选：整条轨迹任一点越界就剔除整条，可与末端截取叠加
.venv\Scripts\python.exe deployment/deployment_viewer.py --ground-truth-only --data-root .cache/data_final_v6_slide_contact --filter-xy-range --trim-stationary-xy-mm 0.029
```

显示图像、XY/位置时间曲线、六维力真值和频谱，并支持回放、拖动与缩放。`--trim-stationary-xy-mm` 与训练共用实现：首次窗口内 XY 最大两点距离严格小于阈值时，剔除该窗口及之后全部窗口；省略或设为 `0` 关闭。`--filter-xy-range` 在末端截取前检查完整轨迹，只要任意 XY 真值超出固定的 x∈[-10,10]、y∈[-8,10] mm（边界包含），就去除整条轨迹，ID 列表仅显示通过筛选的轨迹。两个开关只影响加载结果，不改写缓存；区域开关默认关闭。也支持同格式的折线缓存。完整参数和指定轨迹的示例见 [deployment/README.md](deployment/README.md)。

---

# v6 分目录缓存、多目录训练与末端截取（2026-11-23 更新）

`data/data_final` 下的三个目录已分别预处理为以下缓存。**v6 是本次数据集的命名，内部仍为 v5 接触监督格式**（`version=5`、`schema=contact_gated_v1`），`model_triple_head` 和 `model_contact_gated` 均可直接使用。缓存保存完整有效窗口，不预先按运动阈值截断。

| 原始目录 | 缓存目录 | 窗口数 | 子事件数 |
| --- | --- | ---: | ---: |
| `data/data_final/data_final_press` | `.cache/data_final_v6_press_contact` | 13,159 | 13,159 |
| `data/data_final/data_final_slide` | `.cache/data_final_v6_slide_contact` | 10,059 | 100,590 |
| `data/data_final/data_final_polyline` | `.cache/data_final_v6_polyline_contact` | 9,816 | 98,160 |

按压保留六套来源，其中背景4,783个、接触8,376个。滑移仅包含两批3mm直线数据各300条轨迹；折线包含597条轨迹。**2026-11-24 已从 v6 滑移缓存剔除 `slide_fullframes_formal_4mm_20261118_233432_824` 的500条轨迹、12,599个窗口和125,990个子事件及其文件。** 本次来源剔除仅作用于 v6 滑移缓存。无效/裁剪区间、因果音频历史不足的区间及无法提供十个不同力采样的短区间按原规则排除，清单见各缓存的 `report.json`；全量验收结果见 `verification.json`。

已有缓存可直接训练。仅需从原始数据重新生成时执行以下命令，`--output` 必须是尚不存在的目录，不要对现有缓存直接重跑：

```powershell
uv run preprocess.py --contact-aware --data-root data/data_final/data_final_press --output .cache/data_final_v6_press_contact
uv run preprocess.py --contact-aware --data-root data/data_final/data_final_slide --slide-source sensor3_probe3mm_fast_slide_fullframes_training_20261121 --slide-source sensor3_probe3mm_fast_straight_300trials_20261122 --output .cache/data_final_v6_slide_contact
uv run preprocess.py --contact-aware --data-root data/data_final/data_final_polyline --output .cache/data_final_v6_polyline_contact

# 单份缓存独立验收，不需要 v4 baseline
uv run verify_contact_cache.py --prepared-root .cache/data_final_v6_slide_contact
```

`--prepared-root` 支持一次传入多个目录，也支持重复指定该参数。加载时合并数据，按完整按压位置/滑移轨迹分组划分，然后只用训练划分拟合统一归一化；发现重复目录或重叠样本 ID 会报错。两个模型可在相同目录组合上分别训练：

```powershell
$datasets = @(
  ".cache/data_final_v6_press_contact",
  ".cache/data_final_v6_slide_contact",
  ".cache/data_final_v6_polyline_contact"
)

# 默认不截取，使用三份缓存的完整有效窗口
uv run train.py --model model_triple_head --prepared-root $datasets --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/triple_head_v6
uv run train.py --model model_contact_gated --prepared-root $datasets --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/contact_gated_v6

# 需要末端截取时增加该参数，并使用独立的新训练目录
uv run train.py --model model_triple_head --prepared-root $datasets --trim-stationary-xy-mm 0.03 --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/triple_head_v6_trim_0p03mm
uv run train.py --model model_contact_gated --prepared-root $datasets --trim-stationary-xy-mm 0.03 --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/contact_gated_v6_trim_0p03mm
```

`--trim-stationary-xy-mm MM` 的规则是：对每条滑移/折线轨迹，计算每个图像窗口内10个子事件位置真值的**最大两点 XY 欧氏距离**，忽略 Z；第一次严格小于阈值时，剔除该窗口及这条轨迹后续的全部窗口，后续抖动增大也不恢复。等于阈值保留，全部按压样本保留。新训练默认关闭，也可显式设置 `--trim-stationary-xy-mm 0`。

训练支持同名 `--filter-xy-range` 开关，不需要配置数值，固定范围为 **x∈[-10,10] mm、y∈[-8,10] mm**，包含边界。滑移和折线只要完整轨迹中任意一点越界，就去除整条轨迹；按压则逐独立样本筛选，越界就去除该样本，包括无接触背景。先按完整分组划分，再执行区域筛选、末端截取，最后仅用保留的训练数据拟合归一化。区域判断先于末端截取，因此原轨迹末端的越界点也会触发整条剔除。缓存不变，`data_report.json` 的 `xy_range_selection` 记录各划分的剔除统计。

```powershell
uv run train.py --model model_triple_head --prepared-root $datasets --filter-xy-range --trim-stationary-xy-mm 0.029 --device cuda --output runs/v6_triple_head_xy_range
uv run train.py --model model_contact_gated --prepared-root $datasets --filter-xy-range --trim-stationary-xy-mm 0.029 --device cuda --output runs/v6_contact_gated_xy_range
```

当前区域筛选保留11,719个按压样本、206条直线滑移轨迹（3,202个窗口）和154条折线轨迹（2,457个窗口）。默认关闭，可省略或显式使用 `--no-filter-xy-range`；续训和评估会自动继承 checkpoint 中保存的开关，改变开关应新开训练目录。

**截取不需要额外预处理或派生缓存。** 它直接使用现有 manifest 中的位置真值，只在内存中选择窗口，不重算或复制图像、Log-Mel，不改写 `.cache`。先按完整分组划分，再分别截取，最后拟合保留训练数据的归一化，保证阈值变化不会让存活轨迹跨训练/验证/测试划分移动。运行输出中的 `data_report.json` 记录各划分的保留/删除数量及每条轨迹的截断序号。

0.03mm 阈值下，当前滑移缓存保留4,093个窗口、全部600条轨迹，折线保留5,262个窗口，按压仍为13,159个窗口。该参数按“首次低于阈值”判断，不会自动区分起始静止和末端静止；如果某条轨迹的首个有效窗口就低于阈值，会整条剔除。单独滑移/折线数据只有接触正样本，接触模型允许加载并提示单类别限制；需要学习接触/无接触分类时应与按压背景联合训练。

续训和独立评估自动继承 checkpoint 中保存的缓存目录组合及截取阈值；改变阈值或实际训练样本应开始新训练。续训的 batch size 等原有训练选项仍须与 checkpoint 一致；数据迁移后，评估可显式传入多个新的缓存路径。

```powershell
# 接着上面的三头截取训练继续到40轮；不必重传缓存列表或截取阈值
uv run train.py --resume runs/triple_head_v6_trim_0p03mm/last.pt --device cuda --epochs 40 --batch-size 8 --workers 10 --output runs/triple_head_v6_trim_0p03mm

uv run evaluate.py --checkpoint runs/triple_head_v6_trim_0p03mm/best.pt --device cuda --output runs/triple_head_v6_trim_0p03mm/evaluation
uv run evaluate.py --checkpoint runs/contact_gated_v6_trim_0p03mm/best.pt --device cuda --output runs/contact_gated_v6_trim_0p03mm/evaluation
```

---

# 两个模型共用 v5 数据集（2026-11-23 更新）

`model_triple_head` 和 `model_contact_gated` 现在都支持 `.cache/data_final_v5_faster_contact`，无需重新预处理。三头模型也继续支持 v4 缓存。两者使用相同的分组划分规则、训练集音频统计和有效回归标签统计，背景实测值不会混入回归归一化。

三头模型在 v5 上使用 `target_valid` 掩码计算位置、六维力、滑移的 Smooth L1；无接触背景只通过物理零六维力损失监督力头，权重为 `--zero-wrench-weight`（默认1）。接触标签只用于训练监督和评估，不作为网络输入。三头模型仍输出三个回归头；四头模型继续预测接触并进行滞回门控。

```powershell
# 在同一份 v5 缓存上分别开始新训练，使用独立输出目录
uv run train.py --model model_triple_head --prepared-root .cache/data_final_v5_faster_contact --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/triple_head_v5
uv run train.py --model model_contact_gated --prepared-root .cache/data_final_v5_faster_contact --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/contact_gated_v5

# 评估默认使用 checkpoint 中记录的训练缓存，也可用 --prepared-root 指定迁移后的路径
uv run evaluate.py --checkpoint runs/triple_head_v5/best.pt --device cuda --output runs/triple_head_v5/evaluation
uv run evaluate.py --checkpoint runs/contact_gated_v5/best.pt --device cuda --output runs/contact_gated_v5/evaluation
```

三头模型的 `raw_mae/raw_rmse` 只统计有效实测标签；`mae/rmse` 还将无接触六维力与物理零比较，背景位置/滑移仍排除。另报 `no_contact_wrench_mae`、`target_valid_counts`，v5 预测 CSV 增加 `label_valid_*` 列。两个模型均校验 checkpoint 中的监督摘要，数据或掩码变化需要新训练；从 v4 切换到完整 v5 时应使用新的输出目录，不沿用 v4 的 `--resume`。

---

# 接触感知四预测头（2026-11-21 新增）

新增模型 `mvts/model_contact_gated/model_contact_gated.py`，通过 `--model model_contact_gated` 选择；原三头模型及其训练命令继续保留。共享视觉/声学编码、联合注意力、GRU(128,256) 后，四个 MLP 分别输出位置3维、六维力6维、滑移增量2维和接触 logit 1维。没有额外的不确定性头、动力学专家或真值接触输入。

接触真值按已确定的采集协议生成：**有效按压 background=0，按压 target/limit=1，快速滑移的全部子事件=1**。不使用 `starts_in_contact`，也不按力阈值重新推翻这些标签。原始实测回归标签保留；background 的位置、六维力和滑移常规回归均设为无效，其中六维力另由无接触零力 loss 监督。按压接触样本继续使用已知的零切向运动标签，滑移沿用实测 XY 增量。

新缓存版本为5，建议目录 `.cache/data_final_v5_faster_contact`。它从 `data_final_v4_faster` 所用的原始来源重新计算图像和 Log-Mel，保留全部六套按压和300条3mm快速滑移轨迹，加入此前排除的按压背景；旧4mm滑移仍排除。`--selection-from-cache` **只读取旧缓存的来源选择，不复制其特征**。原始数据、旧缓存和旧 checkpoint 不覆盖。

完整缓存已生成：**18,203个窗口、63,599个子事件**，其中无接触背景4,783个、接触事件58,816个。按压为13,159个单步窗口（原8,376个接触样本+背景），滑移为5,044个十步窗口、50,440个子事件。拒绝记录0条。默认划分的训练/验证/测试事件数为44,175 / 9,635 / 9,789；对应背景数3,349 / 714 / 720。

全量验收已通过：原13,420个窗口全部保留，原有回归标签不变，重新计算的原有 Log-Mel 与旧缓存比较的最大绝对差为0，全部事件的因果时序及接触监督掩码正确。完整结果保存在新缓存 `verification.json`。

```powershell
# 完整重新预处理；output 必须是尚不存在的新目录
uv run preprocess.py --contact-aware --selection-from-cache .cache/data_final_v4_faster --output .cache/data_final_v5_faster_contact

# 对生成的缓存检查标签、掩码、因果时序及原有样本/特征的一致性
uv run verify_contact_cache.py --prepared-root .cache/data_final_v5_faster_contact --baseline .cache/data_final_v4_faster --compare-audio

# 启动独立的新训练（不要 resume 三头模型）
uv run train.py --model model_contact_gated --prepared-root .cache/data_final_v5_faster_contact --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/contact_gated_faster

# 评估新 checkpoint
uv run evaluate.py --checkpoint runs/contact_gated_faster/best.pt --prepared-root .cache/data_final_v5_faster_contact --device cuda --output runs/contact_gated_faster/evaluation
```

损失包括三项带真值有效性掩码的 Smooth L1、接触 BCE 和无接触物理零六维力损失；相应权重为 `--position-weight`、`--wrench-weight`、`--slip-weight`、`--contact-weight`、`--zero-wrench-weight`，默认均为1。零力 loss 使用**门控前的反归一化预测**，按各力/力矩维度的训练集标准差缩放；掩码来自接触真值，不能靠预测无接触逃避回归。归一化只拟合训练划分，位置等回归统计不包含无效背景值。BCE 类别权重由训练窗口类别组成确定，与1步按压/10步滑移分别取均值的批次训练方式一致；事件类别计数另行记录。

推理的双阈值滞回默认 `off=.4, initial=.5, on=.6`，可用 `--contact-off-threshold` / `--contact-on-threshold` 调整两端阈值（须满足 off < .5 < on）。状态比较使用**预测**的上一状态，四种转移为 `0=00`、`1=01`、`2=10`、`3=11`；`-1` 表示缺少连续历史。无接触时物理六维力严格为0，接触位置/滑移用数值0占位并返回 `position_valid` / `slip_valid`；不能把占位当物理真值。建立接触时初始化滑移参考，首步或缺历史时滑移无效。**接触切换不清空 GRU**；新轨迹/数据中断则重置两个状态。稀疏按压事件仍相互独立，不把它们拼成伪150Hz序列；滑移记录首步已经接触时不会被标成建立事件。

程序接口：`forward()` / `forward_sequence()` 返回标准化空间的原始三项回归、`contact_logits` 和 `contact_probability`。调用 `model.set_normalization(stats)` 后，`model.gate_outputs(result, previous_contact)` 返回物理单位的 `position_physical`、`wrench_physical`、`slip_physical`、有效性、转移和 `last_contact`。`model.eval(); model.step(...)` 自动缓存视觉 token、GRU 隐状态和预测接触状态，并返回上述门控输出；新轨迹调用 `reset_state()`。归一化保存在模型权重中，未配置时禁止物理门控。`--disable-mic-channels 5` 等通道选择继续支持。

`predict.py --state-in/--state-out` 为新模型同时传递 hidden 与预测 contact_state，不从原始 NPZ 读取接触真值。`--previous-contact` 如使用，语义必须是上一预测步的**预测状态**；缺历史默认未知。新模型 JSON/CSV 包含接触概率、接触状态、转移、位置/滑移有效性；评估同时报告原始回归误差、门控误差、接触混淆矩阵/平衡准确率及无接触残余力，避免只看多数类准确率。

当前数据足以训练这版分类与规则门控，不需要额外补采。建立/脱离瞬间的检测延迟和专用动力学残差精度仍不属于现有数据已验证的结论。小规模流程验证结果见 `TEST_REPORT.md`。

---

# 三预测头统一训练（2026-11-20 更新）

当前默认模型为 `mvts/model_triple_head/model_triple_head.py`。按图片实现：视觉 ResNet-18 经 4×4 池化形成 16×128 token；共享单通道声学 ResNet-18 分别编码六路麦克风，频率维池化并对麦克风取均值形成 28×128 token；最后一个声学 token 产生 Query，44 个声学/视觉 token 组成 Key/Value；融合结果送入 GRU(128,256)，三个 MLP 分别输出位置3维、六维力6维、滑移位移2维。图中虚线的接触概率/不确定性分支是可选项，此次没有启用。

网络输入只有图像和音频；六维力、位置、滑移位移均为监督真值。默认随机初始化，不下载预训练权重。

- **按压：每张图像只取导出包中最后一条（当前）六维力，以及该条力之前4000点/通道音频；前9条力不用于训练。** 每个按压事件只有1个样本，滑移标签严格为 `[0,0]`，独立事件之间重置 GRU。
- **滑移：** 每对真正相邻图像在半开区间 `[t_j,t_(j+1))` 形成10个等分子事件，包含左端图像时刻和9个内部时刻，不重复右端。每个子事件使用最新可用的左端图像、事件之前4000点/通道音频，以及该时刻之前最近的实测六维力标签。400Hz到约150Hz存在量化偏差，缓存记录 `force_age_ns`；超过3ms的配对排除。
- 滑移位置由实测位置在子事件时刻插值得到，滑移标签是当前位置减上一子事件位置的 x、y 分量（mm）；不使用整段位移，也不跨轨迹做差。按实际相机间隔十等分，所以150Hz是名义频率，真实时间戳抖动被保留。缺少历史音频、存在采样缺口、裁剪不完整的区间会排除并写入报告，不补零。
- 音频50kHz，4000点为80ms。STFT `n_fft=512, hop=128, center=False`，使用窗口最右侧3968点，输出64×28 Log-Mel，避免忽略最后32点。
- 训练按完整轨迹/位置分组划分，按压和滑移分别分层。GRU在同一滑移轨迹内跨窗口携带状态，窗口间截断梯度；按压1步和滑移10步的批次交替训练，批内长度一致。每张新图像在一个窗口内只编码一次。流式 `model.eval(); model.step(...)` 缓存视觉 token 和 GRU 状态，新轨迹调用 `reset_state()`。
- `sensor3_probe5mm_slide_training_20261113` 按用户要求排除；其他5mm按压数据继续保留。

建议一次性预处理全部原始数据：缓存原始尺度 Log-Mel、缩放图像、标签、时间戳和配对索引，之后改网络时直接复用。**不缓存训练中的视觉特征，也不使用全量数据拟合归一化**；归一化只在训练划分上拟合。修改网络层数、注意力和预测头无需重新处理；修改对齐、采样率、音频窗口或 STFT 时需生成新缓存。原始数据更新时也应生成新的缓存目录；完整缓存不会被覆盖。

本次全量缓存已经生成在 `.cache/data_final_v4`（约7.34GB）：8376个按压样本、12599个滑移图像区间，共134366个子事件。可以直接执行下面的训练命令，无需再次预处理。测试范围与结果见 `TEST_REPORT.md`。

另有筛选缓存 `.cache/data_final_v4_small`：仅保留 `sensor3_probe4mm_dynamic_training_20261119`（1199个）和 `sensor3_probe4mm_staircase_training_20261119`（1599个）按压样本；500条滑移轨迹、12599个区间、125990个滑移子事件全部保留，总计128788个子事件。新缓存独立复制图像和事件文件，原缓存保持完整。新数据集需启动新训练运行，重新拟合训练集归一化，不使用旧数据集的 `--resume`。

```powershell
# 筛选缓存的训练与验证
uv run train.py --prepared-root .cache/data_final_v4_small --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/triple_head_small

# 需要重新生成时，output 指定一个尚不存在的新目录
uv run preprocess.py --from-cache .cache/data_final_v4 --press-source sensor3_probe4mm_dynamic_training_20261119 --press-source sensor3_probe4mm_staircase_training_20261119 --output .cache/data_final_v4_small
```

该筛选主要减少按压来源；由于滑移全部保留，总子事件数只减少约4.2%，整体训练提速幅度可能有限。

2026-11-21 新增缓存 `.cache/data_final_v4_faster`：按压使用全部六套来源；滑移仅选择 `sensor3_probe3mm_fast_slide_fullframes_training_20261121`，不包含旧4mm滑移。预处理同时兼容原始 `attempts/` 和整理导出的 `trajectories/` 目录，十等分、因果音频、位置插值和滑移差分规则相同，六路音频均保留。

滑移音频先按同一轨迹拼接连续时间轴，再取每个子事件之前4000点，不受当前相邻图像区间边界限制。80ms音频窗口可以向前跨一个或多个图像区间，不会跨不同轨迹或真实采样缺口；轨迹起始确实不足历史时，相应不完整区间会排除。该缓存包含8376个按压样本，以及300条快速滑移轨迹的5044个有效图像区间、50440个滑移子事件，总计58816个子事件；数据清单与验收结果见缓存内 `report.json` 和 `verification.json`。

```powershell
# 生成指定快速滑移来源 + 全部按压；output 必须使用新的完整缓存目录
uv run preprocess.py --data-root data/data_final --slide-source sensor3_probe3mm_fast_slide_fullframes_training_20261121 --output .cache/data_final_v4_faster

# 使用该缓存开始独立训练，训练集归一化会重新拟合
uv run train.py --prepared-root .cache/data_final_v4_faster --device cuda --epochs 30 --batch-size 8 --workers 10 --output runs/triple_head_faster
```

2026-11-23 增加滑移末尾截断缓存：对每条轨迹，计算每个图像区间内10个位置真值的最大两点 XY 欧氏距离（忽略 z）；**第一次小于0.05mm时，从该区间开始删除全部后续区间**，后续偶尔较大的抖动也不重新保留。等于0.05mm不触发截断。全部按压样本保留，包括 v5 的无接触背景；原始缓存保留，派生缓存独立复制所需图像和 NPZ，保留原有音频、回归标签、时间戳及轨迹序号。

两个截断缓存各保留300条轨迹、1524个连续滑移区间（15240个子事件），各删除3520个区间（35200个子事件）。`.cache/data_final_v4_faster_trim_xy_0p05mm` 保留8376个按压样本，总计9900个窗口、23616个子事件；`.cache/data_final_v5_faster_contact_trim_xy_0p05mm` 保留13159个按压样本，总计14683个窗口、28399个子事件。各目录的 `filtered_out.csv` 记录删除清单和截断序号，`filter_verification.json` 记录连续性及文件校验结果。

```powershell
# 需要重新生成时，output 必须是尚不存在的新目录
uv run filter_stationary_cache.py --source .cache/data_final_v4_faster --output .cache/data_final_v4_faster_trim_xy_0p05mm --threshold-mm 0.05
uv run filter_stationary_cache.py --source .cache/data_final_v5_faster_contact --output .cache/data_final_v5_faster_contact_trim_xy_0p05mm --threshold-mm 0.05 --baseline-cache .cache/data_final_v4_faster_trim_xy_0p05mm
uv run verify_contact_cache.py --prepared-root .cache/data_final_v5_faster_contact_trim_xy_0p05mm --baseline .cache/data_final_v4_faster_trim_xy_0p05mm --compare-audio
```

训练时将 `--prepared-root` 改为对应的截断缓存，并使用新的训练输出目录，归一化只在新的训练划分上重新拟合。

屏蔽第五路麦克风无需重建缓存：训练命令增加 `--disable-mic-channels 5`（编号从1开始），并使用新的输出目录，例如 `--output runs/triple_head_no_mic5`。模型在声学编码前排除该路，对剩余5路特征求平均，缓存仍保持6路。多路可写 `--disable-mic-channels 2 5`，至少保留一路。选择保存在模型配置和checkpoint中，评估、推理、测速及恢复训练会自动沿用；恢复时显式指定不同通道会报错。不能对原六路实验的 `last.pt` 直接添加屏蔽参数继续训练。

`model_triple_head` 支持仅相机输入：训练时增加 `--image-only`，不创建或运行声学骨干和声学投影，也不加入声学 token。16 个视觉 token 的均值生成 Query，视觉 token 作为 Key/Value，经注意力和 GRU 后仍输出位置、六维力和滑移三个预测头。默认不加此参数时保留原来的视听融合行为；`--image-only` 不能与 `--disable-mic-channels` 同用。

```powershell
uv run train.py --model model_triple_head --prepared-root .cache/data_final_v4 --image-only --device cuda --epochs 30 --batch-size 8 --output runs/triple_head_image_only

# 纯视觉 checkpoint 只需图像即可预测，无需 --npz 或 --prepared-window
uv run predict.py --checkpoint runs/triple_head_image_only/best.pt --image PATH_TO_IMAGE --output runs/image_only_prediction.json
```

纯视觉训练可复用原缓存（现有数据加载/统计仍读取缓存音频，但音频不传入网络），按压/滑移事件数量、时序和监督标签保持原规则。同一图像窗口的多个事件复用视觉特征，由 GRU 逐事件更新状态；图像之间没有新增的声学观测。`image_only` 保存到 checkpoint，评估、推理和恢复训练自动沿用；恢复时省略该参数即可，显式指定与 checkpoint 不同的 `--image-only` / `--no-image-only` 会报错。切换输入模态需使用新的输出目录重新训练。

Python 调用：`build_model("model_triple_head", {"image_only": True})`，然后使用 `model(image)`、`model.forward_sequence(image, sequence_length=10)`，或 `model.eval(); model.step(image=image, frame_id=frame_id)`。后续同帧事件可调用 `model.step()`，新轨迹调用 `reset_state()`。原先传入 audio 的接口仍兼容，纯视觉模式忽略音频数值（序列接口仅借其形状确定事件数）。

单事件推理延迟可在停止GPU训练后测量：`uv run benchmark_inference.py --device cuda --iterations 200 --warmup 30 --output runs/inference_latency_cuda.json`。默认使用 `runs/triple_head_small/best.pt`，同时区分视觉token复用/刷新、纯网络/包含内存数据预处理。此次4070 SUPER实测纯网络平均约2.99ms/4.66ms；包含预处理约3.70ms/9.98ms。详细范围、P95和150Hz预算限制见 `TEST_REPORT.md`。

```powershell
# 一次性全量预处理，原始 data 不修改
.venv/Scripts/python.exe preprocess.py --data-root data/data_final --output .cache/data_final_v4 --skip-unsupported

# 训练；正式训练去掉所有 max-* 限制
.venv/Scripts/python.exe train.py --model model_triple_head --prepared-root .cache/data_final_v4 --device cuda --epochs 30 --batch-size 2 --output runs/triple_head

# 评估保存的完整测试划分
.venv/Scripts/python.exe evaluate.py --checkpoint runs/triple_head/best.pt --prepared-root .cache/data_final_v4 --device cuda --output runs/triple_head/evaluation

# 恢复训练时保留原来的 batch-size、max-* 和损失权重等参数
.venv/Scripts/python.exe train.py --prepared-root .cache/data_final_v4 --device cuda --epochs 40 --batch-size 2 --output runs/triple_head --resume runs/triple_head/last.pt

# 原始按压 NPZ：默认只预测当前最后一组
.venv/Scripts/python.exe predict.py --checkpoint runs/triple_head/best.pt --image PATH_TO_IMAGE --npz PATH_TO_PRESS_NPZ --output runs/press_prediction.json

# 缓存滑移窗口：已包含跨原始区间的历史音频
.venv/Scripts/python.exe predict.py --checkpoint runs/triple_head/best.pt --image CACHE_IMAGE_PATH --prepared-window CACHE_EVENTS_NPZ --state-out runs/state.pt --output runs/slide_prediction.json
```

`predict.py --state-in/--state-out` 在相邻滑移窗口间传递 GRU 状态；新轨迹不要传入上一轨迹状态。在线应用优先使用 `model.step`，新相机帧到达时提供图像，其余子事件复用缓存。`--force-index` 仍允许显式选择原始按压包中某一组，但默认只预测当前组。输出11维，单位为位置/位移 mm、力 N、力矩 N*m。

数据质量边界：相机使用主机读帧完成时间，没有真实曝光时间；按压的“当前力”沿用采集端关联，是图像时刻之前最近一条力。不能据此宣称硬件严格同步，`--strict-timing` 会明确拒绝。数据排除和时间对齐说明见缓存 `report.json`。下面保留旧版说明及历史；旧模型、2000点音频和双头相关段落不适用于当前默认模型。

---

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

默认环境使用官方 **CUDA 12.6 版 PyTorch**，由 `pyproject.toml` 的 `pytorch-cu126` 索引和 `uv.lock` 固定。目标平台为带兼容 NVIDIA 驱动的 Windows / Linux；当前部署机器为 RTX 4070 SUPER（12 GB，计算能力 8.9）、驱动 610.88。新驱动向后兼容 CUDA 12.6；`nvidia-smi` 显示的 CUDA 13.3 表示驱动支持的 CUDA 版本，不要求 wheel 使用相同版本。安装的是 CUDA 12.6 runtime 的 wheel，不要求另外安装完整 CUDA Toolkit。迁移时应重新检查目标机器的 GPU、驱动以及实际 CUDA 运算。版本选择依据 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/) 与 [NVIDIA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

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
