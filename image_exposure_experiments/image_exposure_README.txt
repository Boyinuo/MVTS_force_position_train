曝光时间与纯视觉位置预测实验（2026-11-24）

数据范围
只读取 .cache/data_final_v6_slide_contact 和 .cache/data_final_v6_polyline_contact。
基线：runs/v6_triple_head_slide_polyline_image_only_trim0p029_xyfilter/best.pt。
复用其保存的 train/val/test IDs；不重新分组，不重新应用不同的筛选。
1891/443/368个图像窗口，252/59/49条轨迹，每窗口10个子事件。
不含按压，不使用音频、六维力或真实位置/速度作为预测输入。

目录
image_exposure_experiments：全部新代码、因果性/时间积分测试。
runs/image_exposure_v1：冻结特征实验、原始位置插值衍生数据、报告。
runs/image_exposure_anchor_v2：单图位置锚点端到端微调与因果运动补偿。
原始模型、原缓存、原始数据不写入；source_sha256保存关键文件校验。

物理假设
15.6ms是曝光持续时间。假设曝光结束后无额外延迟，中点比读图完成早7.8ms。
额外读出/传输/缓存时间未知，滚动快门性质未知。
曝光平均位置是监督近似，不代表图像外观严格等于平均位置的静态图像。
23.4ms仅为验证集诊断提出的总延迟敏感性实验，不是测量到的硬件延迟。

比较规则
主指标是原始当前时刻XY欧氏距离的均值，单位mm。
标签均不变；速度由当前/过去图像的预测位置估计，不使用真值速度。
先由验证集选择模型、epoch、运动平滑参数，再评价测试集。
相邻子事件相关；bootstrap以整条轨迹为单位。
第二轮是在查看第一轮测试结果后设计，复用测试集的结果属于探索性证据。
图像曝光时刻定位与当前时刻预测是不同任务，必须分别报告。

复现（仓库根目录）
.venv/Scripts/python.exe -m image_exposure_experiments.prepare --output runs/image_exposure_NEW
.venv/Scripts/python.exe -m image_exposure_experiments.run --output runs/image_exposure_NEW
.venv/Scripts/python.exe -m image_exposure_experiments.train_anchor --source runs/image_exposure_NEW --output runs/image_exposure_anchor_NEW
.venv/Scripts/python.exe -m pytest image_exposure_experiments -q -p no:cacheprovider
.venv/Scripts/python.exe -m image_exposure_experiments.report
新输出必须使用未被实验占用的目录，命名image_exposure_xxx。

第二轮推理
from image_exposure_experiments.predict import ImageExposurePredictor
predictor = ImageExposurePredictor('runs/image_exposure_anchor_v2/image_exposure_best.pt')
predictor.update(image_path_or_normalized_tensor, frame_pc_ns, reset=True)
result = predictor.predict(event_pc_ns)
result['position_mm']是当前位置，result['anchor_mm']是估计图像物理时刻的位置。
frame_pc_ns/event_pc_ns必须使用同一单调时钟，不能混入Unix时间或设备时钟。
每次新轨迹/断流重置；每张新图update一次，中间时刻调用predict。
第一张图无法从历史图像估计运动，速度初始化为0。
此接口只用于第二轮anchor-v2模型。若验证集选择原基线回退，则继续使用原model.step。
不要直接将独立图像模型复制覆盖原部署best.pt：它要求每张图位置网络重置GRU，
并由外层维护相邻预测位置与时间。原力/滑移头未在第二轮微调，不能宣称其精度改善。

新增：contact_gated / slip_gated 两数据集受控实验
.venv/Scripts/python.exe -m image_exposure_experiments.gated_compare --output runs/image_exposure_gated_NEW
.venv/Scripts/python.exe -m image_exposure_experiments.gated_report --output runs/image_exposure_gated_NEW

现有部署contact/slip权重包含按压预训练。为严格遵循仅用两份数据，本实验从
runs/v6_triple_head_slide_polyline_no_mic5_trim0p029_xyfilter/best.pt
迁移共有权重，建立新的contact/slip架构基线，没有加载按压预训练权重。
因此它是两种架构的独立受控验证，不是现有部署checkpoint的直接升级验证。

固定真实声学编码器和投影，复用其实际编码token；视觉、融合、GRU等继续训练。
缓存编码在真实样本上与正常推理核对一致，保存的权重仍包含正常声学编码器。
连续基线30轮、多头监督、每窗口10个真实音频更新；每个单图方案再微调30轮，
仅位置监督，每张图重置GRU，只使用该图对应第一个因果音频窗口。
零偏移和7.8ms方案均仅按验证集选择epoch/运动平滑参数，基线保留为回退候选。

单图模型的标准调用：
model.forward_sequence(image, audio_first[:,None], hidden=None, sequence_length=1, ...)
slip_gated还必须传入真实首事件delta_t[:,None]、图像年龄加设定偏移、has_history=False。
位置输出先按保存stats还原为mm，再根据anchor_config维护相邻预测位置与时间。
contact迟滞状态需要独立跨图像保留，不能跟随GRU每图重置。
此方法舍弃后续9个音频更新，不能当作保持了原150Hz多模态预测能力。

全部数据均为接触正样本，只能检查接触漏检，不能评价背景识别/无接触特异性。
门控拒绝的有效标签仍计入位置误差；其他输出头固定也不保证其行为保持不变。
实际新训练配置以对应protocol.json和anchor_config为准，基线args包含初始化来源的历史信息。
