# t0232 — 150 Hz 部署测试数据

来源：`slide_t0232_raw_and_test_subset.zip` 的 `prepared_test_subset/`，共 8 个连续测试窗口、80 个事件、8 张图像。
窗口 order 为 2–9；这里只导出压缩包提供的测试子集，不扩充到整条原始轨迹。

格式与 `deployment_test_150hz` 相同：`events.jsonl` 每行对应一次调用，`events/*.npz` 保存
uint16 ADC `[4000,6]`、未标准化 Log-Mel `[6,64,28]`、标准化 `audio_input`、物理单位 `target[11]` 与原始时间戳。
`labels.csv` 保存全部标签；`images/` 保存训练预处理后的 224×224 RGB 图像。
首事件重置 GRU，此后连续传递状态；每 10 个事件更新一次图像。名义调度频率 150 Hz，事件跨度 0.526667 s。
原始采集时间、标签（包括首事件的滑移增量）均保持原样，没有按 150 Hz 重采样或补零。

`normalization.json` 与 `model_config` 沿用参考包，并非由本测试集拟合。
部署界面会使用所选 checkpoint 自身的标准化参数；独立 reader 使用其他模型时请传 `--normalization`。
输入始终保留全部 6 路麦克风，模型负责禁用通道。

```powershell
python reader.py --kind slide
python reader.py --kind slide --realtime
# 在项目根目录：
.venv/Scripts/python.exe deployment/deployment_viewer.py --model model_triple_head --data-root deployment/test_data/deployment_test_150hz_t0232
```

`verification.json` 记录源 ZIP 校验、逐事件特征重算与标签核对结果；`checksums.sha256` 覆盖包内其余文件。
