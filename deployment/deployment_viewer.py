"""Precompute one MVTS slide trajectory, then replay it on a nominal 150 Hz clock.

python deployment_viewer.py --model model_contact_gated
python deployment_viewer.py --model model_triple_head.py
"""
import argparse
from copy import copy
from pathlib import Path
import sys
import time
import traceback

import numpy as np

# Support both `python deployment/deployment_viewer.py` and `python -m deployment.deployment_viewer`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.replay import ROOT, precompute, save_results, select_sequence


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
        help="Required model name or .py path; automatically loads best.pt from its model directory")
    p.add_argument("--data-root", type=Path, default=ROOT / "deployment/test_data/deployment_test_150hz_t0232")
    p.add_argument("--sequence", help="Slide sequence ID; defaults to first slide (slide_t0242 in sample data)")
    p.add_argument("--hz", type=float, default=150, help="Nominal inference-event playback frequency")
    p.add_argument("--render-fps", type=int, default=60, help="GUI refresh target (default: 60), separate from event frequency")
    p.add_argument("--image-only-comparison", action=argparse.BooleanOptionalAction, default=True,
        help="Add a 15 Hz camera-only comparison using best_image_only.pt beside the selected best.pt (default: enabled)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, help="Results directory; default deployment/results/TIMESTAMP")
    p.add_argument("--headless", action="store_true", help="Infer/export without importing Qt")
    p.add_argument("--screenshot", type=Path, help="Render the final frame to PNG and exit (UI verification)")
    p.add_argument("--random-weights", action="store_true", help="Explicit UI-only test; untrained outputs marked prominently")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not np.isfinite(args.hz) or args.hz <= 0 or args.render_fps <= 0 or args.threads <= 0:
        p.error("hz/render-fps/threads must be positive and finite")
    if args.headless and args.screenshot:
        p.error("--screenshot requires the GUI; omit --headless")
    if args.output is None:
        args.output = ROOT / "deployment/results" / time.strftime("%Y%m%d_%H%M%S")
    if args.headless:
        data = precompute(args, lambda stage, n, total: print(f"\r{stage}: {n}/{total}", end="", flush=True))
        save_results(data, args.output)
        print(f"\nSaved {len(data.metadata)} events to {args.output}")
        return
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg
    except ImportError as exc:
        p.error(f"Viewer dependencies missing: {exc}. Install: uv pip install -r deployment/requirements.txt")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = create_window(args, QtCore, QtGui, QtWidgets, pg)
    window.show()
    QtCore.QTimer.singleShot(0, window.start_precompute)
    sys.exit(app.exec_())


def create_window(args, QtCore, QtGui, QtWidgets, pg):
    """Qt imports stay optional for the headless inference/export path."""
    pg.setConfigOptions(antialias=True, background="#f7f9fc", foreground="#4c6380", imageAxisOrder="row-major")
    teal, orange, purple = "#119ba1", "#d88935", "#7954c5"

    class Worker(QtCore.QThread):
        progress = QtCore.pyqtSignal(str, int, int)
        ready = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, run_args, parent):
            super().__init__(parent)
            self.args = copy(run_args)

        def run(self):
            try:
                data = precompute(self.args, self.progress.emit)
                save_results(data, self.args.output)
                self.ready.emit(data)
            except Exception:
                self.failed.emit(traceback.format_exc())

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("接触位置与轨迹 · MVTS 部署模拟")
            self.resize(1600, 960)
            self.setMinimumSize(1050, 700)
            self.data = None
            self.current = 0
            self.trail_start = 0
            self.playing = False
            self.speed = 1.
            self.anchor_position = 0.
            self.anchor_time = 0.
            self.last_image = None
            self.last_render = None
            self.render_fps = 0.
            self.output_base = args.output
            self.run_count = 0
            self.setStyleSheet("""
                QMainWindow, QWidget { background: #edf1f7; color: #365478; font-family: 'Microsoft YaHei'; font-size: 11px; }
                QGroupBox { background: #f7f9fc; border: 1px solid #abc0d9; border-radius: 9px; margin-top: 8px; padding: 12px 8px 5px; font-weight: bold; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
                QPushButton, QComboBox, QLineEdit { background: #fff; border: 1px solid #b6c8dd; border-radius: 4px; padding: 5px 9px; }
                QPushButton:disabled { color: #a5b1c0; }
                QPushButton:hover { border-color: #119ba1; }
                QTabWidget::pane { border: 0; }
                QTabBar::tab { padding: 5px 14px; background: #e3eaf4; }
                QTabBar::tab:selected { background: #fff; color: #119ba1; }
            """)
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            outer = QtWidgets.QVBoxLayout(central)
            outer.setContentsMargins(10, 8, 10, 8)
            outer.setSpacing(5)
            top = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel("接触位置与历史轨迹")
            title.setStyleSheet("font-size: 20px; font-weight: bold; color: #284f7d")
            top.addWidget(title)
            top.addStretch()
            self.mode = QtWidgets.QLabel(f"测试数据回放 / {args.model} / {args.hz:g} Hz 时间轴")
            self.mode.setStyleSheet("color: #b86c23; font-weight: bold; font-size: 13px")
            top.addWidget(self.mode)
            outer.addLayout(top)
            data_controls = QtWidgets.QHBoxLayout()
            data_controls.addWidget(QtWidgets.QLabel("测试数据目录"))
            self.data_root_edit = QtWidgets.QLineEdit(str(args.data_root))
            self.data_root_edit.setToolTip("选择包含 manifest.json 和 events.jsonl 的部署测试数据目录")
            data_controls.addWidget(self.data_root_edit, 1)
            self.browse_data = QtWidgets.QPushButton("浏览…")
            self.browse_data.clicked.connect(self.choose_data_root)
            data_controls.addWidget(self.browse_data)
            self.load_data = QtWidgets.QPushButton("加载并推理")
            self.load_data.clicked.connect(self.start_precompute)
            self.data_root_edit.returnPressed.connect(self.start_precompute)
            data_controls.addWidget(self.load_data)
            outer.addLayout(data_controls)
            self.status = QtWidgets.QLabel("等待预计算；完成全部推理后开始播放")
            self.details = QtWidgets.QLabel(str(args.data_root))
            outer.addWidget(self.status)
            outer.addWidget(self.details)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setMaximumHeight(14)
            outer.addWidget(self.progress)
            grid = QtWidgets.QGridLayout()
            grid.setSpacing(8)
            outer.addLayout(grid, 1)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)
            grid.setRowStretch(0, 11)
            grid.setRowStretch(1, 10)

            camera, camera_layout = self.panel("01  相机原始图像", "224 × 224 · 仅在 image_update 时换帧，保持原始宽高比")
            self.camera = QtWidgets.QLabel("准备图像…")
            self.camera.setAlignment(QtCore.Qt.AlignCenter)
            self.camera.setStyleSheet("background: #dfe5ee; border-radius: 5px")
            camera_layout.addWidget(self.camera, 1)
            grid.addWidget(camera, 0, 0)

            trajectory, trajectory_layout = self.panel("02  预测接触点与实际接触点", "青色圆点：主模型   紫色方点：纯视觉 15 Hz   橙色叉点：GT   ·   滚轮缩放 / 左键平移")
            view_controls = QtWidgets.QHBoxLayout()
            self.show_image_only = QtWidgets.QCheckBox("显示纯视觉 15 Hz")
            self.show_image_only.setChecked(True)
            self.show_image_only.toggled.connect(self.update_comparison_visibility)
            view_controls.addWidget(self.show_image_only)
            view_controls.addStretch()
            self.fit_xy = QtWidgets.QPushButton("适应已播轨迹")
            self.fit_xy.clicked.connect(self.fit_trajectory)
            view_controls.addWidget(self.fit_xy)
            self.reset_xy = QtWidgets.QPushButton("重置视图")
            self.reset_xy.clicked.connect(self.reset_trajectory_view)
            view_controls.addWidget(self.reset_xy)
            trajectory_layout.addLayout(view_controls)
            self.tabs = QtWidgets.QTabWidget()
            trajectory_layout.addWidget(self.tabs, 1)
            self.xy = self.plot("X (mm)", "Y (mm)")
            self.xy.setAspectLocked(True)
            self.xy.setXRange(-35, 35, padding=0)
            self.xy.setYRange(-15, 15, padding=0)
            self.xy.disableAutoRange()
            self.xy.setMouseEnabled(x=True, y=True)
            self.xy.getViewBox().setMouseMode(pg.ViewBox.PanMode)
            self.xy.setToolTip("每个已播放事件均显示标记点；滚轮缩放，左键拖动平移，右键拖动缩放。标记大小保持固定，便于放大检查。")
            self.tabs.addTab(self.xy, "XY 轨迹")
            self.xy_pred = self.xy.plot(pen=pg.mkPen(teal, width=1.7), connect="finite",
                symbol="o", symbolSize=4, symbolBrush=teal, symbolPen=pg.mkPen(teal, width=.7), pxMode=True)
            self.xy_true = self.xy.plot(pen=pg.mkPen(orange, width=1.7, style=QtCore.Qt.DashLine), connect="finite",
                symbol="x", symbolSize=5, symbolPen=pg.mkPen(orange, width=1), pxMode=True)
            self.xy_image_only = self.xy.plot(pen=pg.mkPen(purple, width=1.8), connect="finite",
                symbol="s", symbolSize=5, symbolPen=purple, symbolBrush=purple, pxMode=True)
            self.image_only_dot = self.xy.plot(pen=None, symbol="s", symbolSize=9, symbolPen=purple, symbolBrush=purple)
            self.image_only_dot.setZValue(9)
            self.pred_dot = self.xy.plot(pen=None, symbol="o", symbolBrush=teal, symbolPen=teal, symbolSize=8)
            self.true_dot = self.xy.plot(pen=None, symbol="x", symbolPen=pg.mkPen(orange, width=2), symbolSize=10)
            self.pred_dot.setZValue(10)
            self.true_dot.setZValue(10)
            position_widget = pg.GraphicsLayoutWidget()
            self.position_plots, self.position_lines, self.position_image_only = [], [], []
            for i, label in enumerate(("X (mm)", "Y (mm)", "Δz (mm)")):
                plot = position_widget.addPlot(row=i, col=0)
                self.style_plot(plot, "时间 (s)", label)
                self.position_plots.append(plot)
                self.position_lines.append((plot.plot(pen=pg.mkPen(teal, width=1.5), connect="finite"),
                    plot.plot(pen=pg.mkPen(orange, style=QtCore.Qt.DashLine), connect="finite")))
                self.position_image_only.append(plot.plot(pen=pg.mkPen(purple, width=1.6),
                    symbol="s", symbolSize=4, symbolPen=purple, symbolBrush=purple, connect="finite"))
            self.tabs.addTab(position_widget, "位置时间曲线")
            self.position_text = QtWidgets.QLabel("预测 / 实测位置")
            self.metrics = QtWidgets.QLabel("XY 误差与接触状态")
            self.comparison_text = QtWidgets.QLabel("纯视觉 15 Hz 对照：等待预计算")
            self.comparison_text.setStyleSheet(f"color: {purple}; font-size: 10px")
            trajectory_layout.addWidget(self.position_text)
            trajectory_layout.addWidget(self.metrics)
            trajectory_layout.addWidget(self.comparison_text)
            grid.addWidget(trajectory, 0, 1)

            wrench, wrench_layout = self.panel("03  六维力数据", "青色：主模型   紫色：纯视觉 15 Hz   橙色虚线：GT   ·   力 N / 力矩 N·m")
            self.wrench_widget = pg.GraphicsLayoutWidget()
            wrench_layout.addWidget(self.wrench_widget, 1)
            self.wrench_plots, self.wrench_lines, self.wrench_image_only = [], [], []
            for i, label in enumerate(("Fx / N", "Fy / N", "Fz / N", "Mx / N·m", "My / N·m", "Mz / N·m")):
                plot = self.wrench_widget.addPlot(row=i // 3, col=i % 3, title=label)
                self.style_plot(plot, "时间 (s)" if i >= 3 else None, None)
                self.wrench_plots.append(plot)
                self.wrench_lines.append((plot.plot(pen=pg.mkPen(teal, width=1.4)),
                    plot.plot(pen=pg.mkPen(orange, style=QtCore.Qt.DashLine))))
                self.wrench_image_only.append(plot.plot(pen=pg.mkPen(purple, width=1.5),
                    symbol="s", symbolSize=3, symbolPen=purple, symbolBrush=purple))
            grid.addWidget(wrench, 1, 0)

            microphones, mic_layout = self.panel("04  六通道麦克风梅尔频谱", "当前事件之前 80 ms · 训练 Log-Mel（未标准化）· 固定色阶 −10 ～ −2.5")
            self.mic_widget = pg.GraphicsLayoutWidget()
            mic_layout.addWidget(self.mic_widget, 1)
            self.mic_images, self.mic_plots = [], []
            # Fixed purple-to-amber log-power palette, consistent across all channels/events.
            colors = ["#100b26", "#281044", "#53136e", "#8f2376", "#cf4770", "#f58b55", "#f8d277"]
            lut = pg.ColorMap(np.linspace(0, 1, len(colors)), [pg.mkColor(c) for c in colors]).getLookupTable(nPts=256)
            for i in range(6):
                plot = self.mic_widget.addPlot(row=i // 3, col=i % 3, title=f"MIC {i + 1} / A{i}")
                self.style_plot(plot, "时间 (ms)" if i >= 3 else None, "Mel bin" if i % 3 == 0 else None)
                image = pg.ImageItem()
                image.setLookupTable(lut)
                plot.addItem(image)
                plot.setXRange(-80, 0, padding=0)
                plot.setYRange(0, 64, padding=0)
                plot.setMouseEnabled(x=False, y=False)
                self.mic_images.append(image)
                self.mic_plots.append(plot)
            grid.addWidget(microphones, 1, 1)

            controls = QtWidgets.QHBoxLayout()
            self.play = QtWidgets.QPushButton("暂停")
            self.play.clicked.connect(self.toggle_play)
            self.replay = QtWidgets.QPushButton("重新回放")
            self.replay.clicked.connect(self.restart)
            self.clear = QtWidgets.QPushButton("清空轨迹")
            self.clear.clicked.connect(self.clear_trail)
            self.freeze = QtWidgets.QPushButton("冻结显示")
            self.freeze.setCheckable(True)
            self.freeze.toggled.connect(self.toggle_freeze)
            self.speed_box = QtWidgets.QComboBox()
            self.speed_box.addItems(["0.25×", "0.5×", "1×", "2×"])
            self.speed_box.setCurrentIndex(2)
            self.speed_box.currentIndexChanged.connect(self.change_speed)
            for widget in (self.play, self.replay, self.clear, self.freeze, self.speed_box):
                controls.addWidget(widget)
                widget.setEnabled(False)
            self.seek = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.seek.setEnabled(False)
            self.seek.valueChanged.connect(self.seek_to)
            controls.addWidget(self.seek, 1)
            self.counter = QtWidgets.QLabel("0 / 0")
            controls.addWidget(self.counter)
            outer.addLayout(controls)
            self.saved = QtWidgets.QLabel("结果将在推理完成后自动保存")
            outer.addWidget(self.saved)
            self.timer = QtCore.QTimer(self)
            self.timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.timer.timeout.connect(self.tick)

        def panel(self, title, subtitle):
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            layout.setSpacing(3)
            text = QtWidgets.QLabel(subtitle)
            text.setStyleSheet("font-size: 10px; color: #738ba6; font-weight: normal")
            layout.addWidget(text)
            return group, layout

        def style_plot(self, plot, bottom, left):
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.hideButtons()
            for side in ("left", "bottom"):
                plot.getAxis(side).setTickFont(QtGui.QFont("Microsoft YaHei", 8))
            item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
            if item.titleLabel.text:
                plot.setTitle(item.titleLabel.text, size="10pt", **{"font-family": "Microsoft YaHei"})
            if bottom:
                plot.setLabel("bottom", bottom, **{"font-family": "Microsoft YaHei", "font-size": "9pt"})
            if left:
                plot.setLabel("left", left, **{"font-family": "Microsoft YaHei", "font-size": "9pt"})

        def plot(self, bottom, left):
            widget = pg.PlotWidget()
            self.style_plot(widget, bottom, left)
            return widget

        def choose_data_root(self):
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self, "选择部署测试数据目录", self.data_root_edit.text())
            if directory:
                self.data_root_edit.setText(directory)

        def set_loading(self, loading):
            for widget in (self.data_root_edit, self.browse_data, self.load_data):
                widget.setEnabled(not loading)
            for widget in (self.play, self.replay, self.clear, self.freeze, self.speed_box, self.seek):
                widget.setEnabled(not loading and self.data is not None)

        def start_precompute(self):
            if hasattr(self, "worker") and self.worker.isRunning():
                return
            try:
                text = self.data_root_edit.text().strip()
                if not text:
                    raise ValueError("请选择测试数据目录")
                root = Path(text).expanduser().resolve()
                if not (root / "manifest.json").is_file():
                    raise ValueError(f"缺少 manifest.json：{root}")
                # A different package gets its own first sequence, rather than a stale CLI ID.
                sequence = getattr(args, "sequence", None) if root == Path(args.data_root).resolve() else None
                rows = select_sequence(root, sequence)
            except (OSError, ValueError, KeyError) as exc:
                self.failed(str(exc))
                return
            self.pause()
            args.data_root, args.sequence = root, rows[0]["sequence"]
            if self.run_count:
                args.output = self.output_base / f"reload_{time.time_ns()}_{args.sequence}"
            self.run_count += 1
            self.data_root_edit.setText(str(root))
            self.set_loading(True)
            self.progress.setRange(0, 0)
            self.progress.show()
            self.status.setText("正在读取测试数据并预计算…")
            self.worker = Worker(args, self)
            self.worker.progress.connect(self.on_progress)
            self.worker.ready.connect(self.loaded)
            self.worker.failed.connect(self.failed)
            self.worker.finished.connect(lambda: self.set_loading(False))
            self.worker.start()

        def on_progress(self, stage, number, total):
            self.progress.setRange(0, total)
            self.progress.setValue(number)
            self.status.setText(f"{stage} {number}/{total} · 全部完成后开始显示，保持序列状态连续")

        def failed(self, error):
            self.pause()
            self.progress.hide()
            self.status.setText("预计算失败；未开始播放")
            self.details.setText(error.strip().splitlines()[-1])
            print(error, file=sys.stderr)
            if args.screenshot:
                QtWidgets.QApplication.exit(1)
            else:
                QtWidgets.QMessageBox.critical(self, "预计算失败", error)

        def loaded(self, data):
            self.data = data
            self.last_image = None
            self.render_fps = 0.
            self.current = self.trail_start = 0
            self.progress.hide()
            self.times = data.times(args.hz)
            self.seek.setRange(0, len(data.metadata) - 1)
            for widget in (self.play, self.replay, self.clear, self.freeze, self.speed_box, self.seek):
                widget.setEnabled(True)
            info = data.summary
            if info["random_weights"]:
                self.mode.setText("随机未训练权重 · 仅界面测试 · 输出不代表模型性能")
                self.mode.setStyleSheet("color: #c13939; font-size: 14px; font-weight: bold")
            self.details.setText(f"{info['sequence']} · {len(data.metadata)} 子事件 / {info['image_updates']} 张图像 · "
                f"{info['device']} · 预计算 {info['precompute_seconds']:.2f} s · 显示刷新上限 {args.render_fps} FPS")
            self.saved.setText(f"结果已自动保存：{args.output}")
            self.show_image_only.setEnabled(data.image_only is not None)
            self.update_comparison_visibility()
            if data.image_only is None:
                self.comparison_text.setText("纯视觉 15 Hz 对照未启用")
            for i, plot in enumerate(self.mic_plots, 1):
                suffix = " · 禁用" if i in info["disabled_mic_channels"] else ""
                plot.setTitle(f"MIC {i} / A{i - 1}{suffix}", size="10pt", **{"font-family": "Microsoft YaHei"})
            self.restart()

        def save_screenshot(self):
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            ok = self.grab().save(str(args.screenshot))
            QtWidgets.QApplication.exit(0 if ok else 1)

        def pause(self):
            self.playing = False
            self.timer.stop()
            self.play.setText("播放")

        def resume(self):
            self.anchor_time = time.perf_counter()
            self.anchor_position = float(self.current)
            self.playing = True
            self.play.setText("暂停")
            self.timer.start(max(1, round(1000 / args.render_fps)))

        def toggle_play(self):
            if self.playing:
                self.pause()
            else:
                if self.current == len(self.data.metadata) - 1:
                    self.current = self.trail_start = 0
                if self.freeze.isChecked():
                    self.freeze.setChecked(False)
                self.resume()

        def toggle_freeze(self, frozen):
            if frozen:
                self.pause()
                self.freeze.setText("解除冻结")
            else:
                self.freeze.setText("冻结显示")
                self.resume()

        def restart(self):
            self.pause()
            self.freeze.blockSignals(True)
            self.freeze.setChecked(False)
            self.freeze.setText("冻结显示")
            self.freeze.blockSignals(False)
            self.current = self.trail_start = 0
            self.last_render = None
            self.render()
            self.resume()

        def change_speed(self, index):
            self.speed = (.25, .5, 1., 2.)[index]
            if self.playing:
                self.resume()

        def seek_to(self, index):
            self.current = index
            self.trail_start = min(self.trail_start, index)
            self.render()
            if self.playing:
                self.resume()

        def clear_trail(self):
            self.trail_start = self.current
            self.render()

        def update_comparison_visibility(self):
            visible = self.data is not None and self.data.image_only is not None and self.show_image_only.isChecked()
            for item in (self.xy_image_only, self.image_only_dot, *self.position_image_only, *self.wrench_image_only):
                item.setVisible(visible)

        def fit_trajectory(self):
            if self.data is not None:
                self.xy.getViewBox().autoRange(padding=.12)

        def reset_trajectory_view(self):
            self.xy.setRange(xRange=(-35, 35), yRange=(-15, 15), padding=0)

        def tick(self):
            # Absolute deadlines avoid timer drift. Every elapsed event contributes to curves,
            # even when multiple events share a paint; no event-wise GUI redraw requirement.
            now = time.perf_counter()
            if self.last_render is not None and now > self.last_render:
                self.render_fps = 1 / (now - self.last_render)
            self.last_render = now
            index = int(self.anchor_position + (now - self.anchor_time) * args.hz * self.speed)
            self.current = min(index, len(self.data.metadata) - 1)
            if self.current == len(self.data.metadata) - 1:
                self.pause()
                if args.screenshot:
                    QtCore.QTimer.singleShot(300, self.save_screenshot)
            self.render()

        def render(self):
            d, i = self.data, self.current
            if d is None:
                return
            row = d.metadata[i]
            if row["image"] != self.last_image:
                self.last_image = row["image"]
                pixels = d.images[self.last_image]
                h, w, _ = pixels.shape
                image = QtGui.QImage(pixels.data, w, h, pixels.strides[0], QtGui.QImage.Format_RGB888).copy()
                self.camera_pixmap = QtGui.QPixmap.fromImage(image)
            self.camera.setPixmap(self.camera_pixmap.scaled(self.camera.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            end = i + 1
            start = self.trail_start
            pred_xy = d.prediction[start:end, :2].copy()
            pred_xy[~d.position_valid[start:end]] = np.nan
            # Do not bridge across explicit sequence resets or missing contact.
            for k in range(start + 1, end):
                if d.metadata[k]["reset_state"]:
                    pred_xy[k - start] = np.nan
            self.xy_pred.setData(pred_xy[:, 0], pred_xy[:, 1])
            self.xy_true.setData(d.target[start:end, 0], d.target[start:end, 1])
            self.pred_dot.setData([d.prediction[i, 0]] if d.position_valid[i] else [],
                                  [d.prediction[i, 1]] if d.position_valid[i] else [])
            self.true_dot.setData([d.target[i, 0]], [d.target[i, 1]])
            if d.image_only is not None:
                baseline = d.image_only
                count = int(np.searchsorted(baseline.times, self.times[i] + 1e-9, side="right"))
                first = int(np.searchsorted(baseline.times[:count], self.times[start] - 1e-9, side="left"))
                values = baseline.prediction[:count].copy()
                values[~baseline.position_valid[:count], :3] = np.nan
                self.xy_image_only.setData(values[first:count, 0], values[first:count, 1])
                if count:
                    self.image_only_dot.setData([values[-1, 0]], [values[-1, 1]])
                    for axis, curve in enumerate(self.position_image_only):
                        curve.setData(baseline.times[:count], values[:, axis])
                    for axis, curve in enumerate(self.wrench_image_only):
                        curve.setData(baseline.times[:count], values[:, axis + 3])
                    mask = baseline.position_valid[:count]
                    error = np.linalg.norm(values[mask, :2] - d.target[baseline.event_indices[:count][mask], :2], axis=1)
                    rmse = f"{np.sqrt(np.mean(error**2)):.3f} mm" if len(error) else "无有效位置"
                    last = " / ".join(f"{v:.2f}" for v in values[-1, :3])
                    self.comparison_text.setText(f"纯视觉 15 Hz：{count}/{len(baseline.times)} 点 · 最近 x/y/Δz {last} mm\n"
                        f"输出时刻 {baseline.times[count - 1]:.3f} s · XY RMSE {rmse}（15 Hz 对应 GT）")
                else:
                    self.image_only_dot.setData([], [])
                    for curve in (*self.position_image_only, *self.wrench_image_only):
                        curve.setData([], [])
            t = self.times[:end]
            for axis, (prediction, truth) in enumerate(self.position_lines):
                values = d.prediction[:end, axis].copy()
                values[~d.position_valid[:end]] = np.nan
                prediction.setData(t, values)
                truth.setData(t, d.target[:end, axis])
                self.position_plots[axis].setXRange(max(0, t[-1] - 10), max(1., t[-1]), padding=0)
            for axis, (prediction, truth) in enumerate(self.wrench_lines):
                prediction.setData(t, d.prediction[:end, axis + 3])
                truth.setData(t, d.target[:end, axis + 3])
                self.wrench_plots[axis].setXRange(max(0, t[-1] - 10), max(1., t[-1]), padding=0)
            for channel, image in enumerate(self.mic_images):
                image.setImage(d.mel[i, channel], autoLevels=False, levels=(-10, -2.5))
                image.setRect(QtCore.QRectF(-80, 0, 80, 64))
            p, target = d.prediction[i], d.target[i]
            probability = d.contact_probability[i]
            contact = f"接触概率 {probability:.3f}" if np.isfinite(probability) else "模型无接触分类头"
            position = " / ".join(f"{v:.2f}" for v in p[:3]) if d.position_valid[i] else "无有效位置"
            slip = f"Δsx/Δsy {p[9]:.4f}/{p[10]:.4f} mm" if d.slip_valid[i] else "滑移增量无效"
            self.position_text.setText(f"预测 x/y/Δz：{position} mm  ·  {contact}  ·  {slip}\n"
                f"实测 x/y/Δz：{target[0]:.2f} / {target[1]:.2f} / {target[2]:.2f} mm · 实测接触（采集协议）")
            valid = d.position_valid[:end]
            errors = np.linalg.norm(d.prediction[:end, :2][valid] - d.target[:end, :2][valid], axis=1)
            metric = f"XY RMSE {np.sqrt(np.mean(errors ** 2)):.3f} / P95 {np.percentile(errors, 95):.3f} mm" if len(errors) else "无有效位置误差"
            contact_metric = f" · 接触漏判 {(~valid).sum()}/{end}" if np.isfinite(probability) else " · 无接触分类评估"
            self.metrics.setText(f"已播放 {end} 点 · {metric} · 位置覆盖率 {valid.mean():.1%}{contact_metric}")
            s = d.summary
            refresh = f"{self.render_fps:.1f} FPS" if self.playing else "暂停"
            self.status.setText(f"目标 {args.hz:g} Hz · 速度 {self.speed:g}× · UI 刷新 {refresh} · "
                f"预计算 step 均值 {s['step_mean_ms']:.2f} / P95 {s['step_p95_ms']:.2f} ms · "
                f"超预算 {s['step_over_budget']}/{len(d.metadata)} · 已全部推理，显示至 {end}/{len(d.metadata)}")
            if d.image_only is not None:
                self.status.setText(self.status.text() + f" · 纯视觉 15 Hz / 均值 {d.image_only.latency_ms.mean():.2f} ms")
            self.counter.setText(f"{end}/{len(d.metadata)} · {self.times[i]:.3f} s")
            self.seek.blockSignals(True)
            self.seek.setValue(i)
            self.seek.blockSignals(False)

        def closeEvent(self, event):
            # Keep the worker alive until torch completes; never destroy a running QThread.
            if hasattr(self, "worker") and self.worker.isRunning():
                self.status.setText("正在完成预计算，请稍后关闭窗口")
                event.ignore()
                return
            self.timer.stop()
            event.accept()

    return Window()


if __name__ == "__main__":
    main()
