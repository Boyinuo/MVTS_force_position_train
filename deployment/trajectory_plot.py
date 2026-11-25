"""Portable Matplotlib trajectory sheets for post-training external evaluation."""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


DISPLAY_NAMES = {"model_triple_head": "Triple-head", "model_slip_gated": "Slip-gated",
                 "model_contact_gated": "Contact-gated"}


def check_panel_alignment(fig, axes, path):
    """Measure final axes in points; keep runtime independent of external skills."""
    fig.canvas.draw()
    width, height = fig.get_size_inches() * 72.
    boxes = np.array([[a.get_position().x0 * width, a.get_position().y0 * height,
                       a.get_position().width * width, a.get_position().height * height]
                      for a in axes.ravel() if a.get_visible()])
    if len(boxes) > 1 and (np.ptp(boxes[:, 2]) > 1.5 or np.ptp(boxes[:, 3]) > 1.5):
        raise ValueError("Trajectory panels have unequal rendered dimensions")
    for row in axes:
        selected = [a for a in row if a.get_visible()]
        if len(selected) > 1:
            bottoms = [a.get_position().y0 * height for a in selected]
            if np.ptp(bottoms) > 1.5:
                raise ValueError("Trajectory panel row is misaligned")
    report = {"verdict": "PASS", "units": "pt", "panel_rectangles": boxes.tolist(), "tolerance_pt": 1.5}
    # Optional authoring QA uses the same final Matplotlib geometry, never an
    # alternate renderer. Normal training needs only the portable checks above.
    audit_root = os.environ.get("NATURE_FIGURE_AUDIT_SCRIPTS")
    if audit_root:
        sys.path.insert(0, audit_root)
        from audit_panel_alignment import require_matplotlib_panel_alignment
        report = require_matplotlib_panel_alignment(fig, tolerance_pt=1.5,
            gutter_tolerance_pt=1.5, strict=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_trajectory_sheet(output, dataset_name, inputs, prediction, result_metrics,
                          model_name, epoch, trim_mm, filter_xy, *, position_valid=None, image_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    target = inputs["target"]
    sequences = np.asarray([r["sequence"] for r in inputs["rows"]])
    resets = np.asarray([r["reset_state"] for r in inputs["rows"]])
    names = list(dict.fromkeys(sequences.tolist()))
    cols, rows = min(5, len(names)), math.ceil(len(names) / 5)
    if not names:
        raise ValueError("Cannot draw an empty external test selection")
    finite = np.concatenate([a[:, :2].ravel() for a in (target, prediction)])
    limit = max(10., float(np.ceil(np.abs(finite).max() / 2.) * 2.))
    label = DISPLAY_NAMES.get(model_name, model_name)
    if image_only:
        label += " (image only)"
    color = {"model_slip_gated": "#db8336", "model_contact_gated": "#8064a2"}.get(model_name, "#3976ae")
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": .7, "legend.frameon": False}):
        fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 3.3 * rows), squeeze=False)
        fig.subplots_adjust(left=.05, right=.985, bottom=.095, top=.81, wspace=.29, hspace=.53)
        handles = None
        for index, ax in enumerate(axes.ravel()):
            if index >= len(names):
                ax.set_visible(False)
                continue
            seq = names[index]
            indices = np.flatnonzero(sequences == seq)
            current = prediction[indices, :2].copy()
            if position_valid is not None:
                current[~position_valid[indices]] = np.nan
            artists = []
            for values, line_color, text, lw in ((target[indices, :2], "#343a40", "Ground truth", 1.5),
                                                (current, color, label, 1.15)):
                starts = np.r_[0, np.flatnonzero(resets[indices][1:]) + 1]
                ends = np.r_[starts[1:], len(indices)]
                first = None
                for start, end in zip(starts, ends):
                    line, = ax.plot(values[start:end, 0], values[start:end, 1],
                                    color=line_color, lw=lw, label=text if first is None else None)
                    if first is None:
                        first = line
                artists.append(first)
            handles = artists
            error = np.linalg.norm(prediction[indices, :2] - target[indices, :2], axis=1)
            mask = inputs["valid"][indices, :2].all(1)
            xy = float(error[mask].mean()) if mask.any() else None
            value = f"{xy:.2f} mm" if xy is not None else "N/A"
            ax.set_title(f"{seq.rsplit('/',1)[-1]}   n={len(indices)}\nXY mean error: {value}", pad=9)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.grid(color="#dce2e7", lw=.5, alpha=.8)
        for ax in axes[-1]:
            ax.set_xlabel("X (mm)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Y (mm)")
        fig.suptitle(f"{dataset_name}: best.pt trajectory predictions", fontsize=14, y=.99)
        fig.text(.5, .944,
            f"{len(names)} trajectories | {len(target):,} events | best epoch {epoch} | "
            f"tail trim {trim_mm:g} mm | test XY filter {'on' if filter_xy else 'off'}",
            ha="center", fontsize=9)
        fig.legend(handles, ["Ground truth", label], loc="upper center",
                   bbox_to_anchor=(.5, .919), ncol=2, fontsize=9, handlelength=3., columnspacing=2.8)
        details = output / "demo_tests" / dataset_name
        details.mkdir(parents=True, exist_ok=True)
        check_panel_alignment(fig, axes, details / "figure_alignment.json")
        for extension in ("png", "svg"):
            destination = output / f"{dataset_name}.{extension}"
            temporary = destination.with_suffix(".tmp")
            fig.savefig(temporary, format=extension, dpi=300)
            temporary.replace(destination)
        fig.savefig(details / "figure.pdf")
        plt.close(fig)
    (details / "figure_contract.json").write_text(json.dumps({
        "backend": "python", "question": "How does this saved estimator reconstruct each external trajectory?",
        "archetype": "quantitative grid", "events_plotted": len(target), "trajectories": len(names),
        "all_retained_events_used": True, "coordinate_alignment": "none", "smoothing": "none",
        "line_breaks": "acquisition gaps and predicted-invalid positions", "font_floor_pt": 8,
        "uncertainty": "Individual trajectories; no aggregate uncertainty bars apply",
        "export_intent": "Inspection/report sheet, 300-dpi preview and editable SVG/PDF; no journal target",
        "xy_mean_euclidean_mm": result_metrics["xy_mean_euclidean_mm"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return {"png": str(output / f"{dataset_name}.png"), "svg": str(output / f"{dataset_name}.svg")}
