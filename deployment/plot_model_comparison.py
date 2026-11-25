"""Render every retained circle trajectory from the saved comparison arrays."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--audit-scripts", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.audit_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    root = args.results.resolve()
    summary = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    with np.load(root / "predictions.npz", allow_pickle=False) as z:
        truth, baseline, candidate = [z[k].copy() for k in ("target", "baseline", "candidate")]
        sequences, resets = z["sequence"].copy(), z["reset_state"].copy()
    names = list(dict.fromkeys(sequences.tolist()))
    if len(names) != 10:
        raise ValueError("The comparison panel requires all ten retained circle trajectories")
    # All retained samples and unmodified absolute positions enter the figure.
    extent = np.concatenate([a[:, :2].ravel() for a in (truth, baseline, candidate)])
    limit = max(10., np.ceil(np.abs(extent).max() / 2.) * 2.)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "legend.frameon": False})
    fig, axes = plt.subplots(2, 5, figsize=(14., 6.6))
    fig.subplots_adjust(left=.05, right=.985, bottom=.095, top=.81, wspace=.29, hspace=.53)
    handles = None
    for ax, name in zip(axes.ravel(), names):
        indices = np.flatnonzero(sequences == name)
        lines = []
        for values, color, label, width, style in [
            (truth, "#343a40", "Ground truth", 1.5, "-"),
            (baseline, "#3976ae", "Triple-head", 1.1, "-"),
            (candidate, "#db8336", "Slip-gated", 1.1, "--"),
        ]:
            # Break lines at acquisition gaps; show the first point of each
            # segment rather than joining separate recurrent-state segments.
            starts = np.r_[0, np.flatnonzero(resets[indices][1:]) + 1]
            ends = np.r_[starts[1:], len(indices)]
            handle = None
            for start, end in zip(starts, ends):
                h, = ax.plot(values[indices[start:end], 0], values[indices[start:end], 1],
                    color=color, lw=width, ls=style, label=label if handle is None else None)
                handle = h if handle is None else handle
            lines.append(handle)
        handles = lines
        detail = next(x for x in summary["per_trajectory"] if x["sequence"] == name)
        a = detail["baseline"]["xy_mean_euclidean_mm"]
        b = detail["candidate"]["xy_mean_euclidean_mm"]
        ax.set_title(f"{name.rsplit('/',1)[-1]}   n={len(indices)}\nXY error: {a:.2f} / {b:.2f} mm", pad=9)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(color="#dce2e7", lw=.5, alpha=.8)
    for ax in axes[1]:
        ax.set_xlabel("X (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Y (mm)")
    fig.suptitle("External circle trajectories: matched best.pt comparison", fontsize=14, y=.99)
    fig.text(.5, .944,
        "10 trajectories | 1,150 events | tail trim 0.029 mm | test XY filter off | microphone 5 disabled",
        ha="center", fontsize=9)
    fig.legend(handles, ["Ground truth", "Triple-head", "Slip-gated"],
        loc="upper center", bbox_to_anchor=(.5, .919), ncol=3, fontsize=9,
        handlelength=3., columnspacing=2.8)
    fig.canvas.draw()
    stem = root / "circle_xy_comparison"
    alignment = require_matplotlib_panel_alignment(fig,
        json_out=str(stem)+".alignment.json", overlay_svg=str(stem)+".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    for extension in ("svg", "pdf"):
        fig.savefig(str(stem)+"."+extension)
    fig.savefig(str(stem)+".png", dpi=300)
    plt.close(fig)
    (root/"figure_contract.json").write_text(json.dumps({
        "question": "How do the two saved estimators reconstruct the same ten external circle trajectories?",
        "archetype": "quantitative grid", "backend": "python",
        "claim_boundary": "Descriptive paired comparison on ten trajectories; no significance or population-generalization claim.",
        "all_retained_events_plotted": len(truth), "sample_exclusions": "Only the shared requested tail trimming; no plotting subsampling",
        "coordinates": "Raw absolute physical XY positions; no fitted alignment, smoothing or integration",
        "uncertainty": "Each panel shows one complete retained trajectory; no uncertainty bars apply to individual traces",
        "panel_size": "Equal rendered XY areas with identical limits and aspect ratios",
        "alignment_verdict": alignment.get("verdict", alignment.get("status")),
        "dimensions_inches": [14., 6.6], "font_floor_pt": 8,
        "display": "Panel errors are baseline / candidate mean XY Euclidean error in mm"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("Exported circle_xy_comparison.png / .svg / .pdf")


if __name__ == "__main__":
    main()
