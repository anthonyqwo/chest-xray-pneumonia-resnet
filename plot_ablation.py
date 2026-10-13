"""
Ablation figures: how each change affects ResNet-18 / ResNet-50.

Reads results/{run}_history.json and results/{run}_summary.json written by train.py and produces:
  figures/ablation_test_curves.png  - test accuracy per epoch, one panel per architecture,
                                      with the validation-selected epoch marked
  figures/ablation_final_metrics.png - final test accuracy / F1 / precision of every run

Usage:
  python plot_ablation.py
"""
import os
import argparse

import matplotlib.pyplot as plt

from utils import load_json

# Fixed order: the same configuration always gets the same color
CONFIGS = [
    ("", "Baseline (random val split)", "#2a78d6"),
    ("_split", "+ Patient-level val split", "#eb6834"),
    ("_reg", "+ Patient split & regularization", "#1baf7a"),
]
INK, MUTED = "#0b0b0b", "#898781"


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(True, axis="y", linestyle=":", color=MUTED, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Plot ablation figures")
    parser.add_argument("--models", nargs="+", default=["resnet18", "resnet50"])
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--fig_dir", type=str, default="figures")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    runs = {}
    for model in args.models:
        for suffix, label, color in CONFIGS:
            run = model + suffix
            h_path = os.path.join(args.result_dir, f"{run}_history.json")
            s_path = os.path.join(args.result_dir, f"{run}_summary.json")
            if os.path.exists(h_path) and os.path.exists(s_path):
                runs[run] = (model, label, color, load_json(h_path), load_json(s_path))

    # 1. Test accuracy per epoch, one panel per architecture (shared y-axis)
    fig, axes = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 4.8), sharey=True)
    axes = [axes] if len(args.models) == 1 else axes
    for ax, model in zip(axes, args.models):
        for run, (m, label, color, h, s) in runs.items():
            if m != model:
                continue
            ep = list(range(1, len(h["test_acc"]) + 1))
            ax.plot(ep, h["test_acc"], color=color, linewidth=2, label=label)
            best = s["best_epoch"]
            ax.plot(best, h["test_acc"][best - 1], "o", color=color, markersize=8,
                    markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax.axhline(90, color=MUTED, linewidth=1, linestyle="--")
        ax.text(1, 90.3, "90%", color=MUTED, fontsize=9)
        ax.set_title(f"{model.upper()} - Test Accuracy per Epoch", fontsize=13, fontweight="bold", color=INK)
        ax.set_xlabel("Epoch", fontsize=11)
        style_axis(ax)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", color=MUTED, linestyle="", markersize=8))
    labels.append("Epoch selected on validation")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(args.fig_dir, "ablation_test_curves.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {path}")

    # 2. Final test metrics (validation-selected checkpoint) - grouped bars
    metrics = [("accuracy", "Test Accuracy (%)", 1.0), ("f1", "Test F1-score", 100.0),
               ("precision", "Test Precision", 100.0)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.6))
    n_cfg = len(CONFIGS)
    width = 0.8 / n_cfg
    for ax, (key, title, scale) in zip(axes, metrics):
        for j, (suffix, label, color) in enumerate(CONFIGS):
            for i, model in enumerate(args.models):
                run = model + suffix
                x = i + (j - (n_cfg - 1) / 2) * width
                if run not in runs:
                    ax.text(x, 2, "not run", ha="center", va="bottom", rotation=90, fontsize=8, color=MUTED)
                    continue
                value = runs[run][4]["final_test"][key] * scale
                ax.bar(x, value, width=width - 0.02, color=color, label=label if i == 0 else None)
                shown = f"{value:.1f}" if key == "accuracy" else f"{value / 100:.3f}"
                ax.text(x, value + 1, shown, ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(args.models)))
        ax.set_xticklabels([m.upper() for m in args.models])
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 20))
        if key != "accuracy":
            ax.set_yticklabels([f"{t / 100:.1f}" for t in range(0, 101, 20)])
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(args.fig_dir, "ablation_final_metrics.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {path}")


if __name__ == "__main__":
    main()
