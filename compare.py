"""
Compare multiple trained models (e.g. ResNet-18 vs ResNet-50).

Reads results/{run}_history.json and results/{run}_summary.json written by train.py and produces:
  figures/comparison_accuracy.png   - train & test accuracy per epoch for every model
  figures/comparison_test_f1.png    - test F1-score per epoch for every model
  figures/comparison_train.png      - training accuracy & F1 per epoch
  figures/comparison_val.png        - validation accuracy & F1 per epoch (bonus)
  figures/{run}_peak_test_cm.png    - confusion matrix at the epoch with the highest test accuracy
  results/comparison_table.md       - Markdown table to paste into README / report

With --tag NAME, the comparison_* files become comparison_NAME_*.

Usage:
  python compare.py --runs resnet18 resnet50
  python compare.py --runs resnet18_reg resnet50_reg --tag v2
"""
import os
import argparse

import matplotlib.pyplot as plt

from utils import load_json, plot_confusion_matrix

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _finish(fig, ax, title, ylabel, save_path):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def _plot_acc_f1(histories, split, split_title, save_path):
    """Side-by-side accuracy and F1-score per epoch of one split (train / val) for every model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (run, h) in enumerate(histories.items()):
        if not h.get(f"{split}_acc"):
            continue
        c = COLORS[i % len(COLORS)]
        ep = range(1, len(h[f"{split}_acc"]) + 1)
        axes[0].plot(ep, h[f"{split}_acc"], color=c, linewidth=2, label=f"{run} {split}")
        axes[1].plot(ep, h[f"{split}_f1"], color=c, linewidth=2, label=f"{run} {split}")
    for ax, t, y in ((axes[0], f"{split_title} Accuracy", "Accuracy (%)"),
                     (axes[1], f"{split_title} F1-score", "F1-score")):
        ax.set_title(t, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare training histories of several models")
    parser.add_argument("--runs", nargs="+", default=["resnet18", "resnet50"],
                        help="Run names (same as --run_name / --model used in train.py)")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--fig_dir", type=str, default="figures")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for output files, e.g. 'v2' -> comparison_v2_accuracy.png")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    prefix = f"comparison_{args.tag}_" if args.tag else "comparison_"

    histories, summaries = {}, {}
    for run in args.runs:
        h_path = os.path.join(args.result_dir, f"{run}_history.json")
        s_path = os.path.join(args.result_dir, f"{run}_summary.json")
        if not os.path.exists(h_path):
            print(f"[Skip] {h_path} not found - train '{run}' first.")
            continue
        histories[run] = load_json(h_path)
        summaries[run] = load_json(s_path) if os.path.exists(s_path) else None

    if not histories:
        raise SystemExit("No histories found. Run train.py first.")

    # 1. Train & Test accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (run, h) in enumerate(histories.items()):
        c = COLORS[i % len(COLORS)]
        ep = range(1, len(h["train_acc"]) + 1)
        ax.plot(ep, h["train_acc"], color=c, linewidth=2, label=f"{run} train")
        if h.get("test_acc"):
            ax.plot(ep, h["test_acc"], color=c, linewidth=2, linestyle="--", label=f"{run} test")
    _finish(fig, ax, "Training vs Testing Accuracy", "Accuracy (%)",
            os.path.join(args.fig_dir, f"{prefix}accuracy.png"))

    # 2. Test F1-score
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (run, h) in enumerate(histories.items()):
        if h.get("test_f1"):
            ep = range(1, len(h["test_f1"]) + 1)
            ax.plot(ep, h["test_f1"], color=COLORS[i % len(COLORS)], linewidth=2, label=f"{run} test F1")
    _finish(fig, ax, "Testing F1-score", "F1-score", os.path.join(args.fig_dir, f"{prefix}test_f1.png"))

    # 3. Training accuracy & F1 curves (spec 3.b.i)
    _plot_acc_f1(histories, "train", "Training", os.path.join(args.fig_dir, f"{prefix}train.png"))

    # 4. Validation accuracy & F1 curves (spec 3.b.ii, bonus)
    if any(h.get("val_acc") for h in histories.values()):
        _plot_acc_f1(histories, "val", "Validation", os.path.join(args.fig_dir, f"{prefix}val.png"))

    # 5. Confusion matrix at the epoch with the highest testing accuracy
    for run, h in histories.items():
        if h.get("test_cm"):
            idx = max(range(len(h["test_acc"])), key=lambda k: h["test_acc"][k])
            plot_confusion_matrix(
                h["test_cm"][idx],
                os.path.join(args.fig_dir, f"{run}_peak_test_cm.png"),
                title=f"{run.upper()} - Highest Test Acc (Epoch {idx + 1}: {h['test_acc'][idx]:.2f}%)",
            )

    # 5. Markdown table
    lines = [
        "| Model | Best epoch (val) | Threshold | Test Acc (%) | Recall | Precision | F1 | Peak Test Acc (%) @ epoch |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for run, s in summaries.items():
        if s is None:
            continue
        t = s["final_test"]
        lines.append(
            f"| {run} | {s['best_epoch']} | {s.get('threshold', 0.5):.2f} | {t['accuracy']:.2f} | {t['recall']:.4f} | "
            f"{t['precision']:.4f} | {t['f1']:.4f} | {s['peak_test_acc']:.2f} @ {s['peak_test_acc_epoch']} |"
        )
    table = "\n".join(lines)
    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, f"{prefix}table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
