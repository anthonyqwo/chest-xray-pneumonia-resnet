import os
import json
import random
import textwrap
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def set_seed(seed: int = 42):
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes accuracy, precision, recall, f1-score, and confusion matrix.
    0: NORMAL (Negative)
    1: PNEUMONIA (Positive)
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1) * 100.0
    precision = tp / max(tp + fp, 1e-10)
    recall = tp / max(tp + fn, 1e-10)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-10)

    # Standard 2x2 confusion matrix:
    # [[TN, FP],
    #  [FN, TP]]
    c_matrix = [[tn, fp], [fn, tp]]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": c_matrix,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_json(obj, save_path: str):
    """Save a dict (history / metrics / args) as UTF-8 JSON."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_split(ax, epochs, history, key, label, color, linestyle):
    values = history.get(key)
    if values:
        ax.plot(epochs, values, label=label, color=color, linewidth=2, linestyle=linestyle)


def plot_learning_curves(history: Dict[str, List[float]], save_path: str, model_name: str = "Model"):
    """
    Plots per-epoch Accuracy and F1-score curves for train / validation / test.
    (Spec: training & testing accuracy curve, testing F1-score curve;
     validation curves are the +5 bonus.)
    """
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    _plot_split(axes[0], epochs, history, "train_acc", "Train Accuracy", "#1f77b4", "-")
    _plot_split(axes[0], epochs, history, "val_acc", "Val Accuracy", "#ff7f0e", "--")
    _plot_split(axes[0], epochs, history, "test_acc", "Test Accuracy", "#2ca02c", ":")
    axes[0].set_title(f"{model_name} - Accuracy vs Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # F1-Score Plot
    _plot_split(axes[1], epochs, history, "train_f1", "Train F1-Score", "#1f77b4", "-")
    _plot_split(axes[1], epochs, history, "val_f1", "Val F1-Score", "#ff7f0e", "--")
    _plot_split(axes[1], epochs, history, "test_f1", "Test F1-Score", "#2ca02c", ":")
    axes[1].set_title(f"{model_name} - F1-Score vs Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("F1-Score", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[Plot Saved] Learning curves saved to: {save_path}")


def plot_confusion_matrix(
    cm: List[List[int]],
    save_path: str,
    title: str = "Test Set Confusion Matrix",
    labels: List[str] = None,
):
    """
    Plots a 2x2 confusion matrix heatmap using seaborn.
    cm format:
      [[TN, FP],
       [FN, TP]]
    """
    if labels is None:
        labels = ["NORMAL (0)", "PNEUMONIA (1)"]

    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"},
    )

    # Wrap long titles so they don't get clipped by the narrow figure,
    # preferring to break before a trailing "(...)" so it stays on one line
    if len(title) > 40:
        head, sep, tail = title.rpartition(" (")
        title = f"{head}\n({tail}" if sep and len(head) <= 45 else textwrap.fill(title, width=40)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual Ground Truth", fontsize=11, labelpad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot Saved] Confusion matrix saved to: {save_path}")
