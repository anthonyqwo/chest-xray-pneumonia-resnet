import os
import argparse
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import build_dataloaders
from models import get_model, list_available_models
from utils import (
    set_seed, compute_binary_metrics, plot_learning_curves, plot_confusion_matrix, save_json,
)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    epoch_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_binary_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
    threshold: float = 0.5,
    return_probs: bool = False,
) -> Dict[str, float]:
    """Predicts PNEUMONIA when P(pneumonia) >= threshold (0.5 == argmax)."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    probs = np.array(all_probs)
    targets = np.array(all_targets)
    eval_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_binary_metrics(targets, (probs >= threshold).astype(int))
    metrics["loss"] = eval_loss
    if return_probs:
        metrics["probs"] = probs
        metrics["targets"] = targets
    return metrics


def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Picks the decision threshold that maximises balanced accuracy on the validation set.
    Balanced accuracy weights both classes equally, so the threshold is not biased by the
    ~3:1 Pneumonia/Normal ratio of the training data. Ties go to the threshold closest to 0.5.
    """
    best_t, best_score = 0.5, -1.0
    for t in sorted(np.arange(0.05, 0.96, 0.01), key=lambda x: abs(x - 0.5)):
        m = compute_binary_metrics(targets, (probs >= t).astype(int))
        specificity = m["tn"] / max(m["tn"] + m["fp"], 1)
        score = (m["recall"] + specificity) / 2
        if score > best_score + 1e-9:
            best_t, best_score = round(float(t), 2), score
    return best_t


def main():
    parser = argparse.ArgumentParser(description="Lab 1: Pneumonia Classification Training")
    parser.add_argument("--model", type=str, default="resnet18", choices=list_available_models(),
                        help="ResNet architecture to train (e.g. resnet18, resnet50)")
    parser.add_argument("--dataset", type=str, default="chest_xray", help="Path to chest_xray folder")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--val_split", type=float, default=0.15, help="Ratio of training set to use as validation")
    parser.add_argument("--img_size", type=int, default=224, help="Image input dimension")
    parser.add_argument("--rotation_deg", type=int, default=15, help="Data augmentation rotation degrees")
    parser.add_argument("--horizontal_flip", action="store_true", help="Apply random horizontal flip")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                        help="Use ImageNet pretrained weights (disable with --no-pretrained)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights")
    parser.add_argument("--monitor", type=str, default="f1", choices=["acc", "f1", "loss"],
                        help="Validation metric used to select the best checkpoint")
    parser.add_argument("--split_mode", type=str, default="patient", choices=["patient", "random"],
                        help="Train/val split: grouped by patient (no leakage) or random by image")
    parser.add_argument("--aug", type=str, default="basic", choices=["basic", "strong"],
                        help="Data augmentation strength")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for CE loss")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout before the final FC layer")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping: stop after N epochs without val improvement (0 = off)")
    parser.add_argument("--tune_threshold", action=argparse.BooleanOptionalAction, default=True,
                        help="Tune the decision threshold on the validation set after training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--fig_dir", type=str, default="figures", help="Directory to save plots")
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Directory to save per-epoch history & final metrics (JSON)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name used for output files (default: model name, e.g. resnet18)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    run_name = args.run_name or args.model

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Training for {run_name} ({args.model}) on {device} ===")
    print(f"Pretrained: {args.pretrained} | Freeze backbone: {args.freeze_backbone} | Monitor: val {args.monitor}")
    print(f"Split: {args.split_mode} | Aug: {args.aug} | Label smoothing: {args.label_smoothing} | "
          f"Dropout: {args.dropout} | Patience: {args.patience}")

    for d in (args.save_dir, args.fig_dir, args.result_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Build DataLoaders
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        data_dir=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        val_split=args.val_split,
        random_seed=args.seed,
        horizontal_flip=args.horizontal_flip,
        rotation_deg=args.rotation_deg,
        split_mode=args.split_mode,
        aug=args.aug,
    )
    if len(train_loader.dataset) == 0:
        raise RuntimeError(f"No training images found under '{args.dataset}/train'. Check --dataset path.")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")
    print(f"Class weights: {class_weights.numpy()} (0: Normal, 1: Pneumonia)")

    # 2. Build Model
    model = get_model(
        model_name=args.model,
        num_classes=2,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout,
    ).to(device)

    # 3. Loss & Optimizer (only optimise trainable params, e.g. when backbone is frozen)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=args.label_smoothing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # 4. Training Loop
    #    Test set is evaluated every epoch ONLY for plotting the curves required by the spec.
    #    Checkpoint selection uses the validation set, so the test set never influences training.
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "test_loss": [], "test_acc": [], "test_f1": [],
        "test_recall": [], "test_precision": [], "test_cm": [],
        "lr": [],
    }
    best_metric = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_weights_path = os.path.join(args.save_dir, f"{run_name}_best.pth")

    for epoch in range(1, args.num_epochs + 1):
        history["lr"].append(optimizer.param_groups[0]["lr"])
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        history["train_loss"].append(train_res["loss"])
        history["train_acc"].append(train_res["accuracy"])
        history["train_f1"].append(train_res["f1"])

        log = (f"[Epoch {epoch:02d}/{args.num_epochs:02d}] "
               f"Train Loss: {train_res['loss']:.4f} | Acc: {train_res['accuracy']:.2f}% | F1: {train_res['f1']:.4f}")

        if val_loader:
            val_res = evaluate(model, val_loader, criterion, device, desc="Validation")
            history["val_loss"].append(val_res["loss"])
            history["val_acc"].append(val_res["accuracy"])
            history["val_f1"].append(val_res["f1"])
            monitor_res = val_res
            log += f" || Val Loss: {val_res['loss']:.4f} | Acc: {val_res['accuracy']:.2f}% | F1: {val_res['f1']:.4f}"
        else:
            monitor_res = train_res
        # Higher is better for every monitored score (loss is negated)
        monitor_score = {"acc": monitor_res["accuracy"], "f1": monitor_res["f1"] * 100.0,
                         "loss": -monitor_res["loss"]}[args.monitor]

        test_res = evaluate(model, test_loader, criterion, device, desc="Testing")
        history["test_loss"].append(test_res["loss"])
        history["test_acc"].append(test_res["accuracy"])
        history["test_f1"].append(test_res["f1"])
        history["test_recall"].append(test_res["recall"])
        history["test_precision"].append(test_res["precision"])
        history["test_cm"].append(test_res["confusion_matrix"])
        log += f" || Test Acc: {test_res['accuracy']:.2f}% | F1: {test_res['f1']:.4f}"
        print(log)

        # Save Best Model (selected on validation, never on test)
        if monitor_score > best_metric:
            best_metric = monitor_score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "monitor": args.monitor,
                "best_score": float(best_metric),
                "dropout": args.dropout,
                "threshold": 0.5,
            }, best_weights_path)
            print(f" -> Best model saved to {best_weights_path} (val {args.monitor}: {abs(best_metric):.4f})")
        else:
            epochs_no_improve += 1

        # Save history every epoch so a crash doesn't lose everything
        save_json(history, os.path.join(args.result_dir, f"{run_name}_history.json"))

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f" -> Early stopping: no val {args.monitor} improvement for {args.patience} epochs")
            break

    # 5. Plot and Save Learning Curves
    curves_path = os.path.join(args.fig_dir, f"{run_name}_learning_curves.png")
    plot_learning_curves(history, curves_path, model_name=run_name.upper())

    # 6. Final Test Evaluation using Best (validation-selected) Weights
    print(f"\n=== Evaluating Best Model (epoch {best_epoch}) on Test Set ===")
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Tune the decision threshold on validation only, then store it with the weights
    threshold = 0.5
    if args.tune_threshold and val_loader:
        val_res = evaluate(model, val_loader, criterion, device, desc="Validation", return_probs=True)
        threshold = tune_threshold(val_res["probs"], val_res["targets"])
        checkpoint["threshold"] = threshold
        torch.save(checkpoint, best_weights_path)
        print(f"Decision threshold tuned on validation (balanced accuracy): {threshold:.2f}")

    def report(res, tag):
        print(
            f"Test Results for {run_name.upper()} ({tag}):\n"
            f"  Accuracy:  {res['accuracy']:.2f}%\n"
            f"  Recall:    {res['recall']:.4f}\n"
            f"  Precision: {res['precision']:.4f}\n"
            f"  F1-Score:  {res['f1']:.4f}\n"
            f"  TP: {res['tp']} | TN: {res['tn']} | FP: {res['fp']} | FN: {res['fn']}"
        )

    test_res_default = evaluate(model, test_loader, criterion, device, desc="Testing")
    report(test_res_default, "threshold 0.50")
    test_res = test_res_default
    if threshold != 0.5:
        test_res = evaluate(model, test_loader, criterion, device, desc="Testing", threshold=threshold)
        report(test_res, f"tuned threshold {threshold:.2f}")

    # Plot Confusion Matrix
    cm_path = os.path.join(args.fig_dir, f"{run_name}_confusion_matrix.png")
    thr_note = f", thr {threshold:.2f}" if threshold != 0.5 else ""
    plot_confusion_matrix(
        test_res["confusion_matrix"],
        cm_path,
        title=f"{run_name.upper()} - Test Confusion Matrix (Acc: {test_res['accuracy']:.2f}%{thr_note})",
    )

    # Save final summary for compare.py / README
    metric_keys = ("accuracy", "recall", "precision", "f1", "tp", "tn", "fp", "fn", "confusion_matrix")
    peak_idx = int(np.argmax(history["test_acc"]))
    save_json({
        "run_name": run_name,
        "model": args.model,
        "args": vars(args),
        "epochs_trained": len(history["train_acc"]),
        "best_epoch": best_epoch,
        "best_val_score": float(best_metric),
        "threshold": threshold,
        "final_test": {k: test_res[k] for k in metric_keys},
        "final_test_default_threshold": {k: test_res_default[k] for k in metric_keys},
        "peak_test_acc_epoch": peak_idx + 1,
        "peak_test_acc": history["test_acc"][peak_idx],
    }, os.path.join(args.result_dir, f"{run_name}_summary.json"))
    print(f"=== Completed Training & Evaluation for {run_name} ===\n")


if __name__ == "__main__":
    main()
