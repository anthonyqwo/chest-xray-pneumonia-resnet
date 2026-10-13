import os
import argparse
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import build_dataloaders, get_transforms
from models import get_model, list_available_models
from utils import compute_binary_metrics, plot_confusion_matrix


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_cm_path: str = None,
    model_name: str = "Model",
    threshold: float = 0.5,
):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, 1]
            preds = (probs >= threshold).long().cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    metrics = compute_binary_metrics(np.array(all_targets), np.array(all_preds))

    print(f"\n==================== Evaluation Results ({model_name}, threshold {threshold:.2f}) ====================")
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"TP: {metrics['tp']:4d} | FP: {metrics['fp']:4d}")
    print(f"FN: {metrics['fn']:4d} | TN: {metrics['tn']:4d}")
    print("========================================================================\n")

    if save_cm_path:
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            save_cm_path,
            title=f"{model_name.upper()} - Test Confusion Matrix (Acc: {metrics['accuracy']:.2f}%)",
        )

    return metrics


def predict_single_image(model: nn.Module, image_path: str, device: torch.device, img_size: int = 224,
                         threshold: float = 0.5):
    _, eval_tf = get_transforms(img_size=img_size)
    image = Image.open(image_path).convert("RGB")
    tensor_img = eval_tf(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
        predicted_class_id = int(probabilities[1] >= threshold)

    result_class = CLASS_NAMES[predicted_class_id]
    confidence = probabilities[predicted_class_id] * 100.0

    print(f"\n[Single Image Prediction]")
    print(f"Image:      {image_path}")
    print(f"Prediction: {result_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities -> Normal: {probabilities[0]*100:.2f}%, Pneumonia: {probabilities[1]*100:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Lab 1: Pneumonia Classification Inference")
    parser.add_argument("--model", type=str, default="resnet18", choices=list_available_models())
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint weights (.pth)")
    parser.add_argument("--dataset", type=str, default="chest_xray", help="Path to chest_xray folder")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image for inference")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--fig_dir", type=str, default="figures")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold on P(pneumonia); default: value tuned on val and saved "
                             "in the checkpoint (0.5 for older checkpoints)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found at: {args.weights}")

    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    dropout = checkpoint.get("dropout", 0.2)

    # Build model and load state
    model = get_model(model_name=args.model, num_classes=2, pretrained=False, dropout_rate=dropout).to(device)
    model.load_state_dict(state_dict)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)
    print(f"Successfully loaded weights from: {args.weights} (threshold {threshold:.2f})")

    if args.image:
        predict_single_image(model, args.image, device, img_size=args.img_size, threshold=threshold)
    else:
        _, _, test_loader, _ = build_dataloaders(
            data_dir=args.dataset,
            batch_size=args.batch_size,
            num_workers=2,
            img_size=args.img_size,
            val_split=0.0,
        )
        cm_save_path = os.path.join(args.fig_dir, f"{args.model}_eval_cm.png")
        evaluate_test_set(model, test_loader, device, save_cm_path=cm_save_path, model_name=args.model,
                          threshold=threshold)


if __name__ == "__main__":
    main()
