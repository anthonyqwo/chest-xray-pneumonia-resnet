import torch
import torch.nn as nn
from torchvision import models


MODEL_REGISTRY = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "resnext50_32x4d": (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.DEFAULT),
}


def list_available_models():
    """Returns a list of supported model architectures."""
    return list(MODEL_REGISTRY.keys())


def get_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.2,
) -> nn.Module:
    """
    Build and return a ResNet architecture for chest X-ray classification.

    Args:
        model_name: Name of the ResNet architecture ('resnet18', 'resnet50', etc.)
        num_classes: Number of target classes (default 2: Normal, Pneumonia)
        pretrained: Whether to initialize with ImageNet pretrained weights
        freeze_backbone: If True, freezes all parameters except the classifier head
        dropout_rate: Dropout probability applied before the final Linear layer
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are: {list_available_models()}"
        )

    model_fn, weights = MODEL_REGISTRY[model_name]
    model = model_fn(weights=weights if pretrained else None)

    # Optionally freeze feature extractor backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer (classification head)
    in_features = model.fc.in_features
    if dropout_rate > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)

    return model
