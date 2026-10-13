import os
import re
from typing import Tuple, List, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


class ChestXRayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray Pneumonia Classification.
    Labels:
        0: NORMAL
        1: PNEUMONIA
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def scan_image_folder(folder_path: str) -> Tuple[List[str], List[int]]:
    """
    Scans a folder structured with NORMAL and PNEUMONIA subfolders.
    """
    file_paths = []
    labels = []
    classes = {"NORMAL": 0, "PNEUMONIA": 1}

    for class_name, label in classes.items():
        class_dir = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_dir):
            # Check lowercase as fallback
            class_dir = os.path.join(folder_path, class_name.lower())
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    return file_paths, labels


def patient_id(path: str) -> str:
    """
    Extracts the patient ID from a Kaggle chest X-ray file name, e.g.
      person1_bacteria_1.jpeg  -> person1
      IM-0115-0001.jpeg        -> IM-0115
      NORMAL2-IM-0123-0001.jpeg -> NORMAL2-IM-0123
    Many patients have several images, so splitting by image leaks patients across splits.
    """
    fname = os.path.basename(path)
    m = re.match(r"(person\d+)_", fname) or re.match(r"((?:NORMAL2-)?IM-\d+)-", fname)
    return m.group(1) if m else fname


def split_train_val(
    paths: List[str],
    labels: List[int],
    val_split: float,
    random_seed: int,
    split_mode: str = "patient",
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """
    Splits train into train / val stratified by class.
      split_mode="patient": all images of a patient go to the same split (no patient leakage)
      split_mode="random":  plain image-level split
    """
    if split_mode == "random":
        return train_test_split(paths, labels, test_size=val_split, stratify=labels, random_state=random_seed)

    n_splits = max(2, round(1.0 / val_split))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    groups = [patient_id(p) for p in paths]
    tr_idx, vl_idx = next(sgkf.split(paths, labels, groups))
    return (
        [paths[i] for i in tr_idx], [paths[i] for i in vl_idx],
        [labels[i] for i in tr_idx], [labels[i] for i in vl_idx],
    )


def get_transforms(
    img_size: int = 224,
    rotation_deg: int = 15,
    brightness: float = 0.1,
    contrast: float = 0.1,
    horizontal_flip: bool = False,
    aug: str = "basic",
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation transforms tailored for medical chest X-rays.
    aug="strong" adds random resized crops, larger shifts / jitter and occasional blur
    to reduce overfitting.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    strong = aug == "strong"
    if strong:
        brightness, contrast = max(brightness, 0.2), max(contrast, 0.2)

    train_transform_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1))
        if strong else transforms.Resize((img_size, img_size)),
    ]

    if horizontal_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if rotation_deg > 0:
        train_transform_list.append(
            transforms.RandomRotation(degrees=(-rotation_deg, rotation_deg))
        )

    # Slight translation and scaling to simulate patient posture variation
    train_transform_list.append(
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1) if strong else (0.05, 0.05),
                                scale=(0.95, 1.05))
    )

    if brightness > 0 or contrast > 0:
        train_transform_list.append(
            transforms.ColorJitter(brightness=brightness, contrast=contrast)
        )

    if strong:
        train_transform_list.append(
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2)
        )

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return transforms.Compose(train_transform_list), eval_transform


def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    img_size: int = 224,
    val_split: float = 0.15,
    random_seed: int = 42,
    horizontal_flip: bool = False,
    rotation_deg: int = 15,
    split_mode: str = "patient",
    aug: str = "basic",
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, torch.Tensor]:
    """
    Builds and returns (train_loader, val_loader, test_loader, class_weights).

    If val_split > 0, splits the training set into Train and Validation sets
    (stratified by class, grouped by patient unless split_mode="random") to produce
    reliable validation curves for the bonus requirement.
    """
    train_folder = os.path.join(data_dir, "train")
    test_folder = os.path.join(data_dir, "test")
    val_folder = os.path.join(data_dir, "val")

    train_paths, train_labels = scan_image_folder(train_folder)
    test_paths, test_labels = scan_image_folder(test_folder)

    train_tf, eval_tf = get_transforms(
        img_size=img_size,
        rotation_deg=rotation_deg,
        horizontal_flip=horizontal_flip,
        aug=aug,
    )

    # Handle Validation Set Split
    if val_split > 0 and len(train_paths) > 0:
        tr_paths, vl_paths, tr_labels, vl_labels = split_train_val(
            train_paths, train_labels, val_split, random_seed, split_mode=split_mode,
        )
        train_dataset = ChestXRayDataset(tr_paths, tr_labels, transform=train_tf)
        val_dataset = ChestXRayDataset(vl_paths, vl_labels, transform=eval_tf)
    else:
        # Fall back to folder val if available
        train_dataset = ChestXRayDataset(train_paths, train_labels, transform=train_tf)
        vl_paths, vl_labels = scan_image_folder(val_folder)
        val_dataset = (
            ChestXRayDataset(vl_paths, vl_labels, transform=eval_tf)
            if vl_paths
            else None
        )

    test_dataset = ChestXRayDataset(test_paths, test_labels, transform=eval_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        if val_dataset is not None
        else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # Compute class weights for handling class imbalance (Normal: 0, Pneumonia: 1)
    if len(train_dataset) > 0:
        labels_list = [label for _, label in (zip(tr_paths, tr_labels) if val_split > 0 else zip(train_paths, train_labels))]
        n_normal = sum(1 for y in labels_list if y == 0)
        n_pneumonia = sum(1 for y in labels_list if y == 1)
        total = n_normal + n_pneumonia

        # Weight inversely proportional to class frequency: total / (2 * count)
        w_normal = total / (2.0 * max(n_normal, 1))
        w_pneumonia = total / (2.0 * max(n_pneumonia, 1))
        class_weights = torch.FloatTensor([w_normal, w_pneumonia])
    else:
        class_weights = torch.FloatTensor([1.0, 1.0])

    return train_loader, val_loader, test_loader, class_weights
