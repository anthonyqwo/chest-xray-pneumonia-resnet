# Lab 1: Pneumonia Classification on Chest X-Ray Images

This repository contains the implementation of deep learning models (ResNet-18 and ResNet-50) for binary classification of chest X-ray images into **Normal** and **Pneumonia**.

**Final results** (checkpoint and decision threshold selected on the validation set only):

| Model | Test Accuracy | Test F1-Score |
| :--- | :---: | :---: |
| ResNet-18 | **92.79%** | **0.9451** |
| ResNet-50 | **93.43%** | **0.9496** |

---

## 📁 Project Structure

```text
chest-xray-pneumonia-resnet/
├── models/
│   ├── __init__.py
│   └── resnet_models.py      # ResNet model builder with pretrained weights & new classification head
├── dataset.py                # Custom Dataset, patient-level train/val split, data augmentation & DataLoader
├── utils.py                  # Evaluation metrics & plotting (learning curves / confusion matrix)
├── train.py                  # Main training script (train / val / test curves every epoch)
├── compare.py                # Cross-model comparison figures & results table
├── plot_ablation.py          # Ablation figures (baseline vs. improved settings)
├── inference.py              # Test set evaluation and single image prediction
├── requirements.txt          # Python dependencies
├── .gitignore
├── checkpoints/              # Trained weights (.pth)
├── figures/                  # Learning curves, confusion matrices, comparison & ablation figures
├── results/                  # Per-epoch history & final metrics (JSON) written by train.py
└── README.md
```

---

## 🛠️ Environment Setup

Tested with Python 3.14, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0 on Windows 11 with an NVIDIA RTX 4060 Ti (8 GB).

### 1. Create a virtual environment
```bash
python -m venv .venv
```
Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Linux/macOS).

### 2. Install PyTorch with CUDA
A plain `pip install torch` installs the **CPU-only** build on Windows. Install the CUDA build from the PyTorch index first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install the remaining dependencies
```bash
pip install -r requirements.txt
```

> **Windows note**: if `import torch` fails with `WinError 126 ... c10.dll`, install the
> [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).
> The first `import torch` after installation can take a few minutes while Windows scans the new DLLs.

---

## 📊 Dataset Preparation

1. Download the **[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)** dataset from Kaggle (`archive.zip`, ~2.3 GB).
2. Extract it and place the `train/`, `val/` and `test/` folders in a `chest_xray/` folder inside the project. The archive also contains a duplicated `chest_xray/chest_xray/` folder and a `__MACOSX/` folder, which are not needed.
3. The directory should be structured as follows:

```text
chest_xray/
├── train/   NORMAL (1,341)  PNEUMONIA (3,875)
├── val/     NORMAL (8)      PNEUMONIA (8)
└── test/    NORMAL (234)    PNEUMONIA (390)
```

> **Validation set**: the original `val/` folder has only 16 images, so `dataset.py` builds a proper validation set from `train/` (~15%), stratified by class. By default the split is done **by patient**: the file names identify the patient (`person123_...`, `IM-0115-...`), and all images of one patient go to the same split. 2,846 patients contribute the 5,216 training images, and 1,075 patients have more than one image. With a plain random split, 62.5% of the validation images belong to a patient who also appears in the training set (see [Experimental Results](#-experimental-results)).

---

## 🚀 How to Train

### 1. Final setting (recommended)
Patient-level validation split, strong augmentation, label smoothing, dropout 0.5 and early stopping on validation loss:

```bash
python train.py --model resnet18 --run_name resnet18_reg --num_epochs 15 --batch_size 64 --lr 1e-4 --split_mode patient --aug strong --label_smoothing 0.1 --dropout 0.5 --patience 5 --monitor loss
```
```bash
python train.py --model resnet50 --run_name resnet50_reg --num_epochs 15 --batch_size 64 --lr 1e-4 --split_mode patient --aug strong --label_smoothing 0.1 --dropout 0.5 --patience 5 --monitor loss
```

On an RTX 4060 Ti, ResNet-18 takes ~18 s per epoch and ResNet-50 ~35 s per epoch.

> **GPU memory**: ResNet-50 with batch size 64 needs ~7 GB of GPU memory. On an 8 GB card, close other programs that use the GPU (browsers, other training jobs, ...) before training. Otherwise Windows moves part of the memory to system RAM and training becomes 4–5× slower. If memory is still short, use `--batch_size 32`.

### 2. Ablation runs (to reproduce the comparison)
```bash
python train.py --model resnet18 --run_name resnet18 --num_epochs 25 --split_mode random
```
```bash
python train.py --model resnet50 --run_name resnet50 --num_epochs 25 --split_mode random
```
```bash
python train.py --model resnet18 --run_name resnet18_split --num_epochs 25 --split_mode patient
```

> The baseline runs `resnet18` / `resnet50` in the results below were trained with an earlier version of the DataLoader (2 workers, restarted every epoch). Random augmentations are drawn per worker, so rerunning them with the current code gives slightly different numbers.

### Options

| Option | Default | Description |
| :--- | :---: | :--- |
| `--model` | `resnet18` | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnext50_32x4d` |
| `--run_name` | model name | Name of the output files, so several runs of one model don't overwrite each other |
| `--split_mode` | `patient` | `patient`: train/val split grouped by patient; `random`: split by image |
| `--aug` | `basic` | `basic`: resize, ±15° rotation, small shift, light brightness/contrast jitter. `strong`: random resized crop (scale 0.75–1.0), larger shift (10%), stronger jitter (0.2), occasional Gaussian blur |
| `--label_smoothing` | `0.0` | Label smoothing for the cross-entropy loss |
| `--dropout` | `0.2` | Dropout before the final fully connected layer |
| `--patience` | `0` | Early stopping after N epochs without validation improvement (0 = off) |
| `--monitor` | `f1` | Validation metric used to select the best checkpoint: `f1`, `acc` or `loss` |
| `--tune_threshold` / `--no-tune_threshold` | on | Tune the decision threshold on the validation set after training |
| `--num_workers` | `4` | DataLoader workers |
| `--no-pretrained` | — | Train from scratch instead of from ImageNet weights |
| `--freeze_backbone` | — | Only train the new classification head |

### Training outputs
- Best weights (selected on **validation**, never on test): `checkpoints/{run}_best.pth`. The checkpoint also stores the tuned decision threshold.
- Train / Val / Test accuracy & F1 curves: `figures/{run}_learning_curves.png`
- Final test confusion matrix (best checkpoint): `figures/{run}_confusion_matrix.png`
- Per-epoch history and final metrics: `results/{run}_history.json` and `results/{run}_summary.json`

> The test set is evaluated after every epoch **only to draw the curves required by the spec**. It is never used for checkpoint selection, threshold tuning, early stopping or hyper-parameter tuning.

**Decision threshold**: after training, the best checkpoint is evaluated on the validation set, and the threshold on P(pneumonia) that maximizes **balanced accuracy** (the mean of recall and specificity) is chosen. Balanced accuracy weights both classes equally, so the ~3:1 Pneumonia/Normal ratio of the training data does not bias the threshold. The summary JSON stores test metrics both at threshold 0.5 and at the tuned threshold.

### 3. Compare models
```bash
python compare.py --runs resnet18_reg resnet50_reg --tag v2
```
```bash
python compare.py --runs resnet18 resnet50
```
```bash
python plot_ablation.py
```

`compare.py` produces (with `--tag NAME`, files are named `comparison_NAME_*`):
- `figures/comparison_accuracy.png`: training vs testing accuracy of every model
- `figures/comparison_test_f1.png`: testing F1-score of every model
- `figures/comparison_train.png`: training accuracy & F1 of every model
- `figures/comparison_val.png`: validation accuracy & F1 (bonus)
- `figures/{run}_peak_test_cm.png`: confusion matrix at the epoch with the highest testing accuracy
- `results/comparison_table.md`: Markdown results table

`plot_ablation.py` produces:
- `figures/ablation_test_curves.png`: test accuracy per epoch for each setting, with the epoch selected on validation marked
- `figures/ablation_final_metrics.png`: final test accuracy / F1 / precision for each setting

---

## 🔍 How to Run Inference

### 1. Evaluate a model on the test set
```bash
python inference.py --model resnet18 --weights checkpoints/resnet18_reg_best.pth --dataset chest_xray
```

### 2. Predict a single chest X-ray image
```bash
python inference.py --model resnet18 --weights checkpoints/resnet18_reg_best.pth --image path/to/xray.jpeg
```

`inference.py` uses the decision threshold saved in the checkpoint (0.5 for older checkpoints). Use `--threshold` to override it.

---

## 📈 Experimental Results

Common setup: ImageNet-pretrained weights with a new classification head, batch size 64, AdamW (lr 1e-4, weight decay 1e-4), cosine learning-rate schedule, inverse-frequency class weights in the loss, seed 42, image size 224.

### Final models

| Model Architecture | Pretrained | Epochs trained | Best Epoch (val) | Threshold | Test Accuracy (%) | Test Recall | Test Precision | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | Yes (ImageNet) | 11 (early stop) | 6 | 0.50 | **92.79** | 0.9923 | 0.9021 | **0.9451** |
| **ResNet-50** | Yes (ImageNet) | 12 (early stop) | 7 | 0.76 | **93.43** | 0.9897 | 0.9125 | **0.9496** |

Test confusion matrices:

| Model | TP | TN | FP | FN |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-18 | 387 | 192 | 42 | 3 |
| ResNet-50 | 386 | 197 | 37 | 4 |

At the default threshold of 0.5, ResNet-50 reaches 90.06% test accuracy (F1 0.9260). For ResNet-18, the tuned threshold is 0.5 itself.

Highest testing accuracy reached during training (the spec asks for the confusion matrix at this epoch; see `figures/{run}_peak_test_cm.png`):

| Model | Peak Test Accuracy (%) | Epoch |
| :--- | :---: | :---: |
| ResNet-18 | 95.51 | 4 |
| ResNet-50 | 95.35 | 4 |

### Ablation

| Run | Val split | Augmentation / regularization | Best epoch (val) | Threshold | Test Acc (%) | F1 | Precision | FP |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `resnet18` | random | basic aug, dropout 0.2, 25 epochs, select on val F1 | 17 | 0.50 | 84.94 | 0.8920 | 0.8083 | 92 |
| `resnet18_split` | patient | same as above | 11 | 0.41 | 85.10 | 0.8932 | 0.8087 | 92 |
| `resnet18_reg` | patient | strong aug, label smoothing 0.1, dropout 0.5, early stop on val loss | 6 | 0.50 | **92.79** | **0.9451** | **0.9021** | **42** |
| `resnet50` | random | basic aug, dropout 0.2, 25 epochs, select on val F1 | 12 | 0.50 | 84.29 | 0.8881 | 0.8004 | 97 |
| `resnet50_reg` | patient | strong aug, label smoothing 0.1, dropout 0.5, early stop on val loss | 7 | 0.76 | **93.43** | **0.9496** | **0.9125** | **37** |

> The baseline runs were trained before threshold tuning existed and use 0.5. `resnet18_split` reaches 85.58% at threshold 0.5.

Figures:
- `figures/comparison_v2_train.png`, `figures/comparison_v2_val.png`, `figures/{run}_confusion_matrix.png`: final models (training curves, validation curves, final test confusion matrices)
- `figures/comparison_v2_accuracy.png`, `figures/comparison_v2_test_f1.png`: final models (training vs testing accuracy, testing F1)
- `figures/comparison_accuracy.png`, `figures/comparison_test_f1.png`, `figures/comparison_val.png`: baseline models
- `figures/ablation_test_curves.png`, `figures/ablation_final_metrics.png`: ablation

### Observations
- **Overfitting was the main problem.** With the baseline setting, training accuracy reaches ~100% within a few epochs, while test accuracy jumps between ~78% and ~94% from one epoch to the next. Which of these values the validation-selected epoch lands on is largely a matter of luck (here 84–85%).
- **Random validation splits leak patients.** 62.5% of the random validation images belong to patients who also appear in training, so validation accuracy stays at 98–99% and says little about unseen patients. Fixing the split alone barely changes the test result (84.94% → 85.10%), because the model still overfits.
- **Regularization makes test accuracy stable.** With strong augmentation, label smoothing, dropout 0.5 and early stopping, test accuracy stays between ~90% and ~95.5% across epochs. The validation-selected checkpoints therefore reach 92.79% (ResNet-18) and 93.43% (ResNet-50).
- **The errors are mostly false positives.** Almost every pneumonia case is detected (recall ≥ 0.99). Regularization cut the number of normal images classified as pneumonia from 92–97 to 37–42, raising precision from ~0.80 to ~0.90–0.91.
- **Threshold tuning on validation is inconsistent.** It helped ResNet-50 (0.76: 90.06% → 93.43%), left ResNet-18 at 0.50, and slightly hurt `resnet18_split` (0.41: 85.58% → 85.10%). The validation set comes from `train/`, whose class ratio (~3:1) and image distribution differ from `test/` (~1.7:1).
- **ResNet-50 brings only a small gain over ResNet-18** (+0.6% accuracy), while it has ~2× the parameters (23.5M vs 11.2M) and needs nearly all of an 8 GB GPU at batch size 64. With ~5k training images and a binary task, ResNet-18 already has enough capacity.

---

## 🔬 Key Technical Highlights
- **Patient-level validation split**: stratified by class and grouped by patient (`StratifiedGroupKFold`), so no patient appears in both training and validation.
- **Medical-specific augmentation**: modest rotations (±15°), random resized crops, small shifts, contrast/brightness jitter and occasional blur. No vertical flips, which would produce anatomically unrealistic images.
- **Class imbalance handling**: inverse-frequency class weights in `CrossEntropyLoss` for the ~3:1 Pneumonia/Normal ratio.
- **Regularization**: label smoothing, dropout before the classifier, and early stopping on validation loss.
- **Validation-only model selection**: the checkpoint, the stopping epoch and the decision threshold are all chosen on validation; the test set is used only for reporting.
- **Validation bonus**: validation accuracy and F1 curves are plotted for every run.
