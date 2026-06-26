"""
Stage 2 of 5 - Model Training
================================================================================
Project : Shout / scream vs. neutral sound classification with a 2-D CNN
Purpose : Train a MobileNetV2 convolutional network on the cached base-MFCC
          features and report standard classification metrics.

Overview
--------
The cached MFCC matrices (produced by feature_extraction.py) are treated as
single-channel images. A MobileNetV2 network - chosen for its high accuracy at
low computational cost - is adapted to accept this 1-channel input and to emit
two output classes. The model is trained from scratch and evaluated on a
held-out test split.

Why MobileNetV2 / the 64 x 64 resize
------------------------------------
MFCC matrices have shape (13, T) with a small frequency axis (13) and a
variable time axis (T). MobileNetV2 down-samples spatially by a factor of 32,
so a height of 13 would collapse to zero. Each MFCC is therefore resized to a
fixed 64 x 64 square ("spectrogram-as-image"), giving every sample an identical
shape and a frequency axis large enough for the network.

Pipeline
--------
    1. Read manifest.csv (the index of cached MFCC CSV files).
    2. Optionally take a balanced subsample (equal clips per class).
    3. Stratified 70 / 30 train / test split.
    4. Train MobileNetV2 (CrossEntropyLoss + Adam) for a fixed number of epochs.
    5. Report Accuracy / Precision / Recall / F1 and a confusion matrix.
    6. Save the trained weights.

Inputs
------
    manifest.csv             produced by feature_extraction.py
    features/<class>/*.csv   base-MFCC matrices (13 x T)

Outputs
-------
    mobilenetv2_shout_classifier.pth   trained model weights
    confusion_matrix.png               test-set confusion matrix

Usage (after Stage 1)
---------------------
    pip install torch torchvision numpy scikit-learn matplotlib tqdm
    python train_model.py

References
----------
    Sandler et al. (2018), "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks", CVPR.
================================================================================
"""

import csv
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==========================================================================
#  CONFIGURATION  (training-side only; feature parameters live in Stage 1)
# ==========================================================================
@dataclass
class TrainConfig:
    manifest_path: str = "manifest.csv"
    class_map = {"neutral": 0, "shouts": 1}     # "shouts" is positive (1)

    # Fixed square the (13 x T) MFCC is resized to, so MobileNetV2's
    # down-sampling stages never collapse the small frequency axis to 0.
    img_size: int = 64

    # Balanced subsample: take this many clips PER CLASS before splitting.
    # Keeps the two classes even (avoids the 500-vs-288 imbalance) and makes
    # CPU training quick. Set to 0 (or None) to use every available file.
    samples_per_class: int = 100

    epochs: int = 100                  # number of training epochs
    batch_size: int = 32
    lr: float = 1e-3                   # Adam learning rate
    weight_decay: float = 1e-4         # L2 regularisation
    test_size: float = 0.30            # 70 / 30 train / test split
    num_workers: int = 0               # 0 is safe on Windows
    seed: int = 42                     # reproducibility
    weights_out: str = "mobilenetv2_shout_classifier.pth"
    cm_out: str = "confusion_matrix.png"


CFG = TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
#  DATA LOADING
# ==========================================================================
def read_manifest(path: str) -> Tuple[List[str], List[int]]:
    """Read the feature manifest into parallel path / label lists.

    Args:
        path: Path to manifest.csv (columns: feature_path, label, class_name).

    Returns:
        (feature_paths, labels) as two parallel lists.

    Raises:
        RuntimeError: if the manifest is empty.
    """
    paths: List[str] = []
    labels: List[int] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["feature_path"])
            labels.append(int(row["label"]))
    if not paths:
        raise RuntimeError(f"Manifest '{path}' is empty. Run feature_extraction.py first.")
    return paths, labels


def balance_subsample(paths: List[str], labels: List[int],
                      n_per_class: int, seed: int
                      ) -> Tuple[List[str], List[int]]:
    """Randomly keep n_per_class items from each label so classes are even.

    Args:
        paths:        Feature file paths.
        labels:       Integer labels parallel to paths.
        n_per_class:  Items to keep per class. If falsy (0 / None) the data is
                      returned unchanged. A class with fewer files keeps all.
        seed:         RNG seed for a reproducible selection.

    Returns:
        The balanced (paths, labels) lists.
    """
    if not n_per_class:
        return paths, labels

    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[str]] = {}
    for p, y in zip(paths, labels):
        by_label.setdefault(y, []).append(p)

    sel_paths: List[str] = []
    sel_labels: List[int] = []
    for y, plist in sorted(by_label.items()):
        idx = rng.permutation(len(plist))[:n_per_class]   # random pick
        for i in idx:
            sel_paths.append(plist[i])
            sel_labels.append(y)
    return sel_paths, sel_labels


class MFCCDataset(Dataset):
    """PyTorch dataset that loads a cached MFCC CSV and returns it as a
    single-channel (1, img_size, img_size) image tensor plus its label."""

    def __init__(self, paths: List[str], labels: List[int],
                 cfg: TrainConfig = CFG):
        self.paths = paths
        self.labels = labels
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the cached MFCC matrix from CSV (13 rows x T columns).
        mfcc = np.loadtxt(self.paths[idx], delimiter=",", dtype=np.float32)
        # A clip with a single time frame loads as 1-D -> restore (13, 1).
        if mfcc.ndim == 1:
            mfcc = mfcc[:, None]

        # (13, T) -> (1, 1, 13, T) so it can be bilinearly resized.
        x = torch.from_numpy(mfcc).float().unsqueeze(0).unsqueeze(0)
        # Resize to a fixed square so every sample has an identical shape.
        x = F.interpolate(
            x,
            size=(self.cfg.img_size, self.cfg.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.squeeze(0)                                       # (1, H, W)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ==========================================================================
#  MODEL  (MobileNetV2 adapted to single-channel MFCC input)
# ==========================================================================
def build_model(num_classes: int = 2) -> nn.Module:
    """Build a MobileNetV2 adapted for MFCC images.

    Two modifications are made to the standard torchvision MobileNetV2:
      * the first convolution accepts 1 input channel (MFCC) rather than 3
        (RGB), and
      * the classifier head emits `num_classes` logits.

    Args:
        num_classes: Number of output classes (2: neutral, shouts).

    Returns:
        The adapted, untrained MobileNetV2 module.
    """
    model = torchvision.models.mobilenet_v2(weights=None)

    # Input layer: 3-channel RGB -> 1-channel MFCC spectrogram.
    old_conv = model.features[0][0]            # Conv2d(3, 32, k=3, s=2, p=1)
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # Output layer: 1000 ImageNet classes -> num_classes.
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)
    return model


# ==========================================================================
#  TRAINING AND EVALUATION
# ==========================================================================
def train(model: nn.Module, train_loader: DataLoader,
          test_loader: DataLoader, cfg: TrainConfig = CFG) -> None:
    """Train the model in place, printing train/val metrics each epoch.

    Args:
        model:        The network to train (moved to DEVICE internally).
        train_loader: DataLoader over the training split.
        test_loader:  DataLoader over the test split (for per-epoch val acc).
        cfg:          Training configuration.
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.epochs}",
                         leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, test_loader, verbose=False)["accuracy"]
        print(f"Epoch {epoch:3d}/{cfg.epochs} | loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             verbose: bool = True) -> Dict[str, Any]:
    """Evaluate the model and compute classification metrics.

    The positive class is "shouts" (label 1), so precision / recall / F1 are
    reported for that class specifically.

    Args:
        model:   Trained network.
        loader:  DataLoader to evaluate over.
        verbose: If True, also print a full classification report.

    Returns:
        A dict with accuracy / precision / recall / f1 and the raw
        y_true / y_pred arrays (used to draw the confusion matrix).
    """
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())

    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0),
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    }

    if verbose:
        print("\n================ TEST-SET EVALUATION ================")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}   (positive = 'shouts')")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print("\nClassification report:")
        names = [k for k, _ in sorted(CFG.class_map.items(), key=lambda kv: kv[1])]
        print(classification_report(y_true_arr, y_pred_arr,
                                    target_names=names, zero_division=0))
    return metrics


def plot_confusion_matrix(metrics: Dict[str, Any], cfg: TrainConfig = CFG) -> None:
    """Render and save the confusion matrix from an evaluate() result."""
    names = [k for k, _ in sorted(cfg.class_map.items(), key=lambda kv: kv[1])]
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Shouts vs. Neutral")
    plt.tight_layout()
    plt.savefig(cfg.cm_out, dpi=150)
    print(f"\nConfusion matrix saved to: {cfg.cm_out}")
    plt.show()


# ==========================================================================
#  MAIN
# ==========================================================================
def main() -> None:
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    paths, labels = read_manifest(CFG.manifest_path)
    print(f"Loaded {len(paths)} cached features "
          f"({sum(labels)} shouts / {len(labels) - sum(labels)} neutral).")

    # Balanced subsample (e.g. 100 per class) for an even, fast-to-train set.
    paths, labels = balance_subsample(paths, labels, CFG.samples_per_class, CFG.seed)
    if CFG.samples_per_class:
        print(f"Balanced to {CFG.samples_per_class}/class -> {len(paths)} total "
              f"({sum(labels)} shouts / {len(labels) - sum(labels)} neutral).")

    # Stratified 70 / 30 split keeps the class ratio identical in both parts.
    p_tr, p_te, y_tr, y_te = train_test_split(
        paths, labels,
        test_size=CFG.test_size,
        random_state=CFG.seed,
        stratify=labels,
    )
    print(f"Train: {len(p_tr)}  |  Test: {len(p_te)}")

    train_loader = DataLoader(
        MFCCDataset(p_tr, y_tr), batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        MFCCDataset(p_te, y_te), batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model(num_classes=len(CFG.class_map))
    print(f"\nModel: MobileNetV2 (1-channel input, {len(CFG.class_map)} classes)\n")

    train(model, train_loader, test_loader, CFG)

    metrics = evaluate(model, test_loader, verbose=True)
    plot_confusion_matrix(metrics, CFG)

    torch.save(model.state_dict(), CFG.weights_out)
    print(f"Model weights saved to: {CFG.weights_out}")


if __name__ == "__main__":
    main()
