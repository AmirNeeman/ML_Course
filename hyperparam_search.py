"""
Stage 3 of 5 - Hyperparameter Search and Cross-Validation
================================================================================
Project : Shout / scream vs. neutral sound classification with a 2-D CNN
Purpose : Tune learning rate and optimiser with scikit-learn's GridSearchCV,
          cross-validate the best configuration, and visualise the learning
          curve and confusion matrix - all with plotly express.

Bridging PyTorch and scikit-learn
---------------------------------
scikit-learn's model-selection tools (GridSearchCV, cross_val_score,
StratifiedKFold) operate on estimators that implement fit / predict / score.
`TorchCNNClassifier` below subclasses BaseEstimator / ClassifierMixin and wraps
the MobileNetV2 training loop, so those sklearn tools drive the PyTorch network
directly. The MFCC features are resized to (1, 64, 64) once and kept in memory
as a single numpy array, which makes every cross-validation fit fast.

Outputs (interactive plotly figures, HTML + PNG)
------------------------------------------------
    grid_search       GridSearchCV: mean CV accuracy per configuration
    cross_validation  per-fold accuracy of the best configuration
    learning_curve    train / validation accuracy versus epoch
    confusion_matrix  confusion matrix of the best configuration

Usage
-----
    pip install plotly pandas scikit-learn torch torchvision
    python hyperparam_search.py

References
----------
    Pedregosa et al. (2011), "scikit-learn: Machine Learning in Python", JMLR.
================================================================================
"""

import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px

# Reuse the cached-feature loaders and the model builder from Stage 2.
from train_model import read_manifest, balance_subsample, build_model, TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ==========================================================================
#  CONFIGURATION  (kept small so the search finishes in minutes on CPU)
# ==========================================================================
IMG_SIZE = TrainConfig().img_size            # 64
SAMPLES_PER_CLASS = 100                       # balanced subsample (200 total)

# IMPORTANT: this MobileNetV2 stays collapsed at ~0.50 validation accuracy
# until ~epoch 17 (see learning_curve), then jumps to ~0.97. SEARCH_EPOCHS
# must sit AFTER that onset, otherwise GridSearchCV scores every configuration
# at chance and cannot tell them apart. 35 gives each config room to generalise
# during the search.
SEARCH_EPOCHS = 35                            # epochs per fit during the search
FINAL_EPOCHS = 60                             # longer fit for the learning curve
CV_FOLDS = 3

# The grid swept by GridSearchCV. Each key maps to a TorchCNNClassifier
# __init__ argument; 'optimizer' is the learning-method comparison
# (Adam vs. SGD + momentum).
PARAM_GRID = {
    "lr": [1e-3, 3e-4],
    "optimizer": ["adam", "sgd"],
}


# ==========================================================================
#  FEATURE LOADING  ->  in-memory X, y   (resize each MFCC to 64 x 64 once)
# ==========================================================================
def load_xy() -> Tuple[np.ndarray, np.ndarray]:
    """Load the balanced MFCC features into memory as CNN-ready tensors.

    Returns:
        X: float32 array of shape (N, 1, IMG_SIZE, IMG_SIZE).
        y: int64 label array of shape (N,).
    """
    paths, labels = read_manifest("manifest.csv")
    paths, labels = balance_subsample(paths, labels, SAMPLES_PER_CLASS, SEED)

    X = np.empty((len(paths), 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, fp in enumerate(paths):
        mfcc = np.loadtxt(fp, delimiter=",", dtype=np.float32)
        if mfcc.ndim == 1:
            mfcc = mfcc[:, None]
        t = torch.from_numpy(mfcc).unsqueeze(0).unsqueeze(0)        # (1,1,13,T)
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False)
        X[i] = t.squeeze(0).numpy()                                 # (1,H,W)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


# ==========================================================================
#  SCIKIT-LEARN-COMPATIBLE WRAPPER AROUND MobileNetV2
# ==========================================================================
class TorchCNNClassifier(BaseEstimator, ClassifierMixin):
    """A scikit-learn estimator wrapping the MobileNetV2 training loop.

    The constructor exposes the hyperparameters GridSearchCV tunes. After
    fit(), per-epoch metrics are available in `self.history_` (used for the
    learning curve); passing X_val / y_val to fit() also logs validation
    accuracy each epoch.
    """

    def __init__(self, lr: float = 1e-3, optimizer: str = "adam",
                 weight_decay: float = 1e-4, batch_size: int = 32,
                 epochs: int = 15, img_size: int = IMG_SIZE, seed: int = SEED):
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.seed = seed

    # -- helpers --------------------------------------------------------
    def _make_optimizer(self, params):
        """Construct the optimiser named by self.optimizer."""
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.lr,
                                    weight_decay=self.weight_decay)
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=0.9,
                                   weight_decay=self.weight_decay)
        raise ValueError(f"unknown optimizer: {self.optimizer}")

    def _accuracy(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))

    # -- scikit-learn API -----------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the network. Records per-epoch history in self.history_."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.classes_ = np.unique(y)

        self.model_ = build_model(num_classes=len(self.classes_)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._make_optimizer(self.model_.parameters())

        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        n = len(Xt)

        self.history_ = {"epoch": [], "train_acc": [], "train_loss": [],
                         "val_acc": []}

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            perm = torch.randperm(n)
            run_loss, correct = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb, yb = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                run_loss += loss.item() * len(idx)
                correct += (logits.argmax(1) == yb).sum().item()

            self.history_["epoch"].append(epoch)
            self.history_["train_loss"].append(run_loss / n)
            self.history_["train_acc"].append(correct / n)
            self.history_["val_acc"].append(
                self._accuracy(X_val, y_val) if X_val is not None else np.nan
            )
        return self

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        """Return predicted class indices for X (batched inference)."""
        self.model_.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(DEVICE)
        out = []
        for s in range(0, len(Xt), 256):
            out.append(self.model_(Xt[s:s + 256]).argmax(1).cpu().numpy())
        return np.concatenate(out)

    def score(self, X, y) -> float:
        """Mean accuracy on (X, y) - the score sklearn maximises."""
        return accuracy_score(y, self.predict(X))


# ==========================================================================
#  PLOTS  (plotly express)
# ==========================================================================
def save_fig(fig, name: str) -> None:
    """Save a plotly figure as interactive HTML, plus PNG if kaleido exists."""
    fig.write_html(f"{name}.html")
    try:
        fig.write_image(f"{name}.png", scale=2)
        print(f"  saved {name}.html + {name}.png")
    except Exception:
        print(f"  saved {name}.html  (PNG skipped: kaleido unavailable)")


def plot_grid_search(cv_results: dict) -> None:
    """Bar chart of mean CV accuracy per grid configuration."""
    df = pd.DataFrame(cv_results)
    df["config"] = df["params"].apply(
        lambda d: ", ".join(f"{k}={v}" for k, v in d.items())
    )
    df = df.sort_values("mean_test_score", ascending=False)
    fig = px.bar(
        df, x="config", y="mean_test_score", error_y="std_test_score",
        title="GridSearchCV - mean CV accuracy per configuration",
        labels={"mean_test_score": "CV accuracy", "config": "hyperparameters"},
        text=df["mean_test_score"].map(lambda v: f"{v:.3f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1])
    save_fig(fig, "grid_search")


def plot_cross_validation(scores: np.ndarray) -> None:
    """Bar chart of per-fold accuracy for the best configuration."""
    df = pd.DataFrame({"fold": [f"fold {i+1}" for i in range(len(scores))],
                       "accuracy": scores})
    fig = px.bar(
        df, x="fold", y="accuracy",
        title=(f"{len(scores)}-fold cross-validation (best config) - "
               f"mean={scores.mean():.3f} ± {scores.std():.3f}"),
        text=df["accuracy"].map(lambda v: f"{v:.3f}"),
    )
    fig.add_hline(y=scores.mean(), line_dash="dash",
                  annotation_text=f"mean {scores.mean():.3f}")
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1])
    save_fig(fig, "cross_validation")


def plot_learning_curve(history: dict) -> None:
    """Line chart of train and validation accuracy versus epoch."""
    df = pd.DataFrame(history)
    long = df.melt(id_vars="epoch",
                   value_vars=["train_acc", "val_acc"],
                   var_name="metric", value_name="accuracy")
    fig = px.line(
        long, x="epoch", y="accuracy", color="metric", markers=True,
        title="Learning curve - accuracy vs epoch (best config)",
    )
    fig.update_yaxes(range=[0, 1])
    save_fig(fig, "learning_curve")


def plot_confusion(cm: np.ndarray, class_names: List[str]) -> None:
    """Heatmap of the confusion matrix for the best configuration."""
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=class_names, y=class_names,
        labels=dict(x="Predicted", y="Actual", color="count"),
        title="Confusion matrix - best config (held-out test set)",
    )
    save_fig(fig, "confusion_matrix")


# ==========================================================================
#  MAIN
# ==========================================================================
def main() -> None:
    t0 = time.time()
    print(f"Device: {DEVICE}")
    print("Loading + resizing features into memory ...")
    X, y = load_xy()
    print(f"  X={X.shape}  y={y.shape}  "
          f"({int((y==1).sum())} shouts / {int((y==0).sum())} neutral)")

    class_map = TrainConfig().class_map
    class_names = [k for k, _ in sorted(class_map.items(), key=lambda kv: kv[1])]

    # Hold out a test set for the final confusion matrix; search / CV on the rest.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # 1. GridSearchCV - hyperparameter tuning (short epochs per fit).
    print(f"\n[1/4] GridSearchCV over {PARAM_GRID} "
          f"({CV_FOLDS}-fold, {SEARCH_EPOCHS} epochs/fit) ...")
    base = TorchCNNClassifier(epochs=SEARCH_EPOCHS)
    gs = GridSearchCV(base, PARAM_GRID, scoring="accuracy", cv=cv,
                      n_jobs=1, refit=True, return_train_score=False)
    gs.fit(X_tr, y_tr)
    print(f"  best params : {gs.best_params_}")
    print(f"  best CV acc : {gs.best_score_:.3f}")
    plot_grid_search(gs.cv_results_)

    # 2. Cross-validation of the best configuration.
    print(f"\n[2/4] {CV_FOLDS}-fold cross_val_score on best config ...")
    best = TorchCNNClassifier(epochs=SEARCH_EPOCHS, **gs.best_params_)
    cv_scores = cross_val_score(best, X_tr, y_tr, cv=cv, scoring="accuracy",
                                n_jobs=1)
    print(f"  fold scores : {np.round(cv_scores, 3)}")
    print(f"  mean +/- std: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    plot_cross_validation(cv_scores)

    # 3. Learning curve versus epoch (best config, longer training).
    print(f"\n[3/4] Training best config for {FINAL_EPOCHS} epochs "
          f"to record the learning curve ...")
    final = TorchCNNClassifier(epochs=FINAL_EPOCHS, **gs.best_params_)
    final.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)   # validation curve = test set
    plot_learning_curve(final.history_)

    # 4. Confusion matrix on the held-out test set.
    print("\n[4/4] Confusion matrix on held-out test set ...")
    y_pred = final.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    test_acc = accuracy_score(y_te, y_pred)
    print(f"  held-out test accuracy: {test_acc:.3f}")
    print(f"  confusion matrix:\n{cm}")
    plot_confusion(cm, class_names)

    print(f"\nDone in {time.time()-t0:.0f}s. Open the .html files in a browser.")


if __name__ == "__main__":
    main()
