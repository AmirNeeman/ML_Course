# CNN Model for Human Emotion Classification Based on MFCC

Final project — Machine Learning course.

This repository contains the Python pipeline for an MFCC-based emotion
classification model, used to test the hypothesis that **base MFCC features
perform as well as MFCC + Δ + ΔΔ** (delta and delta-delta).

## Scripts

| File | Stage | Description |
|------|-------|-------------|
| `feature_extraction.py` | 1 | Converts raw audio into base-MFCC feature matrices and caches them to disk (CSV). |
| `train_model.py` | 2 | Trains a MobileNetV2 (2D-CNN) on the MFCC features and reports Accuracy / Precision / Recall / F1 + a confusion matrix. |
| `hyperparam_search.py` | 3 | Hyperparameter search with `GridSearchCV` and cross-validation; produces the learning-curve and grid-search plots. |

## Requirements

Python 3.9+ with:

```
librosa numpy torch torchvision scikit-learn matplotlib plotly pandas tqdm
```

## Notes

The audio datasets (RAVDESS, TESS) and the reference papers are not included in
this repository — only the source code is submitted for review.
