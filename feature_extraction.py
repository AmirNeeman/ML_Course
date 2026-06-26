"""
Stage 1 of 5 - Feature Extraction
================================================================================
Project : Shout / scream vs. neutral sound classification with a 2-D CNN
Purpose : Convert raw 1-D audio waveforms into 2-D base-MFCC feature matrices
          that are cached to disk and later treated as single-channel images
          by the convolutional network.

Overview
--------
For every audio file in the class folders, this script computes a base
Mel-Frequency Cepstral Coefficient (MFCC) matrix of shape (n_mfcc, T) = (13, T)
and writes it to disk. Doing the (relatively slow) audio decoding and MFCC
computation once - rather than on every training epoch - keeps the downstream
training loop fast and fully reproducible.

Feature-extraction methodology (fixed parameters)
-------------------------------------------------
The MFCC chain follows the standard speech-processing pipeline:

    1. Resample to 16 kHz, mono.
    2. Pre-emphasis high-pass filter:  y[n] = x[n] - alpha * x[n-1].
    3. Frame blocking into 20 ms frames with 50 % overlap.
    4. Hamming window per frame.
    5. Short-time FFT -> power spectrum.
    6. Mel filter bank: 40 overlapping triangular filters.
    7. Logarithm, then DCT-II; keep the first 13 cepstral coefficients.

Only the 13 *base* coefficients are kept. Delta (first-derivative) and
delta-delta (second-derivative) temporal features are deliberately NOT
computed, in line with the reference methodology for this task.

Inputs
------
    data/
      shouts/    *.wav, *.flac, *.mp3, ...   (label 1, positive class)
      neutral/   *.wav, *.flac, *.mp3, ...   (label 0)

Outputs
-------
    features/
      neutral/<name>.csv      one (13 x T) MFCC matrix per audio file
      shouts/<name>.csv
    manifest.csv              index of (feature_path, label, class_name)
    extraction_params.json    the exact parameters used (for reproducibility)

Usage
-----
    pip install librosa numpy tqdm
    python feature_extraction.py

References
----------
    Davis & Mermelstein (1980), "Comparison of parametric representations for
        monosyllabic word recognition" - the MFCC formulation.
    McFee et al., "librosa: Audio and music signal analysis in Python".
================================================================================
"""

import os
import csv
import glob
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import librosa
from tqdm import tqdm


# ==========================================================================
#  CONFIGURATION
#  Every fixed MFCC parameter is declared here so the methodology is
#  transparent and changeable in a single place.
# ==========================================================================
@dataclass
class FeatureConfig:
    # ---- Paths and class labels ---------------------------------------
    data_root: str = "data"             # input audio, organised by class
    out_root: str = "features"          # where the MFCC .csv files are written
    manifest_path: str = "manifest.csv"
    params_path: str = "extraction_params.json"
    # Folder name -> integer label. "shouts" is the positive class (1).
    class_map = {"neutral": 0, "shouts": 1}

    # ---- Audio / MFCC parameters --------------------------------------
    sample_rate: int = 16_000           # 16 kHz target sampling rate
    pre_emphasis: float = 0.97          # alpha in [0.9, 1.0] (high-pass filter)
    frame_ms: float = 20.0              # frame length in milliseconds
    overlap: float = 0.50               # fractional overlap between frames
    n_mels: int = 40                    # number of triangular mel filters
    n_mfcc: int = 13                    # number of base cepstral coefficients
    n_fft: int = 512                    # FFT size (>= win_length, power of two)
    fmin: float = 0.0                   # lowest mel-filter frequency
    fmax: Optional[float] = None        # highest; None -> Nyquist (sr / 2)
    cmn: bool = True                    # per-utterance cepstral mean/var norm

    # ---- Derived values (computed in __post_init__) -------------------
    win_length: int = 0                 # frame length in samples
    hop_length: int = 0                 # hop between frames in samples

    def __post_init__(self) -> None:
        # Frame length in samples, e.g. 20 ms * 16 000 Hz = 320 samples.
        self.win_length = int(round(self.sample_rate * self.frame_ms / 1000.0))
        # 50 % overlap implies a hop of half the frame length.
        self.hop_length = int(round(self.win_length * (1.0 - self.overlap)))
        if self.fmax is None:
            self.fmax = self.sample_rate / 2.0


CFG = FeatureConfig()


# ==========================================================================
#  CORE TRANSFORM:  one audio file -> base-MFCC matrix (13, T)
# ==========================================================================
def extract_mfcc(file_path: str, cfg: FeatureConfig = CFG) -> np.ndarray:
    """Compute the base-MFCC matrix for a single audio file.

    Args:
        file_path: Path to an audio file readable by librosa.
        cfg:       Feature configuration (parameters of the MFCC chain).

    Returns:
        A float32 array of shape (n_mfcc, T) = (13, num_frames) containing the
        13 base cepstral coefficients per frame. No delta / delta-delta rows.
    """
    # Step 0: load the raw 1-D signal, resampled to 16 kHz, mixed to mono.
    signal, _ = librosa.load(file_path, sr=cfg.sample_rate, mono=True)

    # Guard against empty / silent files so downstream code never sees length 0.
    if signal.size == 0:
        signal = np.zeros(cfg.win_length, dtype=np.float32)

    # Step 1: pre-emphasis high-pass filter  y[n] = x[n] - alpha * x[n-1].
    # Boosts high frequencies (consonants, sharp scream onsets) and flattens
    # the spectrum prior to the FFT.
    emphasized = np.append(signal[0], signal[1:] - cfg.pre_emphasis * signal[:-1])

    # Steps 2-6 are performed internally by librosa.feature.mfcc:
    #   2. Frame blocking : win_length frames, hop_length stride (50 % overlap).
    #   3. Windowing      : Hamming window applied to every frame.
    #   4. FFT            : short-time power spectrum (power = 2.0).
    #   5. Mel filter bank: 40 overlapping triangular filters (n_mels = 40).
    #   6. log + DCT-II   : log-compress mel energies, decorrelate with the DCT,
    #                       and keep the first 13 coefficients (n_mfcc = 13).
    mfcc = librosa.feature.mfcc(
        y=emphasized,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,        # 13 base coefficients
        dct_type=2,               # standard DCT-II
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hamming",         # Hamming window
        n_mels=cfg.n_mels,        # 40 mel filters
        power=2.0,                # power spectrum
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        center=True,
    )

    # Methodological constraint: no delta (first-derivative) or delta-delta
    # (second-derivative) features are computed or stacked. Only the 13 base
    # coefficients are returned (note: no call to librosa.feature.delta).

    # Per-utterance cepstral mean / variance normalisation for numerical
    # stability and to reduce channel/recording-condition variation.
    if cfg.cmn:
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (
            mfcc.std(axis=1, keepdims=True) + 1e-8
        )

    return mfcc.astype(np.float32)     # shape: (13, T)


# ==========================================================================
#  FILE DISCOVERY
# ==========================================================================
AUDIO_EXTS = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")


def gather_files(cfg: FeatureConfig = CFG) -> List[Tuple[str, int, str]]:
    """Find every audio file under the class folders.

    Args:
        cfg: Feature configuration (defines data_root and class_map).

    Returns:
        A list of (file_path, label, class_name) tuples, one per audio file.

    Raises:
        FileNotFoundError: if an expected class folder is missing.
        RuntimeError:      if no audio files are found at all.
    """
    items: List[Tuple[str, int, str]] = []
    for cls_name, label in cfg.class_map.items():
        cls_dir = os.path.join(cfg.data_root, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Expected class folder not found: {cls_dir}")
        for ext in AUDIO_EXTS:
            for fp in glob.glob(os.path.join(cls_dir, "**", ext), recursive=True):
                items.append((fp, label, cls_name))
    if not items:
        raise RuntimeError(f"No audio files found under '{cfg.data_root}'.")
    return items


# ==========================================================================
#  MAIN  -  extract every file, then write the MFCC CSVs, manifest and params
# ==========================================================================
def main() -> None:
    cfg = CFG
    print("=" * 70)
    print("STAGE 1 - FEATURE EXTRACTION (base MFCC, no delta / delta-delta)")
    print("=" * 70)
    print(f"sample_rate={cfg.sample_rate} | frame={cfg.frame_ms}ms "
          f"(win={cfg.win_length}, hop={cfg.hop_length}) | "
          f"n_mels={cfg.n_mels} | n_mfcc={cfg.n_mfcc} | alpha={cfg.pre_emphasis}")

    items = gather_files(cfg)
    n_shouts = sum(1 for _, lbl, _ in items if lbl == 1)
    print(f"\nFound {len(items)} files "
          f"({n_shouts} shouts / {len(items) - n_shouts} neutral).\n")

    # Create one output sub-folder per class.
    for cls_name in cfg.class_map:
        os.makedirs(os.path.join(cfg.out_root, cls_name), exist_ok=True)

    manifest_rows: List[Tuple[str, int, str]] = []
    n_frames_seen: List[int] = []
    failures = 0

    for fp, label, cls_name in tqdm(items, desc="Extracting MFCC"):
        try:
            mfcc = extract_mfcc(fp, cfg)               # (13, T)
        except Exception as e:                          # skip unreadable files
            failures += 1
            tqdm.write(f"  [skip] {fp}  ({e})")
            continue

        # Save each MFCC as <out_root>/<class>/<original_stem>.csv.
        # The CSV holds the (13 x T) matrix: 13 coefficient rows, one column
        # per time frame, comma-separated.
        stem = os.path.splitext(os.path.basename(fp))[0]
        feat_path = os.path.join(cfg.out_root, cls_name, f"{stem}.csv")
        np.savetxt(feat_path, mfcc, delimiter=",", fmt="%.6f")

        manifest_rows.append((feat_path, label, cls_name))
        n_frames_seen.append(mfcc.shape[1])

    # Write manifest.csv (the index the training script reads).
    with open(cfg.manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_path", "label", "class_name"])
        writer.writerows(manifest_rows)

    # Write extraction_params.json so the exact parameters are recoverable
    # (the live-classification script reads this to stay consistent).
    params = asdict(cfg)
    params["class_map"] = cfg.class_map
    with open(cfg.params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Summary.
    print("\n" + "=" * 70)
    print(f"Saved {len(manifest_rows)} feature matrices to '{cfg.out_root}/'")
    if failures:
        print(f"Skipped {failures} unreadable file(s).")
    if n_frames_seen:
        arr = np.array(n_frames_seen)
        print(f"Time frames (T) per clip - min {arr.min()}, "
              f"median {int(np.median(arr))}, max {arr.max()}")
        print("(Variable length is expected: Stage 2 resizes each MFCC to a "
              "fixed square.)")
    print(f"Manifest : {cfg.manifest_path}")
    print(f"Params   : {cfg.params_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
