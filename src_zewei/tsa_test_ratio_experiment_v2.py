# Run command: python tsa_test_ratio_experiment_v2.py --pair-mode single --context-mode first_step --include-bce --train-stable-ratio 0.9

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


FILE = Path(__file__).resolve()
ROOT = FILE.parent
SRC = ROOT / "physics_informed_testing-main" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train.eval import encode_to_latent
from train.losses import energy_distance_unif, mmd2_unif
from train.models import EncoderOnly, build_cond_gru_model

TRAINING_CLEAN_FILE = ROOT / "physics_informed_testing-main" / "share_code" / "src" / "train" / "training_clean.py"
_training_clean_spec = importlib.util.spec_from_file_location("tsa_training_clean", TRAINING_CLEAN_FILE)
if _training_clean_spec is None or _training_clean_spec.loader is None:
    raise ImportError(f"Could not load training_clean.py from {TRAINING_CLEAN_FILE}")
_training_clean = importlib.util.module_from_spec(_training_clean_spec)
sys.modules[_training_clean_spec.name] = _training_clean
_training_clean_spec.loader.exec_module(_training_clean)

DivergenceConfig = _training_clean.DivergenceConfig
StabilityConfig = _training_clean.StabilityConfig
TrainingConfig = _training_clean.TrainingConfig
eval_epoch = _training_clean.eval_epoch
train_epoch = _training_clean.train_epoch
train_epoch_mmd_only = _training_clean.train_epoch_mmd_only


DEFAULT_RATIOS = (
    (1.00, 0.00),
    (0.98, 0.02),
    (0.96, 0.04),
    (0.94, 0.06),
    (0.92, 0.08),
    (0.90, 0.10),
    (0.80, 0.20),
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50)
)

TEST_RATIOS = DEFAULT_RATIOS + ((0.0, 1.0),)

TABLE_SETTINGS = (
    ("null", 1.00, 0.00),
    ("transition", 0.80, 0.20),
)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    prefix_len: int
    context_mode: str
    pair_mode: str
    total_train_trajectories: int
    train_stable_ratio: float
    reference_stable_size: int
    batch_train: int
    batch_eval: int
    batch_test_size: int
    epochs_piht: int
    epochs_mmd: int
    epochs_bce: int
    alpha: float
    num_repeats: int
    include_bce: bool
    lr_piht: float
    lr_mmd: float
    lr_bce: float
    weight_decay: float
    lambda_mmd: float
    lambda_stability: float
    asym_recon_delta: float
    d_v: int
    d_h: int
    hidden_enc: int
    hidden_dec: int


class TSALabeledPairDataset(Dataset):
    def __init__(self, x: np.ndarray, o_hist: np.ndarray, o_next: np.ndarray, y: np.ndarray) -> None:
        if not (len(x) == len(o_hist) == len(o_next) == len(y)):
            raise ValueError("Pair dataset arrays must have the same length")
        self.x = torch.tensor(x, dtype=torch.float32)
        self.o_hist = torch.tensor(o_hist, dtype=torch.float32)
        self.o_next = torch.tensor(o_next, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.o_hist[idx], self.o_next[idx], self.y[idx]


class TSAContextDataset(Dataset):
    def __init__(self, x: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.placeholder_hist = torch.zeros((len(x), 1, 1), dtype=torch.float32)
        self.placeholder_next = torch.zeros((len(x), 1, 1), dtype=torch.float32)
        self.placeholder_y = torch.zeros(len(x), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.placeholder_hist[idx], self.placeholder_next[idx], self.placeholder_y[idx]


class SimpleBCEClassifier(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_context(X_seq: np.ndarray, prefix_len: int, context_mode: str) -> np.ndarray:
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_seq.shape}")
    n, t, d = X_seq.shape
    if context_mode == "prefix":
        if not (0 < prefix_len < t):
            raise ValueError(f"Expected 0 < prefix_len < T, got prefix_len={prefix_len}, T={t}")
        return X_seq[:, :prefix_len, :].reshape(n, prefix_len * d).astype(np.float32, copy=False)
    if context_mode == "first_step":
        if t < 1:
            raise ValueError(f"Expected T >= 1, got T={t}")
        return X_seq[:, 0, :].reshape(n, d).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported context_mode='{context_mode}'. Expected one of: prefix, first_step")


def build_sliding_pairs(X_seq: np.ndarray, y_seq: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_seq.shape}")
    n, t, _ = X_seq.shape
    if 2 * k > (t - k):
        suffix_len = t - k
        raise ValueError(f"Need suffix length >= 2*k for sliding pairs, got suffix_len={suffix_len}, k={k}")

    x_list = []
    hist_list = []
    next_list = []
    y_list = []

    for i in range(n):
        seq = X_seq[i]
        label = float(y_seq[i])
        context = seq[:k].reshape(-1)
        suffix = seq[k:]
        l = suffix.shape[0]
        for start in range(l - 2 * k + 1):
            o_hist = suffix[start : start + k]
            o_next = suffix[start + k : start + 2 * k]
            x_list.append(context)
            hist_list.append(o_hist)
            next_list.append(o_next)
            y_list.append(label)

    x = np.asarray(x_list, dtype=np.float32)
    o_hist = np.asarray(hist_list, dtype=np.float32)
    o_next = np.asarray(next_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x, o_hist, o_next, y


def build_single_pairs(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    prefix_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_seq.shape}")
    n, t, _ = X_seq.shape
    if not (0 < prefix_len < t):
        raise ValueError(f"Expected 0 < prefix_len < T, got prefix_len={prefix_len}, T={t}")

    x_list = []
    hist_list = []
    next_list = []
    y_list = []

    for i in range(n):
        seq = X_seq[i]
        label = float(y_seq[i])
        context = seq[:prefix_len].reshape(-1)
        suffix = seq[prefix_len:]
        suffix_len = suffix.shape[0]
        if suffix_len < 2:
            raise ValueError(
                f"Need suffix length >= 2 for single-pair construction, got suffix_len={suffix_len}, prefix_len={prefix_len}"
            )
        usable_len = (suffix_len // 2) * 2
        if usable_len < 2:
            raise ValueError(
                f"Could not derive one usable pair from suffix_len={suffix_len} with prefix_len={prefix_len}"
            )
        usable_suffix = suffix[:usable_len]
        half = usable_len // 2
        o_hist = usable_suffix[:half]
        o_next = usable_suffix[half:usable_len]
        if o_hist.shape[0] != o_next.shape[0]:
            raise RuntimeError("single-pair construction produced mismatched o_hist and o_next lengths")

        x_list.append(context)
        hist_list.append(o_hist)
        next_list.append(o_next)
        y_list.append(label)

    x = np.asarray(x_list, dtype=np.float32)
    o_hist = np.asarray(hist_list, dtype=np.float32)
    o_next = np.asarray(next_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x, o_hist, o_next, y


def build_full_trajectory_pairs(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X_seq.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {X_seq.shape}")
    n, t, d = X_seq.shape
    if t < 2:
        raise ValueError(f"Need trajectory length >= 2 for full-trajectory pairs, got T={t}")

    x_list = []
    hist_list = []
    next_list = []
    y_list = []

    for i in range(n):
        seq = X_seq[i]
        label = float(y_seq[i])
        usable_len = t if t % 2 == 0 else t - 1
        if usable_len < 2:
            raise ValueError(f"Could not derive one usable full-trajectory pair from T={t}")
        half = usable_len // 2
        o_hist = seq[:half]
        o_next = seq[half:usable_len]
        if o_hist.shape[0] != o_next.shape[0]:
            raise RuntimeError("full-trajectory pair construction produced mismatched o_hist and o_next lengths")

        x_list.append(seq[0].reshape(d))
        hist_list.append(o_hist)
        next_list.append(o_next)
        y_list.append(label)

    x = np.asarray(x_list, dtype=np.float32)
    o_hist = np.asarray(hist_list, dtype=np.float32)
    o_next = np.asarray(next_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return x, o_hist, o_next, y


def build_pairs_by_mode(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    prefix_len: int,
    context_mode: str,
    pair_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if context_mode == "first_step":
        return build_full_trajectory_pairs(X_seq, y_seq)
    if context_mode != "prefix":
        raise ValueError(f"Unsupported context_mode='{context_mode}'. Expected one of: prefix, first_step")
    if pair_mode == "sliding":
        return build_sliding_pairs(X_seq, y_seq, prefix_len)
    if pair_mode == "single":
        return build_single_pairs(X_seq, y_seq, prefix_len)
    raise ValueError(f"Unsupported pair_mode='{pair_mode}'. Expected one of: sliding, single")


def load_tsa_from_pickle(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with data_path.open("rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict) and "data" in raw and "target" in raw:
        data = raw
        X = np.asarray(data["data"], dtype=np.float32)
        y = np.asarray(data["target"], dtype=np.float32)
        return X, y

    if isinstance(raw, tuple) and len(raw) == 2:
        seq_list, y_series = raw
        data = {
            "data": np.asarray([df.values for df in seq_list], dtype=np.float32),
            "target": np.asarray(y_series.to_numpy() if hasattr(y_series, "to_numpy") else y_series, dtype=np.float32),
        }
        X = data["data"]
        y = data["target"]
        return X, y

    raise ValueError(f"Unsupported TSA pickle format: {type(raw)}")


def split_train_test_by_class(
    X: np.ndarray,
    y: np.ndarray,
    train_stable_count: int,
    train_unstable_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    stable_idx = np.where(y == 1)[0]
    unstable_idx = np.where(y == 0)[0]
    if train_stable_count > len(stable_idx) or train_unstable_count > len(unstable_idx):
        raise ValueError("Requested train pool exceeds available class count")

    train_stable_idx = rng.choice(stable_idx, size=train_stable_count, replace=False)
    train_unstable_idx = rng.choice(unstable_idx, size=train_unstable_count, replace=False)

    test_stable_idx = np.setdiff1d(stable_idx, train_stable_idx)
    test_unstable_idx = np.setdiff1d(unstable_idx, train_unstable_idx)

    return {
        "X_train_stable": X[train_stable_idx],
        "y_train_stable": y[train_stable_idx],
        "X_train_unstable": X[train_unstable_idx],
        "y_train_unstable": y[train_unstable_idx],
        "X_test_stable": X[test_stable_idx],
        "y_test_stable": y[test_stable_idx],
        "X_test_unstable": X[test_unstable_idx],
        "y_test_unstable": y[test_unstable_idx],
    }


def split_train_reference_eval_by_class(
    X: np.ndarray,
    y: np.ndarray,
    train_stable_count: int,
    train_unstable_count: int,
    reference_stable_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    stable_idx = np.where(y == 1)[0]
    unstable_idx = np.where(y == 0)[0]
    if train_stable_count > len(stable_idx) or train_unstable_count > len(unstable_idx):
        raise ValueError("Requested train pool exceeds available class count")

    train_stable_idx = rng.choice(stable_idx, size=train_stable_count, replace=False)
    train_unstable_idx = rng.choice(unstable_idx, size=train_unstable_count, replace=False)

    heldout_stable_idx = np.setdiff1d(stable_idx, train_stable_idx)
    heldout_unstable_idx = np.setdiff1d(unstable_idx, train_unstable_idx)
    if reference_stable_size > len(heldout_stable_idx):
        raise ValueError(
            f"reference_stable_size={reference_stable_size} exceeds held-out stable pool={len(heldout_stable_idx)}"
        )

    reference_stable_idx = (
        rng.choice(heldout_stable_idx, size=reference_stable_size, replace=False)
        if reference_stable_size > 0
        else np.array([], dtype=int)
    )
    eval_stable_idx = np.setdiff1d(heldout_stable_idx, reference_stable_idx)

    return {
        "X_train_stable": X[train_stable_idx],
        "y_train_stable": y[train_stable_idx],
        "X_train_unstable": X[train_unstable_idx],
        "y_train_unstable": y[train_unstable_idx],
        "X_reference_stable": X[reference_stable_idx],
        "y_reference_stable": y[reference_stable_idx],
        "X_eval_stable": X[eval_stable_idx],
        "y_eval_stable": y[eval_stable_idx],
        "X_eval_unstable": X[heldout_unstable_idx],
        "y_eval_unstable": y[heldout_unstable_idx],
    }


def normalize_with_train_stats(split_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    X_train_all = np.concatenate(
        [split_data["X_train_stable"], split_data["X_train_unstable"]],
        axis=0,
    )
    feat_mean = X_train_all.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train_all.std(axis=(0, 1), keepdims=True)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    out = dict(split_data)
    for key, value in split_data.items():
        if key.startswith("X_"):
            out[key] = ((value - feat_mean) / feat_std).astype(np.float32, copy=False)
    out["feat_mean"] = feat_mean
    out["feat_std"] = feat_std
    return out


# def infer_stability_threshold_from_pairs(o_next: np.ndarray, y: np.ndarray) -> float:
#     peaks = np.max(np.abs(o_next), axis=(1, 2))
#     candidates = np.quantile(peaks, np.linspace(0.01, 0.99, 199))
#     best_threshold = float(np.median(peaks))
#     best_acc = -1.0
#     for threshold in candidates:
#         preds = (peaks <= threshold).astype(np.float32)
#         acc = float(np.mean(preds == y))
#         if acc > best_acc:
#             best_acc = acc
#             best_threshold = float(threshold)
#     return best_threshold

def infer_stability_threshold_from_pairs(o_next: np.ndarray, y: np.ndarray) -> float:
    """
    Infer a stability threshold from trajectory pairs using a label-free quantile rule.

    Args:
        o_next: np.ndarray of shape (N, T, D)
        y: unused (kept for signature compatibility)

    Returns:
        threshold: float
    """
    peaks = np.max(np.abs(o_next), axis=(1, 2))
    threshold = float(np.quantile(peaks, 0.9))  # 90th percentile
    return threshold


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=device.type == "cuda",
    )


def resolve_fixed_train_counts(
    y: np.ndarray,
    total_train_trajectories: int,
    train_stable_ratio: float,
) -> tuple[int, int, int, int]:
    stable_total = int(np.sum(y == 1))
    unstable_total = int(np.sum(y == 0))
    if stable_total == 0 or unstable_total == 0:
        raise ValueError("Dataset must contain both stable and unstable samples")
    if not (0.0 <= train_stable_ratio <= 1.0):
        raise ValueError(f"train_stable_ratio must be in [0, 1], got {train_stable_ratio}")
    if total_train_trajectories <= 0:
        raise ValueError("total_train_trajectories must be positive")

    n_train_stable = int(round(total_train_trajectories * train_stable_ratio))
    n_train_unstable = total_train_trajectories - n_train_stable
    if n_train_stable > stable_total or n_train_unstable > unstable_total:
        raise ValueError(
            f"Requested train split exceeds available counts: "
            f"stable={n_train_stable}/{stable_total}, unstable={n_train_unstable}/{unstable_total}"
        )
    return stable_total, unstable_total, n_train_stable, n_train_unstable


def compute_confidence_intervals(k: int, n: int, z: float = 1.645):
    """
    Returns:
        p_hat,
        wilson_low, wilson_high,
        normal_low, normal_high
    """
    if n <= 0:
        raise ValueError("n must be positive")
    k = max(0, min(int(k), int(n)))
    p_hat = float(k / n)

    z2 = float(z * z)
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    margin = (z * math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)))) / denominator
    wilson_low = float(np.clip(center - margin, 0.0, 1.0))
    wilson_high = float(np.clip(center + margin, 0.0, 1.0))

    normal_margin = z * math.sqrt(max(0.0, p_hat * (1.0 - p_hat) / n))
    normal_low = float(np.clip(p_hat - normal_margin, 0.0, 1.0))
    normal_high = float(np.clip(p_hat + normal_margin, 0.0, 1.0))
    return p_hat, wilson_low, wilson_high, normal_low, normal_high


def sample_with_replacement(pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 0:
        shape = (0,) + pool.shape[1:]
        return np.empty(shape, dtype=pool.dtype)
    if len(pool) == 0:
        raise ValueError("Cannot sample from an empty pool")
    idx = rng.choice(len(pool), size=size, replace=True)
    return pool[idx]


def sample_mixed_batch(
    stable_pool: np.ndarray,
    unstable_pool: np.ndarray,
    stable_ratio: float,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    n_stable = int(round(batch_size * stable_ratio))
    n_unstable = batch_size - n_stable
    stable_batch = sample_with_replacement(stable_pool, n_stable, rng)
    unstable_batch = sample_with_replacement(unstable_pool, n_unstable, rng)

    parts = []
    if n_stable > 0:
        parts.append(stable_batch)
    if n_unstable > 0:
        parts.append(unstable_batch)
    mixed = np.concatenate(parts, axis=0)
    perm = rng.permutation(len(mixed))
    return mixed[perm], n_stable, n_unstable


def sample_reference_and_test_batches(
    reference_pool: np.ndarray,
    stable_pool: np.ndarray,
    unstable_pool: np.ndarray,
    stable_ratio: float,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    reference_batch = sample_with_replacement(reference_pool, batch_size, rng)
    test_batch, n_stable, n_unstable = sample_mixed_batch(
        stable_pool=stable_pool,
        unstable_pool=unstable_pool,
        stable_ratio=stable_ratio,
        batch_size=batch_size,
        rng=rng,
    )
    return reference_batch, test_batch, n_stable, n_unstable


def scalarize_scores(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    return arr.reshape(arr.shape[0], -1).mean(axis=1).astype(np.float32, copy=False)


def evaluate_uniformity(v_np: np.ndarray, device: torch.device, chi_bins: int = 10) -> dict[str, float]:
    v = np.asarray(v_np, dtype=np.float32).reshape(-1)
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    ks_res = stats.kstest(v, "uniform", args=(0.0, 1.0))
    cvm_res = stats.cramervonmises(v, "uniform")
    counts, _ = np.histogram(v, bins=chi_bins, range=(0.0, 1.0))
    expected = np.full(chi_bins, v.size / chi_bins, dtype=np.float64)
    chi2_stat = float(((counts - expected) ** 2 / expected).sum())
    chi2_p = float(1.0 - stats.chi2.cdf(chi2_stat, df=chi_bins - 1))

    v_t = torch.tensor(v[:, None], dtype=torch.float32, device=device)
    with torch.inference_mode():
        mmd_stat = float(mmd2_unif(v_t, sigmas=None, use_median=True, low_discrepancy=True).item())
        energy_stat = float(energy_distance_unif(v_t.cpu()).item())

    return {
        "ks_stat": float(ks_res.statistic),
        "ks_pvalue": float(ks_res.pvalue),
        "cvm_stat": float(cvm_res.statistic),
        "cvm_pvalue": float(cvm_res.pvalue),
        "chi2_stat": chi2_stat,
        "chi2_pvalue": chi2_p,
        "mmd_stat": mmd_stat,
        "energy_stat": energy_stat,
    }


def evaluate_uniformity_chi_square(v_np: np.ndarray, chi_bins: int = 10) -> tuple[float, float]:
    v = np.asarray(v_np, dtype=np.float32).reshape(-1)
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    counts, _ = np.histogram(v, bins=chi_bins, range=(0.0, 1.0))
    expected = np.full(chi_bins, max(v.size / chi_bins, 1e-12), dtype=np.float64)
    chi2_stat = float(((counts - expected) ** 2 / expected).sum())
    chi2_p = float(1.0 - stats.chi2.cdf(chi2_stat, df=chi_bins - 1))
    return chi2_stat, chi2_p


def compute_rbf_mmd_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    y = np.asarray(y, dtype=np.float32).reshape(len(y), -1)
    if len(x) < 2 or len(y) < 2:
        return 0.0

    xy = np.concatenate([x, y], axis=0)
    sq_norms = np.sum(xy * xy, axis=1, keepdims=True)
    dists = np.maximum(sq_norms + sq_norms.T - 2.0 * xy @ xy.T, 0.0)
    upper = dists[np.triu_indices_from(dists, k=1)]
    positive = upper[upper > 0]
    sigma2 = float(np.median(positive)) if positive.size > 0 else 1.0
    sigma2 = max(sigma2, 1e-6)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = np.sum(a * a, axis=1, keepdims=True)
        b_norm = np.sum(b * b, axis=1, keepdims=True)
        dist2 = np.maximum(a_norm + b_norm.T - 2.0 * a @ b.T, 0.0)
        return np.exp(-dist2 / (2.0 * sigma2))

    k_xx = kernel(x, x)
    k_yy = kernel(y, y)
    k_xy = kernel(x, y)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    term_xx = float(k_xx.sum() / (len(x) * (len(x) - 1)))
    term_yy = float(k_yy.sum() / (len(y) * (len(y) - 1)))
    term_xy = float(k_xy.mean())
    return term_xx + term_yy - 2.0 * term_xy


def calibrate_mmd_threshold(
    reference_pool: np.ndarray,
    alpha: float,
    batch_size: int,
    num_repeats: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    null_stats = []
    for _ in range(num_repeats):
        ref_a = sample_with_replacement(reference_pool, batch_size, rng)
        ref_b = sample_with_replacement(reference_pool, batch_size, rng)
        null_stats.append(compute_rbf_mmd_statistic(ref_a, ref_b))
    return float(np.quantile(np.asarray(null_stats, dtype=np.float64), 1.0 - alpha))


def welch_test_from_scores(reference_scores: np.ndarray, test_scores: np.ndarray) -> tuple[float, float]:
    ref = scalarize_scores(reference_scores)
    test = scalarize_scores(test_scores)
    if np.allclose(ref, ref[0]) and np.allclose(test, test[0]) and np.isclose(ref[0], test[0]):
        return 0.0, 1.0
    res = stats.ttest_ind(ref, test, equal_var=False, nan_policy="omit")
    statistic = float(0.0 if np.isnan(res.statistic) else res.statistic)
    pvalue = float(1.0 if np.isnan(res.pvalue) else res.pvalue)
    return statistic, pvalue


def train_bce_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    if np.unique(y_train).size < 2:
        raise ValueError("BCE classifier requires both stable and unstable labels in the training set")
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = make_loader(dataset, cfg.batch_train, shuffle=True, device=device)
    model = SimpleBCEClassifier(d_in=x_train.shape[1], hidden_dim=cfg.hidden_enc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_bce, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg.epochs_bce):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    return model


def predict_bce_probabilities(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, device=device)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            outputs.append(torch.sigmoid(model(x_batch)).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def train_mixed_piht(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    d_x: int,
    d_o: int,
    cfg: ExperimentConfig,
    device: torch.device,
    stability_threshold: float,
) -> tuple[torch.nn.Module, dict[str, float]]:
    divergence_cfg = DivergenceConfig(
        lambda_mmd=cfg.lambda_mmd,
        kwargs={"sigmas": None, "low_discrepancy": True},
    )
    stability_cfg = StabilityConfig(
        lambda_stability=0.0,
        threshold=stability_threshold,
        window=None,
        dt=1.0,
        t0=0.0,
        smooth_beta=10.0,
        loss_type="stability_asymmetric_recon",
        asym_recon_delta=cfg.asym_recon_delta,
    )
    training_cfg = TrainingConfig(divergence=divergence_cfg, stability=stability_cfg)

    model = build_cond_gru_model(
        d_x=d_x,
        d_o=d_o,
        d_v=cfg.d_v,
        d_h=cfg.d_h,
        hidden_enc=cfg.hidden_enc,
        hidden_dec=cfg.hidden_dec,
        enc_layers=2,
        dec_layers=1,
        output_activation="sigmoid",
        dropout_rnn=0.1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_piht, weight_decay=cfg.weight_decay)

    best_state = None
    best_metric = float("inf")
    for _ in range(cfg.epochs_piht):
        train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=training_cfg,
        )
        val_rec, val_mmd = eval_epoch(
            loader=eval_loader,
            model=model,
            device=device,
            config=training_cfg,
        )
        total_metric = float(val_rec + cfg.lambda_mmd * val_mmd)
        if total_metric < best_metric:
            best_metric = total_metric
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    rec, div = eval_epoch(
        loader=eval_loader,
        model=model,
        device=device,
        config=training_cfg,
    )
    return model, {
        "eval_recon": float(rec),
        "eval_mmd": float(div),
    }


def train_mmd_baseline(
    train_loader: DataLoader,
    d_x: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    divergence_cfg = DivergenceConfig(
        lambda_mmd=cfg.lambda_mmd,
        kwargs={"sigmas": None, "low_discrepancy": True},
    )
    model = build_cond_gru_model(
        d_x=d_x,
        d_o=50,
        d_v=cfg.d_v,
        d_h=cfg.d_h,
        hidden_enc=cfg.hidden_enc,
        hidden_dec=cfg.hidden_dec,
        enc_layers=2,
        dec_layers=1,
        output_activation="sigmoid",
        dropout_rnn=0.1,
    ).to(device)
    model_enc = EncoderOnly(copy.deepcopy(model.enc)).to(device)
    optimizer = torch.optim.Adam(model_enc.parameters(), lr=cfg.lr_mmd, weight_decay=cfg.weight_decay)

    for _ in range(cfg.epochs_mmd):
        train_epoch_mmd_only(
            model_enc=model_enc,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            divergence=divergence_cfg,
        )
    return model_enc


def train_trajectory_only(
    train_loader: DataLoader,
    eval_loader: DataLoader,
    d_x: int,
    d_o: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, float]]:
    training_cfg = TrainingConfig(
        divergence=DivergenceConfig(
            lambda_mmd=0.0,
            kwargs={"sigmas": None, "low_discrepancy": True},
        ),
        stability=StabilityConfig(lambda_stability=0.0),
    )
    model = build_cond_gru_model(
        d_x=d_x,
        d_o=d_o,
        d_v=cfg.d_v,
        d_h=cfg.d_h,
        hidden_enc=cfg.hidden_enc,
        hidden_dec=cfg.hidden_dec,
        enc_layers=2,
        dec_layers=1,
        output_activation="sigmoid",
        dropout_rnn=0.1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_piht, weight_decay=cfg.weight_decay)

    best_state = None
    best_metric = float("inf")
    for _ in range(cfg.epochs_piht):
        train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=training_cfg,
        )
        val_rec, val_mmd = eval_epoch(
            loader=eval_loader,
            model=model,
            device=device,
            config=training_cfg,
        )
        total_metric = float(val_rec)
        if total_metric < best_metric:
            best_metric = total_metric
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    rec, div = eval_epoch(
        loader=eval_loader,
        model=model,
        device=device,
        config=training_cfg,
    )
    return model, {
        "eval_recon": float(rec),
        "eval_mmd": float(div),
    }


def collect_cached_latents(
    model: torch.nn.Module,
    x_reference: np.ndarray,
    x_eval_stable: np.ndarray,
    x_eval_unstable: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    return {
        "reference": encode_to_latent(model, x_reference, device=device, batch_size=batch_size),
        "stable_eval": encode_to_latent(model, x_eval_stable, device=device, batch_size=batch_size),
        "unstable_eval": encode_to_latent(model, x_eval_unstable, device=device, batch_size=batch_size),
    }


def summarize_repeat_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_type",
        "test_setting",
        "ratio",
        "stable_ratio",
        "unstable_ratio",
        "context_mode",
        "pair_mode",
        "alpha",
        "batch_test_size",
        "num_repeats",
        "train_stable_ratio",
        "total_train_trajectories",
        "train_stable_trajectories",
        "train_unstable_trajectories",
        "reference_stable_size",
        "stable_eval_pool_size",
        "unstable_eval_pool_size",
        "stable_total",
        "unstable_total",
        "mmd_threshold",
    ]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=False):
        record = dict(zip(group_cols, keys))
        reject_count = int(group["reject"].sum())
        reject_rate, wilson_low, wilson_high, normal_low, normal_high = compute_confidence_intervals(
            k=reject_count,
            n=len(group),
        )
        record.update(
            {
                "reject_count": reject_count,
                "reject_rate": reject_rate,
                "reject_ci_wilson_low": wilson_low,
                "reject_ci_wilson_high": wilson_high,
                "reject_ci_normal_low": normal_low,
                "reject_ci_normal_high": normal_high,
                "statistic_mean": float(group["statistic"].mean()),
                "pvalue_mean": float(group["pvalue"].mean()) if group["pvalue"].notna().any() else np.nan,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def build_compact_summary(summary_df: pd.DataFrame, only_table: bool) -> pd.DataFrame:
    keep_cols = [
        "model_type",
        "test_setting",
        "ratio",
        "stable_ratio",
        "unstable_ratio",
        "context_mode",
        "pair_mode",
        "reject_count",
        "reject_rate",
        "reject_ci_wilson_low",
        "reject_ci_wilson_high",
        "reject_ci_normal_low",
        "reject_ci_normal_high",
        "alpha",
        "batch_test_size",
        "num_repeats",
    ]
    compact = summary_df.loc[:, [col for col in keep_cols if col in summary_df.columns]].copy()
    if only_table:
        compact = compact[compact["test_setting"].isin(["null", "transition"])].reset_index(drop=True)
    else:
        compact = (
            compact[compact["test_setting"] == "sweep"]
            .drop(columns=["test_setting", "ratio"], errors="ignore")
            .sort_values(["model_type", "stable_ratio", "unstable_ratio"])
            .reset_index(drop=True)
        )
    return compact


def run_experiment(cfg: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(cfg.seed)
    device = choose_device()

    data_path = ROOT / "data" / "tsa_data.pkl"
    X, y = load_tsa_from_pickle(data_path)

    stable_total, unstable_total, train_stable_count, train_unstable_count = resolve_fixed_train_counts(
        y=y,
        total_train_trajectories=cfg.total_train_trajectories,
        train_stable_ratio=cfg.train_stable_ratio,
    )
    print(
        f"Loaded {data_path} with X shape={X.shape}, y shape={y.shape}, "
        f"stable={stable_total}, unstable={unstable_total}",
        flush=True,
    )
    print(
        f"Fixed training split: train_stable={train_stable_count}, train_unstable={train_unstable_count}, "
        f"total_train={cfg.total_train_trajectories}",
        flush=True,
    )

    splits = split_train_reference_eval_by_class(
        X=X,
        y=y,
        train_stable_count=train_stable_count,
        train_unstable_count=train_unstable_count,
        reference_stable_size=cfg.reference_stable_size,
        seed=cfg.seed,
    )
    splits = normalize_with_train_stats(splits)

    train_pool_X = np.concatenate([splits["X_train_stable"], splits["X_train_unstable"]], axis=0)
    train_pool_y = np.concatenate([splits["y_train_stable"], splits["y_train_unstable"]], axis=0).astype(np.float32)
    _, _, train_pool_o_next, train_pool_pair_y = build_pairs_by_mode(
        train_pool_X,
        train_pool_y,
        cfg.prefix_len,
        cfg.context_mode,
        cfg.pair_mode,
    )
    stability_threshold = infer_stability_threshold_from_pairs(train_pool_o_next, train_pool_pair_y)

    X_train_model = train_pool_X
    y_train_model = train_pool_y
    x_train_context = build_context(X_train_model, cfg.prefix_len, cfg.context_mode)
    x_train, o_hist_train, o_next_train, y_train_pairs = build_pairs_by_mode(
        X_train_model,
        y_train_model,
        cfg.prefix_len,
        cfg.context_mode,
        cfg.pair_mode,
    )
    train_pair_ds = TSALabeledPairDataset(x_train, o_hist_train, o_next_train, y_train_pairs)
    train_pair_loader = make_loader(train_pair_ds, cfg.batch_train, shuffle=True, device=device)
    train_context_ds = TSAContextDataset(x_train_context)
    train_context_loader = make_loader(train_context_ds, cfg.batch_train, shuffle=True, device=device)

    X_eval_all = np.concatenate([splits["X_eval_stable"], splits["X_eval_unstable"]], axis=0)
    y_eval_all = np.concatenate([splits["y_eval_stable"], splits["y_eval_unstable"]], axis=0).astype(np.float32)
    x_eval_all, o_hist_eval_all, o_next_eval_all, y_eval_pairs = build_pairs_by_mode(
        X_eval_all,
        y_eval_all,
        cfg.prefix_len,
        cfg.context_mode,
        cfg.pair_mode,
    )
    eval_pair_ds = TSALabeledPairDataset(x_eval_all, o_hist_eval_all, o_next_eval_all, y_eval_pairs)
    eval_pair_loader = make_loader(eval_pair_ds, cfg.batch_eval, shuffle=False, device=device)

    reference_stable_pool_size = len(splits["X_reference_stable"])
    stable_eval_pool_size = len(splits["X_eval_stable"])
    unstable_eval_pool_size = len(splits["X_eval_unstable"])
    if reference_stable_pool_size <= 0 or stable_eval_pool_size <= 0 or unstable_eval_pool_size <= 0:
        raise ValueError("Reference stable pool, stable eval pool, and unstable eval pool must all be non-empty")
    if cfg.batch_test_size <= 0:
        raise ValueError("batch_test_size must be positive")

    print(
        f"Held-out pools: reference_stable={reference_stable_pool_size}, "
        f"stable_eval={stable_eval_pool_size}, unstable_eval={unstable_eval_pool_size}",
        flush=True,
    )

    train_start = time.time()
    mixed_piht_model, mixed_piht_metrics = train_mixed_piht(
        train_loader=train_pair_loader,
        eval_loader=eval_pair_loader,
        d_x=x_train.shape[1],
        d_o=o_hist_train.shape[2],
        cfg=cfg,
        device=device,
        stability_threshold=stability_threshold,
    )
    mmd_only_model = train_mmd_baseline(
        train_loader=train_context_loader,
        d_x=x_train_context.shape[1],
        cfg=cfg,
        device=device,
    )
    trajectory_only_model, trajectory_only_metrics = train_trajectory_only(
        train_loader=train_pair_loader,
        eval_loader=eval_pair_loader,
        d_x=x_train.shape[1],
        d_o=o_hist_train.shape[2],
        cfg=cfg,
        device=device,
    )
    print(f"Finished one-time model training in {time.time() - train_start:.1f}s", flush=True)

    x_reference = build_context(splits["X_reference_stable"], cfg.prefix_len, cfg.context_mode)
    x_eval_stable = build_context(splits["X_eval_stable"], cfg.prefix_len, cfg.context_mode)
    x_eval_unstable = build_context(splits["X_eval_unstable"], cfg.prefix_len, cfg.context_mode)

    method_caches = {
        "mixed_piht": {
            "type": "two_sample_mmd",
            "metrics": mixed_piht_metrics,
            **collect_cached_latents(
                mixed_piht_model,
                x_reference=x_reference,
                x_eval_stable=x_eval_stable,
                x_eval_unstable=x_eval_unstable,
                device=device,
                batch_size=cfg.batch_eval,
            ),
        },
        "mmd_only": {
            "type": "two_sample_mmd",
            "metrics": {},
            **collect_cached_latents(
                mmd_only_model,
                x_reference=x_reference,
                x_eval_stable=x_eval_stable,
                x_eval_unstable=x_eval_unstable,
                device=device,
                batch_size=cfg.batch_eval,
            ),
        },
        "trajectory_only": {
            "type": "two_sample_mmd",
            "metrics": trajectory_only_metrics,
            **collect_cached_latents(
                trajectory_only_model,
                x_reference=x_reference,
                x_eval_stable=x_eval_stable,
                x_eval_unstable=x_eval_unstable,
                device=device,
                batch_size=cfg.batch_eval,
            ),
        },
        "context_mmd": {
            "type": "two_sample_mmd",
            "metrics": {},
            "reference": x_reference,
            "stable_eval": x_eval_stable,
            "unstable_eval": x_eval_unstable,
        },
    }

    if cfg.include_bce:
        bce_model = train_bce_classifier(x_train=x_train_context, y_train=y_train_model, cfg=cfg, device=device)
        method_caches["bce_classifier"] = {
            "type": "welch",
            "metrics": {},
            "reference": predict_bce_probabilities(bce_model, x_reference, batch_size=cfg.batch_eval, device=device),
            "stable_eval": predict_bce_probabilities(bce_model, x_eval_stable, batch_size=cfg.batch_eval, device=device),
            "unstable_eval": predict_bce_probabilities(bce_model, x_eval_unstable, batch_size=cfg.batch_eval, device=device),
        }

    for model_type, cache in method_caches.items():
        if cache["type"] == "two_sample_mmd":
            cache["mmd_threshold"] = calibrate_mmd_threshold(
                reference_pool=cache["reference"],
                alpha=cfg.alpha,
                batch_size=cfg.batch_test_size,
                num_repeats=cfg.num_repeats,
                seed=cfg.seed + 17000 + len(model_type),
            )
        else:
            cache["mmd_threshold"] = np.nan

    common_meta = {
        "context_mode": cfg.context_mode,
        "pair_mode": cfg.pair_mode,
        "alpha": cfg.alpha,
        "batch_test_size": cfg.batch_test_size,
        "num_repeats": cfg.num_repeats,
        "train_stable_ratio": cfg.train_stable_ratio,
        "total_train_trajectories": cfg.total_train_trajectories,
        "train_stable_trajectories": train_stable_count,
        "train_unstable_trajectories": train_unstable_count,
        "reference_stable_size": reference_stable_pool_size,
        "stable_eval_pool_size": stable_eval_pool_size,
        "unstable_eval_pool_size": unstable_eval_pool_size,
        "stable_total": stable_total,
        "unstable_total": unstable_total,
        "stability_threshold": stability_threshold,
    }

    results = []
    all_settings = list(TABLE_SETTINGS) + [("sweep", stable_ratio, unstable_ratio) for stable_ratio, unstable_ratio in DEFAULT_RATIOS]
    for setting_idx, (test_setting, stable_ratio, unstable_ratio) in enumerate(all_settings):
        for model_idx, (model_type, cache) in enumerate(method_caches.items()):
            rng = np.random.default_rng(cfg.seed + 1000 * (setting_idx + 1) + 100_000 * (model_idx + 1))
            for repeat_id in range(cfg.num_repeats):
                if cache["type"] == "two_sample_mmd":
                    reference_batch, test_batch, test_stable, test_unstable = sample_reference_and_test_batches(
                        reference_pool=cache["reference"],
                        stable_pool=cache["stable_eval"],
                        unstable_pool=cache["unstable_eval"],
                        stable_ratio=stable_ratio,
                        batch_size=cfg.batch_test_size,
                        rng=rng,
                    )
                    statistic = compute_rbf_mmd_statistic(reference_batch, test_batch)
                    pvalue = np.nan
                    reject = int(statistic > cache["mmd_threshold"])
                elif cache["type"] == "welch":
                    reference_batch, test_batch, test_stable, test_unstable = sample_reference_and_test_batches(
                        reference_pool=cache["reference"],
                        stable_pool=cache["stable_eval"],
                        unstable_pool=cache["unstable_eval"],
                        stable_ratio=stable_ratio,
                        batch_size=cfg.batch_test_size,
                        rng=rng,
                    )
                    statistic, pvalue = welch_test_from_scores(reference_batch, test_batch)
                    reject = int(pvalue < cfg.alpha)

                results.append(
                    {
                        "model_type": model_type,
                        "test_setting": test_setting,
                        "ratio": f"{stable_ratio:.2f}/{unstable_ratio:.2f}",
                        "stable_ratio": stable_ratio,
                        "unstable_ratio": unstable_ratio,
                        "repeat_id": repeat_id,
                        "reject": reject,
                        "pvalue": pvalue,
                        "statistic": statistic,
                        "mmd_threshold": cache["mmd_threshold"],
                        "test_stable_trajectories": test_stable,
                        "test_unstable_trajectories": test_unstable,
                        **common_meta,
                        **cache["metrics"],
                    }
                )
        print(
            f"[{test_setting} ratio {stable_ratio:.2f}/{unstable_ratio:.2f}] "
            f"batch={cfg.batch_test_size}, repeats={cfg.num_repeats}",
            flush=True,
        )

    detailed_df = pd.DataFrame(results)
    summary_df = summarize_repeat_results(detailed_df)
    table_df = build_compact_summary(summary_df, only_table=True)
    sweep_df = build_compact_summary(summary_df, only_table=False)
    return detailed_df, summary_df, table_df, sweep_df


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="TSA test-ratio PIHT experiment v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-len", type=int, default=10)
    parser.add_argument("--context-mode", type=str, choices=("prefix", "first_step"), default="prefix")
    parser.add_argument("--pair-mode", type=str, choices=("sliding", "single"), default="sliding")
    parser.add_argument("--total-train-trajectories", type=int, default=5000)
    parser.add_argument("--train-stable-ratio", type=float, default=1.0)
    parser.add_argument("--reference-stable-size", type=int, default=1000)
    parser.add_argument("--batch-train", type=int, default=256)
    parser.add_argument("--batch-eval", type=int, default=512)
    parser.add_argument("--batch-test-size", type=int, default=512)
    parser.add_argument("--epochs-piht", type=int, default=15)
    parser.add_argument("--epochs-mmd", type=int, default=15)
    parser.add_argument("--epochs-bce", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num-repeats", type=int, default=200)
    parser.add_argument("--include-bce", action="store_true")
    parser.add_argument("--lr-piht", type=float, default=5e-3)
    parser.add_argument("--lr-mmd", type=float, default=5e-3)
    parser.add_argument("--lr-bce", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-mmd", type=float, default=1)
    parser.add_argument("--lambda-stability", type=float, default=1e-1)
    parser.add_argument("--asym-recon-delta", type=float, default=3.0)
    parser.add_argument("--d-v", type=int, default=1)
    parser.add_argument("--d-h", type=int, default=64)
    parser.add_argument("--hidden-enc", type=int, default=128)
    parser.add_argument("--hidden-dec", type=int, default=128)
    args = parser.parse_args()

    return ExperimentConfig(
        seed=args.seed,
        prefix_len=args.prefix_len,
        context_mode=args.context_mode,
        pair_mode=args.pair_mode,
        total_train_trajectories=args.total_train_trajectories,
        train_stable_ratio=args.train_stable_ratio,
        reference_stable_size=args.reference_stable_size,
        batch_train=args.batch_train,
        batch_eval=args.batch_eval,
        batch_test_size=args.batch_test_size,
        epochs_piht=args.epochs_piht,
        epochs_mmd=args.epochs_mmd,
        epochs_bce=args.epochs_bce,
        alpha=args.alpha,
        num_repeats=args.num_repeats,
        include_bce=args.include_bce,
        lr_piht=args.lr_piht,
        lr_mmd=args.lr_mmd,
        lr_bce=args.lr_bce,
        weight_decay=args.weight_decay,
        lambda_mmd=args.lambda_mmd,
        lambda_stability=args.lambda_stability,
        asym_recon_delta=args.asym_recon_delta,
        d_v=args.d_v,
        d_h=args.d_h,
        hidden_enc=args.hidden_enc,
        hidden_dec=args.hidden_dec,
    )


def main() -> None:
    cfg = parse_args()
    detailed_df, summary_df, table_df, sweep_df = run_experiment(cfg)
    output_path = ROOT / "tsa_test_ratio_experiment_v2_results.csv"
    summary_output_path = ROOT / "tsa_test_ratio_experiment_v2_results_summary.csv"
    table_output_path = ROOT / "tsa_test_ratio_experiment_v2_results_table_summary.csv"
    sweep_output_path = ROOT / "tsa_test_ratio_experiment_v2_results_sweep_summary.csv"
    detailed_df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)
    table_df.to_csv(table_output_path, index=False)
    sweep_df.to_csv(sweep_output_path, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(summary_df)
    print(f"\nSaved detailed results to {output_path}")
    print(f"Saved summary results to {summary_output_path}")
    print(f"Saved table summary to {table_output_path}")
    print(f"Saved sweep summary to {sweep_output_path}")


if __name__ == "__main__":
    main()
