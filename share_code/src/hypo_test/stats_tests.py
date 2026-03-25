"""
Statistical tests for uniformity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
import warnings

from data.sample_x import sample_x
from train.eval import encode_to_latent

from train.losses import energy_distance_unif, mmd2_unif


def chisq_uniform_1d(v: np.ndarray, b: int = 20, eps: float = 1e-12) -> tuple[float, float, np.ndarray]:
    """
    Chi-square test for Unif(0,1).

    v: (N,) values in (0,1)
    """
    v = np.clip(v, 0.0 + eps, 1.0 - eps)
    counts, _ = np.histogram(v, bins=b, range=(0.0, 1.0))
    expected = np.ones(b) * (len(v) / b)
    chi2 = ((counts - expected) ** 2 / expected).sum()
    p_val = 1.0 - stats.chi2.cdf(chi2, df=b - 1)
    return chi2, p_val, counts


def stats_for_latents(v: np.ndarray, b: int = 20) -> tuple[float, float, float]:
    """
    Return p-values for KS, Cramer-von Mises, and chi-square uniformity tests.

    v: (N, d_v) or (N,) latent array in [0, 1]
    """
    v = np.asarray(v)
    v1 = v[:, 0] if v.ndim > 1 else v
    ks_p = stats.kstest(v1, "uniform", args=(0, 1)).pvalue
    cvm_p = stats.cramervonmises(v1, "uniform", args=(0, 1)).pvalue
    _, chi2_p, _ = chisq_uniform_1d(v1, b=b)
    return ks_p, cvm_p, chi2_p


def pvals_ks_cvm_to_uniform(v_np: np.ndarray) -> tuple[float, float]:
    v = np.asarray(v_np, float).ravel()
    v = np.clip(v, 0.0, 1.0)  # if your latent is intended in [0,1]
    p_ks = float(stats.kstest(v, "uniform", args=(0.0, 1.0)).pvalue)
    p_cvm = float(stats.cramervonmises(v, "uniform").pvalue)
    return p_ks, p_cvm


@torch.no_grad()
def mmd_stat(v_np, device, ref_u) -> float:
    v = torch.as_tensor(v_np, device=device, dtype=torch.float32).view(-1, 1)
    return float(mmd2_unif(v, sigmas=None, ref_u=ref_u, use_median=False).item())


@torch.no_grad()
def energy_stat(v_np, device, ref_u) -> float:
    v = torch.as_tensor(v_np, device=device, dtype=torch.float32).view(-1, 1)
    # Use CPU-safe energy distance for robust execution across devices.
    return float(energy_distance_unif(v.cpu(), ref_u.cpu()).item())


@torch.no_grad()
def perm_pvalue_two_sample(stat_fn, x, y, *, B: int = 300, device: str = "cpu", seed: int = 0) -> tuple[float, float]:
    """
    Two-sample permutation test p-value for statistic stat_fn(x,y),
    where stat_fn uses a fixed reference y (uniform) passed in.

    Returns p = (1 + #{T_perm >= T_obs})/(B+1).
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = x.shape[0]
    assert y.shape[0] == n, "Use equal sample sizes for clean permutation test."

    # torch refs
    y_t = torch.as_tensor(y, device=device, dtype=torch.float32).view(-1, 1)
    T_obs = stat_fn(x, device, ref_u=y_t)

    z = np.concatenate([x, y], axis=0)
    ge = 0
    for _ in range(B):
        perm = rng.permutation(2 * n)
        x_b = z[perm[:n]]
        y_b = z[perm[n:]]
        y_b_t = torch.as_tensor(y_b, device=device, dtype=torch.float32).view(-1, 1)
        T_b = stat_fn(x_b, device, ref_u=y_b_t)
        ge += int(T_b >= T_obs)

    p = (ge + 1.0) / (B + 1.0)
    return float(T_obs), float(p)


def chi2_pvalue_to_uniform(v_np, B: int = 20) -> tuple[float, float]:
    v = np.asarray(v_np, float).ravel()
    v = np.clip(v, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, B + 1)
    counts, _ = np.histogram(v, bins=edges)

    expected = v.size / B
    if expected < 5:
        raise ValueError(f"Expected count n/B={expected:.2f} < 5; reduce B or increase n.")

    x2 = np.sum((counts - expected) ** 2 / expected)
    p = 1.0 - stats.chi2.cdf(x2, df=B - 1)
    return float(x2), float(p)


def ks_2sample_pvalue(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    return float(stats.ks_2samp(x, y, alternative="two-sided", mode="auto").pvalue)


def sweep_direction_parallel_twosample_ks(
    model,
    model_enc,
    device,
    mu_base,
    direction,
    deltas,
    *,
    n_samples: int = 2000,
    cov: np.ndarray | None = None,
    truncate_box: tuple[float, float] | None = None,
    seed: int = 1,
    n_reps: int = 10,
    n_jobs: int = -1,
    backend: str = "threading",
    batch_size: int = 1024,
    latent_index: int = 0,
) -> pd.DataFrame:
    """
    Sweep mean shifts along a direction and compare latent two-sample KS p-values.

    Returns one row per delta with replicate p-values and summary stats.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if n_reps <= 0:
        raise ValueError("n_reps must be positive")

    mu_base = np.asarray(mu_base, dtype=float).ravel()
    direction = np.asarray(direction, dtype=float).ravel()
    deltas = np.asarray(list(deltas), dtype=float).ravel()
    if deltas.size == 0:
        raise ValueError("deltas must be non-empty")
    if mu_base.shape != direction.shape:
        raise ValueError(f"mu_base and direction must have the same shape; got {mu_base.shape} vs {direction.shape}")

    if cov is None:
        cov = np.eye(mu_base.size, dtype=float)
    else:
        cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != mu_base.size:
        raise ValueError("cov must be square and match the dimension of mu_base")

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=deltas.size * n_reps, dtype=np.uint32)

    # Torch models are typically safer with thread-based parallelism on WSL.
    backend_use = backend
    if backend != "threading" and str(device) != "cpu":
        warnings.warn(
            "Non-thread backend with non-CPU device can be unstable on WSL/Torch; switching backend to 'threading'.",
            RuntimeWarning,
            stacklevel=2,
        )
        backend_use = "threading"

    def one(delta: float, rep_seed: int):
        rep_seed = int(rep_seed)
        mu_shift = mu_base + float(delta) * direction
        try:
            X0 = sample_x(
                n_samples=n_samples,
                dist_type="gaussian",
                params={"mean": mu_base, "cov": cov},
                seed=rep_seed,
                truncate_box=truncate_box,
            ).astype(np.float32, copy=False)

            # Use a different seed for shifted draws to avoid shared RNG trajectories.
            X1 = sample_x(
                n_samples=n_samples,
                dist_type="gaussian",
                params={"mean": mu_shift, "cov": cov},
                seed=rep_seed + 99991,
                truncate_box=truncate_box,
            ).astype(np.float32, copy=False)

            v0_p = encode_to_latent(model, X0, device, batch_size=batch_size)[:, latent_index]
            v1_p = encode_to_latent(model, X1, device, batch_size=batch_size)[:, latent_index]
            v0_b = encode_to_latent(model_enc, X0, device, batch_size=batch_size)[:, latent_index]
            v1_b = encode_to_latent(model_enc, X1, device, batch_size=batch_size)[:, latent_index]
        except Exception as exc:
            raise RuntimeError(f"Failed sweep worker for delta={delta}, seed={rep_seed}") from exc

        p_p = ks_2sample_pvalue(v0_p, v1_p)
        p_b = ks_2sample_pvalue(v0_b, v1_b)
        mu1 = float(mu_shift[0]) if mu_shift.size >= 1 else float("nan")
        mu2 = float(mu_shift[1]) if mu_shift.size >= 2 else float("nan")
        return float(delta), mu1, mu2, p_p, p_b

    tasks = [(float(delta), int(seeds[i * n_reps + j])) for i, delta in enumerate(deltas) for j in range(n_reps)]

    if n_jobs == 1:
        results = [one(delta, s) for delta, s in tasks]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend=backend_use)(delayed(one)(delta, s) for delta, s in tasks)

    rows = []
    for i, delta in enumerate(deltas):
        chunk = results[i * n_reps : (i + 1) * n_reps]
        p_p = np.array([r[3] for r in chunk], dtype=float)
        p_b = np.array([r[4] for r in chunk], dtype=float)
        mu1, mu2 = chunk[0][1], chunk[0][2]

        rows.append(
            {
                "delta": float(delta),
                "mu1": float(mu1),
                "mu2": float(mu2),
                "pval": p_p,
                "pval_base": p_b,
                "ks_p_prop_mean": float(np.mean(p_p)),
                "ks_p_base_mean": float(np.mean(p_b)),
                "ks_p_prop_std": float(np.std(p_p)),
                "ks_p_base_std": float(np.std(p_b)),
            }
        )

    return pd.DataFrame(rows)
def ks_to_uniform(x):
    x = np.sort(x)
    n = len(x)
    Fn = np.arange(1, n + 1) / n
    return float(np.max(np.abs(Fn - x)))


def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def uniformity_pvalues_all(v_np, v0_np, *, device, B=300, seed=0):
    v_np  = np.clip(np.asarray(v_np,  float).ravel(), 0.0, 1.0)
    v0_np = np.clip(np.asarray(v0_np, float).ravel(), 0.0, 1.0)

    n = v_np.shape[0]
    rng = np.random.default_rng(seed)
    u = rng.random(n)  # Uniform(0,1) reference, fixed for all tests

    # KS / CvM
    ks_p,  cvm_p  = pvals_ks_cvm_to_uniform(v_np)
    ks_p0, cvm_p0 = pvals_ks_cvm_to_uniform(v0_np)

    # MMD / Energy (permutation)
    mmd_stat_v,  mmd_p  = perm_pvalue_two_sample(mmd_stat,    v_np,  u, B=B, device=device, seed=seed+1)
    mmd_stat_v0, mmd_p0 = perm_pvalue_two_sample(mmd_stat,    v0_np, u, B=B, device=device, seed=seed+2)

    ed_stat_v,   ed_p   = perm_pvalue_two_sample(energy_stat, v_np,  u, B=B, device=device, seed=seed+3)
    ed_stat_v0,  ed_p0  = perm_pvalue_two_sample(energy_stat, v0_np, u, B=B, device=device, seed=seed+4)
    _,chi2_p = chi2_pvalue_to_uniform(v_np,B=10)
    _,chi2_p0 = chi2_pvalue_to_uniform(v_np,B=10)
    return {
        "proposed": {"KS_p": ks_p,  "CvM_p": cvm_p,  "MMD_stat": mmd_stat_v,  "MMD_p": mmd_p,  "ED_stat": ed_stat_v,  "ED_p": ed_p,'Chi2':chi2_p},
        "baseline": {"KS_p": ks_p0, "CvM_p": cvm_p0, "MMD_stat": mmd_stat_v0, "MMD_p": mmd_p0, "ED_stat": ed_stat_v0, "ED_p": ed_p0,'Chi2':chi2_p0},
        "n": int(n),
        "B": int(B),
    }
