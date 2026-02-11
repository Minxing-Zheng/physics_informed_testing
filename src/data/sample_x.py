"""
Sampling helpers for the Physics-Informed-Hypo-Test project.

Primary entry point: ``sample_x`` which currently supports Gaussian draws
and can be extended with other distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _check_cov(cov: np.ndarray) -> None:
    """Validate symmetric positive semi-definite covariance."""
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square; got {cov.shape}")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-12):
        raise ValueError("covariance must be positive semi-definite")


def _validate_gaussian_params(mean: np.ndarray, cov: np.ndarray) -> None:
    if mean.ndim != 1:
        raise ValueError(f"mean must be 1D; got shape {mean.shape}")
    _check_cov(cov)
    if cov.shape[0] != mean.shape[0]:
        raise ValueError("covariance dimension must match mean dimension")


# --------------------------------------------------------------------------- #
# Distribution specs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GaussianSpec:
    """Specification of a d-dimensional Gaussian."""

    mean: Tuple[float, ...] = (0.0, 0.0)
    cov: Tuple[Tuple[float, ...], ...] = ((1.0, 0.0), (0.0, 1.0))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(self.mean, dtype=float)
        sigma = np.asarray(self.cov, dtype=float)
        _validate_gaussian_params(mu, sigma)
        return mu, sigma


# --------------------------------------------------------------------------- #
# Sampling functions
# --------------------------------------------------------------------------- #


def sample_bivariate_gaussian(
    n_samples: int,
    mean: Iterable[float] | np.ndarray = (0.0, 0.0),
    cov: Iterable[Iterable[float]] | np.ndarray = ((1.0, 0.0), (0.0, 1.0)),
    seed: int | None = None,
    truncate_box: tuple[float, float] | None = None,
    oversample: int = 2,
    max_rounds: int = 50,
) -> np.ndarray:
    """Backward-compatible helper for 2D Gaussian draws."""
    return sample_x(
        n_samples,
        dist_type="gaussian",
        params={"mean": mean, "cov": cov},
        seed=seed,
        truncate_box=truncate_box,
        oversample=oversample,
        max_rounds=max_rounds,
    )


def _filter_to_box(X: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep rows within [low, high]^d."""
    mask = np.all((X >= low) & (X <= high), axis=1)
    return X[mask]


def sample_x(
    n_samples: int,
    dist_type: str = "gaussian",
    params: dict | None = None,
    seed: int | None = None,
    truncate_box: tuple[float, float] | None = None,
    oversample: int = 2,
    max_rounds: int = 50,
) -> np.ndarray:
    """
    Generic sampler for input X.

    Supported types
    ---------------
    - ``gaussian``: expects params with ``mean`` (1D array-like) and ``cov`` (square PSD matrix).

    Returns
    -------
    np.ndarray of shape (n_samples, d)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    params = params or {}

    if dist_type.lower() == "gaussian":
        if "mean" not in params or "cov" not in params:
            raise ValueError("gaussian params must include 'mean' and 'cov'")
        mean = np.asarray(params["mean"], dtype=float)
        cov = np.asarray(params["cov"], dtype=float)
        _validate_gaussian_params(mean, cov)
        if truncate_box is None:
            return rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)

        low, high = truncate_box
        kept = []
        total = 0
        for _ in range(max_rounds):
            X = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples * oversample)
            X = _filter_to_box(X, low, high)
            if X.size > 0:
                kept.append(X)
                total += X.shape[0]
            if total >= n_samples:
                return np.concatenate(kept, axis=0)[:n_samples]

        raise RuntimeError(
            f"Not enough accepted samples inside box {truncate_box}. "
            f"Try enlarging the box or increasing oversample/max_rounds."
        )

    raise ValueError(f"Unsupported dist_type '{dist_type}'. Supported: gaussian.")


__all__ = [
    "GaussianSpec",
    "sample_bivariate_gaussian",
    "sample_x",
]
