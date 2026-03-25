"""
Loss and regularization utilities.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.2, 0.5, 1.0)) -> torch.Tensor:
    """Multi-kernel RBF on pairs of samples.

    Shapes: x -> (B, d), y -> (B, d). Returns (B, B).
    """
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).T
    dist2 = x2 + y2 - 2.0 * (x @ y.T)
    k_val = 0.0
    for s in sigmas:
        k_val = k_val + torch.exp(-dist2 / (2.0 * s * s))
    return k_val / len(sigmas)


def _median_heuristic_sigma(x: torch.Tensor) -> float:
    # Median of pairwise distances (robust bandwidth guess)
    with torch.no_grad():
        d = torch.cdist(x, x, p=2)
        med = d.median()
        return med.item() if torch.isfinite(med) else 0.5


def mmd2_unif(
    v: torch.Tensor,
    sigmas=None,
    ref_u: torch.Tensor | None = None,
    use_median: bool = True,
    low_discrepancy: bool = False,
) -> torch.Tensor:
    """Unbiased MMD^2 between encoder samples ``v`` and U[0,1]^d.

    Args:
        v: (B, d) latent samples.
        sigmas: list/tuple of RBF bandwidths. If None, use median heuristic grid.
        ref_u: optional fixed reference uniforms shaped like ``v``. Re-use to cut variance.
        use_median: if True and sigmas is None, build a 4-bandwidth grid around the median distance.
        low_discrepancy: use Sobol reference instead of random for lower variance.
    """

    if sigmas is None:
        med = _median_heuristic_sigma(v) if use_median else 0.5
        sigmas = (0.5 * med + 1e-3, med + 1e-3, 2 * med + 1e-3, 4 * med + 1e-3)

    if ref_u is None:
        if low_discrepancy:
            # Sobol reference in [0,1]^d for lower discrepancy than iid noise.
            dim = v.shape[1]
            n = v.shape[0]
            sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=None)
            u = sobol.draw(n).to(v)
        else:
            u = torch.rand_like(v)
    else:
        u = ref_u

    kvv = _rbf_kernel(v, v, sigmas=sigmas)
    kuu = _rbf_kernel(u, u, sigmas=sigmas)
    kvu = _rbf_kernel(v, u, sigmas=sigmas)

    bsz = v.shape[0]
    kvv = kvv - torch.diag(torch.diagonal(kvv))
    kuu = kuu - torch.diag(torch.diagonal(kuu))

    return kvv.sum() / (bsz * (bsz - 1)) + kuu.sum() / (bsz * (bsz - 1)) - 2.0 * kvu.mean()


def energy_distance_unif(v: torch.Tensor, ref_u: torch.Tensor | None = None) -> torch.Tensor:
    """Energy distance to U[0,1]^d (bias-reduced, low-variance alternative to MMD)."""
    if ref_u is None:
        ref_u = torch.rand_like(v)
    bsz = v.shape[0]
    c_vv = torch.pdist(v, p=2).mean()  # E||v-v'||
    c_uu = torch.pdist(ref_u, p=2).mean()  # E||u-u'||
    c_vu = torch.cdist(v, ref_u, p=2).mean()  # E||v-u||
    return 2 * c_vu - c_vv - c_uu


def _flatten_latent_1d(v: torch.Tensor) -> torch.Tensor:
    """Flatten latent samples to 1D for scalar-uniform regularizers."""
    if v.ndim == 0:
        return v.reshape(1)
    if v.ndim == 1:
        return v
    if v.ndim == 2 and v.shape[1] == 1:
        return v[:, 0]
    return v.reshape(-1)


def cvm_unif(v: torch.Tensor) -> torch.Tensor:
    """Differentiable 1D Cramer-von Mises loss to Uniform(0, 1).

    Assumes the latent has already been mapped into (0, 1), e.g. by sigmoid.
    This compares sorted samples to uniform quantile targets.
    """
    u = torch.sort(_flatten_latent_1d(v))[0]
    n = u.numel()
    if n < 2:
        return torch.zeros((), device=v.device, dtype=v.dtype)
    target = (torch.arange(n, device=u.device, dtype=u.dtype) + 0.5) / n
    return torch.mean((u - target) ** 2)


def wasserstein1_unif(v: torch.Tensor) -> torch.Tensor:
    """Differentiable 1D Wasserstein-1 loss to Uniform(0, 1).

    Assumes the latent has already been mapped into (0, 1), e.g. by sigmoid.
    In 1D, W1 reduces to the mean absolute deviation between sorted samples and
    the target quantile grid.
    """
    u = torch.sort(_flatten_latent_1d(v))[0]
    n = u.numel()
    if n < 2:
        return torch.zeros((), device=v.device, dtype=v.dtype)
    target = (torch.arange(n, device=u.device, dtype=u.dtype) + 0.5) / n
    return torch.mean(torch.abs(u - target))


def asymmetric_interval_l1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    lower: float,
    upper: float,
    alpha_plus: float = 2.0,
    beta_plus: float = 1.0,
    alpha_minus: float = 2.0,
    beta_minus: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Asymmetric L1-style loss with interval-dependent directional penalties.

    This implements:
      - y > U: alpha_plus * (y - yhat)_+ + beta_plus * (yhat - y)_+
      - L <= y <= U: |y - yhat|
      - y < L: alpha_minus * (yhat - y)_+ + beta_minus * (y - yhat)_+

    Where (a)_+ = max(a, 0). To encourage:
      - overestimation above ``upper``: set ``alpha_plus > beta_plus``
      - underestimation below ``lower``: set ``alpha_minus > beta_minus``

    Args:
        y_true: Ground-truth tensor.
        y_pred: Prediction tensor (same shape as ``y_true``).
        lower: Lower threshold L.
        upper: Upper threshold U, must satisfy lower < upper.
        alpha_plus: Penalty for underestimation when y_true > upper.
        beta_plus: Penalty for overestimation when y_true > upper.
        alpha_minus: Penalty for overestimation when y_true < lower.
        beta_minus: Penalty for underestimation when y_true < lower.
        reduction: ``"none"``, ``"mean"``, or ``"sum"``.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have same shape, got {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
        )
    if float(lower) >= float(upper):
        raise ValueError(f"Expected lower < upper, got lower={lower}, upper={upper}")
    for name, val in {
        "alpha_plus": alpha_plus,
        "beta_plus": beta_plus,
        "alpha_minus": alpha_minus,
        "beta_minus": beta_minus,
    }.items():
        if float(val) < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction='{reduction}', expected 'none'|'mean'|'sum'")

    y_true_f = y_true.to(dtype=y_pred.dtype)

    # Directional signed errors.
    under_err = F.relu(y_true_f - y_pred)  # y_true > y_pred
    over_err = F.relu(y_pred - y_true_f)   # y_pred > y_true

    mask_hi = y_true_f > float(upper)
    mask_lo = y_true_f < float(lower)
    mask_mid = ~(mask_hi | mask_lo)

    loss_hi = float(alpha_plus) * under_err + float(beta_plus) * over_err
    loss_mid = torch.abs(y_true_f - y_pred)
    loss_lo = float(alpha_minus) * over_err + float(beta_minus) * under_err

    loss = (
        mask_hi.to(loss_hi.dtype) * loss_hi
        + mask_mid.to(loss_mid.dtype) * loss_mid
        + mask_lo.to(loss_lo.dtype) * loss_lo
    )

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
