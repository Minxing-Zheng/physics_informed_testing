"""
Loss and regularization utilities.
"""

from __future__ import annotations

import torch


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
