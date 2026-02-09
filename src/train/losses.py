"""
Loss and regularization utilities.
"""

from __future__ import annotations

import torch


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas=(0.1, 0.2, 0.5, 1.0)) -> torch.Tensor:
    """
    x: (B, d), y: (B, d)
    returns: (B, B) kernel matrix (multi-kernel RBF)
    """
    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True).T
    dist2 = x2 + y2 - 2.0 * (x @ y.T)
    k_val = 0.0
    for s in sigmas:
        k_val = k_val + torch.exp(-dist2 / (2.0 * s * s))
    return k_val / len(sigmas)


def mmd2_unif(v: torch.Tensor, sigmas=(0.1, 0.2, 0.5, 1.0)) -> torch.Tensor:
    u = torch.rand_like(v)
    kvv = _rbf_kernel(v, v, sigmas=sigmas)
    kuu = _rbf_kernel(u, u, sigmas=sigmas)
    kvu = _rbf_kernel(v, u, sigmas=sigmas)

    bsz = v.shape[0]
    kvv = kvv - torch.diag(torch.diagonal(kvv))
    kuu = kuu - torch.diag(torch.diagonal(kuu))

    return kvv.sum() / (bsz * (bsz - 1)) + kuu.sum() / (bsz * (bsz - 1)) - 2.0 * kvu.mean()
