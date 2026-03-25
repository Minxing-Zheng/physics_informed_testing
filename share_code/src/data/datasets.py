"""
Dataset helpers for spring trajectories.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SpringDataset(Dataset):
    """
    Simple dataset wrapper for (x, o_hist, o_next) tuples.

    x:      (N, d_x)
    o_hist: (N, L, d_o)
    o_next: (N, L, d_o)
    y:      (N,) optional binary stability/safety label
    """

    def __init__(
        self,
        x: np.ndarray,
        o_hist: np.ndarray,
        o_next: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.o_hist = torch.as_tensor(o_hist, dtype=torch.float32)
        self.o_next = torch.as_tensor(o_next, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

        n = int(self.x.shape[0])
        if int(self.o_hist.shape[0]) != n or int(self.o_next.shape[0]) != n:
            raise ValueError("x, o_hist, and o_next must have the same first dimension")
        if self.y is not None and int(self.y.shape[0]) != n:
            raise ValueError("y must have the same first dimension as x")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.y is None:
            return self.x[idx], self.o_hist[idx], self.o_next[idx]
        return self.x[idx], self.o_hist[idx], self.o_next[idx], self.y[idx]


def make_sequence_pairs(
    s: np.ndarray,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (o_hist, o_next) pairs for next-step prediction.

    Parameters
    ----------
    s : np.ndarray
        Trajectories array of shape (N, T) or (T,).
    stride : int, optional
        Downsampling stride. If None, uses stride=1.

    Returns
    -------
    o_hist, o_next : np.ndarray
        Both of shape (N, L, 1) where L = T_ds - 1.
    """
    if stride is None:
        stride = 1
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    s = np.asarray(s, dtype=np.float32)
    if s.ndim == 1:
        s = s[None, :]
    if s.ndim != 2:
        raise ValueError("s must have shape (N, T) or (T,)")

    s_ds = s[:, ::stride]
    if s_ds.shape[1] < 2:
        raise ValueError("trajectory too short after downsampling")

    o_hist = s_ds[:, :-1][:, :, None]
    o_next = s_ds[:, 1:][:, :, None]
    return o_hist, o_next
