"""
Stability/safety checks for trajectory data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_and_unpack_traj(traj: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    t = traj.get("t")
    s = traj.get("s")
    if t is None or s is None:
        raise ValueError("traj must contain keys 't' and 's'")

    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    if t.ndim != 1:
        raise ValueError("traj['t'] must be 1D")
    if s.ndim not in (1, 2):
        raise ValueError("traj['s'] must be 1D or 2D")
    if s.shape[-1] != t.shape[0]:
        raise ValueError("traj['s'] last dimension must match len(traj['t'])")
    return t, s


def _window_mask(t: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    t_l, t_u = float(window[0]), float(window[1])
    if t_l > t_u:
        raise ValueError(f"window must satisfy t_l <= t_u, got {window}")
    mask = (t >= t_l) & (t <= t_u)
    if not np.any(mask):
        raise ValueError(f"window {window} does not overlap with trajectory time range")
    return mask


def check_violation(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float],
    use_abs: bool = True,
) -> np.ndarray | bool:
    """
    Return violation indicator(s) for trajectory(ies) in a time window.

    Violation is defined as:
    - max(|s(t)|) > threshold if use_abs=True (default)
    - max(s(t)) > threshold if use_abs=False
    """
    threshold = float(threshold)
    if use_abs and threshold < 0:
        raise ValueError("threshold must be non-negative when use_abs=True")

    t, s = _validate_and_unpack_traj(traj)
    mask_t = _window_mask(t, window)

    s_window = s[..., mask_t]
    if use_abs:
        s_window = np.abs(s_window)
    peak = np.max(s_window, axis=-1)
    violated = peak > threshold

    if s.ndim == 1:
        return bool(violated)
    return np.asarray(violated, dtype=bool)


def check_safety(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float],
    use_abs: bool = True,
) -> np.ndarray | bool:
    """
    Return safety indicator(s): True if trajectory is safe in window, else False.
    """
    violated = check_violation(
        traj=traj,
        threshold=threshold,
        window=window,
        use_abs=use_abs,
    )
    if isinstance(violated, np.ndarray):
        return ~violated
    return not violated


def stability_labels(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float],
    use_abs: bool = True,
) -> np.ndarray | int:
    """
    Return binary labels per trajectory: 1 if safe, 0 if violated.
    """
    safe = check_safety(
        traj=traj,
        threshold=threshold,
        window=window,
        use_abs=use_abs,
    )
    if isinstance(safe, np.ndarray):
        return safe.astype(np.int64)
    return int(safe)


__all__ = ["check_violation", "check_safety", "stability_labels"]
