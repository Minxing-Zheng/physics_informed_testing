"""
Small plotting helpers for trajectory visualization.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_traj(
    traj: dict[str, np.ndarray],
    title: str = "Trajectories",
    threshold: Optional[float] = None,
    window: Optional[Tuple[float, float]] = None,
    n_max: Optional[int] = None,
    save_path: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = (-2, 2),
) -> None:
    """
    Plot trajectories from simulate_spring output.

    Parameters
    ----------
    traj : dict
        Output from simulate_spring with keys "t", "s", "s_dot".
    title : str
        Plot title.
    threshold : float, optional
        If provided, draw horizontal bounds at +/- threshold.
    window : tuple, optional
        If provided, shade a time window (t1, t2).
    n_max : int, optional
        Max number of trajectories to draw if s is 2D. None draws all.
    save_path : str, optional
        If provided, save figure to this path.
    """
    tau = threshold
    t = traj.get("t")
    s = traj.get("s")
    if t is None or s is None:
        raise ValueError("traj must contain keys 't' and 's'")

    t = np.asarray(t)
    s = np.asarray(s)

    plt.figure()

    if s.ndim == 1:
        plt.plot(t, s, alpha=0.85, label="traj")
    elif s.ndim == 2:
        rng_local = np.random.default_rng(0)
        n = s.shape[0]
        if n_max is None or n_max <= 0:
            idx = np.arange(n)
        else:
            k = min(n_max, n)
            idx = rng_local.choice(np.arange(n), size=k, replace=False)
        for i, j in enumerate(idx):
            plt.plot(t, s[j], color="tab:blue", alpha=0.7, label="Trajectories" if i == 0 else None)
    else:
        raise ValueError("traj['s'] must be 1D or 2D")

    if tau is not None:
        tau = float(tau)
        plt.axhline(+tau, linestyle="--", linewidth=2, label="Safety Threshold", color='red')
        plt.axhline(-tau, linestyle="--", linewidth=2, color='red')

    if window is not None:
        t1, t2 = float(window[0]), float(window[1])
        plt.axvspan(t1, t2, alpha=0.2, label="Inspect Window", color='tab:orange')
    if ylim is not None:
        y0, y1 = float(ylim[0]), float(ylim[1])
        if y0 >= y1:
            raise ValueError(f"ylim must satisfy ylim[0] < ylim[1], got {ylim}")
        plt.ylim(y0, y1)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("State s(t)")
    
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def check_safety(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float],
) -> np.ndarray | bool:
    """
    Check safety for each trajectory based on max value in a time window.

    Parameters
    ----------
    traj : dict
        Output from simulate_spring with keys "t" and "s".
    threshold : float
        Safety threshold; violation if max(s) in window exceeds this value.
    window : tuple
        Time window (t1, t2) to check.
    """
    t = traj.get("t")
    s = traj.get("s")
    if t is None or s is None:
        raise ValueError("traj must contain keys 't' and 's'")

    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    t1, t2 = float(window[0]), float(window[1])
    if t1 >= t2:
        raise ValueError("window must satisfy t1 < t2")

    mask = (t >= t1) & (t <= t2)
    if not np.any(mask):
        raise ValueError("window does not overlap the time grid")

    if s.ndim == 1:
        max_val = np.max(s[mask])
        return bool(max_val <= threshold)

    if s.ndim == 2:
        max_vals = np.max(s[:, mask], axis=1)
        return max_vals <= threshold

    raise ValueError("traj['s'] must be 1D or 2D")


def plot_x_safety_map(
    params,
    t_final: float,
    dt: float,
    window: Tuple[float, float],
    threshold: float,
    n_grid: int = 10,
    box_limits: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (-1.0, 1.0),
    forcing_type: str = "sin",
    config: np.ndarray | None = None,
    x0: float = 0.0,
    v0: float = 0.0,
    title: str = "Safety map in X box",
    cmap: str = "RdYlGn",
    mean: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    contour_levels: int = 6,
    contour_color: str = "k",
    contour_alpha: float = 0.8,
    eta: float = 0.1,
    print_msg: bool = True,
    save_path: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """
    Sample X uniformly in a box, simulate s(t), check safety, and plot a color map.

    Parameters
    ----------
    params : SpringParams
        Physical parameters for simulate_spring.
    t_final : float
        Final time horizon.
    dt : float
        Time step.
    window : tuple
        Time window (t1, t2) to check safety.
    threshold : float
        Safety threshold for max(s) in window.
    n_grid : int
        Grid size per axis.
    box_limits : tuple
        Either (low, high) shared for both axes or ((lowA, highA), (lowW, highW)).
    forcing_type : str
        Forcing type built from X. Supported: "sin", "constant", "zero".
    config : np.ndarray, optional
        Initial state; shape (2,) or (N, 2).
    x0, v0 : float
        Initial conditions if config is None.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap.
    mean, cov : np.ndarray, optional
        If provided, overlay Gaussian contour plot for X ~ N(mean, cov) and
        estimate safety probability under the Gaussian.
    contour_levels : int
        Number of contour levels for the Gaussian overlay.
    contour_color : str
        Color for contour lines.
    contour_alpha : float
        Alpha for contour lines.
    eta : float
        Chance-constraint parameter. Safety means P(safe) >= 1 - eta.
    print_msg : bool
        If True, print the safety decision message when mean/cov are provided.
    save_path : str, optional
        If provided, save figure to this path.
    """
    from data.solve_ode import simulate_spring
    tau = threshold
    if n_grid <= 1:
        raise ValueError("n_grid must be > 1")

    if isinstance(box_limits[0], (tuple, list)):
        if len(box_limits) != 2:
            raise ValueError("box_limits must be (low, high) or ((lowA, highA), (lowW, highW))")
        a_low, a_high = float(box_limits[0][0]), float(box_limits[0][1])
        w_low, w_high = float(box_limits[1][0]), float(box_limits[1][1])
    else:
        if len(box_limits) != 2:
            raise ValueError("box_limits must be (low, high) or ((lowA, highA), (lowW, highW))")
        a_low, a_high = float(box_limits[0]), float(box_limits[1])
        w_low, w_high = float(box_limits[0]), float(box_limits[1])

    a_vals = np.linspace(a_low, a_high, n_grid)
    w_vals = np.linspace(w_low, w_high, n_grid)
    A_grid, W_grid = np.meshgrid(a_vals, w_vals, indexing="xy")
    X = np.stack([A_grid.ravel(), W_grid.ravel()], axis=1)

    traj = simulate_spring(
        params=params,
        t_final=t_final,
        dt=dt,
        X=X,
        forcing_type=forcing_type,
        config=config,
        x0=x0,
        v0=v0,
    )
    safe_mask = check_safety(traj, threshold=tau, window=window)
    safe_grid = np.asarray(safe_mask, dtype=float).reshape(W_grid.shape)

    plt.figure()
    plt.pcolormesh(a_vals, w_vals, safe_grid, cmap=cmap, shading="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="safe (1) / unsafe (0)")
    p_safe = None
    threshold_value = None
    config_safe = None
    if mean is not None and cov is not None:
        mu = np.asarray(mean, dtype=float).reshape(2)
        sigma = np.asarray(cov, dtype=float).reshape(2, 2)
        inv = np.linalg.inv(sigma)
        det = np.linalg.det(sigma)
        if det <= 0:
            raise ValueError("cov determinant must be > 0 for contour plotting")
        grid = np.stack([A_grid.ravel(), W_grid.ravel()], axis=1)
        diff = grid - mu.reshape(1, 2)
        quad = np.einsum("bi,ij,bj->b", diff, inv, diff)
        Z = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))
        Z = Z.reshape(A_grid.shape)
        plt.contour(
            A_grid,
            W_grid,
            Z,
            levels=contour_levels,
            colors=contour_color,
            linewidths=1.0,
            alpha=contour_alpha,
        )
        if not (0.0 <= eta < 1.0):
            raise ValueError("eta must be in [0, 1)")
        p_safe = float(np.sum(safe_grid * Z) / np.sum(Z))
        threshold_value = 1.0 - float(eta)
        config_safe = 1 if p_safe >= threshold_value else 0
        if print_msg:
            status = "SAFE" if config_safe == 1 else "UNSAFE"
            print(
                f"Empirical safety probability={p_safe:.4f} vs threshold {threshold_value:.4f} "
                f"(eta={eta:.4f}) -> {status}"
            )
    plt.title(title)
    plt.xlabel("A")
    plt.ylabel("omega")
    plt.grid(False)

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return {
        "A": a_vals,
        "omega": w_vals,
        "safe": safe_grid,
        "p_safe": p_safe,
        "threshold": threshold_value,
        "config_safe": config_safe,
    }
    
