"""
Utility and visualization helpers for sampling and safety evaluation.

Only a small subset is used by the notebooks:
- plot_traj, check_safety, plot_x_safety_map
- plot_mu_safety_map, compute_mu_safety_map
- filter_to_box, (alias of sample_x for backward compatibility)

The original file was removed inadvertently; this reconstruction keeps the
API stable for the existing notebooks.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from data.solve_ode import simulate_spring
from data.sample_x import sample_x, _filter_to_box as filter_to_box


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sample_gaussian_and_filter(
    n: int,
    mean: np.ndarray,
    cov: np.ndarray,
    rng: np.random.Generator | None = None,
    truncate_box: tuple[float, float] | None = None,
    max_rounds: int = 50,
    oversample: int = 2,
) -> np.ndarray:
    """
    Backward-compatible wrapper that routes to sample_x with truncation support.
    """
    rng = np.random.default_rng() if rng is None else rng
    seed = int(rng.integers(0, 2**32 - 1))
    return sample_x(
        n_samples=n,
        dist_type="gaussian",
        params={"mean": mean, "cov": cov},
        seed=seed,
        truncate_box=truncate_box,
        oversample=oversample,
        max_rounds=max_rounds,
    )



# --------------------------------------------------------------------------- #
# Trajectory utilities
# --------------------------------------------------------------------------- #


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
        plt.axhline(+tau, linestyle="--", linewidth=2, label="Safety Threshold", color="red")
        plt.axhline(-tau, linestyle="--", linewidth=2, color="red")

    if window is not None:
        t1, t2 = float(window[0]), float(window[1])
        plt.axvspan(t1, t2, alpha=0.2, label="Inspect Window", color="tab:orange")
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
    Check safety for each trajectory based on max |s| inside a time window.
    Returns a boolean mask (one per trajectory) or a single bool for 1D s.
    """
    t = traj.get("t")
    s = traj.get("s")
    if t is None or s is None:
        raise ValueError("traj must contain keys 't' and 's'")

    t = np.asarray(t)
    s = np.asarray(s)
    t1, t2 = float(window[0]), float(window[1])
    mask_t = (t >= t1) & (t <= t2)
    if s.ndim == 1:
        safe = np.max(np.abs(s[mask_t])) <= threshold
    elif s.ndim == 2:
        safe = np.max(np.abs(s[:, mask_t]), axis=1) <= threshold
    else:
        raise ValueError("traj['s'] must be 1D or 2D")
    return safe


# --------------------------------------------------------------------------- #
# Sampling and safety maps
# --------------------------------------------------------------------------- #



def compute_x_safety_map(
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
) -> dict[str, np.ndarray]:
    """
    Sample X uniformly in a box, simulate s(t), and return safety grid (no plotting).

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

    return {
        "A": a_vals,
        "omega": w_vals,
        "mu1": a_vals,
        "mu2": w_vals,
        "safe": safe_grid,
        "t": traj.get("t"),
        "s": traj.get("s"),
    }
    


def _bin_edges(vals: np.ndarray) -> np.ndarray:
    """Convert 1D grid centers to bin edges for pcolormesh."""
    vals = np.asarray(vals, float)
    if vals.size < 2:
        raise ValueError("Need at least 2 grid points to form bin edges.")
    mid = 0.5 * (vals[1:] + vals[:-1])
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = mid
    edges[0] = vals[0] - (mid[0] - vals[0])
    edges[-1] = vals[-1] + (vals[-1] - mid[-1])
    return edges


def plot_mu_safety_map(
    res: dict,
    *,
    mode: str = "indicator",  # "indicator" or "prob"
    title: str = r"Safety map over $\mu$",
    cmap: str = "RdYlGn",
    eta=0.1,
    show_cell_edges: bool = True,
    edgecolor: str = "k",
    linewidth: float = 0.2,
    draw_boundary: bool = True,
    boundary_color: str = "k",
    boundary_lw: float = 1.0,
    boundary_alpha: float = 0.9,
    colorbar: bool = True,
    save_path: Optional[str] = None,
    vmin=0, vmax = 1
) -> None:
    """
    Plot a precomputed mu safety map. Keeps plotting separate from computation.
    """
    mu1_vals = res["mu1"] if "mu1" in res else res.get("A")
    mu2_vals = res["mu2"] if "mu2" in res else res.get("omega")
    if mu1_vals is None or mu2_vals is None:
        raise ValueError("res must contain mu1/mu2 (or A/omega) grid coordinates.")

    if mode == "prob":
        Z = res.get("p_safe_grid", res.get("safe"))
        cbar_label = r"$\hat p_{\mathrm{safe}}(\mu)$"
        threshold_value = 1.0 - float(eta)
    elif mode == "indicator":
        Z = res.get("config_safe_grid")
        if Z is None:
            safe_grid = res.get("safe")
            thr = res.get("threshold", 1.0)
            if safe_grid is None:
                raise ValueError("Need 'safe' grid to derive indicator map.")
            Z = (np.asarray(safe_grid) >= thr).astype(float)
        cbar_label = r"$\mathbb{1}\{\hat p_{\mathrm{safe}}(\mu)\geq 1-\eta\}$"
    else:
        raise ValueError("mode must be 'indicator' or 'prob'")

    Z = np.asarray(Z)
    mu1_vals = np.asarray(mu1_vals)
    mu2_vals = np.asarray(mu2_vals)

    # Align axes: pcolormesh expects (Ny, Nx) = (len(mu2), len(mu1))
    if Z.shape == (mu1_vals.size, mu2_vals.size):
        Z_plot = Z.T
    elif Z.shape == (mu2_vals.size, mu1_vals.size):
        Z_plot = Z
    else:
        raise ValueError(f"Shape mismatch: Z{Z.shape} vs grid ({mu1_vals.size}, {mu2_vals.size})")

    mu1_edges = _bin_edges(mu1_vals)
    mu2_edges = _bin_edges(mu2_vals)

    plt.figure()
    mesh_kwargs = dict(cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    if show_cell_edges:
        mesh_kwargs.update(edgecolors=edgecolor, linewidth=linewidth)

    plt.pcolormesh(mu1_edges, mu2_edges, Z_plot, **mesh_kwargs)

    if colorbar:
        plt.colorbar(label=cbar_label)

    if draw_boundary:
        if mode == "indicator":
            Zc = res.get("config_safe_grid", res.get("config_safe"))
            if Zc is not None:
                Zc = np.asarray(Zc)
                if Zc.shape == (mu1_vals.size, mu2_vals.size):
                    Zc = Zc.T
                plt.contour(mu1_vals, mu2_vals, Zc, levels=[0.5],
                            colors=boundary_color, linewidths=boundary_lw, alpha=boundary_alpha)
        elif mode == "prob":
            plt.contour(mu1_vals, mu2_vals, Z_plot, levels=[threshold_value],
                        colors=boundary_color, linewidths=boundary_lw,
                        alpha=boundary_alpha, linestyles="--")

    plt.title(title)
    plt.xlabel(r"$\mu_1$")
    plt.ylabel(r"$\mu_2$")
    plt.grid(False)

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def _eval_one_mu_cell(
    j: int,
    i: int,
    mu: np.ndarray,
    *,
    params,
    t_final: float,
    dt: float,
    window: Tuple[float, float],
    tau: float,
    sigma: np.ndarray,
    n_per_mu: int,
    forcing_type: str,
    config: np.ndarray | None,
    truncate_X_box: Optional[tuple[float, float]],
    batch_size: int,
    seed: int,
) -> tuple[int, int, float]:
    """
    Evaluate p_safe(mu) for one grid cell (j, i). Returns (j, i, p_safe).
    Each cell uses its own RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    X = sample_x(
        n_samples=n_per_mu,
        dist_type="gaussian",
        params={"mean": mu, "cov": sigma},
        seed=seed,
        truncate_box=truncate_X_box,
    )

    safe_count = 0
    n_done = 0
    while n_done < n_per_mu:
        Xb = X[n_done : n_done + batch_size]
        traj = simulate_spring(
            params=params,
            t_final=t_final,
            dt=dt,
            X=Xb,
            forcing_type=forcing_type,
            config=config,
        )
        safe_mask = check_safety(traj, threshold=tau, window=window)
        safe_count += int(np.sum(safe_mask))
        n_done += Xb.shape[0]

    return j, i, safe_count / float(n_per_mu)


def compute_mu_safety_map(
    params,
    t_final: float,
    dt: float,
    window: Tuple[float, float],
    threshold: float,
    cov: np.ndarray,
    n_mu_grid: int = 25,
    mu_limits: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = (-1.0, 1.0),
    n_per_mu: int = 10_000,
    forcing_type: str = "sin",
    config: np.ndarray | None = None,
    eta: float = 0.1,
    truncate_X_box: Optional[tuple[float, float]] = None,
    batch_size: int = 2000,
    seed: int = 0,
    n_jobs: int = -1,
) -> dict:
    """
    Estimate p_safe(mu) on a grid of initial-condition means.
    """
    cov = np.asarray(cov, float)
    if isinstance(mu_limits[0], (tuple, list, np.ndarray)):
        A_lim, w_lim = mu_limits  # type: ignore
    else:
        A_lim = w_lim = mu_limits  # symmetric box
    A_vals = np.linspace(A_lim[0], A_lim[1], n_mu_grid)
    w_vals = np.linspace(w_lim[0], w_lim[1], n_mu_grid)
    safe_grid = np.zeros((n_mu_grid, n_mu_grid), dtype=float)

    seeds = np.random.SeedSequence(seed).spawn(n_mu_grid * n_mu_grid)
    tasks = []
    idx = 0
    for j, A in enumerate(A_vals):
        for i, w in enumerate(w_vals):
            mu = np.array([A, w], dtype=float)
            tasks.append((j, i, mu, seeds[idx]))
            idx += 1

    def _run(task):
        j, i, mu, ss = task
        return _eval_one_mu_cell(
            j=j,
            i=i,
            mu=mu,
            params=params,
            t_final=t_final,
            dt=dt,
            window=window,
            tau=threshold,
            sigma=cov,
            n_per_mu=n_per_mu,
            forcing_type=forcing_type,
            config=config,
            truncate_X_box=truncate_X_box,
            batch_size=batch_size,
            seed=ss.generate_state(1)[0],
        )

    results = Parallel(n_jobs=n_jobs)(delayed(_run)(task) for task in tasks)
    for j, i, p_safe in results:
        safe_grid[j, i] = p_safe

    # Basic safety decision on nominal config
    if not (0.0 <= eta < 1.0):
        raise ValueError("eta must be in [0, 1)")
    threshold_value = 1.0 - float(eta)
    config_safe_grid = (safe_grid >= threshold_value).astype(float)
    config_safe = 1 if float(np.mean(safe_grid)) >= threshold_value else 0

    return {
        "A": A_vals,
        "omega": w_vals,
        "mu1": A_vals,
        "mu2": w_vals,
        "safe": safe_grid,
        "config_safe_grid": config_safe_grid,
        "config_safe": config_safe,
        "threshold": threshold_value,
    }


__all__ = [
    "plot_traj",
    "check_safety",
    "compute_x_safety_map",
    "filter_to_box",
    "sample_gaussian_and_filter",
    "compute_mu_safety_map",
    "plot_mu_safety_map",
]
