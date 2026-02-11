"""
Plotting helpers split out from compute functions to keep side effects separate.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from data.utils import _bin_edges


def plot_x_safety_map(
    res: dict,
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
) -> None:
    """
    Plot safety grid from compute_x_safety_map output, with optional Gaussian contour overlay.
    """
    a_vals = res["A"]
    w_vals = res["omega"]
    safe_grid = res["safe"]
    A_grid, W_grid = np.meshgrid(a_vals, w_vals, indexing="xy")

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


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
    vmin=0,
    vmax=1,
) -> None:
    """
    Plot a precomputed mu safety map (probability or indicator).
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


__all__ = ["plot_x_safety_map", "plot_mu_safety_map"]
