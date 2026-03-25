"""
Plotting helpers split out from compute functions to keep side effects separate.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from data.utils import _bin_edges



def plot_x_safety_map(
    res: dict,
    title: str = "Safety map in X box",
    cmap: str = "RdYlGn",
    norm=None,
    mean: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    contour_levels: int = 6,
    contour_mass: bool = False,
    contour_mass_max: float = 0.9,
    contour_color: str = "k",
    contour_alpha: float = 0.8,
    eta: float = 0.1,
    print_msg: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot safety grid from compute_x_safety_map output, with optional Gaussian contour overlay.
    """
    a_vals = res["A"]
    w_vals = res["omega"]
    safe_grid = res["safe"]
    A_grid, W_grid = np.meshgrid(a_vals, w_vals, indexing="xy")

    fig, ax = plt.subplots()
    mesh_kwargs = dict(cmap=cmap, shading="auto")
    if norm is not None:
        mesh_kwargs["norm"] = norm
    else:
        mesh_kwargs["vmin"] = 0.0
        mesh_kwargs["vmax"] = 1.0
    mesh = ax.pcolormesh(a_vals, w_vals, safe_grid, **mesh_kwargs)
    fig.colorbar(mesh, ax=ax, label="safe (1) / unsafe (0)")
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
        if contour_mass:
            if not (0.0 < contour_mass_max < 1.0):
                raise ValueError("contour_mass_max must be in (0, 1)")
            if contour_levels < 1:
                raise ValueError("contour_levels must be >= 1")
            mass_levels = np.linspace(contour_mass_max / contour_levels, contour_mass_max, contour_levels)
            density_peak = 1.0 / (2 * np.pi * np.sqrt(det))
            density_levels = density_peak * np.exp(-0.5 * chi2.ppf(mass_levels, df=2))
            cs = ax.contour(
                A_grid,
                W_grid,
                Z,
                levels=np.sort(density_levels),
                colors=contour_color,
                linewidths=1.0,
                alpha=contour_alpha,
            )
            # fmt = {
            #     level: f"{mass:.0%}"
            #     for level, mass in zip(np.sort(density_levels), mass_levels[::-1])
            # }
            # plt.clabel(cs, fmt=fmt, inline=True, fontsize=8)
        else:
            ax.contour(
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

    ax.set_title(title)
    ax.set_xlabel("A")
    ax.set_ylabel("omega")
    ax.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# def plot_mu_safety_map(
#     res: dict,
#     *,
#     mode: str = "indicator",  # "indicator" or "prob"
#     title: str = r"Safety map over $\mu$",
#     cmap: str = "RdYlGn",
#     norm: Optional[plt.Normalize] = None,
#     eta=0.1,
#     show_cell_edges: bool = True,
#     edgecolor: str = "k",
#     linewidth: float = 0.2,
#     draw_boundary: bool = True,
#     boundary_color: str = "k",
#     boundary_lw: float = 1.0,
#     boundary_alpha: float = 0.9,
#     colorbar: bool = True,
#     save_path: Optional[str] = None,
#     vmin=0,
#     vmax=1,
# ) -> None:
#     """
#     Plot a precomputed mu safety map (probability or indicator).
#     """
#     mu1_vals = res["mu1"] if "mu1" in res else res.get("A")
#     mu2_vals = res["mu2"] if "mu2" in res else res.get("omega")
#     if mu1_vals is None or mu2_vals is None:
#         raise ValueError("res must contain mu1/mu2 (or A/omega) grid coordinates.")

#     if mode == "prob":
#         Z = res.get("p_safe_grid", res.get("safe"))
#         cbar_label = "Safety Probability (Chance)"#r"$\hat p_{\mathrm{safe}}(\mu)$"
#         threshold_value = 1.0 - float(eta)
#     elif mode == "indicator":
#         Z = res.get("config_safe_grid")
#         if Z is None:
#             safe_grid = res.get("safe")
#             thr = res.get("threshold", 1.0)
#             if safe_grid is None:
#                 raise ValueError("Need 'safe' grid to derive indicator map.")
#             Z = (np.asarray(safe_grid) >= thr).astype(float)
#         cbar_label ="Safety Indicator" #r"$\mathbb{1}\{\hat p_{\mathrm{safe}}(\mu)\geq 1-\eta\}$"
#     else:
#         raise ValueError("mode must be 'indicator' or 'prob'")

#     Z = np.asarray(Z)
#     mu1_vals = np.asarray(mu1_vals)
#     mu2_vals = np.asarray(mu2_vals)

#     if Z.shape == (mu1_vals.size, mu2_vals.size):
#         Z_plot = Z.T
#     elif Z.shape == (mu2_vals.size, mu1_vals.size):
#         Z_plot = Z
#     else:
#         raise ValueError(f"Shape mismatch: Z{Z.shape} vs grid ({mu1_vals.size}, {mu2_vals.size})")

#     mu1_edges = _bin_edges(mu1_vals)
#     mu2_edges = _bin_edges(mu2_vals)
#     cmap_default, norm_default, vmin_default, vmax_default, cbar_ticks = _mu_plot_style(
#         mode, cmap, vmin, vmax
#     )
#     cmap = cmap if cmap is not None else cmap_default
#     norm = norm if norm is not None else norm_default
#     vmin = vmin if vmin is not None else vmin_default
#     vmax = vmax if vmax is not None else vmax_default

#     plt.figure()
#     mesh_kwargs = dict(cmap=cmap, shading="auto")
#     if norm is not None:
#         mesh_kwargs["norm"] = norm
#     else:
#         mesh_kwargs["vmin"] = vmin
#         mesh_kwargs["vmax"] = vmax
#     if show_cell_edges:
#         mesh_kwargs.update(edgecolors=edgecolor, linewidth=linewidth)

#     plt.pcolormesh(mu1_edges, mu2_edges, Z_plot, **mesh_kwargs)

#     if colorbar:
#         plt.colorbar(label=cbar_label, ticks=cbar_ticks)

#     if draw_boundary:
#         if mode == "indicator":
#             Zc = res.get("config_safe_grid", res.get("config_safe"))
#             if Zc is not None:
#                 Zc = np.asarray(Zc)
#                 if Zc.shape == (mu1_vals.size, mu2_vals.size):
#                     Zc = Zc.T
#                 plt.contour(mu1_vals, mu2_vals, Zc, levels=[0.5],
#                             colors=boundary_color, linewidths=boundary_lw, alpha=boundary_alpha)
#         elif mode == "prob":
#             plt.contour(mu1_vals, mu2_vals, Z_plot, levels=[threshold_value],
#                         colors=boundary_color, linewidths=boundary_lw,
#                         alpha=boundary_alpha, linestyles="--")

#     plt.title(title)
#     plt.xlabel(r"$\mu_1$")
#     plt.ylabel(r"$\mu_2$")
#     plt.grid(False)

#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.show()

#     return fig, ax

def plot_mu_safety_map(
    res: dict,
    *,
    mode: str = "indicator",
    title: str = r"Safety map over $\mu$",
    cmap: str = "RdYlGn",
    norm=None,
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
):
    mu1_vals = res["mu1"] if "mu1" in res else res.get("A")
    mu2_vals = res["mu2"] if "mu2" in res else res.get("omega")

    if mode == "prob":
        Z = res.get("p_safe_grid", res.get("safe"))
        cbar_label = "Stability Probability"
        threshold_value = 1.0 - float(eta)
    elif mode == "indicator":
        Z = res.get("config_safe_grid")
        if Z is None:
            safe_grid = res.get("safe")
            thr = res.get("threshold", 1.0)
            if safe_grid is None:
                raise ValueError("Need 'safe' grid to derive indicator map.")
            Z = (np.asarray(safe_grid) >= thr).astype(float)
        cbar_label = "Safety Indicator"
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

    fig, ax = plt.subplots()

    mesh_kwargs = dict(cmap=cmap, shading="auto")
    if norm is not None:
        mesh_kwargs["norm"] = norm
    else:
        mesh_kwargs["vmin"] = vmin
        mesh_kwargs["vmax"] = vmax

    if show_cell_edges:
        mesh_kwargs.update(edgecolors=edgecolor, linewidth=linewidth)

    mesh = ax.pcolormesh(mu1_edges, mu2_edges, Z_plot, **mesh_kwargs)

    if colorbar:
        fig.colorbar(mesh, ax=ax, label=cbar_label)

    if draw_boundary:
        if mode == "indicator":
            Zc = res.get("config_safe_grid", res.get("config_safe"))
            if Zc is not None:
                Zc = np.asarray(Zc)
                if Zc.shape == (mu1_vals.size, mu2_vals.size):
                    Zc = Zc.T
                cs =ax.contour(
                    mu1_vals, mu2_vals, Zc, levels=[0.5],
                    colors=boundary_color, linewidths=boundary_lw, alpha=boundary_alpha,
                   
                )
                handles, _ = cs.legend_elements()
                if handles:
                    handles[0].set_label("Stability Boundary")
                    ax.legend(handles=handles, loc="best")

        elif mode == "prob":
            cs = ax.contour(
                mu1_vals, mu2_vals, Z_plot, levels=[threshold_value],
                colors=boundary_color, linewidths=boundary_lw,
                alpha=boundary_alpha, linestyles="--",label='Stability Boundary',
            )
            handles, _ = cs.legend_elements()
            if handles:
                handles[0].set_label("Stability Boundary")
                ax.legend(handles=handles, loc="best")

    ax.set_title(title)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    from matplotlib.lines import Line2D

    boundary_handle = Line2D(
        [0], [0],
        color=boundary_color,
        lw=boundary_lw,
        alpha=boundary_alpha,
        linestyle="--" if mode == "prob" else "-",
        label="Stability Boundary",
    )
    ax.legend(handles=[boundary_handle], loc="lower right")
    ax.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax

def annotate_mu_shift_directions(
    ax,
    initial_positions,
    directions,
    labels,
    *,
    arrow_scale=0.5,
    arrow_scale_each=None,
    arrow_start_pad=0.0,
    arrow_start_pad_each=None,
    colors=None,
    text_offset=(0.05, 0.05),
    text_offset_each=None,
    text_side=None,
    arrow_head_width=0.06,
    arrow_head_length=0.08,
    fontsize=10,
    linewidth=2.0,
    scatter_size=20,
):
    
    if len(initial_positions) != len(directions) or len(labels) != len(initial_positions):
        raise ValueError("initial_positions, directions, and labels must have the same length.")

    if colors is None:
        colors = ["black", "#d95f02", "#1f77b4"]
    if arrow_scale_each is None:
        arrow_scale_each = [arrow_scale] * len(labels)
    if len(arrow_scale_each) != len(labels):
        raise ValueError("arrow_scale_each must have the same length as labels.")
    if arrow_start_pad_each is None:
        arrow_start_pad_each = [arrow_start_pad] * len(labels)
    if len(arrow_start_pad_each) != len(labels):
        raise ValueError("arrow_start_pad_each must have the same length as labels.")
    if text_offset_each is None:
        text_offset_each = [text_offset] * len(labels)
    if len(text_offset_each) != len(labels):
        raise ValueError("text_offset_each must have the same length as labels.")
    if text_side is None:
        text_side = ["auto"] * len(labels)
    if len(text_side) != len(labels):
        raise ValueError("text_side must have the same length as labels.")

    for i, ((x0, y0), (vx, vy), label, color) in enumerate(zip(initial_positions, directions, labels, colors)):
        direction_norm = float(np.hypot(vx, vy))
        if direction_norm == 0.0:
            raise ValueError("direction vectors must be non-zero.")
        ux = vx / direction_norm
        uy = vy / direction_norm

        x0_pad = x0 + arrow_start_pad_each[i] * ux
        y0_pad = y0 + arrow_start_pad_each[i] * uy
        dx = arrow_scale_each[i] * vx
        dy = arrow_scale_each[i] * vy

        ax.scatter([x0], [y0], color=color, s=scatter_size, zorder=5)
        ax.arrow(
            x0_pad, y0_pad, dx, dy,
            color=color,
            linewidth=linewidth,
            head_width=arrow_head_width,
            head_length=arrow_head_length,
            length_includes_head=True,
            zorder=5,
        )

        x1 = x0_pad + dx
        y1 = y0_pad + dy

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x1_vis = min(max(x1, xlim[0]), xlim[1])
        y1_vis = min(max(y1, ylim[0]), ylim[1])

        x_mid = 0.5 * (x0_pad + x1_vis)
        y_mid = 0.5 * (y0_pad + y1_vis)

        side = text_side[i]
        offset_x, offset_y = text_offset_each[i]
        if side not in {"auto", "top", "bottom", "left", "right"}:
            raise ValueError("Each text_side entry must be one of 'auto', 'top', 'bottom', 'left', 'right'.")
        if side == "auto":
            side = "top" if abs(dx) >= abs(dy) else "right"

        if side == "top":
            ax.text(
                x_mid,
                y_mid + offset_y,
                label,
                color=color,
                fontsize=fontsize,
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )
        elif side == "bottom":
            ax.text(
                x_mid,
                y_mid - offset_y,
                label,
                color=color,
                fontsize=fontsize,
                ha="center",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )
        elif side == "left":
            ax.text(
                x_mid - offset_x,
                y_mid,
                label,
                color=color,
                fontsize=fontsize,
                ha="right",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )
        else:
            ax.text(
                x_mid + offset_x,
                y_mid,
                label,
                color=color,
                fontsize=fontsize,
                ha="left",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )




def annotate_x_shift_directions(
    ax,
    initial_positions,
    directions,
    labels,
    *,
    arrow_scale=0.5,
    colors=None,
    horizontal_text_position="top",
    vertical_text_position="right",
    text_pad=0.05,
    fontsize=10,
    linewidth=2.0,
):
    """
    Overlay labeled shift arrows on an X-safety-map axis.

    Call this right after plot_x_safety_map(...) using ax=plt.gca().
    """
    if len(initial_positions) != len(directions) or len(labels) != len(initial_positions):
        raise ValueError("initial_positions, directions, and labels must have the same length.")

    if colors is None:
        colors = ["black", "#f39c12", "#1f77b4"]
    if horizontal_text_position not in {"top", "bottom"}:
        raise ValueError("horizontal_text_position must be 'top' or 'bottom'.")
    if vertical_text_position not in {"left", "right"}:
        raise ValueError("vertical_text_position must be 'left' or 'right'.")

    for (x0, y0), (vx, vy), label, color in zip(initial_positions, directions, labels, colors):
        dx = arrow_scale * vx
        dy = arrow_scale * vy

        ax.scatter([x0], [y0], color=color, s=40, zorder=5)
        ax.arrow(
            x0, y0, dx, dy,
            color=color,
            linewidth=linewidth,
            head_width=0.06,
            head_length=0.08,
            length_includes_head=True,
            zorder=5,
        )

        x1 = x0 + dx
        y1 = y0 + dy

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        x1_vis = min(max(x1, xlim[0]), xlim[1])
        y1_vis = min(max(y1, ylim[0]), ylim[1])

        x_mid = 0.5 * (x0 + x1_vis)
        y_mid = 0.5 * (y0 + y1_vis)

        if abs(dx) >= abs(dy):
            y_text = y_mid + text_pad if horizontal_text_position == "top" else y_mid - text_pad
            va = "bottom" if horizontal_text_position == "top" else "top"
            ax.text(
                x_mid,
                y_text,
                label,
                color=color,
                fontsize=fontsize,
                ha="center",
                va=va,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )
        else:
            x_text = x_mid + text_pad if vertical_text_position == "right" else x_mid - text_pad
            ha = "left" if vertical_text_position == "right" else "right"
            ax.text(
                x_text,
                y_mid,
                label,
                color=color,
                fontsize=fontsize,
                ha=ha,
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=6,
            )



def plot_x_safety_map_shift_panels(
    res,
    initial_positions,
    directions,
    labels,
    *,
    title_prefix="",
    cmap="RdYlGn",
    norm=None,
    mean=None,
    cov=None,
    contour_levels=5,
    contour_mass=False,
    contour_mass_max=0.9,
    contour_color="black",
    contour_alpha=0.7,
    cmap_alpha=1.0,
    shift_mode="arrow",
    shift_length=None,
    original_contour_color="gray",
    original_contour_alpha=None,
    original_contour_linestyle="--",
    shifted_contour_linestyle="-",
    show_distribution_markers=True,
    distribution_marker_size=40,
    show_distribution_legend=True,
    legend_fontsize=None,
    show_contour_labels=False,
    contour_label_fontsize=None,
    contour_label_fmt=".2f",
    arrow_scale=1.5,
    show_arrow_text=True,
    horizontal_text_position="bottom",
    vertical_text_position="left",
    text_pad=0.2,
    colors=None,
    fontsize=10,
    linewidth=2.0,
    figsize=None,
    colorbar=True,
    title_fontsize=None,
    label_fontsize=None,
    tick_fontsize=None,
    cbar_label_fontsize=None,
    cbar_tick_fontsize=None,
    save_path=None,
):
    """Plot X-safety-map panels with either shift arrows or Gaussian-shift comparisons."""
    if len(initial_positions) != len(directions) or len(labels) != len(initial_positions):
        raise ValueError("initial_positions, directions, and labels must have the same length.")
    if shift_mode not in {"arrow", "distribution", "both"}:
        raise ValueError("shift_mode must be one of 'arrow', 'distribution', or 'both'.")
    if shift_mode in {"distribution", "both"} and cov is None:
        raise ValueError("cov must be provided when shift_mode is 'distribution' or 'both'.")

    if colors is None:
        colors = ["black", "#f39c12", "#1f77b4"]
    if cbar_label_fontsize is None:
        cbar_label_fontsize = label_fontsize
    if cbar_tick_fontsize is None:
        cbar_tick_fontsize = tick_fontsize
    if contour_label_fontsize is None:
        contour_label_fontsize = tick_fontsize
    if legend_fontsize is None:
        legend_fontsize = tick_fontsize
    if original_contour_alpha is None:
        original_contour_alpha = contour_alpha

    n = len(initial_positions)
    if figsize is None:
        figsize = (4.5 * n, 4.2)

    a_vals = res["A"]
    w_vals = res["omega"]
    safe_grid = res["safe"]
    A_grid, W_grid = np.meshgrid(a_vals, w_vals, indexing="xy")

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    mesh = None
    sigma = None if cov is None else np.asarray(cov, dtype=float).reshape(2, 2)

    def _format_contour_label(value):
        return format(float(value), contour_label_fmt)

    def _resolve_shift_delta(v):
        vec = np.asarray(v, dtype=float).reshape(2)
        if shift_length is None:
            return arrow_scale * vec
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0.0:
            raise ValueError("direction vectors must be non-zero when shift_length is provided.")
        return shift_length * vec / vec_norm

    def _get_mass_levels():
        if np.isscalar(contour_levels):
            count = int(contour_levels)
            if count < 1:
                raise ValueError("contour_levels must be >= 1.")
            if not np.isclose(contour_levels, count):
                raise ValueError("contour_levels must be an integer count or an explicit list of masses.")
            if not (0.0 < contour_mass_max < 1.0):
                raise ValueError("contour_mass_max must be in (0, 1).")
            return np.linspace(contour_mass_max / count, contour_mass_max, count)

        mass_levels = np.asarray(contour_levels, dtype=float).reshape(-1)
        if mass_levels.size == 0:
            raise ValueError("contour_levels must not be empty.")
        if np.any((mass_levels <= 0.0) | (mass_levels >= 1.0)):
            raise ValueError("When contour_mass=True, explicit contour_levels must lie in (0, 1).")
        return np.sort(mass_levels)

    mass_levels = _get_mass_levels() if contour_mass else None

    def _draw_gaussian_contours(
        ax,
        mu,
        sigma,
        *,
        color,
        alpha,
        linestyle,
        label_contours,
    ):
        inv = np.linalg.inv(sigma)
        det = np.linalg.det(sigma)
        if det <= 0:
            raise ValueError("cov determinant must be > 0 for contour plotting")

        grid = np.stack([A_grid.ravel(), W_grid.ravel()], axis=1)
        diff = grid - mu.reshape(1, 2)
        quad = np.einsum("bi,ij,bj->b", diff, inv, diff)
        Z = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))
        Z = Z.reshape(A_grid.shape)

        if contour_mass:
            density_peak = 1.0 / (2 * np.pi * np.sqrt(det))
            density_levels = density_peak * np.exp(-0.5 * chi2.ppf(mass_levels, df=2))
            cs = ax.contour(
                A_grid,
                W_grid,
                Z,
                levels=np.sort(density_levels),
                colors=color,
                linewidths=1.0,
                alpha=alpha,
                linestyles=linestyle,
            )
            if label_contours:
                contour_label_map = {
                    level: _format_contour_label(mass)
                    for level, mass in zip(cs.levels, mass_levels[::-1])
                }
                ax.clabel(
                    cs,
                    cs.levels,
                    fmt=contour_label_map,
                    inline=True,
                    fontsize=contour_label_fontsize,
                )
            return cs

        cs = ax.contour(
            A_grid,
            W_grid,
            Z,
            levels=contour_levels,
            colors=color,
            linewidths=1.0,
            alpha=alpha,
            linestyles=linestyle,
        )
        if label_contours:
            ax.clabel(
                cs,
                cs.levels,
                fmt=lambda level: _format_contour_label(level),
                inline=True,
                fontsize=contour_label_fontsize,
            )
        return cs

    for ax, mu0, v, label, color in zip(axes, initial_positions, directions, labels, colors):
        mesh_kwargs = dict(cmap=cmap, shading="auto", alpha=cmap_alpha)
        if norm is not None:
            mesh_kwargs["norm"] = norm
        else:
            mesh_kwargs["vmin"] = 0.0
            mesh_kwargs["vmax"] = 1.0

        mesh = ax.pcolormesh(a_vals, w_vals, safe_grid, **mesh_kwargs)

        mu = np.asarray(mu0, dtype=float).reshape(2)
        delta = _resolve_shift_delta(v)
        shifted_mu = mu + delta

        if sigma is not None and shift_mode == "arrow":
            _draw_gaussian_contours(
                ax,
                mu,
                sigma,
                color=contour_color,
                alpha=contour_alpha,
                linestyle="-",
                label_contours=show_contour_labels,
            )

        if shift_mode in {"distribution", "both"}:
            _draw_gaussian_contours(
                ax,
                mu,
                sigma,
                color=original_contour_color,
                alpha=original_contour_alpha,
                linestyle=original_contour_linestyle,
                label_contours=False,
            )
            _draw_gaussian_contours(
                ax,
                shifted_mu,
                sigma,
                color=color,
                alpha=contour_alpha,
                linestyle=shifted_contour_linestyle,
                label_contours=show_contour_labels,
            )
            if show_distribution_markers:
                ax.scatter(
                    [mu[0]],
                    [mu[1]],
                    facecolors="none",
                    edgecolors=original_contour_color,
                    s=distribution_marker_size,
                    zorder=5,
                )
                ax.scatter(
                    [shifted_mu[0]],
                    [shifted_mu[1]],
                    color=color,
                    s=distribution_marker_size,
                    zorder=5,
                )
            if show_distribution_legend:
                from matplotlib.lines import Line2D

                handles = [
                    Line2D(
                        [0],
                        [0],
                        color=original_contour_color,
                        lw=1.5,
                        alpha=original_contour_alpha,
                        linestyle=original_contour_linestyle,
                        label="Baseline",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        lw=1.5,
                        alpha=contour_alpha,
                        linestyle=shifted_contour_linestyle,
                        label="Shifted",
                    ),
                ]
                ax.legend(handles=handles, loc="lower right", fontsize=legend_fontsize)

        if shift_mode in {"arrow", "both"}:
            x0, y0 = mu
            dx, dy = delta

            ax.scatter([x0], [y0], color=color, s=40, zorder=5)
            ax.arrow(
                x0, y0, dx, dy,
                color=color,
                linewidth=linewidth,
                head_width=0.06,
                head_length=0.08,
                length_includes_head=True,
                zorder=5,
            )

            x1 = x0 + dx
            y1 = y0 + dy

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            x1_vis = min(max(x1, xlim[0]), xlim[1])
            y1_vis = min(max(y1, ylim[0]), ylim[1])

            x_mid = 0.5 * (x0 + x1_vis)
            y_mid = 0.5 * (y0 + y1_vis)

            if show_arrow_text:
                if abs(dx) >= abs(dy):
                    y_text = y_mid + text_pad if horizontal_text_position == "top" else y_mid - text_pad
                    va = "bottom" if horizontal_text_position == "top" else "top"
                    ax.text(
                        x_mid, y_text, label,
                        color=color,
                        fontsize=fontsize,
                        ha="center",
                        va=va,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                        zorder=6,
                    )
                else:
                    x_text = x_mid + text_pad if vertical_text_position == "right" else x_mid - text_pad
                    ha = "left" if vertical_text_position == "right" else "right"
                    ax.text(
                        x_text, y_mid, label,
                        color=color,
                        fontsize=fontsize,
                        ha=ha,
                        va="center",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                        zorder=6,
                    )

        ax.set_title(rf"{label}", fontsize=title_fontsize)
        ax.set_xlabel("A", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(False)

    axes[0].set_ylabel(r"$\omega$", fontsize=label_fontsize)
    if title_prefix:
        fig.suptitle(title_prefix, y=1.02, fontsize=title_fontsize)

    if colorbar:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.70])
        # fig.colorbar(mesh, cax=cbar_ax, label="safe (1) / unsafe (0)")
        cbar = fig.colorbar(
            mesh,
            cax=cbar_ax,
            label="Stable (1) / Unstable (0)",
            ticks=[0, 1],
        )
        cbar.ax.set_yticklabels(["0", "1"])
        if cbar_label_fontsize is not None:
            cbar.set_label("Stable (1) / Unstable (0)", fontsize=cbar_label_fontsize)
        if cbar_tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
        fig.tight_layout(rect=[0, 0, 0.92, 1])
    else:
        fig.tight_layout()

    # if title_prefix:
    #     fig.suptitle(title_prefix, y=1.02)

    # fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes

def plot_x_safety_map_cov_panels(
    res,
    covariances,
    labels,
    *,
    mean=(0.0, 0.0),
    reference_cov=None,
    show_reference=True,
    reference_color="gray",
    reference_alpha=0.8,
    reference_linestyle="--",
    reference_linewidth=1.0,
    title_prefix="",
    cmap="RdYlGn",
    norm=None,
    contour_levels=5,
    contour_mass=True,
    contour_mass_max=0.8,
    contour_color="black",
    contour_alpha=0.7,
    cmap_alpha=1.0,
    show_mean_marker=True,
    mean_marker_color="black",
    mean_marker_size=40,
    figsize=None,
    colorbar=True,
    show_distribution_legend=True,
    legend_fontsize=None,
    reference_label="Baseline",
    shifted_label="Shifted",
    show_contour_labels=False,
    contour_label_fontsize=None,
    contour_label_fmt=".2f",
    title_fontsize=None,
    label_fontsize=None,
    tick_fontsize=None,
    cbar_label_fontsize=None,
    cbar_tick_fontsize=None,
    save_path=None,
):
    """
    Plot one row of X-safety-map panels, each with the same background safety map
    and a different Gaussian covariance overlay.

    Parameters
    ----------
    res : dict
        Output from compute_x_safety_map.
    covariances : list of array-like, each shape (2, 2)
        One covariance matrix per panel.
    labels : list[str]
        One title label per panel.
    mean : tuple[float, float] or array-like shape (2,)
        Common Gaussian mean for all panels.
    contour_mass : bool
        If True, contour_levels are probability-mass levels up to contour_mass_max.
        If False, uses standard matplotlib contour level behavior on the pdf.
    """
    if len(covariances) != len(labels):
        raise ValueError("covariances and labels must have the same length.")
    if cbar_label_fontsize is None:
        cbar_label_fontsize = label_fontsize
    if cbar_tick_fontsize is None:
        cbar_tick_fontsize = tick_fontsize
    if contour_label_fontsize is None:
        contour_label_fontsize = tick_fontsize
    if legend_fontsize is None:
        legend_fontsize = tick_fontsize

    mean = np.asarray(mean, dtype=float).reshape(2)
    reference_cov = None if reference_cov is None else np.asarray(reference_cov, dtype=float).reshape(2, 2)

    a_vals = res["A"]
    w_vals = res["omega"]
    safe_grid = res["safe"]
    A_grid, W_grid = np.meshgrid(a_vals, w_vals, indexing="xy")

    n = len(covariances)
    if figsize is None:
        figsize = (4.5 * n, 4.2)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    mesh = None

    def _format_contour_label(value):
        return format(float(value), contour_label_fmt)

    def _get_mass_levels():
        if np.isscalar(contour_levels):
            count = int(contour_levels)
            if count < 1:
                raise ValueError("contour_levels must be >= 1.")
            if not np.isclose(contour_levels, count):
                raise ValueError("contour_levels must be an integer count or an explicit list of masses.")
            if not (0.0 < contour_mass_max < 1.0):
                raise ValueError("contour_mass_max must be in (0, 1).")
            return np.linspace(contour_mass_max / count, contour_mass_max, count)

        mass_levels = np.asarray(contour_levels, dtype=float).reshape(-1)
        if mass_levels.size == 0:
            raise ValueError("contour_levels must not be empty.")
        if np.any((mass_levels <= 0.0) | (mass_levels >= 1.0)):
            raise ValueError("When contour_mass=True, explicit contour_levels must lie in (0, 1).")
        return np.sort(mass_levels)

    mass_levels = _get_mass_levels() if contour_mass else None

    for ax, sigma, label in zip(axes, covariances, labels):
        sigma = np.asarray(sigma, dtype=float).reshape(2, 2)

        mesh_kwargs = dict(cmap=cmap, shading="auto", alpha=cmap_alpha)
        if norm is not None:
            mesh_kwargs["norm"] = norm
        else:
            mesh_kwargs["vmin"] = 0.0
            mesh_kwargs["vmax"] = 1.0

        mesh = ax.pcolormesh(a_vals, w_vals, safe_grid, **mesh_kwargs)

        grid = np.stack([A_grid.ravel(), W_grid.ravel()], axis=1)
        diff = grid - mean.reshape(1, 2)

        def _draw_cov_contour(cov_mat, *, color, alpha, linestyle, linewidth, label_contours):
            inv = np.linalg.inv(cov_mat)
            det = np.linalg.det(cov_mat)
            if det <= 0:
                raise ValueError("Each covariance determinant must be > 0.")
            quad = np.einsum("bi,ij,bj->b", diff, inv, diff)
            Z = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))
            Z = Z.reshape(A_grid.shape)

            if contour_mass:
                density_peak = 1.0 / (2 * np.pi * np.sqrt(det))
                density_levels = density_peak * np.exp(-0.5 * chi2.ppf(mass_levels, df=2))
                cs = ax.contour(
                    A_grid,
                    W_grid,
                    Z,
                    levels=np.sort(density_levels),
                    colors=color,
                    linewidths=linewidth,
                    alpha=alpha,
                    linestyles=linestyle,
                )
                if label_contours:
                    contour_label_map = {
                        level: _format_contour_label(mass)
                        for level, mass in zip(cs.levels, mass_levels[::-1])
                    }
                    ax.clabel(
                        cs,
                        cs.levels,
                        fmt=contour_label_map,
                        inline=True,
                        fontsize=contour_label_fontsize,
                    )
            else:
                cs = ax.contour(
                    A_grid,
                    W_grid,
                    Z,
                    levels=contour_levels,
                    colors=color,
                    linewidths=linewidth,
                    alpha=alpha,
                    linestyles=linestyle,
                )
                if label_contours:
                    ax.clabel(
                        cs,
                        cs.levels,
                        fmt=lambda level: _format_contour_label(level),
                        inline=True,
                        fontsize=contour_label_fontsize,
                    )

        if show_reference and reference_cov is not None:
            _draw_cov_contour(
                reference_cov,
                color=reference_color,
                alpha=reference_alpha,
                linestyle=reference_linestyle,
                linewidth=reference_linewidth,
                label_contours=False,
            )

        _draw_cov_contour(
            sigma,
            color=contour_color,
            alpha=contour_alpha,
            linestyle="-",
            linewidth=1.0,
            label_contours=show_contour_labels,
        )

        if show_mean_marker:
            ax.scatter(
                [mean[0]],
                [mean[1]],
                color=mean_marker_color,
                s=mean_marker_size,
                zorder=5,
            )

        if show_distribution_legend:
            from matplotlib.lines import Line2D

            handles = []
            if show_reference and reference_cov is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=reference_color,
                        lw=reference_linewidth,
                        alpha=reference_alpha,
                        linestyle=reference_linestyle,
                        label=reference_label,
                    )
                )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=contour_color,
                    lw=1.5,
                    alpha=contour_alpha,
                    linestyle="-",
                    label=shifted_label,
                )
            )
            ax.legend(handles=handles, loc="lower right", fontsize=legend_fontsize)

        ax.set_title(label, fontsize=title_fontsize)
        ax.set_xlabel("A", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(False)

    axes[0].set_ylabel(r"$\omega$", fontsize=label_fontsize)

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02, fontsize=title_fontsize)

    if colorbar:
        subplot_right = 0.90
        cbar_left = 0.92
        fig.subplots_adjust(right=subplot_right)
        cbar_ax = fig.add_axes([cbar_left, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(mesh, cax=cbar_ax, label="Stable (1) / Unstable (0)", ticks=[0, 1])
        if cbar_label_fontsize is not None:
            cbar.set_label("Stable (1) / Unstable (0)", fontsize=cbar_label_fontsize)
        if cbar_tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
        fig.tight_layout(rect=[0, 0, subplot_right, 1])
    else:
        fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes


def plot_x_safety_map_cov_panels_no_bar(
    res,
    covariances,
    labels,
    *,
    mean=(0.0, 0.0),
    reference_cov=None,
    show_reference=True,
    reference_color="gray",
    reference_alpha=0.8,
    reference_linestyle="--",
    reference_linewidth=1.0,
    title_prefix="",
    cmap="RdYlGn",
    norm=None,
    contour_levels=5,
    contour_mass=True,
    contour_mass_max=0.8,
    contour_color="black",
    contour_alpha=0.7,
    show_mean_marker=True,
    mean_marker_color="black",
    mean_marker_size=40,
    figsize=None,
    save_path=None,
):
    """
    Plot one row of X-safety-map panels, each with the same background safety map
    and a different Gaussian covariance overlay.

    Parameters
    ----------
    res : dict
        Output from compute_x_safety_map.
    covariances : list of array-like, each shape (2, 2)
        One covariance matrix per panel.
    labels : list[str]
        One title label per panel.
    mean : tuple[float, float] or array-like shape (2,)
        Common Gaussian mean for all panels.
    contour_mass : bool
        If True, contour_levels are probability-mass levels up to contour_mass_max.
        If False, uses standard matplotlib contour level behavior on the pdf.
    """
    if len(covariances) != len(labels):
        raise ValueError("covariances and labels must have the same length.")

    mean = np.asarray(mean, dtype=float).reshape(2)
    reference_cov = None if reference_cov is None else np.asarray(reference_cov, dtype=float).reshape(2, 2)

    a_vals = res["A"]
    w_vals = res["omega"]
    safe_grid = res["safe"]
    A_grid, W_grid = np.meshgrid(a_vals, w_vals, indexing="xy")

    n = len(covariances)
    if figsize is None:
        figsize = (4.5 * n, 4.2)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    mesh = None

    for ax, sigma, label in zip(axes, covariances, labels):
        sigma = np.asarray(sigma, dtype=float).reshape(2, 2)

        mesh_kwargs = dict(cmap=cmap, shading="auto")
        if norm is not None:
            mesh_kwargs["norm"] = norm
        else:
            mesh_kwargs["vmin"] = 0.0
            mesh_kwargs["vmax"] = 1.0

        mesh = ax.pcolormesh(a_vals, w_vals, safe_grid, **mesh_kwargs)

        grid = np.stack([A_grid.ravel(), W_grid.ravel()], axis=1)
        diff = grid - mean.reshape(1, 2)

        if contour_mass:
            if not (0.0 < contour_mass_max < 1.0):
                raise ValueError("contour_mass_max must be in (0, 1).")
            if contour_levels < 1:
                raise ValueError("contour_levels must be >= 1.")
            mass_levels = np.linspace(
                contour_mass_max / contour_levels,
                contour_mass_max,
                contour_levels,
            )
        else:
            mass_levels = None

        def _draw_cov_contour(cov_mat, *, color, alpha, linestyle, linewidth):
            inv = np.linalg.inv(cov_mat)
            det = np.linalg.det(cov_mat)
            if det <= 0:
                raise ValueError("Each covariance determinant must be > 0.")
            quad = np.einsum("bi,ij,bj->b", diff, inv, diff)
            Z = np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))
            Z = Z.reshape(A_grid.shape)

            if contour_mass:
                density_peak = 1.0 / (2 * np.pi * np.sqrt(det))
                density_levels = density_peak * np.exp(-0.5 * chi2.ppf(mass_levels, df=2))
                ax.contour(
                    A_grid,
                    W_grid,
                    Z,
                    levels=np.sort(density_levels),
                    colors=color,
                    linewidths=linewidth,
                    alpha=alpha,
                    linestyles=linestyle,
                )
            else:
                ax.contour(
                    A_grid,
                    W_grid,
                    Z,
                    levels=contour_levels,
                    colors=color,
                    linewidths=linewidth,
                    alpha=alpha,
                    linestyles=linestyle,
                )

        if show_reference and reference_cov is not None:
            _draw_cov_contour(
                reference_cov,
                color=reference_color,
                alpha=reference_alpha,
                linestyle=reference_linestyle,
                linewidth=reference_linewidth,
            )

        _draw_cov_contour(
            sigma,
            color=contour_color,
            alpha=contour_alpha,
            linestyle="-",
            linewidth=1.0,
        )

        if show_mean_marker:
            ax.scatter(
                [mean[0]],
                [mean[1]],
                color=mean_marker_color,
                s=mean_marker_size,
                zorder=5,
            )

        ax.set_title(label)
        ax.set_xlabel("Feature 1")
        ax.grid(False)

    axes[0].set_ylabel("Feature 2")

    # subplot_right = 0.90
    # cbar_left = 0.92
    # fig.subplots_adjust(right=subplot_right)
    # cbar_ax = fig.add_axes([cbar_left, 0.15, 0.02, 0.70])
    # fig.colorbar(mesh, cax=cbar_ax, label="Stable (1) / Unstable (0)", ticks=[0, 1])

    # if title_prefix:
    #     fig.suptitle(title_prefix, y=1.02)

    # fig.tight_layout(rect=[0, 0, subplot_right, 1])

    # if save_path:
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes



def annotate_covariance_change(
    ax,
    *,
    mean=(0.0, 0.0),
    mode="scale_down",
    color="black",
    scale=0.35,
    origin_pad=0.0,
    linewidth=2.0,
    alpha=0.9,
    zorder=6,
):
    """
    Add symbolic arrows that describe a covariance change around a mean location.

    Supported modes:
    - "scale_down": inward arrows toward the mean
    - "scale_up": outward arrows away from the mean
    - "correlation": major-axis outward arrows with minor-axis inward arrows
    """
    x0, y0 = np.asarray(mean, dtype=float).reshape(2)

    def _draw_arrow(x_start, y_start, dx, dy):
        ax.arrow(
            x_start,
            y_start,
            dx,
            dy,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            head_width=0.2,
            head_length=0.15,
            length_includes_head=True,
            zorder=zorder,
        )

    def _draw_outward(direction, length):
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        start = np.array([x0, y0]) + origin_pad * direction
        delta = length * direction
        _draw_arrow(start[0], start[1], delta[0], delta[1])

    def _draw_inward(direction, length):
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        end = np.array([x0, y0]) + origin_pad * direction
        start = end + length * direction
        delta = end - start
        _draw_arrow(start[0], start[1], delta[0], delta[1])

    if mode == "scale_down":
        _draw_arrow(x0 + scale, y0, -0.55 * scale, 0.0)
        _draw_arrow(x0 - scale, y0, 0.55 * scale, 0.0)
        _draw_arrow(x0, y0 + scale, 0.0, -0.55 * scale)
        _draw_arrow(x0, y0 - scale, 0.0, 0.55 * scale)
    elif mode == "scale_up":
        _draw_outward((1.0, 0.0), scale)
        _draw_outward((-1.0, 0.0), scale)
        _draw_outward((0.0, 1.0), scale)
        _draw_outward((0.0, -1.0), scale)
    elif mode == "correlation":
        major_scale = scale
        minor_scale = 0.5 * scale
        _draw_outward((1.0, 1.0), major_scale)
        _draw_outward((-1.0, -1.0), major_scale)
        _draw_inward((1.0, -1.0), minor_scale)
        _draw_inward((-1.0, 1.0), minor_scale)
    else:
        raise ValueError("mode must be 'scale_down', 'scale_up', or 'correlation'")

def plot_cov_safety_map(
    res: dict,
    *,
    mode: str = "indicator",   # "indicator" or "prob"
    title: str = r"Safety map over $(\delta_1,\delta_2)$",
    cmap="Greys_r",
    norm=None,
    eta=0.1,
    show_cell_edges: bool = True,
    edgecolor: str = "k",
    linewidth: float = 0.2,
    draw_boundary: bool = True,
    boundary_color: str = "red",
    boundary_lw: float = 2.0,
    boundary_alpha: float = 0.9,
    colorbar: bool = True,
    vmin=None,
    vmax=None,
    save_path=None,
):

    delta1_vals = res["delta1"]
    delta2_vals = res["delta2"]

    if mode == "prob":
        Z = res.get("p_safe_grid", res.get("safe"))
        cbar_label = "Stability Probability"
        threshold_value = 1.0 - float(eta)
    elif mode == "indicator":
        Z = res.get("config_safe_grid")
        if Z is None:
            safe_grid = res.get("safe")
            thr = res.get("threshold", 1.0)
            if safe_grid is None:
                raise ValueError("Need 'safe' grid to derive indicator map.")
            Z = (np.asarray(safe_grid) >= thr).astype(float)
        cbar_label ="Safety Indicator" #r"$\mathbb{1}\{\hat p_{\mathrm{safe}}(\mu)\geq 1-\eta\}$"
    else:
        raise ValueError("mode must be 'indicator' or 'prob'")

    Z = np.asarray(Z)
    delta1_vals = np.asarray(delta1_vals)
    delta2_vals = np.asarray(delta2_vals)

    if Z.shape == (delta1_vals.size, delta2_vals.size):
        Z_plot = Z.T
    elif Z.shape == (delta2_vals.size, delta1_vals.size):
        Z_plot = Z
    else:
        raise ValueError(
            f"Shape mismatch: Z{Z.shape} vs grid ({delta1_vals.size}, {delta2_vals.size})"
        )

    def _bin_edges(vals):
        vals = np.asarray(vals, float)
        mid = 0.5 * (vals[1:] + vals[:-1])
        edges = np.empty(vals.size + 1, dtype=float)
        edges[1:-1] = mid
        edges[0] = vals[0] - (mid[0] - vals[0])
        edges[-1] = vals[-1] + (vals[-1] - mid[-1])
        return edges

    delta1_edges = _bin_edges(delta1_vals)
    delta2_edges = _bin_edges(delta2_vals)

    fig, ax = plt.subplots()

    mesh_kwargs = dict(cmap=cmap, shading="auto")
    if norm is not None:
        mesh_kwargs["norm"] = norm
    else:
        mesh_kwargs["vmin"] = 0.0 if vmin is None else vmin
        mesh_kwargs["vmax"] = 1.0 if vmax is None else vmax

    if show_cell_edges:
        mesh_kwargs.update(edgecolors=edgecolor, linewidth=linewidth)

    mesh = ax.pcolormesh(delta1_edges, delta2_edges, Z_plot, **mesh_kwargs)

    if colorbar:
        if mode == "prob":
            # cbar_ticks = [0.0, threshold_value, 1.0]
            # cbar_ticklabels = ["0", f"{threshold_value:.2f}", "1"]
            # fig.colorbar(mesh, ax=ax, label=cbar_label, ticks=cbar_ticks).ax.set_yticklabels(cbar_ticklabels)
            fig.colorbar(mesh, ax=ax, label=cbar_label)
        # fig.colorbar(mesh, ax=ax, label=cbar_label)
        else:
            fig.colorbar(mesh, ax=ax, label="safe (1) / unsafe (0)", ticks=[0, 1])
    if draw_boundary:
        if mode == "indicator":
            Zc = res.get("config_safe_grid", res.get("config_safe"))
            if Zc is not None:
                Zc = np.asarray(Zc)
                if Zc.shape == (delta1_vals.size, delta2_vals.size):
                    Zc = Zc.T
                ax.contour(
                    delta1_vals,
                    delta2_vals,
                    Zc,
                    levels=[0.5],
                    colors=boundary_color,
                    linestyles='--',
                    linewidths=boundary_lw,
                    alpha=boundary_alpha,
                )
        else:
            cs = ax.contour(
                delta1_vals,
                delta2_vals,
                Z_plot,
                levels=[threshold_value],
                colors=boundary_color,
                linestyles='--',
                linewidths=boundary_lw,
                alpha=boundary_alpha,
            )
            handles, _ = cs.legend_elements()
            if handles:
                handles[0].set_label("Stability Boundary")
                ax.legend(handles=handles, loc="best")


    ax.set_title(title)
    ax.set_xlabel(r"$\delta_1$")
    ax.set_ylabel(r"$\delta_2$")
    ax.grid(False)
    from matplotlib.lines import Line2D

    boundary_handle = Line2D(
        [0], [0],
        color=boundary_color,
        lw=boundary_lw,
        alpha=boundary_alpha,
        linestyle="--" if mode == "prob" else "-",
        label="Stability Boundary",
    )
    ax.legend(handles=[boundary_handle], loc="lower right")
    ax.grid(False)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax

__all__ = [
    "plot_x_safety_map",
    "plot_mu_safety_map",
    "annotate_mu_shift_directions",
    "annotate_x_shift_directions",
    "annotate_covariance_change",
    "plot_x_safety_map_shift_panels",
    "plot_x_safety_map_cov_panels",
]
