"""
Plotting helpers for Cartesian pendulum safety maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import chi2, gaussian_kde

from data.cartesian_pendulum import DistributionConfig, sample_X_shake_pulse


@dataclass(frozen=True)
class KDEContourStyle:
    levels: int = 5
    mass: bool = False
    mass_max: float = 0.9
    color: str = "black"
    alpha: float = 0.7
    linewidth: float = 1.0


@dataclass(frozen=True)
class ArrowStyle:
    scale: float = 1.5
    scale_each: tuple[float, ...] | None = None
    linewidth: float = 2.0
    head_width: float = 0.06
    head_width_each: tuple[float, ...] | None = None
    head_length: float = 0.08
    head_length_each: tuple[float, ...] | None = None
    point_size: float = 40.0


@dataclass(frozen=True)
class TextStyle:
    show: bool = True
    horizontal_position: str = "bottom"
    vertical_position: str = "left"
    pad: float = 0.2
    offset_each: tuple[tuple[float, float], ...] | None = None
    side_each: tuple[str, ...] | None = None
    fontsize: int = 10
    bbox_alpha: float = 0.8


def _bin_edges(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, float)
    if vals.size < 2:
        raise ValueError("Need at least 2 grid points to form bin edges.")
    mid = 0.5 * (vals[1:] + vals[:-1])
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = mid
    edges[0] = vals[0] - (mid[0] - vals[0])
    edges[-1] = vals[-1] + (vals[-1] - mid[-1])
    return edges


def _resolve_box_limits(res: dict, A_vals: np.ndarray, omega_vals: np.ndarray):
    box_limits = res.get("box_limits")
    if box_limits is not None:
        return box_limits
    return ((float(np.min(A_vals)), float(np.max(A_vals))), (float(np.min(omega_vals)), float(np.max(omega_vals))))


def _build_reflected_samples(samples: np.ndarray, box_limits):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape (N, 2)")

    (x_lo, x_hi), (y_lo, y_hi) = box_limits
    x = samples[:, 0]
    y = samples[:, 1]

    return np.column_stack(
        [
            np.concatenate(
                [
                    x,
                    2.0 * x_lo - x,
                    2.0 * x_hi - x,
                    x,
                    x,
                    2.0 * x_lo - x,
                    2.0 * x_lo - x,
                    2.0 * x_hi - x,
                    2.0 * x_hi - x,
                ]
            ),
            np.concatenate(
                [
                    y,
                    y,
                    y,
                    2.0 * y_lo - y,
                    2.0 * y_hi - y,
                    2.0 * y_lo - y,
                    2.0 * y_hi - y,
                    2.0 * y_lo - y,
                    2.0 * y_hi - y,
                ]
            ),
        ]
    )


def _evaluate_reflected_kde(
    samples: np.ndarray,
    box_limits,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
):
    reflected = _build_reflected_samples(samples, box_limits)
    kde = gaussian_kde(reflected.T)
    Xg, Yg = np.meshgrid(x_vals, y_vals, indexing="xy")
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    Z = kde(grid.T).reshape(Xg.shape)
    return Xg, Yg, Z


def _sample_joint_distribution(
    A_config: DistributionConfig,
    omega_config: DistributionConfig,
    correlation: float,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    return sample_X_shake_pulse(
        n_samples=n_samples,
        A_config=A_config,
        omega_config=omega_config,
        sigma_range=1.0,
        correlation=correlation,
        t0=None,
        q=(1.0, 0.0, 0.0),
        a0=(0.0, 0.0, 0.0),
        seed=seed,
    )[:, :2]


def _plot_reflected_kde_contours(
    ax,
    samples: np.ndarray,
    box_limits,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    contour_levels: int,
    contour_mass: bool,
    contour_mass_max: float,
    contour_color: str,
    contour_alpha: float,
    contour_linewidth: float,
):
    Xg, Yg, Z = _evaluate_reflected_kde(samples, box_limits, x_vals, y_vals)

    if contour_mass:
        if not (0.0 < contour_mass_max < 1.0):
            raise ValueError("contour_mass_max must be in (0, 1)")
        if contour_levels < 1:
            raise ValueError("contour_levels must be >= 1")

        cell_area = (x_vals[1] - x_vals[0]) * (y_vals[1] - y_vals[0])
        z_flat = Z.ravel()
        order = np.argsort(z_flat)[::-1]
        z_sorted = z_flat[order]
        mass = np.cumsum(z_sorted * cell_area)
        mass = mass / mass[-1]
        target_masses = np.linspace(contour_mass_max / contour_levels, contour_mass_max, contour_levels)
        density_levels = [z_sorted[np.searchsorted(mass, m, side="left")] for m in target_masses]
        ax.contour(
            Xg,
            Yg,
            Z,
            levels=np.sort(np.unique(density_levels)),
            colors=contour_color,
            linewidths=contour_linewidth,
            alpha=contour_alpha,
        )
    else:
        ax.contour(
            Xg,
            Yg,
            Z,
            levels=contour_levels,
            colors=contour_color,
            linewidths=contour_linewidth,
            alpha=contour_alpha,
        )


def plot_cartesian_mu_safety_map(
    res: dict,
    *,
    mode: str = "indicator",
    title: str = "Cartesian Pendulum Safety Map",
    cmap: str = "RdYlGn",
    norm=None,
    eta: float | None = None,
    show_cell_edges: bool = True,
    edgecolor: str = "k",
    linewidth: float = 0.2,
    draw_boundary: bool = True,
    boundary_color: str = "k",
    boundary_lw: float = 1.0,
    boundary_alpha: float = 0.9,
    colorbar: bool = True,
    save_path: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Plot a precomputed Cartesian safety map over ``(A, omega)``.
    """

    A_vals = np.asarray(res.get("A", res.get("mu1")), dtype=float)
    omega_vals = np.asarray(res.get("omega", res.get("mu2")), dtype=float)
    if A_vals.ndim != 1 or omega_vals.ndim != 1:
        raise ValueError("res must contain 1D A and omega grids")

    eta_value = float(res.get("eta", 0.1) if eta is None else eta)
    threshold_value = float(res.get("threshold", 1.0 - eta_value))

    if mode == "prob":
        Z = np.asarray(res.get("p_safe_grid", res.get("safe")), dtype=float)
        cbar_label = "Stability Probability"
    elif mode == "indicator":
        Z = res.get("config_safe_grid")
        if Z is None:
            safe_grid = res.get("p_safe_grid", res.get("safe"))
            if safe_grid is None:
                raise ValueError("Need 'p_safe_grid' or 'safe' to derive indicator map.")
            Z = (np.asarray(safe_grid, dtype=float) >= threshold_value).astype(float)
        else:
            Z = np.asarray(Z, dtype=float)
        cbar_label = "Safety Indicator"
    else:
        raise ValueError("mode must be 'indicator' or 'prob'")

    if Z.shape == (A_vals.size, omega_vals.size):
        Z_plot = Z.T
    elif Z.shape == (omega_vals.size, A_vals.size):
        Z_plot = Z
    else:
        raise ValueError(f"Shape mismatch: Z{Z.shape} vs grid ({A_vals.size}, {omega_vals.size})")

    A_edges = _bin_edges(A_vals)
    omega_edges = _bin_edges(omega_vals)

    fig, ax = plt.subplots()
    mesh_kwargs = dict(cmap=cmap, shading="auto")
    if norm is not None:
        mesh_kwargs["norm"] = norm
    else:
        mesh_kwargs["vmin"] = vmin
        mesh_kwargs["vmax"] = vmax
    if show_cell_edges:
        mesh_kwargs.update(edgecolors=edgecolor, linewidth=linewidth)

    mesh = ax.pcolormesh(A_edges, omega_edges, Z_plot, **mesh_kwargs)
    if colorbar:
        fig.colorbar(mesh, ax=ax, label=cbar_label)

    if draw_boundary:
        boundary_source = (
            np.asarray(res.get("config_safe_grid"), dtype=float)
            if res.get("config_safe_grid") is not None
            else None
        )
        if mode == "indicator":
            if boundary_source is None:
                boundary_source = (np.asarray(res.get("p_safe_grid", res.get("safe"))) >= threshold_value).astype(float)
            if boundary_source.shape == (A_vals.size, omega_vals.size):
                boundary_plot = boundary_source.T
            else:
                boundary_plot = boundary_source
            ax.contour(
                A_vals,
                omega_vals,
                boundary_plot,
                levels=[0.5],
                colors=boundary_color,
                linewidths=boundary_lw,
                alpha=boundary_alpha,
            )
        else:
            prob_source = np.asarray(res.get("p_safe_grid", res.get("safe")), dtype=float)
            if prob_source.shape == (A_vals.size, omega_vals.size):
                prob_plot = prob_source.T
            else:
                prob_plot = prob_source
            ax.contour(
                A_vals,
                omega_vals,
                prob_plot,
                levels=[threshold_value],
                colors=boundary_color,
                linewidths=boundary_lw,
                alpha=boundary_alpha,
                linestyles="--",
            )
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=boundary_color,
                    lw=boundary_lw,
                    alpha=boundary_alpha,
                    linestyle="--" if mode == "prob" else "-",
                    label="Stability Boundary",
                )
            ],
            loc="lower right",
        )

    ax.set_title(title)
    ax.set_xlabel("A")
    ax.set_ylabel("omega")
    ax.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_cartesian_x_safety_map(
    res: dict,
    title: str = "Cartesian safety map in X box",
    cmap: str = "RdYlGn",
    norm=None,
    A_config: Optional[DistributionConfig] = None,
    omega_config: Optional[DistributionConfig] = None,
    correlation: float = 0.0,
    contour_n_samples: int = 4000,
    contour_seed: int = 0,
    samples: Optional[np.ndarray] = None,
    contour_levels: int = 6,
    contour_mass: bool = False,
    contour_mass_max: float = 0.9,
    contour_color: str = "k",
    contour_alpha: float = 0.8,
    show_contours: bool = True,
    eta: float = 0.1,
    print_msg: bool = True,
    save_path: Optional[str] = None,
):
    """
    Cartesian analogue of ``plot_x_safety_map`` over the ``(A, omega)`` box.

    If ``A_config`` and ``omega_config`` are provided, the function samples the
    implied ``(A, omega)`` distribution and can overlay a reflected KDE
    contour inside the ``box_limits`` stored in ``res``.

    ``samples`` is kept as a fallback for backward compatibility.
    """

    A_vals = np.asarray(res["A"], dtype=float)
    omega_vals = np.asarray(res["omega"], dtype=float)
    safe_grid = np.asarray(res.get("safe", res.get("p_safe_grid")), dtype=float)

    if safe_grid.shape == (A_vals.size, omega_vals.size):
        safe_grid_plot = safe_grid.T
    elif safe_grid.shape == (omega_vals.size, A_vals.size):
        safe_grid_plot = safe_grid
        safe_grid = safe_grid_plot.T
    else:
        raise ValueError(
            f"Shape mismatch: safe{safe_grid.shape} vs grid ({A_vals.size}, {omega_vals.size})"
        )

    A_grid, W_grid = np.meshgrid(A_vals, omega_vals, indexing="xy")

    fig, ax = plt.subplots()
    mesh_kwargs = dict(cmap=cmap, shading="auto")
    if norm is not None:
        mesh_kwargs["norm"] = norm
    else:
        mesh_kwargs["vmin"] = 0.0
        mesh_kwargs["vmax"] = 1.0

    mesh = ax.pcolormesh(A_vals, omega_vals, safe_grid_plot, **mesh_kwargs)
    fig.colorbar(mesh, ax=ax, label="safe (1) / unsafe (0)")

    contour_samples = samples
    if A_config is not None and omega_config is not None:
        contour_samples = sample_X_shake_pulse(
            n_samples=contour_n_samples,
            A_config=A_config,
            omega_config=omega_config,
            sigma_range=1.0,
            correlation=correlation,
            t0=None,
            q=(1.0, 0.0, 0.0),
            a0=(0.0, 0.0, 0.0),
            seed=contour_seed,
        )[:, :2]

    if contour_samples is not None:
        box_limits = _resolve_box_limits(res, A_vals, omega_vals)
        if show_contours:
            _plot_reflected_kde_contours(
                ax=ax,
                samples=contour_samples,
                box_limits=box_limits,
                x_vals=A_vals,
                y_vals=omega_vals,
                contour_levels=contour_levels,
                contour_mass=contour_mass,
                contour_mass_max=contour_mass_max,
                contour_color=contour_color,
                contour_alpha=contour_alpha,
                contour_linewidth=1.0,
            )
        _, _, Z = _evaluate_reflected_kde(contour_samples, box_limits, A_vals, omega_vals)
        if not (0.0 <= eta < 1.0):
            raise ValueError("eta must be in [0, 1)")
        p_safe = float(np.sum(safe_grid_plot * Z) / np.sum(Z))
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


def plot_cartesian_x_safety_map_shift_panels(
    res: dict,
    labels,
    *,
    title_prefix: str = "",
    cmap: str = "RdYlGn",
    norm=None,
    A_config: Optional[DistributionConfig] = None,
    omega_config: Optional[DistributionConfig] = None,
    correlation: float = 0.0,
    panel_A_configs: Optional[list[DistributionConfig] | tuple[DistributionConfig, ...]] = None,
    panel_omega_configs: Optional[list[DistributionConfig] | tuple[DistributionConfig, ...]] = None,
    panel_correlations: Optional[list[float] | tuple[float, ...]] = None,
    contour_n_samples: int = 4000,
    contour_seed: int = 0,
    samples: Optional[np.ndarray] = None,
    show_contours: bool = True,
    contour_style: KDEContourStyle | None = None,
    arrow_style: ArrowStyle | None = None,
    text_style: TextStyle | None = None,
    colors=None,
    figsize=None,
    save_path: Optional[str] = None,
):
    """
    Plot shift panels over a Cartesian ``(A, omega)`` safety map.

    Each panel samples its own shifted distribution from ``panel_A_configs`` and
    ``panel_omega_configs``. The arrow is derived automatically from the baseline
    distribution mean to the shifted distribution mean.
    """

    if colors is None:
        colors = ["black", "#f39c12", "#1f77b4"]
    if contour_style is None:
        contour_style = KDEContourStyle()
    if arrow_style is None:
        arrow_style = ArrowStyle()
    if text_style is None:
        text_style = TextStyle()

    if panel_A_configs is None or panel_omega_configs is None:
        raise ValueError("panel_A_configs and panel_omega_configs are required.")
    n = len(labels)
    if len(panel_A_configs) != n or len(panel_omega_configs) != n:
        raise ValueError("panel_A_configs, panel_omega_configs, and labels must have the same length.")
    if panel_correlations is None:
        panel_correlations = [correlation] * n
    elif len(panel_correlations) != n:
        raise ValueError("panel_correlations must have the same length as labels.")
    if A_config is None or omega_config is None:
        raise ValueError("Baseline A_config and omega_config are required.")

    if arrow_style.scale_each is None:
        arrow_scale_each = [arrow_style.scale] * n
    else:
        arrow_scale_each = list(arrow_style.scale_each)
        if len(arrow_scale_each) != n:
            raise ValueError("arrow_style.scale_each must have the same length as labels.")

    if arrow_style.head_width_each is None:
        head_width_each = [arrow_style.head_width] * n
    else:
        head_width_each = list(arrow_style.head_width_each)
        if len(head_width_each) != n:
            raise ValueError("arrow_style.head_width_each must have the same length as labels.")

    if arrow_style.head_length_each is None:
        head_length_each = [arrow_style.head_length] * n
    else:
        head_length_each = list(arrow_style.head_length_each)
        if len(head_length_each) != n:
            raise ValueError("arrow_style.head_length_each must have the same length as labels.")

    if text_style.offset_each is None:
        text_offset_each = [(text_style.pad, text_style.pad)] * n
    else:
        text_offset_each = list(text_style.offset_each)
        if len(text_offset_each) != n:
            raise ValueError("text_style.offset_each must have the same length as labels.")

    if text_style.side_each is None:
        text_side_each = ["auto"] * n
    else:
        text_side_each = list(text_style.side_each)
        if len(text_side_each) != n:
            raise ValueError("text_style.side_each must have the same length as labels.")

    seed_seq = np.random.SeedSequence(contour_seed)
    seed_children = seed_seq.spawn(n + 1)
    baseline_samples = (
        samples
        if samples is not None
        else _sample_joint_distribution(
            A_config=A_config,
            omega_config=omega_config,
            correlation=correlation,
            n_samples=contour_n_samples,
            seed=int(seed_children[0].generate_state(1)[0]),
        )
    )
    baseline_mean = np.mean(baseline_samples, axis=0)

    panel_samples = []
    panel_means = []
    for idx in range(n):
        samples_i = _sample_joint_distribution(
            A_config=panel_A_configs[idx],
            omega_config=panel_omega_configs[idx],
            correlation=float(panel_correlations[idx]),
            n_samples=contour_n_samples,
            seed=int(seed_children[idx + 1].generate_state(1)[0]),
        )
        panel_samples.append(samples_i)
        panel_means.append(np.mean(samples_i, axis=0))

    A_vals = np.asarray(res["A"], dtype=float)
    omega_vals = np.asarray(res["omega"], dtype=float)
    Z = res.get("config_safe_grid", res.get("safe"))
    if Z is None:
        raise ValueError("res must contain 'config_safe_grid' or 'safe'.")
    Z = np.asarray(Z, dtype=float)

    if Z.shape == (A_vals.size, omega_vals.size):
        safe_grid_plot = Z.T
    elif Z.shape == (omega_vals.size, A_vals.size):
        safe_grid_plot = Z
    else:
        raise ValueError(f"Shape mismatch: Z{Z.shape} vs grid ({A_vals.size}, {omega_vals.size})")

    A_grid, W_grid = np.meshgrid(A_vals, omega_vals, indexing="xy")

    if figsize is None:
        figsize = (4.5 * n, 4.2)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    mesh = None

    box_limits = _resolve_box_limits(res, A_vals, omega_vals)

    for idx, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        mesh_kwargs = dict(cmap=cmap, shading="auto")
        if norm is not None:
            mesh_kwargs["norm"] = norm
        else:
            mesh_kwargs["vmin"] = 0.0
            mesh_kwargs["vmax"] = 1.0

        mesh = ax.pcolormesh(A_vals, omega_vals, safe_grid_plot, **mesh_kwargs)

        if show_contours:
            _plot_reflected_kde_contours(
                ax=ax,
                samples=panel_samples[idx],
                box_limits=box_limits,
                x_vals=A_vals,
                y_vals=omega_vals,
                contour_levels=contour_style.levels,
                contour_mass=contour_style.mass,
                contour_mass_max=contour_style.mass_max,
                contour_color=contour_style.color,
                contour_alpha=contour_style.alpha,
                contour_linewidth=contour_style.linewidth,
            )

        x0, y0 = baseline_mean
        vx, vy = panel_means[idx] - baseline_mean
        dx = arrow_scale_each[idx] * vx
        dy = arrow_scale_each[idx] * vy

        ax.scatter([x0], [y0], color=color, s=arrow_style.point_size, zorder=5)
        ax.arrow(
            x0,
            y0,
            dx,
            dy,
            color=color,
            linewidth=arrow_style.linewidth,
            head_width=head_width_each[idx],
            head_length=head_length_each[idx],
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

        if text_style.show:
            offset_x, offset_y = text_offset_each[idx]
            side = text_side_each[idx]
            if side not in {"auto", "top", "bottom", "left", "right"}:
                raise ValueError("Each text_style.side_each entry must be one of 'auto', 'top', 'bottom', 'left', 'right'.")
            if side == "auto":
                side = "top" if abs(dx) >= abs(dy) else "right"

            if side == "top":
                ax.text(
                    x_mid,
                    y_mid + offset_y,
                    label,
                    color=color,
                    fontsize=text_style.fontsize,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=text_style.bbox_alpha, edgecolor="none"),
                    zorder=6,
                )
            elif side == "bottom":
                ax.text(
                    x_mid,
                    y_mid - offset_y,
                    label,
                    color=color,
                    fontsize=text_style.fontsize,
                    ha="center",
                    va="top",
                    bbox=dict(facecolor="white", alpha=text_style.bbox_alpha, edgecolor="none"),
                    zorder=6,
                )
            elif side == "left":
                ax.text(
                    x_mid - offset_x,
                    y_mid,
                    label,
                    color=color,
                    fontsize=text_style.fontsize,
                    ha="right",
                    va="center",
                    bbox=dict(facecolor="white", alpha=text_style.bbox_alpha, edgecolor="none"),
                    zorder=6,
                )
            else:
                ax.text(
                    x_mid + offset_x,
                    y_mid,
                    label,
                    color=color,
                    fontsize=text_style.fontsize,
                    ha="left",
                    va="center",
                    bbox=dict(facecolor="white", alpha=text_style.bbox_alpha, edgecolor="none"),
                    zorder=6,
                )

        _, _, Z_density = _evaluate_reflected_kde(panel_samples[idx], box_limits, A_vals, omega_vals)
        p_safe = float(np.sum(safe_grid_plot * Z_density) / np.sum(Z_density))
        ax.set_title(f"{label}\np_safe={p_safe:.3f}")
        ax.set_xlabel("A")
        ax.grid(False)

    axes[0].set_ylabel("omega")
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(mesh, cax=cbar_ax, label="Stable (1) / Unstable (0)", ticks=[0, 1])
    cbar.ax.set_yticklabels(["0", "1"])

    if title_prefix:
        fig.suptitle(title_prefix, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.92, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes


__all__ = [
    "ArrowStyle",
    "KDEContourStyle",
    "TextStyle",
    "plot_cartesian_mu_safety_map",
    "plot_cartesian_x_safety_map",
    "plot_cartesian_x_safety_map_shift_panels",
]
