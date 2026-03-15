"""
Cartesian pendulum helpers for pulse-driven pivot motion.

This module keeps the notebook-facing API small:

- ``CartesianPendulumParams`` for physical parameters
- ``sample_X_shake_pulse`` for context sampling
- ``pivot_shake_pulse_batch`` for the pulse pivot trajectory
- ``simulate_cartesian_pendulum_custom`` for RK4 simulation
- ``check_cartesian_safety`` for spring-style safety checks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass(frozen=True)
class CartesianPendulumParams:
    """Physical parameters for a rigid pendulum in Cartesian coordinates."""

    mass: float = 1.0
    length: float = 1.0
    gravity: float = 9.81

    def validate(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.gravity <= 0:
            raise ValueError("gravity must be positive")


def _validate_time_grid(t_final: float, dt: float) -> np.ndarray:
    if t_final <= 0:
        raise ValueError("t_final must be positive")
    if dt <= 0 or dt > t_final:
        raise ValueError("dt must satisfy 0 < dt <= t_final")

    n_steps = int(np.floor(t_final / dt))
    return np.linspace(0.0, n_steps * dt, n_steps + 1)


def _window_mask(t: np.ndarray, window: Tuple[float, float] | None) -> np.ndarray:
    if window is None:
        return np.ones_like(t, dtype=bool)

    t_l, t_u = float(window[0]), float(window[1])
    if t_l > t_u:
        raise ValueError(f"window must satisfy t_l <= t_u, got {window}")

    mask = (t >= t_l) & (t <= t_u)
    if not np.any(mask):
        raise ValueError(f"window {window} does not overlap with trajectory time range")
    return mask


def _as_2d_context(X: np.ndarray, n_features: int | None = None) -> tuple[np.ndarray, bool]:
    X = np.asarray(X, dtype=float)
    single = False

    if X.ndim == 1:
        if n_features is not None and X.shape[0] != n_features:
            raise ValueError(f"X must have shape ({n_features},) when 1D")
        X = X.reshape(1, -1)
        single = True
    elif X.ndim != 2:
        raise ValueError("X must be 1D or 2D")

    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X must have shape (N, {n_features})")

    return X, single


def _lambda_closed_form(
    params: CartesianPendulumParams,
    d: np.ndarray,
    rel_v: np.ndarray,
    a_ddot: np.ndarray,
) -> np.ndarray:
    ell2 = params.length**2
    g0 = np.array([0.0, 0.0, -params.gravity], dtype=float)
    term = np.einsum("ij,ij->i", d, g0 - a_ddot) + np.einsum("ij,ij->i", rel_v, rel_v)
    return -(params.mass / ell2) * term


def sample_X_shake_pulse(
    n_samples: int,
    A_distribution: str = "uniform",
    A_range: tuple[float, float] = (0.10, 0.35),
    A_loc: float = 1.0,
    A_scale: float = 0.3,
    bound_A: tuple[float, float] | None = None,
    omega_distribution: str = "uniform",
    omega_range: tuple[float, float] = (1.0, 3.0),
    omega_loc: float = 2.0,
    omega_scale: float = 0.5,
    bound_omega: tuple[float, float] | None = None,
    tau_range: tuple[float, float] = (0.45, 1.2),
    t0: float | None = None,
    q: tuple[float, float, float] = (1.0, 0.0, 0.0),
    a0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int | None = 21,
) -> np.ndarray:
    """Sample pulse contexts X with columns [A, omega, tau, t0, qx, qy, qz, a0x, a0y, a0z]."""

    def _validate_range(name: str, values: tuple[float, float]) -> tuple[float, float]:
        if len(values) != 2:
            raise ValueError(f"{name} must be a tuple/list of length 2")
        lo, hi = float(values[0]), float(values[1])
        if lo > hi:
            raise ValueError(f"{name} lower bound must be <= upper bound")
        return lo, hi

    def _sample_param(
        rng: np.random.Generator,
        name: str,
        distribution: str,
        uniform_range: tuple[float, float],
        loc: float,
        scale: float,
        bounds: tuple[float, float] | None,
    ) -> np.ndarray:
        dist = str(distribution).strip().lower()
        if dist == "uniform":
            lo, hi = _validate_range(f"{name}_range", uniform_range)
            samples = rng.uniform(lo, hi, size=n_samples)
        elif dist in ("gaussian", "normal"):
            if scale < 0:
                raise ValueError(f"{name}_scale must be >= 0")
            samples = rng.normal(loc=float(loc), scale=float(scale), size=n_samples)
        else:
            raise ValueError(
                f"{name}_distribution must be 'uniform' or 'gaussian', got '{distribution}'"
            )

        if bounds is not None:
            b_lo, b_hi = _validate_range(f"bound_{name}", bounds)
            samples = np.clip(samples, b_lo, b_hi)
        return samples

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = np.random.default_rng(seed)

    A = _sample_param(rng, "A", A_distribution, A_range, A_loc, A_scale, bound_A)
    omega = _sample_param(
        rng,
        "omega",
        omega_distribution,
        omega_range,
        omega_loc,
        omega_scale,
        bound_omega,
    )

    tau_lo, tau_hi = _validate_range("tau_range", tau_range)
    tau = rng.uniform(tau_lo, tau_hi, size=n_samples)
    t0_vals = np.full(n_samples, np.nan if t0 is None else float(t0), dtype=float)

    q_arr = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q_arr)
    if q_norm <= 1e-12:
        raise ValueError("q must be non-zero")
    q_arr = q_arr / q_norm

    a0_arr = np.asarray(a0, dtype=float)

    q_all = np.tile(q_arr[None, :], (n_samples, 1))
    a0_all = np.tile(a0_arr[None, :], (n_samples, 1))

    return np.column_stack([A, omega, tau, t0_vals, q_all, a0_all])


def pivot_shake_pulse_batch(t: float, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate pivot position, velocity, and acceleration for pulse contexts."""

    X, _ = _as_2d_context(X, n_features=10)

    A = X[:, 0]
    omega = X[:, 1]
    tau = np.maximum(X[:, 2], 1e-6)
    t0 = np.where(np.isnan(X[:, 3]), 5.0, X[:, 3])
    q = X[:, 4:7]
    a0 = X[:, 7:10]

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)

    s = t - t0
    env = np.exp(-(s**2) / (2.0 * tau**2))
    sin_term = np.sin(omega * s)
    cos_term = np.cos(omega * s)

    amp = A * env * sin_term
    amp_dot = A * env * ((-s / (tau**2)) * sin_term + omega * cos_term)
    amp_ddot = A * env * (
        (s**2 / (tau**4) - 1.0 / (tau**2) - omega**2) * sin_term
        - 2.0 * omega * s / (tau**2) * cos_term
    )

    a = a0 + amp[:, None] * q
    a_dot = amp_dot[:, None] * q
    a_ddot = amp_ddot[:, None] * q
    return a, a_dot, a_ddot


def simulate_cartesian_pendulum_custom(
    params: CartesianPendulumParams,
    t_final: float,
    dt: float,
    X: np.ndarray,
    pivot_eval: Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    n0: np.ndarray = np.array([0.15, 0.0, -0.9887]),
    rel_v0: np.ndarray = np.array([0.0, 0.8, 0.0]),
    project_each_step: bool = True,
) -> dict[str, np.ndarray]:
    """Simulate a Cartesian pendulum with a custom pivot motion callback."""

    params.validate()
    t = _validate_time_grid(t_final=t_final, dt=dt)
    X, single = _as_2d_context(X)

    n0 = np.asarray(n0, dtype=float)
    n0 = n0 / np.linalg.norm(n0)
    rel_v0 = np.asarray(rel_v0, dtype=float)

    n_traj = X.shape[0]
    r = np.zeros((n_traj, t.size, 3), dtype=float)
    v = np.zeros((n_traj, t.size, 3), dtype=float)
    a_hist = np.zeros((n_traj, t.size, 3), dtype=float)

    a_init, a_dot_init, _ = pivot_eval(t[0], X)
    a_hist[:, 0] = a_init
    r[:, 0] = a_init + params.length * n0[None, :]

    rel_v = np.broadcast_to(rel_v0, (n_traj, 3)).astype(float).copy()
    d0 = r[:, 0] - a_init
    rel_v -= ((np.einsum("ij,ij->i", d0, rel_v) / (params.length**2))[:, None]) * d0
    v[:, 0] = a_dot_init + rel_v

    g0 = np.array([0.0, 0.0, -params.gravity], dtype=float)

    def rhs(ti: float, ri: np.ndarray, vi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ai, ai_dot, ai_ddot = pivot_eval(ti, X)
        d = ri - ai
        rel_vi = vi - ai_dot
        lam = _lambda_closed_form(params, d=d, rel_v=rel_vi, a_ddot=ai_ddot)
        dr = vi
        dv = g0[None, :] + (lam[:, None] / params.mass) * d
        return dr, dv

    for i in range(t.size - 1):
        ti = float(t[i])
        ri = r[:, i]
        vi = v[:, i]

        k1_r, k1_v = rhs(ti, ri, vi)
        k2_r, k2_v = rhs(ti + 0.5 * dt, ri + 0.5 * dt * k1_r, vi + 0.5 * dt * k1_v)
        k3_r, k3_v = rhs(ti + 0.5 * dt, ri + 0.5 * dt * k2_r, vi + 0.5 * dt * k2_v)
        k4_r, k4_v = rhs(ti + dt, ri + dt * k3_r, vi + dt * k3_v)

        r_next = ri + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v_next = vi + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        a_next, a_dot_next, _ = pivot_eval(float(t[i + 1]), X)

        if project_each_step:
            d = r_next - a_next
            d_norm = np.linalg.norm(d, axis=1, keepdims=True)
            d = params.length * d / np.maximum(d_norm, 1e-12)
            r_next = a_next + d

            rel_v_next = v_next - a_dot_next
            rel_v_next -= (
                (np.einsum("ij,ij->i", d, rel_v_next) / (params.length**2))[:, None]
            ) * d
            v_next = a_dot_next + rel_v_next

        r[:, i + 1] = r_next
        v[:, i + 1] = v_next
        a_hist[:, i + 1] = a_next

    theta_deg = trajectory_theta_deg({"r": r, "a": a_hist})
    out = {"t": t, "r": r, "v": v, "a": a_hist, "X": X, "theta_deg": theta_deg}
    if single:
        return {k: (val[0] if k in {"r", "v", "a", "X", "theta_deg"} else val) for k, val in out.items()}
    return out


def trajectory_theta_deg(traj: dict[str, np.ndarray]) -> np.ndarray:
    """Return bob angle from vertical in degrees for each trajectory and time step."""

    r = np.asarray(traj.get("r"), dtype=float)
    a = np.asarray(traj.get("a"), dtype=float)
    if r.ndim not in (2, 3) or a.shape != r.shape or r.shape[-1] != 3:
        raise ValueError("traj['r'] and traj['a'] must have shape (T, 3) or (N, T, 3)")

    d = r - a
    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
    u = d / np.maximum(d_norm, 1e-12)
    cos_th = np.clip(-u[..., 2], -1.0, 1.0)
    return np.degrees(np.arccos(cos_th))


def check_cartesian_violation(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float] | None = None,
) -> np.ndarray | bool:
    """Return True where the pendulum angle exceeds ``threshold`` degrees."""

    threshold = float(threshold)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    t = np.asarray(traj.get("t"), dtype=float)
    if t.ndim != 1:
        raise ValueError("traj['t'] must be 1D")

    theta_deg = np.asarray(traj.get("theta_deg", trajectory_theta_deg(traj)), dtype=float)
    if theta_deg.shape[-1] != t.shape[0]:
        raise ValueError("traj angle history must match len(traj['t'])")

    mask_t = _window_mask(t, window)
    theta_window = theta_deg[..., mask_t]
    violated = np.max(theta_window, axis=-1) > threshold

    if theta_deg.ndim == 1:
        return bool(violated)
    return np.asarray(violated, dtype=bool)


def check_cartesian_safety(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float] | None = None,
) -> np.ndarray | bool:
    """Spring-style safety check: True means angle stays below ``threshold`` degrees."""

    violated = check_cartesian_violation(traj=traj, threshold=threshold, window=window)
    if isinstance(violated, np.ndarray):
        return ~violated
    return not violated


def cartesian_stability_labels(
    traj: dict[str, np.ndarray],
    threshold: float,
    window: Tuple[float, float] | None = None,
) -> np.ndarray | int:
    """Return binary labels: 1 if safe, 0 if violated."""

    safe = check_cartesian_safety(traj=traj, threshold=threshold, window=window)
    if isinstance(safe, np.ndarray):
        return safe.astype(np.int64)
    return int(safe)


__all__ = [
    "CartesianPendulumParams",
    "cartesian_stability_labels",
    "check_cartesian_safety",
    "check_cartesian_violation",
    "pivot_shake_pulse_batch",
    "sample_X_shake_pulse",
    "simulate_cartesian_pendulum_custom",
    "trajectory_theta_deg",
]
