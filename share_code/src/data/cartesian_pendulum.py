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
from scipy.stats import beta as beta_distribution
from scipy.stats import norm, truncnorm


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


@dataclass(frozen=True)
class DistributionConfig:
    """Configuration for a bounded scalar distribution."""

    kind: str
    bounds: tuple[float, float]
    loc: float | None = None
    scale: float | None = None
    alpha: float | None = None
    beta: float | None = None

    def validate(self, name: str) -> tuple[float, float]:
        lo, hi = _validate_range(f"{name}.bounds", self.bounds)
        kind = self.kind.strip().lower()
        if kind == "uniform":
            return lo, hi
        if kind in ("gaussian", "normal"):
            if self.loc is None or self.scale is None:
                raise ValueError(f"{name} gaussian config requires loc and scale")
            if self.scale < 0:
                raise ValueError(f"{name}.scale must be >= 0")
            return lo, hi
        if kind == "beta":
            if self.alpha is None or self.beta is None:
                raise ValueError(f"{name} beta config requires alpha and beta")
            if self.alpha <= 0 or self.beta <= 0:
                raise ValueError(f"{name} beta parameters must be positive")
            return lo, hi
        raise ValueError(f"{name}.kind must be 'uniform', 'gaussian', or 'beta'")


def _validate_range(name: str, values: tuple[float, float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must be a tuple/list of length 2")
    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return lo, hi


def _sample_from_config(
    rng: np.random.Generator,
    n_samples: int,
    name: str,
    config: DistributionConfig,
) -> np.ndarray:
    u = rng.uniform(np.finfo(float).eps, 1.0 - np.finfo(float).eps, size=n_samples)
    return _sample_from_config_with_u(name=name, config=config, u=u)


def _sample_from_config_with_u(
    name: str,
    config: DistributionConfig,
    u: np.ndarray,
) -> np.ndarray:
    lo, hi = config.validate(name)
    kind = config.kind.strip().lower()
    u = np.asarray(u, dtype=float)
    u = np.clip(u, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    if kind == "uniform":
        return lo + (hi - lo) * u
    if kind in ("gaussian", "normal"):
        scale = float(config.scale)
        if scale == 0.0:
            return np.full_like(u, fill_value=np.clip(float(config.loc), lo, hi), dtype=float)
        a = (lo - float(config.loc)) / scale
        b = (hi - float(config.loc)) / scale
        return truncnorm.ppf(u, a=a, b=b, loc=float(config.loc), scale=scale)

    samples01 = beta_distribution.ppf(u, float(config.alpha), float(config.beta))
    return lo + (hi - lo) * samples01


def _normalize_sigma_spec(
    sigma_spec: float | list[float] | tuple[float, ...],
) -> tuple[float, float]:
    if np.isscalar(sigma_spec):
        sigma = float(sigma_spec)
        if sigma < 0:
            raise ValueError("sigma_spec must be non-negative")
        return sigma, sigma

    values = tuple(float(x) for x in sigma_spec)
    if len(values) == 1:
        sigma = values[0]
        if sigma < 0:
            raise ValueError("sigma_spec must be non-negative")
        return sigma, sigma
    if len(values) == 2:
        return _validate_range("sigma_spec", (values[0], values[1]))
    raise ValueError("sigma_spec must be a scalar, [sigma], or (low, high)")


def _sample_correlated_uniform_pair(
    rng: np.random.Generator,
    n_samples: int,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    rho = float(correlation)
    if not (-0.999999 <= rho <= 0.999999):
        raise ValueError("correlation must be in [-0.999999, 0.999999]")

    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    z = rng.multivariate_normal(mean=np.zeros(2, dtype=float), cov=cov, size=n_samples)
    u = norm.cdf(z)
    eps = np.finfo(float).eps
    return np.clip(u[:, 0], eps, 1.0 - eps), np.clip(u[:, 1], eps, 1.0 - eps)


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
    A_config: DistributionConfig | None = None,
    omega_config: DistributionConfig | None = None,
    sigma_range: float | list[float] | tuple[float, ...] = (0.45, 1.2),
    correlation: float = 0.0,
    t0: float | None = None,
    q: tuple[float, float, float] = (1.0, 0.0, 0.0),
    a0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int | None = 21,
) -> np.ndarray:
    """
    Sample pulse contexts X with columns
    [A, omega, sigma, t0, qx, qy, qz, a0x, a0y, a0z].

    ``A_config`` and ``omega_config`` each describe one bounded distribution:
    - ``DistributionConfig(kind="uniform", bounds=(lo, hi))``
    - ``DistributionConfig(kind="gaussian", bounds=(lo, hi), loc=..., scale=...)``
    - ``DistributionConfig(kind="beta", bounds=(lo, hi), alpha=..., beta=...)``

    ``correlation`` couples ``A`` and ``omega`` through a Gaussian copula, so you
    can keep the marginal configs above while controlling dependence separately.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    rng = np.random.default_rng(seed)

    if A_config is None:
        A_config = DistributionConfig(kind="uniform", bounds=(0.10, 0.35))
    if omega_config is None:
        omega_config = DistributionConfig(kind="uniform", bounds=(1.0, 3.0))

    if abs(float(correlation)) < 1e-12:
        A = _sample_from_config(rng, n_samples=n_samples, name="A_config", config=A_config)
        omega = _sample_from_config(rng, n_samples=n_samples, name="omega_config", config=omega_config)
    else:
        u_A, u_omega = _sample_correlated_uniform_pair(
            rng=rng,
            n_samples=n_samples,
            correlation=correlation,
        )
        A = _sample_from_config_with_u(name="A_config", config=A_config, u=u_A)
        omega = _sample_from_config_with_u(name="omega_config", config=omega_config, u=u_omega)

    sigma_lo, sigma_hi = _normalize_sigma_spec(sigma_range)
    if sigma_lo == sigma_hi:
        sigma = np.full(n_samples, sigma_lo, dtype=float)
    else:
        sigma = rng.uniform(sigma_lo, sigma_hi, size=n_samples)
    t0_vals = np.full(n_samples, np.nan if t0 is None else float(t0), dtype=float)

    q_arr = np.asarray(q, dtype=float)
    q_norm = np.linalg.norm(q_arr)
    if q_norm <= 1e-12:
        raise ValueError("q must be non-zero")
    q_arr = q_arr / q_norm

    a0_arr = np.asarray(a0, dtype=float)

    q_all = np.tile(q_arr[None, :], (n_samples, 1))
    a0_all = np.tile(a0_arr[None, :], (n_samples, 1))

    return np.column_stack([A, omega, sigma, t0_vals, q_all, a0_all])


def pivot_shake_pulse_batch(t: float, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate pivot position, velocity, and acceleration for pulse contexts."""

    X, _ = _as_2d_context(X, n_features=10)

    A = X[:, 0]
    omega = X[:, 1]
    sigma = np.maximum(X[:, 2], 1e-6)
    t0 = np.where(np.isnan(X[:, 3]), 5.0, X[:, 3])
    q = X[:, 4:7]
    a0 = X[:, 7:10]

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)

    s = t - t0
    env = np.exp(-(s**2) / (2.0 * sigma**2))
    sin_term = np.sin(omega * s)
    cos_term = np.cos(omega * s)

    amp = A * env * sin_term
    amp_dot = A * env * ((-s / (sigma**2)) * sin_term + omega * cos_term)
    amp_ddot = A * env * (
        (s**2 / (sigma**4) - 1.0 / (sigma**2) - omega**2) * sin_term
        - 2.0 * omega * s / (sigma**2) * cos_term
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


def compute_cartesian_mu_safety_map(
    params: CartesianPendulumParams,
    t_final: float,
    dt: float,
    window: Tuple[float, float] | None,
    threshold: float,
    A_limits: tuple[float, float] = (0.01, 0.5),
    omega_limits: tuple[float, float] = (1.0, 5.0),
    n_A_grid: int = 25,
    n_omega_grid: int = 25,
    n_per_cell: int = 1000,
    sigma_range: float | list[float] | tuple[float, ...] = (0.45, 1.2),
    t0: float | None = None,
    q: tuple[float, float, float] = (1.0, 0.0, 0.0),
    a0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    eta: float = 0.1,
    pivot_eval: Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] = pivot_shake_pulse_batch,
    n0: np.ndarray = np.array([0.15, 0.0, -0.9887]),
    rel_v0: np.ndarray = np.array([0.0, 0.8, 0.0]),
    project_each_step: bool = True,
    seed: int = 21,
) -> dict[str, np.ndarray | float | int | tuple[float, float]]:
    """
    Estimate safety probability on an ``(A, omega)`` grid for the Cartesian pendulum.

    Each grid cell fixes ``A`` and ``omega`` and samples the remaining pulse context
    randomness through ``sigma_range``.
    """

    params.validate()
    if n_A_grid < 2 or n_omega_grid < 2:
        raise ValueError("n_A_grid and n_omega_grid must both be >= 2")
    if n_per_cell <= 0:
        raise ValueError("n_per_cell must be > 0")
    if not (0.0 <= eta < 1.0):
        raise ValueError("eta must be in [0, 1)")

    A_lo, A_hi = _validate_range("A_limits", A_limits)
    omega_lo, omega_hi = _validate_range("omega_limits", omega_limits)
    A_vals = np.linspace(A_lo, A_hi, n_A_grid)
    omega_vals = np.linspace(omega_lo, omega_hi, n_omega_grid)

    p_safe_grid = np.zeros((n_A_grid, n_omega_grid), dtype=float)
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_A_grid * n_omega_grid)

    idx = 0
    for j, A_val in enumerate(A_vals):
        for i, omega_val in enumerate(omega_vals):
            cell_seed = int(child_seeds[idx].generate_state(1)[0])
            idx += 1

            X = sample_X_shake_pulse(
                n_samples=n_per_cell,
                A_config=DistributionConfig(kind="uniform", bounds=(A_val, A_val)),
                omega_config=DistributionConfig(kind="uniform", bounds=(omega_val, omega_val)),
                sigma_range=sigma_range,
                t0=t0,
                q=q,
                a0=a0,
                seed=cell_seed,
            )
            traj = simulate_cartesian_pendulum_custom(
                params=params,
                t_final=t_final,
                dt=dt,
                X=X,
                pivot_eval=pivot_eval,
                n0=n0,
                rel_v0=rel_v0,
                project_each_step=project_each_step,
            )
            safe_mask = check_cartesian_safety(traj, threshold=threshold, window=window)
            p_safe_grid[j, i] = float(np.mean(safe_mask))

    threshold_value = 1.0 - float(eta)
    config_safe_grid = (p_safe_grid >= threshold_value).astype(float)
    config_safe = int(float(np.mean(p_safe_grid)) >= threshold_value)

    return {
        "A": A_vals,
        "omega": omega_vals,
        "mu1": A_vals,
        "mu2": omega_vals,
        "p_safe_grid": p_safe_grid,
        "safe": p_safe_grid,
        "config_safe_grid": config_safe_grid,
        "config_safe": config_safe,
        "threshold": threshold_value,
        "eta": float(eta),
        "window": window,
        "theta_threshold_deg": float(threshold),
        "sigma_range": _normalize_sigma_spec(sigma_range),
    }


def compute_cartesian_x_safety_map(
    params: CartesianPendulumParams,
    t_final: float,
    dt: float,
    window: Tuple[float, float] | None,
    threshold: float,
    n_grid: int = 25,
    box_limits: tuple[float, float] | tuple[tuple[float, float], tuple[float, float]] = ((0.01, 0.5), (1.0, 5.0)),
    sigma_range: float | list[float] | tuple[float, ...] = 1.0,
    n_per_cell: int = 1,
    correlation: float = 0.0,
    t0: float | None = None,
    q: tuple[float, float, float] = (1.0, 0.0, 0.0),
    a0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    eta: float = 0.1,
    pivot_eval: Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] = pivot_shake_pulse_batch,
    n0: np.ndarray = np.array([0.15, 0.0, -0.9887]),
    rel_v0: np.ndarray = np.array([0.0, 0.8, 0.0]),
    project_each_step: bool = True,
    seed: int = 21,
) -> dict[str, np.ndarray | float | int | tuple[float, float] | None]:
    """
    Cartesian analogue of ``compute_x_safety_map`` over an ``(A, omega)`` box.

    Each grid cell fixes the center values of ``A`` and ``omega``. If ``n_per_cell=1``
    and ``sigma_range`` is fixed, this behaves like a deterministic grid evaluation.
    If ``n_per_cell>1`` and/or ``sigma_range`` spans an interval, the returned
    ``safe`` grid is the estimated safety probability at each cell.
    """

    params.validate()
    if n_grid <= 1:
        raise ValueError("n_grid must be > 1")
    if n_per_cell <= 0:
        raise ValueError("n_per_cell must be > 0")
    if not (0.0 <= eta < 1.0):
        raise ValueError("eta must be in [0, 1)")

    if isinstance(box_limits[0], (tuple, list, np.ndarray)):
        if len(box_limits) != 2:
            raise ValueError("box_limits must be (low, high) or ((lowA, highA), (lowW, highW))")
        A_lim = tuple(box_limits[0])  # type: ignore[arg-type]
        omega_lim = tuple(box_limits[1])  # type: ignore[arg-type]
    else:
        if len(box_limits) != 2:
            raise ValueError("box_limits must be (low, high) or ((lowA, highA), (lowW, highW))")
        shared = (float(box_limits[0]), float(box_limits[1]))  # type: ignore[index]
        A_lim = shared
        omega_lim = shared

    A_lo, A_hi = _validate_range("A_limits", (float(A_lim[0]), float(A_lim[1])))
    omega_lo, omega_hi = _validate_range("omega_limits", (float(omega_lim[0]), float(omega_lim[1])))
    sigma_pair = _normalize_sigma_spec(sigma_range)

    A_vals = np.linspace(A_lo, A_hi, n_grid)
    omega_vals = np.linspace(omega_lo, omega_hi, n_grid)
    safe_grid = np.zeros((n_grid, n_grid), dtype=float)

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_grid * n_grid)

    idx = 0
    for j, A_val in enumerate(A_vals):
        for i, omega_val in enumerate(omega_vals):
            cell_seed = int(child_seeds[idx].generate_state(1)[0])
            idx += 1

            X = sample_X_shake_pulse(
                n_samples=n_per_cell,
                A_config=DistributionConfig(kind="uniform", bounds=(A_val, A_val)),
                omega_config=DistributionConfig(kind="uniform", bounds=(omega_val, omega_val)),
                sigma_range=sigma_pair,
                correlation=correlation,
                t0=t0,
                q=q,
                a0=a0,
                seed=cell_seed,
            )
            traj = simulate_cartesian_pendulum_custom(
                params=params,
                t_final=t_final,
                dt=dt,
                X=X,
                pivot_eval=pivot_eval,
                n0=n0,
                rel_v0=rel_v0,
                project_each_step=project_each_step,
            )
            safe_mask = check_cartesian_safety(traj, threshold=threshold, window=window)
            safe_grid[j, i] = float(np.mean(safe_mask))

    threshold_value = 1.0 - float(eta)
    config_safe_grid = (safe_grid >= threshold_value).astype(float)

    return {
        "A": A_vals,
        "omega": omega_vals,
        "mu1": A_vals,
        "mu2": omega_vals,
        "box_limits": ((A_lo, A_hi), (omega_lo, omega_hi)),
        "safe": safe_grid,
        "p_safe_grid": safe_grid,
        "config_safe_grid": config_safe_grid,
        "threshold": threshold_value,
        "eta": float(eta),
        "window": window,
        "theta_threshold_deg": float(threshold),
        "sigma_range": sigma_pair,
        "n_per_cell": int(n_per_cell),
    }


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
    "DistributionConfig",
    "cartesian_stability_labels",
    "compute_cartesian_x_safety_map",
    "compute_cartesian_mu_safety_map",
    "check_cartesian_safety",
    "check_cartesian_violation",
    "pivot_shake_pulse_batch",
    "sample_X_shake_pulse",
    "simulate_cartesian_pendulum_custom",
    "trajectory_theta_deg",
]
