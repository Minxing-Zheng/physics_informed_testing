"""
Lightweight ODE utilities for spring-mass(-damper) dynamics with external forcing.

The core entry point is ``simulate_spring`` which integrates

    x''(t) = (1/m) * ( -c * x'(t) - k * x(t) + u(t) )

using a 4th-order Runge–Kutta scheme. Only NumPy is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Parameter containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpringParams:
    """Physical parameters for a 1D spring-mass-damper system."""

    mass: float = 1.0
    damping: float = 0.0
    stiffness: float = 1.0

    def validate(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.stiffness <= 0:
            raise ValueError("stiffness must be positive")
        # damping can be zero (undamped) but not negative
        if self.damping < 0:
            raise ValueError("damping must be non-negative")


# --------------------------------------------------------------------------- #
# Integrator
# --------------------------------------------------------------------------- #


def simulate_spring(
    params: SpringParams,
    t_final: float,
    dt: float,
    X: np.ndarray,
    forcing_type: str = "sin",
    config: np.ndarray | None = None,
    x0: float = 0.0,
    v0: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Integrate a spring-mass-damper ODE with external forcing.

    Parameters
    ----------
    params : SpringParams
        Physical parameters (mass m, damping c, stiffness k).
    t_final : float
        Final time horizon (seconds). Must be > 0.
    dt : float
        Time step. Must satisfy 0 < dt <= t_final.
    X : np.ndarray
        Array of shape (N, 2) with columns [A, omega] or shape (2,) for a single
        trajectory. A is amplitude, omega is frequency.
    forcing_type : str, optional
        Forcing type built from X. Supported: "sin", "constant", "zero".
    config : np.ndarray, optional
        Initial state as np.array([x0, v0]). If provided, overrides x0 and v0.
    x0 : float, optional
        Initial position (used when config is None).
    v0 : float, optional
        Initial velocity (used when config is None).

    Returns
    -------
    dict with keys
        - t: time grid (np.ndarray of shape (K,))
        - s: positions over time (np.ndarray of shape (N, K) or (K,) for single)
        - s_dot: velocities over time (np.ndarray of shape (N, K) or (K,) for single)
    """
    params.validate()
    if t_final <= 0:
        raise ValueError("t_final must be positive")
    if dt <= 0 or dt > t_final:
        raise ValueError("dt must satisfy 0 < dt <= t_final")

    X = np.asarray(X, dtype=float)
    single = False
    if X.ndim == 1:
        if X.shape[0] != 2:
            raise ValueError("X must have shape (2,) when 1D")
        X = X.reshape(1, 2)
        single = True
    elif X.ndim == 2:
        if X.shape[1] != 2:
            raise ValueError("X must have shape (N, 2)")
    else:
        raise ValueError("X must have shape (2,) or (N, 2)")

    if config is not None:
        config = np.asarray(config, dtype=float)
        if config.shape == (2,):
            x0 = float(config[0])
            v0 = float(config[1])
            config = None
        elif config.shape != (X.shape[0], 2):
            raise ValueError("config must have shape (2,) or (N, 2)")

    n_steps = int(np.floor(t_final / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    N = X.shape[0]
    A = X[:, 0]
    omega = X[:, 1]

    s = np.empty((N, t.size), dtype=float)
    s_dot = np.empty((N, t.size), dtype=float)

    if config is None:
        s[:, 0] = x0
        s_dot[:, 0] = v0
    else:
        s[:, 0] = config[:, 0]
        s_dot[:, 0] = config[:, 1]

    def _u(time: float) -> np.ndarray:
        if forcing_type == "sin":
            return A * np.sin(omega * time)
        if forcing_type == "constant":
            return A
        if forcing_type == "zero":
            return np.zeros_like(A)
        raise ValueError("forcing_type must be 'sin', 'constant', or 'zero'")

    def _accel_vec(x: np.ndarray, v: np.ndarray, time: float) -> np.ndarray:
        u = _u(time)
        return (-params.damping * v - params.stiffness * x + u) / params.mass

    for i in range(n_steps):
        ti = float(t[i])
        xi = s[:, i]
        vi = s_dot[:, i]

        k1_x = vi
        k1_v = _accel_vec(xi, vi, ti)
        k2_x = vi + 0.5 * dt * k1_v
        k2_v = _accel_vec(xi + 0.5 * dt * k1_x, vi + 0.5 * dt * k1_v, ti + 0.5 * dt)
        k3_x = vi + 0.5 * dt * k2_v
        k3_v = _accel_vec(xi + 0.5 * dt * k2_x, vi + 0.5 * dt * k2_v, ti + 0.5 * dt)
        k4_x = vi + dt * k3_v
        k4_v = _accel_vec(xi + dt * k3_x, vi + dt * k3_v, ti + dt)

        s[:, i + 1] = xi + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        s_dot[:, i + 1] = vi + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    if single:
        return {"t": t, "s": s[0], "s_dot": s_dot[0]}

    return {"t": t, "s": s, "s_dot": s_dot}


__all__ = ["SpringParams", "simulate_spring"]
