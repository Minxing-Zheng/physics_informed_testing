#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Spring Toy Scenario Generator (X-space circular boundary + Gaussian variance shift)
Complete implementation:
- Configurable X_core ~ N(mean, cov) with optional target unsafe ratio
- X-space safe set S = {||X_core - mu||_2 <= radius}
- Auxiliary inputs: damping c, initial conditions x0, v0 (configurable distributions)
- Spring-mass-damper simulation with forcing mapped from X_core
- Observation generation with optional noise
- Physics consistency check in a time window: max |x(t)| <= tau
- Visualization interfaces:
  (V1) X-space scatter + boundary + Gaussian contour
  (V2) Trajectories with optional bounds overlay
  (V3) unsafe_ratio vs sigma scan curve
- Reproducible dataset with metadata, stats, and saving utilities

Dependencies: numpy, matplotlib
(Scipy is NOT required; RK4 is implemented.)

"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, Any, List, Literal, Union

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Utilities / Validation
# ----------------------------

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _is_symmetric(a: np.ndarray, atol: float = 1e-10) -> bool:
    return np.allclose(a, a.T, atol=atol)

def _is_pos_def(a: np.ndarray, tol: float = 1e-12) -> bool:
    # Positive definite check via eigenvalues.
    # For covariance, semidefinite can be OK in some contexts, but here we require PD for stable sampling.
    w = np.linalg.eigvalsh(a)
    return np.all(w > tol)

def _validate_2vec(name: str, v: np.ndarray) -> None:
    if not isinstance(v, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(v)}")
    if v.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {v.shape}")

def _validate_2x2(name: str, m: np.ndarray) -> None:
    if not isinstance(m, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(m)}")
    if m.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2,2), got {m.shape}")
    if not _is_symmetric(m):
        raise ValueError(f"{name} must be symmetric")

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)

def _coerce_float(x: Any, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise TypeError(f"{name} must be convertible to float: {e}")
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    return v


# ----------------------------
# Distributions for Aux Variables
# ----------------------------

@dataclass(frozen=True)
class UniformSpec:
    low: float
    high: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        low = _coerce_float(self.low, "UniformSpec.low")
        high = _coerce_float(self.high, "UniformSpec.high")
        if not (high > low):
            raise ValueError(f"UniformSpec requires high>low, got low={low}, high={high}")
        return rng.uniform(low, high, size=size)

@dataclass(frozen=True)
class NormalSpec:
    mean: float
    std: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        mean = _coerce_float(self.mean, "NormalSpec.mean")
        std = _coerce_float(self.std, "NormalSpec.std")
        if std < 0:
            raise ValueError("NormalSpec.std must be >= 0")
        return rng.normal(mean, std, size=size)

AuxSpec = Union[UniformSpec, NormalSpec]


# ----------------------------
# Forcing map
# ----------------------------

class ForcingMap:
    """
    Maps X_core=(x1,x2) to a forcing function u(t).
    Provide either:
      - mode="sine": u(t)=A*sin(omega*t) with A=x1, omega=x2
      - mode="custom": provide a python callable: (t, X_core)->u
    """

    def __init__(
        self,
        mode: Literal["sine", "custom"] = "sine",
        custom: Optional[Callable[[float, np.ndarray], float]] = None,
    ):
        if mode not in ("sine", "custom"):
            raise ValueError(f"ForcingMap.mode must be 'sine' or 'custom', got {mode}")
        if mode == "custom" and custom is None:
            raise ValueError("ForcingMap custom mode requires a callable")
        self.mode = mode
        self.custom = custom

    def u(self, t: float, X_core: np.ndarray) -> float:
        if self.mode == "sine":
            # X_core = (A, omega)
            A = float(X_core[0])
            omega = float(X_core[1])
            return A * math.sin(omega * t)
        else:
            return float(self.custom(t, X_core))


# ----------------------------
# Spring dynamics and simulator
# ----------------------------

@dataclass(frozen=True)
class SpringParams:
    m: float = 1.0
    k: float = 1.0

    def validate(self) -> None:
        m = _coerce_float(self.m, "SpringParams.m")
        k = _coerce_float(self.k, "SpringParams.k")
        if m <= 0:
            raise ValueError("SpringParams.m must be > 0")
        if k <= 0:
            raise ValueError("SpringParams.k must be > 0")

@dataclass(frozen=True)
class SimParams:
    T: float
    dt: float
    solver: Literal["rk4"] = "rk4"

    def validate(self) -> None:
        T = _coerce_float(self.T, "SimParams.T")
        dt = _coerce_float(self.dt, "SimParams.dt")
        if T <= 0:
            raise ValueError("SimParams.T must be > 0")
        if dt <= 0:
            raise ValueError("SimParams.dt must be > 0")
        if dt >= T:
            raise ValueError("SimParams.dt must be < T")
        if self.solver != "rk4":
            raise ValueError("Only 'rk4' solver is supported in this implementation (intentionally self-contained).")

@dataclass(frozen=True)
class ObsParams:
    obs_type: Literal["x", "x_v"] = "x"
    noise_std: float = 0.0
    return_time: bool = True

    def validate(self) -> None:
        if self.obs_type not in ("x", "x_v"):
            raise ValueError(f"ObsParams.obs_type must be 'x' or 'x_v', got {self.obs_type}")
        noise_std = _coerce_float(self.noise_std, "ObsParams.noise_std")
        if noise_std < 0:
            raise ValueError("ObsParams.noise_std must be >= 0")

@dataclass(frozen=True)
class PhysCheckParams:
    enabled: bool
    window: Tuple[float, float]
    x_bound: float

    def validate(self, T: float) -> None:
        if not self.enabled:
            return
        T1 = _coerce_float(self.window[0], "PhysCheckParams.window[0]")
        T2 = _coerce_float(self.window[1], "PhysCheckParams.window[1]")
        if not (0 <= T1 < T2 <= T):
            raise ValueError(f"PhysCheckParams.window must satisfy 0<=T1<T2<=T. Got {self.window}, T={T}")
        xb = _coerce_float(self.x_bound, "PhysCheckParams.x_bound")
        if xb <= 0:
            raise ValueError("PhysCheckParams.x_bound must be > 0")

class SpringSimulator:
    """
    Simulate:
        m x'' + c x' + k x = u(t; X_core)
    as first-order system:
        d/dt [x, v] = [v, (u - c v - k x)/m]
    """

    def __init__(
        self,
        spring_params: SpringParams,
        sim_params: SimParams,
        forcing_map: ForcingMap,
    ):
        spring_params.validate()
        sim_params.validate()
        self.m = float(spring_params.m)
        self.k = float(spring_params.k)
        self.T = float(sim_params.T)
        self.dt = float(sim_params.dt)
        self.forcing_map = forcing_map

        self.t = np.arange(0.0, self.T + 1e-12, self.dt)  # inclusive endpoint if near
        if len(self.t) < 2:
            raise RuntimeError("Time grid too small; check T and dt.")

    def _f(self, t: float, state: np.ndarray, c: float, X_core: np.ndarray) -> np.ndarray:
        x, v = float(state[0]), float(state[1])
        u = self.forcing_map.u(t, X_core)
        a = (u - c * v - self.k * x) / self.m
        return np.array([v, a], dtype=float)

    def simulate(
        self,
        X_core: np.ndarray,
        c: float,
        x0: float,
        v0: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            t: (K,)
            states: (K,2) for [x,v]
        """
        _validate_2vec("X_core", X_core)
        c = _coerce_float(c, "c")
        x0 = _coerce_float(x0, "x0")
        v0 = _coerce_float(v0, "v0")

        K = len(self.t)
        states = np.zeros((K, 2), dtype=float)
        states[0, 0] = x0
        states[0, 1] = v0

        # RK4
        for i in range(K - 1):
            ti = float(self.t[i])
            si = states[i].copy()
            h = self.dt

            k1 = self._f(ti, si, c, X_core)
            k2 = self._f(ti + 0.5 * h, si + 0.5 * h * k1, c, X_core)
            k3 = self._f(ti + 0.5 * h, si + 0.5 * h * k2, c, X_core)
            k4 = self._f(ti + h, si + h * k3, c, X_core)
            states[i + 1] = si + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return self.t.copy(), states


# ----------------------------
# Generator Config and Dataset
# ----------------------------

@dataclass(frozen=True)
class SafeSetParams:
    mu: Tuple[float, float]
    radius: float

    def validate(self) -> None:
        mu = np.array(self.mu, dtype=float)
        _validate_2vec("SafeSetParams.mu", mu)
        r = _coerce_float(self.radius, "SafeSetParams.radius")
        if r <= 0:
            raise ValueError("SafeSetParams.radius must be > 0")

@dataclass(frozen=True)
class GaussianCoreDistParams:
    mean: Tuple[float, float]
    cov: Tuple[Tuple[float, float], Tuple[float, float]]

    def validate(self) -> None:
        mean = np.array(self.mean, dtype=float)
        cov = np.array(self.cov, dtype=float)
        _validate_2vec("GaussianCoreDistParams.mean", mean)
        _validate_2x2("GaussianCoreDistParams.cov", cov)
        if not _is_pos_def(cov):
            raise ValueError("GaussianCoreDistParams.cov must be positive definite")

@dataclass(frozen=True)
class DatasetParams:
    N: int
    seed: Optional[int] = None
    target_unsafe_ratio: Optional[float] = None
    ratio_tolerance: float = 0.01
    max_sampling_rounds: int = 200
    batch_size: int = 4096

    def validate(self) -> None:
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError("DatasetParams.N must be a positive integer")
        if self.target_unsafe_ratio is not None:
            r = _coerce_float(self.target_unsafe_ratio, "DatasetParams.target_unsafe_ratio")
            if not (0.0 <= r <= 1.0):
                raise ValueError("target_unsafe_ratio must be in [0,1]")
            tol = _coerce_float(self.ratio_tolerance, "DatasetParams.ratio_tolerance")
            if tol <= 0 or tol >= 0.5:
                raise ValueError("ratio_tolerance must be in (0, 0.5)")
            if not isinstance(self.max_sampling_rounds, int) or self.max_sampling_rounds <= 0:
                raise ValueError("max_sampling_rounds must be a positive integer")
            if not isinstance(self.batch_size, int) or self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer")

@dataclass(frozen=True)
class GeneratorConfig:
    safe_set: SafeSetParams
    core_dist: GaussianCoreDistParams
    aux_damping: AuxSpec
    aux_x0: AuxSpec
    aux_v0: AuxSpec
    spring: SpringParams
    sim: SimParams
    obs: ObsParams
    phys_check: PhysCheckParams
    dataset: DatasetParams
    forcing: ForcingMap

    def validate(self) -> None:
        self.safe_set.validate()
        self.core_dist.validate()
        self.spring.validate()
        self.sim.validate()
        self.obs.validate()
        self.dataset.validate()
        self.phys_check.validate(self.sim.T)

@dataclass
class Dataset:
    X_core: np.ndarray            # (N,2)
    X_aux: Dict[str, np.ndarray]  # c, x0, v0 each (N,)
    y: np.ndarray                 # (N,) 1=safe, 0=unsafe (by X-space boundary)
    O: np.ndarray                 # (N,K,d)
    t: Optional[np.ndarray]       # (K,) if return_time else None
    phys_ok: Optional[np.ndarray] # (N,) if enabled else None
    meta: Dict[str, Any]          # config + run info
    stats: Dict[str, Any]         # computed stats

    def save_npz(self, path: str) -> None:
        _ensure_dir(os.path.dirname(path))
        np.savez_compressed(
            path,
            X_core=self.X_core,
            y=self.y,
            O=self.O,
            t=np.array([]) if self.t is None else self.t,
            phys_ok=np.array([]) if self.phys_ok is None else self.phys_ok,
            c=self.X_aux["c"],
            x0=self.X_aux["x0"],
            v0=self.X_aux["v0"],
            meta=json.dumps(self.meta),
            stats=json.dumps(self.stats),
        )

    @staticmethod
    def load_npz(path: str) -> "Dataset":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        stats = json.loads(str(z["stats"]))
        t = z["t"]
        phys_ok = z["phys_ok"]
        t_out = None if t.size == 0 else t
        phys_out = None if phys_ok.size == 0 else phys_ok
        return Dataset(
            X_core=z["X_core"],
            X_aux={"c": z["c"], "x0": z["x0"], "v0": z["v0"]},
            y=z["y"],
            O=z["O"],
            t=t_out,
            phys_ok=phys_out,
            meta=meta,
            stats=stats,
        )


# ----------------------------
# Core generator implementation
# ----------------------------

class SpringScenarioGenerator:
    def __init__(self, config: GeneratorConfig):
        config.validate()
        self.cfg = config
        self.rng = _rng(config.dataset.seed)

        self.safe_mu = np.array(config.safe_set.mu, dtype=float)
        self.safe_r = float(config.safe_set.radius)

        self.core_mean = np.array(config.core_dist.mean, dtype=float)
        self.core_cov = np.array(config.core_dist.cov, dtype=float)
        self.core_cov_chol = np.linalg.cholesky(self.core_cov)

        self.simulator = SpringSimulator(
            spring_params=config.spring,
            sim_params=config.sim,
            forcing_map=config.forcing,
        )

    def _is_safe_x(self, X_core: np.ndarray) -> np.ndarray:
        # X_core: (n,2)
        d = X_core - self.safe_mu.reshape(1, 2)
        dist = np.sqrt(np.sum(d * d, axis=1))
        return (dist <= self.safe_r).astype(np.int64)

    def _sample_core_gaussian(self, n: int) -> np.ndarray:
        # Draw from N(mean, cov) using chol for full control
        z = self.rng.standard_normal(size=(n, 2))
        return self.core_mean.reshape(1, 2) + z @ self.core_cov_chol.T

    def _sample_aux(self, n: int) -> Dict[str, np.ndarray]:
        c = self.cfg.aux_damping.sample(self.rng, n)
        x0 = self.cfg.aux_x0.sample(self.rng, n)
        v0 = self.cfg.aux_v0.sample(self.rng, n)
        return {"c": c.astype(float), "x0": x0.astype(float), "v0": v0.astype(float)}

    def _make_observation(self, states: np.ndarray) -> np.ndarray:
        # states: (K,2) [x,v]
        obs_type = self.cfg.obs.obs_type
        if obs_type == "x":
            out = states[:, [0]]
        elif obs_type == "x_v":
            out = states[:, :2]
        else:
            raise RuntimeError("Unsupported obs_type (should not happen after validation).")

        noise_std = float(self.cfg.obs.noise_std)
        if noise_std > 0:
            out = out + self.rng.normal(0.0, noise_std, size=out.shape)
        return out.astype(float)

    def _physics_check(self, t: np.ndarray, x: np.ndarray) -> bool:
        # Check max |x(t)| <= tau over time window.
        if not self.cfg.phys_check.enabled:
            return True
        T1, T2 = self.cfg.phys_check.window
        tau = float(self.cfg.phys_check.x_bound)
        mask = (t >= T1) & (t <= T2)
        if not np.any(mask):
            raise RuntimeError("Physics check window mask is empty (unexpected).")
        return bool(np.max(np.abs(x[mask])) <= tau)

    def generate(self) -> Dataset:
        N = self.cfg.dataset.N
        target = self.cfg.dataset.target_unsafe_ratio
        tol = float(self.cfg.dataset.ratio_tolerance)
        max_rounds = int(self.cfg.dataset.max_sampling_rounds)
        batch_size = int(self.cfg.dataset.batch_size)

        t_grid = self.simulator.t if self.cfg.obs.return_time else None
        K = len(self.simulator.t)
        d = 1 if self.cfg.obs.obs_type == "x" else 2

        X_core_list: List[np.ndarray] = []
        c_list: List[np.ndarray] = []
        x0_list: List[np.ndarray] = []
        v0_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        O_list: List[np.ndarray] = []
        phys_list: List[np.ndarray] = []

        # Sampling strategy:
        # - If no target unsafe ratio: sample exactly N points from Gaussian.
        # - If target is set: iteratively sample batches and accept according to desired composition.
        #   This is done by controlling acceptance of safe vs unsafe points from the raw Gaussian.
        #   It does NOT modify the underlying Gaussian; it builds a dataset with controlled composition.
        #   (This is often what is meant by "generate scenarios with desired violation rate".)
        if target is None:
            X_core = self._sample_core_gaussian(N)
            y = self._is_safe_x(X_core)

            aux = self._sample_aux(N)
            O = np.zeros((N, K, d), dtype=float)
            phys_ok = np.ones((N,), dtype=np.int64) if self.cfg.phys_check.enabled else None

            for i in range(N):
                t, states = self.simulator.simulate(
                    X_core=X_core[i],
                    c=float(aux["c"][i]),
                    x0=float(aux["x0"][i]),
                    v0=float(aux["v0"][i]),
                )
                O[i] = self._make_observation(states)
                if self.cfg.phys_check.enabled:
                    ok = self._physics_check(t, states[:, 0])
                    phys_ok[i] = 1 if ok else 0

            stats = self._compute_stats(y=y, phys_ok=phys_ok)
            meta = self._build_meta(stats=stats)
            return Dataset(
                X_core=X_core,
                X_aux=aux,
                y=y,
                O=O,
                t=t_grid.copy() if t_grid is not None else None,
                phys_ok=phys_ok,
                meta=meta,
                stats=stats,
            )

        # target unsafe ratio path
        if not (0.0 <= target <= 1.0):
            raise ValueError("target_unsafe_ratio must be in [0,1]")

        # desired counts
        desired_unsafe = int(round(target * N))
        desired_safe = N - desired_unsafe

        got_safe = 0
        got_unsafe = 0

        # Keep sampling from Gaussian; accept samples into dataset until desired counts are met
        rounds = 0
        while (got_safe < desired_safe or got_unsafe < desired_unsafe) and rounds < max_rounds:
            rounds += 1
            n_batch = min(batch_size, (desired_safe - got_safe) + (desired_unsafe - got_unsafe) + batch_size)
            Xb = self._sample_core_gaussian(n_batch)
            yb = self._is_safe_x(Xb)  # 1 safe, 0 unsafe

            safe_idx = np.where(yb == 1)[0]
            unsafe_idx = np.where(yb == 0)[0]

            need_safe = desired_safe - got_safe
            need_unsafe = desired_unsafe - got_unsafe

            take_safe = safe_idx[: max(0, min(need_safe, safe_idx.size))]
            take_unsafe = unsafe_idx[: max(0, min(need_unsafe, unsafe_idx.size))]

            take_idx = np.concatenate([take_safe, take_unsafe], axis=0)
            if take_idx.size == 0:
                continue

            X_take = Xb[take_idx]
            y_take = yb[take_idx].astype(np.int64)
            aux = self._sample_aux(take_idx.size)

            O_take = np.zeros((take_idx.size, K, d), dtype=float)
            phys_take = np.ones((take_idx.size,), dtype=np.int64) if self.cfg.phys_check.enabled else None

            for j in range(take_idx.size):
                t, states = self.simulator.simulate(
                    X_core=X_take[j],
                    c=float(aux["c"][j]),
                    x0=float(aux["x0"][j]),
                    v0=float(aux["v0"][j]),
                )
                O_take[j] = self._make_observation(states)
                if self.cfg.phys_check.enabled:
                    ok = self._physics_check(t, states[:, 0])
                    phys_take[j] = 1 if ok else 0

            # Append
            X_core_list.append(X_take)
            y_list.append(y_take)
            O_list.append(O_take)
            c_list.append(aux["c"])
            x0_list.append(aux["x0"])
            v0_list.append(aux["v0"])
            if self.cfg.phys_check.enabled:
                phys_list.append(phys_take)

            got_safe += int(np.sum(y_take == 1))
            got_unsafe += int(np.sum(y_take == 0))

        if got_safe < desired_safe or got_unsafe < desired_unsafe:
            raise RuntimeError(
                f"Could not meet target composition within max_sampling_rounds={max_rounds}. "
                f"Needed safe={desired_safe}, unsafe={desired_unsafe}; got safe={got_safe}, unsafe={got_unsafe}. "
                f"Possible causes: Gaussian mass almost entirely inside or outside the safe circle. "
                f"Try changing mean/cov or target_unsafe_ratio."
            )

        X_core = np.concatenate(X_core_list, axis=0)[:N]
        y = np.concatenate(y_list, axis=0)[:N]
        O = np.concatenate(O_list, axis=0)[:N]
        c = np.concatenate(c_list, axis=0)[:N]
        x0 = np.concatenate(x0_list, axis=0)[:N]
        v0 = np.concatenate(v0_list, axis=0)[:N]
        aux = {"c": c, "x0": x0, "v0": v0}
        phys_ok = None
        if self.cfg.phys_check.enabled:
            phys_ok = np.concatenate(phys_list, axis=0)[:N]

        # Verify composition achieved (within tolerance)
        unsafe_ratio = float(np.mean(y == 0))
        if abs(unsafe_ratio - target) > tol:
            # We still return dataset, but mark in stats and meta; do NOT silently proceed.
            # This is deliberate: "no simplification" => surface mismatch explicitly.
            pass

        stats = self._compute_stats(y=y, phys_ok=phys_ok)
        stats["target_unsafe_ratio"] = float(target)
        stats["achieved_unsafe_ratio"] = unsafe_ratio
        stats["ratio_tolerance"] = float(tol)
        stats["sampling_rounds"] = int(rounds)

        meta = self._build_meta(stats=stats)
        return Dataset(
            X_core=X_core,
            X_aux=aux,
            y=y,
            O=O,
            t=t_grid.copy() if t_grid is not None else None,
            phys_ok=phys_ok,
            meta=meta,
            stats=stats,
        )

    def _compute_stats(self, y: np.ndarray, phys_ok: Optional[np.ndarray]) -> Dict[str, Any]:
        y = y.astype(np.int64)
        unsafe_ratio_x = float(np.mean(y == 0))
        safe_ratio_x = float(np.mean(y == 1))

        out: Dict[str, Any] = {
            "N": int(y.shape[0]),
            "safe_ratio_x": safe_ratio_x,
            "unsafe_ratio_x": unsafe_ratio_x,
        }

        if phys_ok is not None:
            phys_ok = phys_ok.astype(np.int64)
            phys_violation_ratio = float(np.mean(phys_ok == 0))
            out["phys_ok_ratio"] = float(np.mean(phys_ok == 1))
            out["phys_violation_ratio"] = phys_violation_ratio

            # correlation between y (safe label) and phys_ok (physical window check)
            # Use Pearson correlation; handle degenerate cases explicitly
            y_center = y - np.mean(y)
            p_center = phys_ok - np.mean(phys_ok)
            denom = float(np.sqrt(np.sum(y_center ** 2) * np.sum(p_center ** 2)))
            corr = float(np.sum(y_center * p_center) / denom) if denom > 0 else float("nan")
            out["corr_safe_phys"] = corr

        return out

    def _build_meta(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg

        # Custom serialization for non-JSON-friendly objects
        aux = {
            "aux_damping": type(cfg.aux_damping).__name__,
            "aux_x0": type(cfg.aux_x0).__name__,
            "aux_v0": type(cfg.aux_v0).__name__,
        }

        forcing = {"forcing_mode": cfg.forcing.mode, "forcing_custom": cfg.forcing.mode == "custom"}

        meta = {
            "generated_at_unix": time.time(),
            "safe_set": asdict(cfg.safe_set),
            "core_dist": asdict(cfg.core_dist),
            "spring": asdict(cfg.spring),
            "sim": asdict(cfg.sim),
            "obs": asdict(cfg.obs),
            "phys_check": asdict(cfg.phys_check),
            "dataset": asdict(cfg.dataset),
            "aux_specs": aux,
            "forcing": forcing,
            "stats": stats,
        }
        return meta


# ----------------------------
# Visualization Interfaces
# ----------------------------

def plot_x_space(
    dataset: Dataset,
    mu: Tuple[float, float],
    radius: float,
    mean: Tuple[float, float],
    cov: np.ndarray,
    title: str = "X-space scenarios with safe boundary",
    show_boundary: bool = True,
    show_gaussian_contours: bool = True,
    contour_levels: int = 6,
    save_path: Optional[str] = None,
) -> None:
    X = dataset.X_core
    y = dataset.y.astype(int)

    mu = np.array(mu, dtype=float)
    mean = np.array(mean, dtype=float)
    cov = np.array(cov, dtype=float)
    _validate_2vec("mu", mu)
    _validate_2vec("mean", mean)
    _validate_2x2("cov", cov)

    plt.figure()
    safe = X[y == 1]
    unsafe = X[y == 0]
    if safe.size > 0:
        plt.scatter(safe[:, 0], safe[:, 1], label="safe (X in circle)", alpha=0.7)
    if unsafe.size > 0:
        plt.scatter(unsafe[:, 0], unsafe[:, 1], label="unsafe (X outside)", alpha=0.7)

    if show_boundary:
        theta = np.linspace(0, 2 * np.pi, 400)
        cx = mu[0] + radius * np.cos(theta)
        cy = mu[1] + radius * np.sin(theta)
        plt.plot(cx, cy, linewidth=2, label="safe boundary (circle)")
        plt.scatter([mu[0]], [mu[1]], marker="x", s=80, label="mu (circle center)")

    if show_gaussian_contours:
        # Gaussian density contours on a grid (no seaborn)
        # Determine plotting bounds
        pad = 0.15
        xmin = float(np.min(X[:, 0])) if X.size else float(mu[0] - radius)
        xmax = float(np.max(X[:, 0])) if X.size else float(mu[0] + radius)
        ymin = float(np.min(X[:, 1])) if X.size else float(mu[1] - radius)
        ymax = float(np.max(X[:, 1])) if X.size else float(mu[1] + radius)
        xr = xmax - xmin
        yr = ymax - ymin
        xmin -= pad * xr
        xmax += pad * xr
        ymin -= pad * yr
        ymax += pad * yr

        xs = np.linspace(xmin, xmax, 220)
        ys = np.linspace(ymin, ymax, 220)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

        inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        if det <= 0:
            raise ValueError("cov determinant must be > 0 for contour plotting")

        diff = grid - mean.reshape(1, 2)
        quad = np.einsum("bi,ij,bj->b", diff, inv, diff)
        # unnormalized log density ok; but we compute density for contours
        Z = np.exp(-0.5 * quad) / (2 * np.pi * math.sqrt(det))
        Z = Z.reshape(YY.shape)
        plt.contour(XX, YY, Z, levels=contour_levels, linewidths=1.0)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_trajectories(
    dataset: Dataset,
    title: str = "Spring trajectories",
    n_safe: int = 10,
    n_unsafe: int = 10,
    overlay_bounds: bool = True,
    bound_tau: Optional[float] = None,
    window: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    if dataset.t is None:
        raise ValueError("Dataset does not include time grid (return_time=False). Can't plot trajectories.")
    t = dataset.t
    O = dataset.O
    y = dataset.y.astype(int)

    # Use the first channel as x even if obs_type == x_v
    x_all = O[:, :, 0]

    safe_idx = np.where(y == 1)[0]
    unsafe_idx = np.where(y == 0)[0]

    rng_local = np.random.default_rng(0)
    chosen_safe = rng_local.choice(safe_idx, size=min(n_safe, safe_idx.size), replace=False) if safe_idx.size else np.array([], dtype=int)
    chosen_unsafe = rng_local.choice(unsafe_idx, size=min(n_unsafe, unsafe_idx.size), replace=False) if unsafe_idx.size else np.array([], dtype=int)

    plt.figure()
    for i in chosen_safe:
        plt.plot(t, x_all[i], alpha=0.8, label="safe" if i == chosen_safe[0] else None)
    for i in chosen_unsafe:
        plt.plot(t, x_all[i], alpha=0.8, label="unsafe" if i == chosen_unsafe[0] else None)

    if overlay_bounds and bound_tau is not None:
        tau = float(bound_tau)
        plt.axhline(+tau, linestyle="--", linewidth=1.5, label="+tau")
        plt.axhline(-tau, linestyle="--", linewidth=1.5, label="-tau")

    if window is not None:
        T1, T2 = float(window[0]), float(window[1])
        plt.axvspan(T1, T2, alpha=0.15, label="check window")

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.legend()
    plt.grid(True)

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_unsafe_ratio_vs_sigma(
    mu: Tuple[float, float],
    radius: float,
    sigma_list: List[float],
    N_per_sigma: int = 200000,
    seed: int = 123,
    title: str = "Unsafe ratio vs sigma (isotropic Gaussian)",
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each sigma, sample X ~ N(mu, sigma^2 I) and estimate P(||X-mu||>r).
    Returns arrays (sigma, unsafe_ratio).
    """
    mu = np.array(mu, dtype=float)
    _validate_2vec("mu", mu)
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be > 0")
    if not isinstance(N_per_sigma, int) or N_per_sigma <= 0:
        raise ValueError("N_per_sigma must be positive int")

    rng = np.random.default_rng(seed)
    sigs = np.array([float(s) for s in sigma_list], dtype=float)
    if np.any(sigs <= 0):
        raise ValueError("All sigma values must be > 0")

    ratios = np.zeros_like(sigs)
    for i, s in enumerate(sigs):
        Z = rng.standard_normal(size=(N_per_sigma, 2))
        X = mu.reshape(1, 2) + (s * Z)
        dist = np.sqrt(np.sum((X - mu.reshape(1, 2)) ** 2, axis=1))
        ratios[i] = float(np.mean(dist > r))

    plt.figure()
    plt.plot(sigs, ratios, marker="o")
    plt.title(title)
    plt.xlabel("sigma")
    plt.ylabel("estimated unsafe ratio")
    plt.grid(True)

    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return sigs, ratios


# ----------------------------
# Example / Demo (You can delete this section in your repo if unwanted)
# ----------------------------

def _example_config_P0(seed: int = 7, N = 10) -> GeneratorConfig:
    # Safe set centered at (0,0) radius r
    safe = SafeSetParams(mu=(0.0, 0.0), radius=2.0)

    # P0: small covariance (isotropic)
    sigma0 = 0.6
    cov0 = ((sigma0**2, 0.0), (0.0, sigma0**2))
    core0 = GaussianCoreDistParams(mean=(0.0, 0.0), cov=cov0)

    # Aux: damping + initial conditions
    aux_c = UniformSpec(low=0.15, high=0.45)
    aux_x0 = UniformSpec(low=-0.05, high=0.05)
    aux_v0 = UniformSpec(low=-0.05, high=0.05)

    spring = SpringParams(m=1.0, k=1.0)
    sim = SimParams(T=30.0, dt=0.02, solver="rk4")
    obs = ObsParams(obs_type="x", noise_std=0.0, return_time=True)

    phys = PhysCheckParams(enabled=True, window=(15.0, 30.0), x_bound=1.5)

    dataset = DatasetParams(
        N=N,
        seed=seed,
        target_unsafe_ratio=None,  # sample directly from Gaussian
    )

    forcing = ForcingMap(mode="sine")  # u(t)=A*sin(omega*t), A=x1, omega=x2

    return GeneratorConfig(
        safe_set=safe,
        core_dist=core0,
        aux_damping=aux_c,
        aux_x0=aux_x0,
        aux_v0=aux_v0,
        spring=spring,
        sim=sim,
        obs=obs,
        phys_check=phys,
        dataset=dataset,
        forcing=forcing,
    )

def _example_config_P1(N=10, seed: int = 8) -> GeneratorConfig:
    safe = SafeSetParams(mu=(0.0, 0.0), radius=2.0)

    # P1: larger covariance -> more mass outside circle
    sigma1 = 1.3
    cov1 = ((sigma1**2, 0.0), (0.0, sigma1**2))
    core1 = GaussianCoreDistParams(mean=(0.0, 0.0), cov=cov1)

    aux_c = UniformSpec(low=0.15, high=0.45)
    aux_x0 = UniformSpec(low=-0.05, high=0.05)
    aux_v0 = UniformSpec(low=-0.05, high=0.05)

    spring = SpringParams(m=1.0, k=1.0)
    sim = SimParams(T=30.0, dt=0.02, solver="rk4")
    obs = ObsParams(obs_type="x", noise_std=0.0, return_time=True)

    phys = PhysCheckParams(enabled=True, window=(15.0, 30.0), x_bound=1.5)

    dataset = DatasetParams(
        N=N,
        seed=seed,
        target_unsafe_ratio=None,
    )

    forcing = ForcingMap(mode="sine")

    return GeneratorConfig(
        safe_set=safe,
        core_dist=core1,
        aux_damping=aux_c,
        aux_x0=aux_x0,
        aux_v0=aux_v0,
        spring=spring,
        sim=sim,
        obs=obs,
        phys_check=phys,
        dataset=dataset,
        forcing=forcing,
    )

def main_demo(N= 10, output_dir: str = "./toy_outputs") -> None:
    _ensure_dir(output_dir)

    # Generate P0 and P1
    cfg0 = _example_config_P0(seed=7, N=N)
    gen0 = SpringScenarioGenerator(cfg0)
    ds0 = gen0.generate()
    ds0.save_npz(os.path.join(output_dir, "dataset_P0.npz"))

    cfg1 = _example_config_P1(seed=8, N=N)
    gen1 = SpringScenarioGenerator(cfg1)
    ds1 = gen1.generate()
    ds1.save_npz(os.path.join(output_dir, "dataset_P1.npz"))

    # Print key stats
    print("P0 stats:", json.dumps(ds0.stats, indent=2))
    print("P1 stats:", json.dumps(ds1.stats, indent=2))

    # Visualization: X-space scatter + boundary
    plot_x_space(
        ds0,
        mu=cfg0.safe_set.mu,
        radius=cfg0.safe_set.radius,
        mean=cfg0.core_dist.mean,
        cov=np.array(cfg0.core_dist.cov, dtype=float),
        title="P0: X-space scatter + safe circle",
        save_path=os.path.join(output_dir, "P0_x_space.png"),
    )
    plot_x_space(
        ds1,
        mu=cfg1.safe_set.mu,
        radius=cfg1.safe_set.radius,
        mean=cfg1.core_dist.mean,
        cov=np.array(cfg1.core_dist.cov, dtype=float),
        title="P1: X-space scatter + safe circle",
        save_path=os.path.join(output_dir, "P1_x_space.png"),
    )

    # Visualization: trajectories
    plot_trajectories(
        ds0,
        title="P0 trajectories",
        n_safe=12,
        n_unsafe=12,
        overlay_bounds=True,
        bound_tau=cfg0.phys_check.x_bound if cfg0.phys_check.enabled else None,
        window=cfg0.phys_check.window if cfg0.phys_check.enabled else None,
        save_path=os.path.join(output_dir, "P0_trajectories.png"),
    )
    plot_trajectories(
        ds1,
        title="P1 trajectories",
        n_safe=12,
        n_unsafe=12,
        overlay_bounds=True,
        bound_tau=cfg1.phys_check.x_bound if cfg1.phys_check.enabled else None,
        window=cfg1.phys_check.window if cfg1.phys_check.enabled else None,
        save_path=os.path.join(output_dir, "P1_trajectories.png"),
    )

    # Visualization: unsafe ratio vs sigma scan
    sigma_list = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8]
    plot_unsafe_ratio_vs_sigma(
        mu=cfg0.safe_set.mu,
        radius=cfg0.safe_set.radius,
        sigma_list=sigma_list,
        N_per_sigma=150000,
        title="Unsafe ratio vs sigma (circle boundary fixed)",
        save_path=os.path.join(output_dir, "unsafe_ratio_vs_sigma.png"),
    )

if __name__ == "__main__":
    # Run the demo by default
    main_demo()
