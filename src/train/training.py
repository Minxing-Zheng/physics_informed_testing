"""
Training utilities for conditional GRU models.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from train.losses import mmd2_beta, mmd2_unif


def _unpack_seq_batch(batch):
    """
    Accept (x, o_hist, o_next) with optional y stability label.
    """
    if not isinstance(batch, (tuple, list)) or len(batch) < 3:
        raise ValueError("Expected batch to be a tuple/list with at least 3 elements")
    y = batch[3] if len(batch) >= 4 else None
    return batch[0], batch[1], batch[2], y


def _stability_logits_from_o_hat(
    o_hat: torch.Tensor,
    threshold: float,
    window: Optional[Tuple[float, float]],
    dt: float,
    t0: float,
    smooth_beta: float = 10.0,
) -> torch.Tensor:
    """
    Build binary-safe logits from reconstructed sequence:
      logit = threshold - max_t |o_hat(t)| in selected window.
    Positive logit => more likely safe (label 1).
    """
    if o_hat.ndim != 3:
        raise ValueError(f"o_hat must have shape (B, L, d_o), got {tuple(o_hat.shape)}")
    if dt <= 0:
        raise ValueError("stability dt must be positive")
    if threshold < 0:
        raise ValueError("stability threshold must be non-negative")

    bsz, seq_len, _ = o_hat.shape
    t = t0 + dt * torch.arange(seq_len, device=o_hat.device, dtype=o_hat.dtype)
    if window is None:
        mask = torch.ones(seq_len, dtype=torch.bool, device=o_hat.device)
    else:
        t_l, t_u = float(window[0]), float(window[1])
        if t_l > t_u:
            raise ValueError(f"window must satisfy t_l <= t_u, got {window}")
        mask = (t >= t_l) & (t <= t_u)
        if not torch.any(mask):
            raise ValueError(
                f"stability window {window} has no overlap with sequence time "
                f"range [{float(t[0]):.4f}, {float(t[-1]):.4f}]"
            )

    peak_over_dims = torch.abs(o_hat).amax(dim=-1)  # (B, L)
    vals = peak_over_dims[:, mask]
    # Smooth max to avoid sparse gradients from hard max.
    if smooth_beta is not None and float(smooth_beta) > 0:
        beta = float(smooth_beta)
        # Bias-corrected smooth-max: subtract log(K)/beta so scale matches hard max.
        k = vals.shape[1]
        peak_in_window = (torch.logsumexp(beta * vals, dim=1) - torch.log(torch.tensor(float(k), device=vals.device, dtype=vals.dtype))) / beta
    else:
        peak_in_window = vals.amax(dim=1)
    logits = float(threshold) - peak_in_window
    if logits.shape != (bsz,):
        raise RuntimeError("unexpected stability logits shape")
    return logits


def _resolve_stability_cfg(
    lambda_stability: float,
    stability_threshold: Optional[float],
    stability_window: Optional[Tuple[float, float]],
    stability_dt: float,
    stability_t0: float,
    stability_spec: Optional[Dict[str, Any]],
    stability_smooth_beta: float,
) -> tuple[float, Optional[float], Optional[Tuple[float, float]], float, float, float, float, float]:
    """
    Resolve stability settings from either legacy separate args or one wrapped spec.

    Supported wrapped spec forms:
      {
        "lambda": 0.1,
        "fn": check_safety,
        "fn_kwargs": {"threshold": 1.0, "window": (2.0, 8.0)},
        "dt": 0.05,
        "t0": 0.05,
        "smooth_beta": 10.0,
        "safe_weight": 1.0,
        "unsafe_weight": 1.0,
      }
    or:
      {
        "lambda": 0.1,
        "threshold": 1.0,
        "window": (2.0, 8.0),
        "dt": 0.05,
        "t0": 0.05,
        "smooth_beta": 10.0,
        "safe_weight": 1.0,
        "unsafe_weight": 1.0,
      }
    """
    lam = float(lambda_stability)
    thr = stability_threshold
    win = stability_window
    dt = float(stability_dt)
    t0 = float(stability_t0)
    smooth_beta = float(stability_smooth_beta)
    safe_w = 1.0
    unsafe_w = 1.0
    if stability_spec is None:
        return lam, thr, win, dt, t0, smooth_beta, safe_w, unsafe_w
    if not isinstance(stability_spec, dict):
        raise ValueError("stability_spec must be a dict when provided")

    lam = float(stability_spec.get("lambda", lam))
    dt = float(stability_spec.get("dt", dt))
    t0 = float(stability_spec.get("t0", t0))
    smooth_beta = float(stability_spec.get("smooth_beta", smooth_beta))
    safe_w = float(stability_spec.get("safe_weight", safe_w))
    unsafe_w = float(stability_spec.get("unsafe_weight", unsafe_w))
    if safe_w <= 0 or unsafe_w <= 0:
        raise ValueError("safe_weight and unsafe_weight must be positive")

    # Direct key mode
    if "threshold" in stability_spec:
        thr = float(stability_spec["threshold"])
    if "window" in stability_spec and stability_spec["window"] is not None:
        w = stability_spec["window"]
        win = (float(w[0]), float(w[1]))

    # Wrapped function mode (check_safety-style)
    fn = stability_spec.get("fn")
    if fn is not None:
        fn_name = getattr(fn, "__name__", "")
        if fn_name != "check_safety":
            raise ValueError(
                "Only check_safety is supported in stability_spec['fn'] "
                "for differentiable training."
            )
        fn_kwargs = stability_spec.get("fn_kwargs", {})
        if "threshold" not in fn_kwargs or "window" not in fn_kwargs:
            raise ValueError(
                "stability_spec['fn_kwargs'] must include 'threshold' and 'window' for check_safety."
            )
        thr = float(fn_kwargs["threshold"])
        w = fn_kwargs["window"]
        win = (float(w[0]), float(w[1]))

    return lam, thr, win, dt, t0, smooth_beta, safe_w, unsafe_w


def _stability_bce_loss(
    logits_safe: torch.Tensor,
    y_safe: torch.Tensor,
    safe_weight: float = 1.0,
    unsafe_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted BCE for safety labels (1=safe, 0=unsafe).
    """
    per = F.binary_cross_entropy_with_logits(logits_safe, y_safe, reduction="none")
    weights = float(safe_weight) * y_safe + float(unsafe_weight) * (1.0 - y_safe)
    return torch.mean(per * weights)


def _stability_loss(
    logits_safe: torch.Tensor,
    y_safe: torch.Tensor,
    stability_loss_type: str = "bce",
    safe_weight: float = 1.0,
    unsafe_weight: float = 1.0,
) -> torch.Tensor:
    """
    Stability supervision objective selector.
    Supported:
      - "bce": standard unweighted BCE-with-logits
      - "weighted_bce": weighted BCE-with-logits
    """
    loss_type = str(stability_loss_type).lower()
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits_safe, y_safe, reduction="mean")
    if loss_type == "weighted_bce":
        return _stability_bce_loss(
            logits_safe=logits_safe,
            y_safe=y_safe,
            safe_weight=safe_weight,
            unsafe_weight=unsafe_weight,
        )
    raise ValueError(f"Unsupported stability_loss_type='{stability_loss_type}' for _stability_loss.")


def _regime_hinge_loss(
    divergence_val: torch.Tensor,
    p_safe: torch.Tensor,
    gamma: float,
    delta_safe: float,
    delta_unsafe: float,
) -> torch.Tensor:
    """
    Batch-level hinge tied to safety regime:
      if p_safe >= gamma: penalize divergence above delta_safe
      else:               penalize divergence below delta_unsafe
    """
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError("regime_gamma must be in [0, 1]")
    if float(delta_safe) < 0 or float(delta_unsafe) < 0:
        raise ValueError("regime deltas must be non-negative")

    div = divergence_val
    safe_loss = F.relu(div - float(delta_safe))
    unsafe_loss = F.relu(float(delta_unsafe) - div)
    is_safe_regime = (p_safe >= float(gamma)).to(div.dtype)
    return is_safe_regime * safe_loss + (1.0 - is_safe_regime) * unsafe_loss


def _safe_uniform_unsafe_center_loss(
    v: torch.Tensor,
    y_safe: torch.Tensor,
    *,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor],
    divergence_kwargs: Optional[Dict] = None,
    unsafe_center: float = 0.5,
    unsafe_center_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shape the latent by class:
      - safe points (y=1) are pushed toward the reference distribution via divergence_fn
      - unsafe points (y=0) are pushed toward a center value

    Returns (total_latent_loss, safe_divergence, unsafe_center_loss).
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    if y_safe.ndim != 1:
        raise ValueError(f"y_safe must be 1D, got shape {tuple(y_safe.shape)}")
    if v.shape[0] != y_safe.shape[0]:
        raise ValueError(f"v/y_safe batch mismatch: {tuple(v.shape)} vs {tuple(y_safe.shape)}")
    if float(unsafe_center_weight) < 0:
        raise ValueError("unsafe_center_weight must be non-negative")

    safe_mask = y_safe >= 0.5
    unsafe_mask = ~safe_mask

    safe_div = torch.zeros((), device=v.device, dtype=v.dtype)
    if torch.sum(safe_mask) >= 2:
        safe_div = divergence_fn(v[safe_mask], **divergence_kwargs)

    unsafe_center_loss = torch.zeros((), device=v.device, dtype=v.dtype)
    if torch.any(unsafe_mask):
        target = torch.full_like(v[unsafe_mask], float(unsafe_center))
        unsafe_center_loss = F.mse_loss(v[unsafe_mask], target, reduction="mean")

    total = safe_div + float(unsafe_center_weight) * unsafe_center_loss
    return total, safe_div, unsafe_center_loss


def _safe_uniform_unsafe_beta_loss(
    v: torch.Tensor,
    y_safe: torch.Tensor,
    *,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor],
    divergence_kwargs: Optional[Dict] = None,
    unsafe_beta_alpha: float = 5.0,
    unsafe_beta_beta: float = 5.0,
    unsafe_beta_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shape the latent by class:
      - safe points (y=1) are pushed toward the reference distribution via divergence_fn
      - unsafe points (y=0) are pushed toward Beta(alpha, beta) on (0, 1)

    Returns (total_latent_loss, safe_divergence, unsafe_beta_loss).
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    if y_safe.ndim != 1:
        raise ValueError(f"y_safe must be 1D, got shape {tuple(y_safe.shape)}")
    if v.shape[0] != y_safe.shape[0]:
        raise ValueError(f"v/y_safe batch mismatch: {tuple(v.shape)} vs {tuple(y_safe.shape)}")
    if float(unsafe_beta_alpha) <= 0 or float(unsafe_beta_beta) <= 0:
        raise ValueError(
            "unsafe_beta_alpha and unsafe_beta_beta must be positive"
        )
    if float(unsafe_beta_weight) < 0:
        raise ValueError("unsafe_beta_weight must be non-negative")

    safe_mask = y_safe >= 0.5
    unsafe_mask = ~safe_mask

    safe_div = torch.zeros((), device=v.device, dtype=v.dtype)
    if torch.sum(safe_mask) >= 2:
        safe_div = divergence_fn(v[safe_mask], **divergence_kwargs)

    unsafe_beta_loss = torch.zeros((), device=v.device, dtype=v.dtype)
    if torch.sum(unsafe_mask) >= 2:
        beta_kwargs = {}
        if "sigmas" in divergence_kwargs:
            beta_kwargs["sigmas"] = divergence_kwargs["sigmas"]
        if "use_median" in divergence_kwargs:
            beta_kwargs["use_median"] = divergence_kwargs["use_median"]
        unsafe_beta_loss = mmd2_beta(
            v[unsafe_mask],
            alpha=float(unsafe_beta_alpha),
            beta=float(unsafe_beta_beta),
            **beta_kwargs,
        )

    total = safe_div + float(unsafe_beta_weight) * unsafe_beta_loss
    return total, safe_div, unsafe_beta_loss


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
    lambda_stability: float = 0.0,
    stability_threshold: Optional[float] = None,
    stability_window: Optional[Tuple[float, float]] = None,
    stability_dt: float = 1.0,
    stability_t0: float = 0.0,
    stability_smooth_beta: float = 10.0,
    stability_spec: Optional[Dict[str, Any]] = None,
    stability_loss_type: str = "bce",
    stability_weight_stab: float = 1.0,
    lambda_regime: float = 0.0,
    regime_gamma: float = 0.9,
    regime_delta_safe: float = 0.05,
    regime_delta_unsafe: float = 0.2,
    latent_target_mode: Optional[str] = None,
    unsafe_center: float = 0.5,
    unsafe_center_weight: float = 1.0,
    unsafe_beta_alpha: float = 5.0,
    unsafe_beta_beta: float = 5.0,
    unsafe_beta_weight: float = 1.0,
) -> float:
    """
    Train conditional model for one epoch with a configurable divergence.
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    latent_target_mode_eff = None if latent_target_mode is None else str(latent_target_mode).lower()
    if latent_target_mode_eff not in {
        None,
        "safe_uniform_unsafe_center",
        "safe_uniform_unsafe_beta",
    }:
        raise ValueError(
            f"Unsupported latent_target_mode='{latent_target_mode}'. "
            "Use None, 'safe_uniform_unsafe_center', or 'safe_uniform_unsafe_beta'."
        )
    loss_mode = str(stability_loss_type).lower()
    if loss_mode not in {"bce", "weighted", "weighted_bce"}:
        raise ValueError(
            f"Unsupported stability_loss_type='{stability_loss_type}'. "
            "Use 'bce', 'weighted' (misclassification-weighted recon), or 'weighted_bce'."
        )
    (
        lambda_stability_eff,
        stability_threshold_eff,
        stability_window_eff,
        stability_dt_eff,
        stability_t0_eff,
        stability_smooth_beta_eff,
        safe_weight_eff,
        unsafe_weight_eff,
    ) = _resolve_stability_cfg(
        lambda_stability=lambda_stability,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
        stability_dt=stability_dt,
        stability_t0=stability_t0,
        stability_spec=stability_spec,
        stability_smooth_beta=stability_smooth_beta,
    )
    if stability_spec is not None and isinstance(stability_spec, dict):
        stability_weight_stab = float(stability_spec.get("weight_stab", stability_weight_stab))
    if float(stability_weight_stab) < 0:
        raise ValueError("stability_weight_stab must be non-negative")

    if loss_mode == "weighted" and float(lambda_stability_eff) != 0.0:
        raise ValueError(
            "stability_loss_type='weighted' uses reweighted reconstruction; "
            "set lambda_stability=0 or use stability_loss_type='bce'/'weighted_bce'."
        )

    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        x, o_hist_b, o_next_b, y_b = _unpack_seq_batch(batch)
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)
        if y_b is not None:
            y_b = y_b.to(device).reshape(-1).float()

        o_hat, v = model(x, o_hist_b)
        mmd = divergence_fn(v, **divergence_kwargs)
        latent_reg = mmd
        logits_safe = None
        if loss_mode == "weighted" or lambda_stability_eff != 0.0:
            if y_b is None:
                raise ValueError(
                    "stability supervision requires y labels in loader batch "
                    "(dataset should return 4th element y)."
                )
            if stability_threshold_eff is None:
                raise ValueError(
                    "stability_threshold must be provided when using stability supervision "
                    "(weighted recon or lambda_stability != 0)."
                )
            logits_safe = _stability_logits_from_o_hat(
                o_hat=o_hat,
                threshold=float(stability_threshold_eff),
                window=stability_window_eff,
                dt=float(stability_dt_eff),
                t0=float(stability_t0_eff),
                smooth_beta=float(stability_smooth_beta_eff),
            )
            if logits_safe.shape[0] != y_b.shape[0]:
                raise ValueError(
                    f"stability logits and y shape mismatch: {tuple(logits_safe.shape)} vs {tuple(y_b.shape)}"
                )
        if latent_target_mode_eff == "safe_uniform_unsafe_center":
            if y_b is None:
                raise ValueError(
                    "latent_target_mode='safe_uniform_unsafe_center' requires y labels "
                    "in loader batch (dataset should return 4th element y)."
                )
            latent_reg, _, _ = _safe_uniform_unsafe_center_loss(
                v=v,
                y_safe=y_b,
                divergence_fn=divergence_fn,
                divergence_kwargs=divergence_kwargs,
                unsafe_center=float(unsafe_center),
                unsafe_center_weight=float(unsafe_center_weight),
            )
        elif latent_target_mode_eff == "safe_uniform_unsafe_beta":
            if y_b is None:
                raise ValueError(
                    "latent_target_mode='safe_uniform_unsafe_beta' requires y labels "
                    "in loader batch (dataset should return 4th element y)."
                )
            latent_reg, _, _ = _safe_uniform_unsafe_beta_loss(
                v=v,
                y_safe=y_b,
                divergence_fn=divergence_fn,
                divergence_kwargs=divergence_kwargs,
                unsafe_beta_alpha=float(unsafe_beta_alpha),
                unsafe_beta_beta=float(unsafe_beta_beta),
                unsafe_beta_weight=float(unsafe_beta_weight),
            )

        if loss_mode == "weighted":
            # Eq-style per-sample reconstruction weighting by stability misclassification.
            # rec_i = mean over prediction horizon/features.
            rec_per_sample = torch.mean((o_hat - o_next_b) ** 2, dim=(1, 2))
            y_pred = logits_safe >= 0.0  # type: ignore[operator]
            y_true = y_b >= 0.5
            misclassified = (y_pred != y_true).to(rec_per_sample.dtype)
            rec_weights = 1.0 + float(stability_weight_stab) * misclassified
            rec = torch.mean(rec_weights * rec_per_sample)
        else:
            rec = F.mse_loss(o_hat, o_next_b)

        loss = rec + lambda_mmd * latent_reg

        if lambda_stability_eff != 0.0:
            stability_loss = _stability_loss(
                logits_safe=logits_safe,  # type: ignore[arg-type]
                y_safe=y_b,
                stability_loss_type=("weighted_bce" if loss_mode == "weighted_bce" else "bce"),
                safe_weight=float(safe_weight_eff),
                unsafe_weight=float(unsafe_weight_eff),
            )
            loss = loss + float(lambda_stability_eff) * stability_loss

        if float(lambda_regime) != 0.0:
            if y_b is None:
                raise ValueError(
                    "lambda_regime != 0 requires y labels in loader batch "
                    "(dataset should return 4th element y)."
                )
            p_safe = torch.mean(y_b)
            regime_loss = _regime_hinge_loss(
                divergence_val=mmd,
                p_safe=p_safe,
                gamma=float(regime_gamma),
                delta_safe=float(regime_delta_safe),
                delta_unsafe=float(regime_delta_unsafe),
            )
            loss = loss + float(lambda_regime) * regime_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


def train_epoch_mmd_only(
    model_enc: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
) -> float:
    """
    Train encoder-only model using a divergence objective.
    Returns the raw divergence (unscaled by lambda) averaged over the epoch.
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    model_enc.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        x, _, _, _ = _unpack_seq_batch(batch)
        x = x.to(device)
        v = model_enc(x)
        div_val = divergence_fn(v, **divergence_kwargs)
        loss = div_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += div_val.item() * bs  # log raw divergence
        n += bs
    return total_loss / n if n > 0 else 0.0


@torch.no_grad()
def eval_epoch(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
    lambda_stability: float = 0.0,
    stability_threshold: Optional[float] = None,
    stability_window: Optional[Tuple[float, float]] = None,
    stability_dt: float = 1.0,
    stability_t0: float = 0.0,
    stability_smooth_beta: float = 10.0,
    stability_spec: Optional[Dict[str, Any]] = None,
    return_stability: bool = False,
    latent_target_mode: Optional[str] = None,
    unsafe_center: float = 0.5,
    unsafe_center_weight: float = 1.0,
    unsafe_beta_alpha: float = 5.0,
    unsafe_beta_beta: float = 5.0,
    unsafe_beta_weight: float = 1.0,
) -> Tuple[float, float] | Tuple[float, float, Optional[float]]:
    if divergence_kwargs is None:
        divergence_kwargs = {}
    latent_target_mode_eff = None if latent_target_mode is None else str(latent_target_mode).lower()
    if latent_target_mode_eff not in {
        None,
        "safe_uniform_unsafe_center",
        "safe_uniform_unsafe_beta",
    }:
        raise ValueError(
            f"Unsupported latent_target_mode='{latent_target_mode}'. "
            "Use None, 'safe_uniform_unsafe_center', or 'safe_uniform_unsafe_beta'."
        )
    (
        lambda_stability_eff,
        stability_threshold_eff,
        stability_window_eff,
        stability_dt_eff,
        stability_t0_eff,
        stability_smooth_beta_eff,
        safe_weight_eff,
        unsafe_weight_eff,
    ) = _resolve_stability_cfg(
        lambda_stability=lambda_stability,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
        stability_dt=stability_dt,
        stability_t0=stability_t0,
        stability_spec=stability_spec,
        stability_smooth_beta=stability_smooth_beta,
    )
    model.eval()
    total_rec, total_mmd, total_stability, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        x, o_hist_b, o_next_b, y_b = _unpack_seq_batch(batch)
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)
        if y_b is not None:
            y_b = y_b.to(device).reshape(-1).float()

        o_hat, v = model(x, o_hist_b)
        rec = F.mse_loss(o_hat, o_next_b, reduction="mean").item()
        if latent_target_mode_eff == "safe_uniform_unsafe_center":
            if y_b is None:
                raise ValueError(
                    "latent_target_mode='safe_uniform_unsafe_center' requires y labels "
                    "in loader batch (dataset should return 4th element y)."
                )
            latent_reg, _, _ = _safe_uniform_unsafe_center_loss(
                v=v,
                y_safe=y_b,
                divergence_fn=divergence_fn,
                divergence_kwargs=divergence_kwargs,
                unsafe_center=float(unsafe_center),
                unsafe_center_weight=float(unsafe_center_weight),
            )
            mmd = float(latent_reg.item())
        elif latent_target_mode_eff == "safe_uniform_unsafe_beta":
            if y_b is None:
                raise ValueError(
                    "latent_target_mode='safe_uniform_unsafe_beta' requires y labels "
                    "in loader batch (dataset should return 4th element y)."
                )
            latent_reg, _, _ = _safe_uniform_unsafe_beta_loss(
                v=v,
                y_safe=y_b,
                divergence_fn=divergence_fn,
                divergence_kwargs=divergence_kwargs,
                unsafe_beta_alpha=float(unsafe_beta_alpha),
                unsafe_beta_beta=float(unsafe_beta_beta),
                unsafe_beta_weight=float(unsafe_beta_weight),
            )
            mmd = float(latent_reg.item())
        else:
            mmd = divergence_fn(v, **divergence_kwargs).item()
        stability_val = None
        if lambda_stability_eff != 0.0:
            if y_b is None:
                raise ValueError(
                    "lambda_stability != 0 requires y labels in loader batch "
                    "(dataset should return 4th element y)."
                )
            if stability_threshold_eff is None:
                raise ValueError("stability_threshold must be provided when lambda_stability != 0")
            logits_safe = _stability_logits_from_o_hat(
                o_hat=o_hat,
                threshold=float(stability_threshold_eff),
                window=stability_window_eff,
                dt=float(stability_dt_eff),
                t0=float(stability_t0_eff),
                smooth_beta=float(stability_smooth_beta_eff),
            )
            stability_val = _stability_bce_loss(
                logits_safe=logits_safe,
                y_safe=y_b,
                safe_weight=float(safe_weight_eff),
                unsafe_weight=float(unsafe_weight_eff),
            ).item()

        bs = x.shape[0]
        total_rec += rec * bs
        total_mmd += mmd * bs
        if stability_val is not None:
            total_stability += stability_val * bs
        n += bs

    rec_avg = total_rec / n
    mmd_avg = total_mmd / n
    stability_avg = (total_stability / n) if lambda_stability_eff != 0.0 else None
    if return_stability:
        return rec_avg, mmd_avg, stability_avg
    return rec_avg, mmd_avg


@torch.no_grad()
def eval_stability_classification(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    stability_threshold: Optional[float] = None,
    stability_window: Optional[Tuple[float, float]] = None,
    stability_dt: float = 1.0,
    stability_t0: float = 0.0,
    stability_smooth_beta: float = 10.0,
    stability_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate class balance and per-class correctness for stability labels.

    Returns a dict with:
      - frac_y1 / frac_y0
      - y1_correct_frac: P(pred=1 | y=1)
      - y0_correct_frac: P(pred=0 | y=0)
      - overall_acc
    """
    (
        _,
        stability_threshold_eff,
        stability_window_eff,
        stability_dt_eff,
        stability_t0_eff,
        stability_smooth_beta_eff,
        _,
        _,
    ) = _resolve_stability_cfg(
        lambda_stability=0.0,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
        stability_dt=stability_dt,
        stability_t0=stability_t0,
        stability_spec=stability_spec,
        stability_smooth_beta=stability_smooth_beta,
    )
    if stability_threshold_eff is None:
        raise ValueError("stability_threshold must be provided (directly or via stability_spec)")

    model.eval()
    n_total = 0
    n_y1 = 0
    n_y0 = 0
    n_correct = 0
    n_y1_correct = 0
    n_y0_correct = 0

    for batch in loader:
        x, o_hist_b, _, y_b = _unpack_seq_batch(batch)
        if y_b is None:
            raise ValueError(
                "stability classification evaluation requires y labels in loader batch."
            )

        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        y_b = y_b.to(device).reshape(-1).float()

        o_hat, _ = model(x, o_hist_b)
        logits_safe = _stability_logits_from_o_hat(
            o_hat=o_hat,
            threshold=float(stability_threshold_eff),
            window=stability_window_eff,
            dt=float(stability_dt_eff),
            t0=float(stability_t0_eff),
            smooth_beta=float(stability_smooth_beta_eff),
        )
        pred_y1 = logits_safe >= 0.0
        y1 = y_b >= 0.5
        y0 = ~y1
        correct = pred_y1 == y1

        n_batch = int(y_b.numel())
        n_total += n_batch
        n_y1_batch = int(y1.sum().item())
        n_y0_batch = int(y0.sum().item())
        n_y1 += n_y1_batch
        n_y0 += n_y0_batch
        n_correct += int(correct.sum().item())
        n_y1_correct += int((pred_y1 & y1).sum().item())
        n_y0_correct += int(((~pred_y1) & y0).sum().item())

    if n_total == 0:
        raise ValueError("loader is empty; cannot compute stability classification metrics")

    return {
        "frac_y1": n_y1 / n_total,
        "frac_y0": n_y0 / n_total,
        "y1_correct_frac": (n_y1_correct / n_y1) if n_y1 > 0 else None,
        "y0_correct_frac": (n_y0_correct / n_y0) if n_y0 > 0 else None,
        "overall_acc": n_correct / n_total,
    }


@torch.no_grad()
def eval_regime_batch_stats(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
    regime_gamma: float = 0.9,
    regime_delta_safe: float = 0.05,
    regime_delta_unsafe: float = 0.2,
) -> Dict[str, Any]:
    """
    Evaluate regime split at batch level using empirical safe rate p_safe = mean(y).

    Returns:
      - num_batches / safe_batches / unsafe_batches
      - frac_safe_batches / frac_unsafe_batches
      - safe_divergences / unsafe_divergences (list[float])
      - safe_div_q50 / safe_div_q90 / unsafe_div_q50 / unsafe_div_q90
      - avg_regime_hinge_loss (with lambda excluded)
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    if not (0.0 <= float(regime_gamma) <= 1.0):
        raise ValueError("regime_gamma must be in [0, 1]")

    model.eval()
    safe_divergences = []
    unsafe_divergences = []
    total_batches = 0
    total_regime_hinge = 0.0

    for batch in loader:
        x, o_hist_b, _, y_b = _unpack_seq_batch(batch)
        if y_b is None:
            raise ValueError("eval_regime_batch_stats requires y labels in loader batch")

        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        y_b = y_b.to(device).reshape(-1).float()

        _, v = model(x, o_hist_b)
        div_val = divergence_fn(v, **divergence_kwargs)
        p_safe = torch.mean(y_b)
        regime_loss = _regime_hinge_loss(
            divergence_val=div_val,
            p_safe=p_safe,
            gamma=float(regime_gamma),
            delta_safe=float(regime_delta_safe),
            delta_unsafe=float(regime_delta_unsafe),
        )

        div_item = float(div_val.item())
        if float(p_safe.item()) >= float(regime_gamma):
            safe_divergences.append(div_item)
        else:
            unsafe_divergences.append(div_item)
        total_regime_hinge += float(regime_loss.item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("loader is empty; cannot compute regime stats")

    safe_n = len(safe_divergences)
    unsafe_n = len(unsafe_divergences)

    def _quantiles(vals):
        if not vals:
            return None, None
        t = torch.tensor(vals, dtype=torch.float32)
        q50 = float(torch.quantile(t, 0.5).item())
        q90 = float(torch.quantile(t, 0.9).item())
        return q50, q90

    safe_q50, safe_q90 = _quantiles(safe_divergences)
    unsafe_q50, unsafe_q90 = _quantiles(unsafe_divergences)

    return {
        "num_batches": total_batches,
        "safe_batches": safe_n,
        "unsafe_batches": unsafe_n,
        "frac_safe_batches": safe_n / total_batches,
        "frac_unsafe_batches": unsafe_n / total_batches,
        "safe_divergences": safe_divergences,
        "unsafe_divergences": unsafe_divergences,
        "safe_div_q50": safe_q50,
        "safe_div_q90": safe_q90,
        "unsafe_div_q50": unsafe_q50,
        "unsafe_div_q90": unsafe_q90,
        "avg_regime_hinge_loss": total_regime_hinge / total_batches,
    }


def eval_epoch_enc(
    loader,
    model_enc: torch.nn.Module,
    device: torch.device,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
) -> float:
    if divergence_kwargs is None:
        divergence_kwargs = {}
    model_enc.eval()
    total_mmd, n = 0.0, 0
    for batch in loader:
        x, _, _, _ = _unpack_seq_batch(batch)
        x = x.to(device)
        v = model_enc(x)
        mmd = divergence_fn(v, **divergence_kwargs).item()
        bs = x.shape[0]
        total_mmd += mmd * bs
        n += bs
    return total_mmd / n if n > 0 else 0.0


def train_eval_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
    val_loader=None,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
    lambda_stability: float = 0.0,
    stability_threshold: Optional[float] = None,
    stability_window: Optional[Tuple[float, float]] = None,
    stability_dt: float = 1.0,
    stability_t0: float = 0.0,
    stability_smooth_beta: float = 10.0,
    stability_spec: Optional[Dict[str, Any]] = None,
    stability_loss_type: str = "bce",
    stability_weight_stab: float = 1.0,
    lambda_regime: float = 0.0,
    regime_gamma: float = 0.9,
    regime_delta_safe: float = 0.05,
    regime_delta_unsafe: float = 0.2,
    return_val_stability: bool = False,
    latent_target_mode: Optional[str] = None,
    unsafe_center: float = 0.5,
    unsafe_center_weight: float = 1.0,
    unsafe_beta_alpha: float = 5.0,
    unsafe_beta_beta: float = 5.0,
    unsafe_beta_weight: float = 1.0,
) -> Tuple[float, Optional[float], Optional[float]] | Tuple[float, Optional[float], Optional[float], Optional[float]]:
    """
    Run one train epoch and optionally evaluate on val_loader.
    Returns (train_loss, val_rec or None, val_mmd or None).
    """
    tr_loss = train_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        lambda_mmd=lambda_mmd,
        device=device,
        divergence_fn=divergence_fn,
        divergence_kwargs=divergence_kwargs,
        lambda_stability=lambda_stability,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
        stability_dt=stability_dt,
        stability_t0=stability_t0,
        stability_smooth_beta=stability_smooth_beta,
        stability_spec=stability_spec,
        stability_loss_type=stability_loss_type,
        stability_weight_stab=stability_weight_stab,
        lambda_regime=lambda_regime,
        regime_gamma=regime_gamma,
        regime_delta_safe=regime_delta_safe,
        regime_delta_unsafe=regime_delta_unsafe,
        latent_target_mode=latent_target_mode,
        unsafe_center=unsafe_center,
        unsafe_center_weight=unsafe_center_weight,
        unsafe_beta_alpha=unsafe_beta_alpha,
        unsafe_beta_beta=unsafe_beta_beta,
        unsafe_beta_weight=unsafe_beta_weight,
    )
    if val_loader is None:
        if return_val_stability:
            return tr_loss, None, None, None
        return tr_loss, None, None
    va_out = eval_epoch(
        loader=val_loader,
        model=model,
        device=device,
        divergence_fn=divergence_fn,
        divergence_kwargs=divergence_kwargs,
        lambda_stability=lambda_stability,
        stability_threshold=stability_threshold,
        stability_window=stability_window,
        stability_dt=stability_dt,
        stability_t0=stability_t0,
        stability_smooth_beta=stability_smooth_beta,
        stability_spec=stability_spec,
        return_stability=return_val_stability,
        latent_target_mode=latent_target_mode,
        unsafe_center=unsafe_center,
        unsafe_center_weight=unsafe_center_weight,
        unsafe_beta_alpha=unsafe_beta_alpha,
        unsafe_beta_beta=unsafe_beta_beta,
        unsafe_beta_weight=unsafe_beta_weight,
    )
    if return_val_stability:
        va_rec, va_mmd, va_stability = va_out  # type: ignore[misc]
        return tr_loss, va_rec, va_mmd, va_stability

    va_rec, va_mmd = va_out  # type: ignore[misc]
    return tr_loss, va_rec, va_mmd


def train_eval_epoch_stability(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
    val_loader=None,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
    stability_threshold: Optional[float] = None,
    stability_spec: Optional[Dict[str, Any]] = None,
    asym_recon_delta: float = 2.0,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Train one epoch with asymmetric stability-weighted reconstruction:
      loss = weighted_recon + lambda_mmd * divergence

    Weighted reconstruction follows the same rule as
    ``stability_asymmetric_recon`` in ``training_clean.py``:
      - build elementwise squared error
      - upweight entries where |target| > threshold and prediction moves
        in the wrong direction, by factor ``asym_recon_delta``.
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    if float(asym_recon_delta) < 1.0:
        raise ValueError("asym_recon_delta must be at least 1.0")

    (
        _,
        stability_threshold_eff,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _resolve_stability_cfg(
        lambda_stability=0.0,
        stability_threshold=stability_threshold,
        stability_window=None,
        stability_dt=1.0,
        stability_t0=0.0,
        stability_spec=stability_spec,
        stability_smooth_beta=10.0,
    )
    if stability_threshold_eff is None:
        raise ValueError(
            "stability_threshold must be provided (directly or via stability_spec) "
            "for train_eval_epoch_stability."
        )

    model.train()
    total_loss, n = 0.0, 0
    for batch in train_loader:
        x, o_hist_b, o_next_b, _ = _unpack_seq_batch(batch)
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)

        o_hat, v = model(x, o_hist_b)
        mmd = divergence_fn(v, **divergence_kwargs)

        sq_err = (o_hat - o_next_b) ** 2
        unstable = torch.abs(o_next_b) > float(stability_threshold_eff)
        wrong_direction = torch.sign(o_next_b) * o_hat < torch.sign(o_next_b) * o_next_b
        rec_weights = torch.where(
            unstable & wrong_direction,
            torch.full_like(o_next_b, float(asym_recon_delta)),
            torch.ones_like(o_next_b),
        )
        rec = torch.mean(rec_weights * sq_err)

        loss = rec + float(lambda_mmd) * mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        n += bs

    tr_loss = total_loss / n

    if val_loader is None:
        return tr_loss, None, None

    va_rec, va_mmd = eval_epoch(
        loader=val_loader,
        model=model,
        device=device,
        divergence_fn=divergence_fn,
        divergence_kwargs=divergence_kwargs,
    )
    return tr_loss, va_rec, va_mmd


def train_eval_epoch_enc_only(
    model_enc: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
    val_loader=None,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
) -> Tuple[float, Optional[float]]:
    """
    Run one train epoch for encoder-only divergence and optionally eval on val_loader.
    Returns (train_divergence, val_divergence or None).
    """
    tr_div = train_epoch_mmd_only(
        model_enc=model_enc,
        loader=train_loader,
        optimizer=optimizer,
        lambda_mmd=lambda_mmd,
        device=device,
        divergence_fn=divergence_fn,
        divergence_kwargs=divergence_kwargs,
    )
    if val_loader is None:
        return tr_div, None
    va_div = eval_epoch_enc(
        loader=val_loader,
        model_enc=model_enc,
        device=device,
        divergence_fn=divergence_fn,
        divergence_kwargs=divergence_kwargs,
    )
    return tr_div, va_div
