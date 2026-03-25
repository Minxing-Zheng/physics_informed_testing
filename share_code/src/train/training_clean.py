"""
Cleanly structured training utilities for conditional GRU models.

This module keeps the functionality of ``training.py`` but replaces long,
+mode-dependent function signatures with typed config objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from train.losses import mmd2_unif


LossFn = Callable[[torch.Tensor], torch.Tensor]
Window = Optional[Tuple[float, float]]


@dataclass(frozen=True)
class SequenceBatch:
    x: torch.Tensor
    o_hist: torch.Tensor
    o_next: torch.Tensor
    y: Optional[torch.Tensor] = None

    @classmethod
    def from_loader_batch(cls, batch: Any, device: torch.device) -> "SequenceBatch":
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError("Expected batch to be a tuple/list with at least 3 elements")

        y = batch[3] if len(batch) >= 4 else None
        x = batch[0].to(device)
        o_hist = batch[1].to(device)
        o_next = batch[2].to(device)
        y_tensor = None if y is None else y.to(device).reshape(-1).float()
        return cls(x=x, o_hist=o_hist, o_next=o_next, y=y_tensor)

    @property
    def batch_size(self) -> int:
        return int(self.x.shape[0])


@dataclass(frozen=True)
class DivergenceConfig:
    lambda_mmd: float
    fn: LossFn = mmd2_unif
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StabilityConfig:
    lambda_stability: float = 0.0
    threshold: Optional[float] = None
    window: Window = None
    dt: float = 1.0
    t0: float = 0.0
    smooth_beta: float = 10.0
    loss_type: str = "bce"
    weight_stab: float = 1.0
    asym_recon_delta: float = 2.0
    safe_weight: float = 1.0
    unsafe_weight: float = 1.0

    @classmethod
    def from_legacy(
        cls,
        *,
        lambda_stability: float = 0.0,
        stability_threshold: Optional[float] = None,
        stability_window: Window = None,
        stability_dt: float = 1.0,
        stability_t0: float = 0.0,
        stability_smooth_beta: float = 10.0,
        stability_spec: Optional[Dict[str, Any]] = None,
        stability_loss_type: str = "bce",
        stability_weight_stab: float = 1.0,
    ) -> "StabilityConfig":
        threshold = stability_threshold
        window = stability_window
        dt = float(stability_dt)
        t0 = float(stability_t0)
        smooth_beta = float(stability_smooth_beta)
        lam = float(lambda_stability)
        safe_weight = 1.0
        unsafe_weight = 1.0
        weight_stab = float(stability_weight_stab)
        asym_recon_delta = 2.0

        if stability_spec is not None:
            if not isinstance(stability_spec, dict):
                raise ValueError("stability_spec must be a dict when provided")

            lam = float(stability_spec.get("lambda", lam))
            dt = float(stability_spec.get("dt", dt))
            t0 = float(stability_spec.get("t0", t0))
            smooth_beta = float(stability_spec.get("smooth_beta", smooth_beta))
            safe_weight = float(stability_spec.get("safe_weight", safe_weight))
            unsafe_weight = float(stability_spec.get("unsafe_weight", unsafe_weight))
            weight_stab = float(stability_spec.get("weight_stab", weight_stab))
            asym_recon_delta = float(
                stability_spec.get("asym_recon_delta", asym_recon_delta)
            )

            if "threshold" in stability_spec:
                threshold = float(stability_spec["threshold"])
            if "window" in stability_spec and stability_spec["window"] is not None:
                win = stability_spec["window"]
                window = (float(win[0]), float(win[1]))

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
                        "stability_spec['fn_kwargs'] must include 'threshold' and "
                        "'window' for check_safety."
                    )
                threshold = float(fn_kwargs["threshold"])
                win = fn_kwargs["window"]
                window = (float(win[0]), float(win[1]))

        cfg = cls(
            lambda_stability=lam,
            threshold=threshold,
            window=window,
            dt=dt,
            t0=t0,
            smooth_beta=smooth_beta,
            loss_type=stability_loss_type,
            weight_stab=weight_stab,
            asym_recon_delta=asym_recon_delta,
            safe_weight=safe_weight,
            unsafe_weight=unsafe_weight,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        loss_mode = self.loss_type.lower()
        if loss_mode not in {"bce", "weighted", "weighted_bce", "stability_asymmetric_recon"}:
            raise ValueError(
                f"Unsupported stability loss_type='{self.loss_type}'. "
                "Use 'bce', 'weighted', 'weighted_bce', or 'stability_asymmetric_recon'."
            )
        if self.dt <= 0:
            raise ValueError("stability dt must be positive")
        if self.threshold is not None and self.threshold < 0:
            raise ValueError("stability threshold must be non-negative")
        if self.window is not None and float(self.window[0]) > float(self.window[1]):
            raise ValueError(f"window must satisfy t_l <= t_u, got {self.window}")
        if self.weight_stab < 0:
            raise ValueError("stability weight_stab must be non-negative")
        if self.asym_recon_delta < 1.0:
            raise ValueError("asym_recon_delta must be at least 1.0")
        if self.safe_weight <= 0 or self.unsafe_weight <= 0:
            raise ValueError("safe_weight and unsafe_weight must be positive")
        if loss_mode == "weighted" and self.lambda_stability != 0.0:
            raise ValueError(
                "loss_type='weighted' uses reweighted reconstruction; "
                "set lambda_stability=0 or use 'bce'/'weighted_bce'."
            )

    @property
    def enabled(self) -> bool:
        return float(self.lambda_stability) != 0.0

    @property
    def needs_labels(self) -> bool:
        return self.loss_type.lower() == "weighted" or self.enabled


@dataclass(frozen=True)
class RegimeConfig:
    lambda_regime: float = 0.0
    gamma: float = 0.9
    delta_safe: float = 0.05
    delta_unsafe: float = 0.2

    def validate(self) -> None:
        if not (0.0 <= float(self.gamma) <= 1.0):
            raise ValueError("regime gamma must be in [0, 1]")
        if float(self.delta_safe) < 0 or float(self.delta_unsafe) < 0:
            raise ValueError("regime deltas must be non-negative")

    @property
    def enabled(self) -> bool:
        return float(self.lambda_regime) != 0.0


@dataclass(frozen=True)
class TrainingConfig:
    divergence: DivergenceConfig
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    def validate(self) -> None:
        self.stability.validate()
        self.regime.validate()


def _stability_logits_from_o_hat(
    o_hat: torch.Tensor,
    threshold: float,
    window: Window,
    dt: float,
    t0: float,
    smooth_beta: float = 10.0,
) -> torch.Tensor:
    if o_hat.ndim != 3:
        raise ValueError(f"o_hat must have shape (B, L, d_o), got {tuple(o_hat.shape)}")

    bsz, seq_len, _ = o_hat.shape
    t = t0 + dt * torch.arange(seq_len, device=o_hat.device, dtype=o_hat.dtype)

    if window is None:
        mask = torch.ones(seq_len, dtype=torch.bool, device=o_hat.device)
    else:
        t_l, t_u = float(window[0]), float(window[1])
        mask = (t >= t_l) & (t <= t_u)
        if not torch.any(mask):
            raise ValueError(
                f"stability window {window} has no overlap with sequence time "
                f"range [{float(t[0]):.4f}, {float(t[-1]):.4f}]"
            )

    peak_over_dims = torch.abs(o_hat).amax(dim=-1)
    vals = peak_over_dims[:, mask]
    if smooth_beta is not None and float(smooth_beta) > 0:
        beta = float(smooth_beta)
        k = vals.shape[1]
        peak = (
            torch.logsumexp(beta * vals, dim=1)
            - torch.log(torch.tensor(float(k), device=vals.device, dtype=vals.dtype))
        ) / beta
    else:
        peak = vals.amax(dim=1)

    logits = float(threshold) - peak
    if logits.shape != (bsz,):
        raise RuntimeError("unexpected stability logits shape")
    return logits


def _weighted_bce_loss(
    logits_safe: torch.Tensor,
    y_safe: torch.Tensor,
    safe_weight: float,
    unsafe_weight: float,
) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits_safe, y_safe, reduction="none")
    weights = float(safe_weight) * y_safe + float(unsafe_weight) * (1.0 - y_safe)
    return torch.mean(per * weights)


def _stability_loss(
    logits_safe: torch.Tensor,
    y_safe: torch.Tensor,
    cfg: StabilityConfig,
) -> torch.Tensor:
    loss_mode = cfg.loss_type.lower()
    if loss_mode == "bce":
        return F.binary_cross_entropy_with_logits(logits_safe, y_safe, reduction="mean")
    if loss_mode == "weighted_bce":
        return _weighted_bce_loss(
            logits_safe=logits_safe,
            y_safe=y_safe,
            safe_weight=cfg.safe_weight,
            unsafe_weight=cfg.unsafe_weight,
        )
    raise ValueError(f"Unsupported stability loss_type='{cfg.loss_type}' for _stability_loss.")


def _regime_hinge_loss(
    divergence_val: torch.Tensor,
    p_safe: torch.Tensor,
    cfg: RegimeConfig,
) -> torch.Tensor:
    safe_loss = F.relu(divergence_val - float(cfg.delta_safe))
    unsafe_loss = F.relu(float(cfg.delta_unsafe) - divergence_val)
    is_safe_regime = (p_safe >= float(cfg.gamma)).to(divergence_val.dtype)
    return is_safe_regime * safe_loss + (1.0 - is_safe_regime) * unsafe_loss


def _require_labels(batch: SequenceBatch, reason: str) -> torch.Tensor:
    if batch.y is None:
        raise ValueError(reason)
    return batch.y


def _compute_stability_logits(
    o_hat: torch.Tensor,
    batch: SequenceBatch,
    cfg: StabilityConfig,
) -> Optional[torch.Tensor]:
    if not cfg.needs_labels:
        return None
    _require_labels(
        batch,
        "stability supervision requires y labels in loader batch "
        "(dataset should return 4th element y).",
    )
    if cfg.threshold is None:
        raise ValueError(
            "stability threshold must be provided when using stability supervision"
        )
    logits = _stability_logits_from_o_hat(
        o_hat=o_hat,
        threshold=float(cfg.threshold),
        window=cfg.window,
        dt=float(cfg.dt),
        t0=float(cfg.t0),
        smooth_beta=float(cfg.smooth_beta),
    )
    if batch.y is not None and logits.shape[0] != batch.y.shape[0]:
        raise ValueError(
            f"stability logits and y shape mismatch: {tuple(logits.shape)} vs "
            f"{tuple(batch.y.shape)}"
        )
    return logits


def _reconstruction_loss(
    o_hat: torch.Tensor,
    target: torch.Tensor,
    logits_safe: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    cfg: StabilityConfig,
) -> torch.Tensor:
    loss_mode = cfg.loss_type.lower()
    if loss_mode == "stability_asymmetric_recon":
        if cfg.threshold is None:
            raise ValueError(
                "stability threshold must be provided for stability_asymmetric_recon"
            )
        sq_err = (o_hat - target) ** 2
        unstable = torch.abs(target) > float(cfg.threshold)
        wrong_direction = torch.sign(target) * o_hat < torch.sign(target) * target
        weights = torch.where(
            unstable & wrong_direction,
            torch.full_like(target, float(cfg.asym_recon_delta)),
            torch.ones_like(target),
        )
        return torch.mean(weights * sq_err)

    if loss_mode != "weighted":
        return F.mse_loss(o_hat, target)

    if logits_safe is None or y is None:
        raise ValueError("weighted stability reconstruction requires logits and labels")

    rec_per_sample = torch.mean((o_hat - target) ** 2, dim=(1, 2))
    y_pred = logits_safe >= 0.0
    y_true = y >= 0.5
    misclassified = (y_pred != y_true).to(rec_per_sample.dtype)
    rec_weights = 1.0 + float(cfg.weight_stab) * misclassified
    return torch.mean(rec_weights * rec_per_sample)


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    config.validate()
    model.train()

    total_loss = 0.0
    n = 0

    for raw_batch in loader:
        batch = SequenceBatch.from_loader_batch(raw_batch, device)
        o_hat, v = model(batch.x, batch.o_hist)
        mmd = config.divergence.fn(v, **config.divergence.kwargs)
        logits_safe = _compute_stability_logits(o_hat, batch, config.stability)

        rec = _reconstruction_loss(
            o_hat=o_hat,
            target=batch.o_next,
            logits_safe=logits_safe,
            y=batch.y,
            cfg=config.stability,
        )
        loss = rec + float(config.divergence.lambda_mmd) * mmd

        if config.stability.enabled:
            y = _require_labels(
                batch,
                "lambda_stability != 0 requires y labels in loader batch "
                "(dataset should return 4th element y).",
            )
            loss = loss + float(config.stability.lambda_stability) * _stability_loss(
                logits_safe=logits_safe,  # type: ignore[arg-type]
                y_safe=y,
                cfg=config.stability,
            )

        if config.regime.enabled:
            y = _require_labels(
                batch,
                "lambda_regime != 0 requires y labels in loader batch "
                "(dataset should return 4th element y).",
            )
            p_safe = torch.mean(y)
            loss = loss + float(config.regime.lambda_regime) * _regime_hinge_loss(
                divergence_val=mmd,
                p_safe=p_safe,
                cfg=config.regime,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch.batch_size
        n += batch.batch_size

    return total_loss / n


def train_epoch_mmd_only(
    model_enc: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    divergence: DivergenceConfig,
) -> float:
    model_enc.train()
    total_div = 0.0
    n = 0

    for raw_batch in loader:
        batch = SequenceBatch.from_loader_batch(raw_batch, device)
        v = model_enc(batch.x)
        div_val = divergence.fn(v, **divergence.kwargs)

        optimizer.zero_grad()
        div_val.backward()
        optimizer.step()

        total_div += float(div_val.item()) * batch.batch_size
        n += batch.batch_size

    return total_div / n if n > 0 else 0.0


@torch.no_grad()
def eval_epoch(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    config: TrainingConfig,
    return_stability: bool = False,
) -> Tuple[float, float] | Tuple[float, float, Optional[float]]:
    config.validate()
    model.eval()

    total_rec = 0.0
    total_mmd = 0.0
    total_stability = 0.0
    n = 0

    for raw_batch in loader:
        batch = SequenceBatch.from_loader_batch(raw_batch, device)
        o_hat, v = model(batch.x, batch.o_hist)
        rec = F.mse_loss(o_hat, batch.o_next, reduction="mean").item()
        mmd = config.divergence.fn(v, **config.divergence.kwargs).item()
        stability_val = None

        if config.stability.enabled:
            y = _require_labels(
                batch,
                "lambda_stability != 0 requires y labels in loader batch "
                "(dataset should return 4th element y).",
            )
            if config.stability.threshold is None:
                raise ValueError(
                    "stability threshold must be provided when lambda_stability != 0"
                )
            logits_safe = _stability_logits_from_o_hat(
                o_hat=o_hat,
                threshold=float(config.stability.threshold),
                window=config.stability.window,
                dt=float(config.stability.dt),
                t0=float(config.stability.t0),
                smooth_beta=float(config.stability.smooth_beta),
            )
            stability_val = _weighted_bce_loss(
                logits_safe=logits_safe,
                y_safe=y,
                safe_weight=float(config.stability.safe_weight),
                unsafe_weight=float(config.stability.unsafe_weight),
            ).item()

        total_rec += rec * batch.batch_size
        total_mmd += mmd * batch.batch_size
        if stability_val is not None:
            total_stability += stability_val * batch.batch_size
        n += batch.batch_size

    rec_avg = total_rec / n
    mmd_avg = total_mmd / n
    stability_avg = (total_stability / n) if config.stability.enabled else None
    if return_stability:
        return rec_avg, mmd_avg, stability_avg
    return rec_avg, mmd_avg


@torch.no_grad()
def eval_stability_classification(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    stability: StabilityConfig,
) -> Dict[str, Optional[float]]:
    stability.validate()
    if stability.threshold is None:
        raise ValueError("stability threshold must be provided")

    model.eval()
    n_total = 0
    n_y1 = 0
    n_y0 = 0
    n_correct = 0
    n_y1_correct = 0
    n_y0_correct = 0

    for raw_batch in loader:
        batch = SequenceBatch.from_loader_batch(raw_batch, device)
        y = _require_labels(
            batch,
            "stability classification evaluation requires y labels in loader batch.",
        )
        o_hat, _ = model(batch.x, batch.o_hist)
        logits_safe = _stability_logits_from_o_hat(
            o_hat=o_hat,
            threshold=float(stability.threshold),
            window=stability.window,
            dt=float(stability.dt),
            t0=float(stability.t0),
            smooth_beta=float(stability.smooth_beta),
        )
        pred_y1 = logits_safe >= 0.0
        y1 = y >= 0.5
        y0 = ~y1
        correct = pred_y1 == y1

        n_batch = int(y.numel())
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
    divergence: DivergenceConfig,
    regime: RegimeConfig,
) -> Dict[str, Any]:
    regime.validate()
    model.eval()

    safe_divergences = []
    unsafe_divergences = []
    total_batches = 0
    total_regime_hinge = 0.0

    for raw_batch in loader:
        batch = SequenceBatch.from_loader_batch(raw_batch, device)
        y = _require_labels(batch, "eval_regime_batch_stats requires y labels in loader batch")
        _, v = model(batch.x, batch.o_hist)
        div_val = divergence.fn(v, **divergence.kwargs)
        p_safe = torch.mean(y)
        regime_loss = _regime_hinge_loss(
            divergence_val=div_val,
            p_safe=p_safe,
            cfg=regime,
        )

        div_item = float(div_val.item())
        if float(p_safe.item()) >= float(regime.gamma):
            safe_divergences.append(div_item)
        else:
            unsafe_divergences.append(div_item)
        total_regime_hinge += float(regime_loss.item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("loader is empty; cannot compute regime stats")

    def _quantiles(values: list[float]) -> tuple[Optional[float], Optional[float]]:
        if not values:
            return None, None
        tensor = torch.tensor(values, dtype=torch.float32)
        q50 = float(torch.quantile(tensor, 0.5).item())
        q90 = float(torch.quantile(tensor, 0.9).item())
        return q50, q90

    safe_q50, safe_q90 = _quantiles(safe_divergences)
    unsafe_q50, unsafe_q90 = _quantiles(unsafe_divergences)

    safe_n = len(safe_divergences)
    unsafe_n = len(unsafe_divergences)
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


@torch.no_grad()
def eval_epoch_enc(
    loader,
    model_enc: torch.nn.Module,
    device: torch.device,
    divergence: DivergenceConfig,
) -> float:
    model_enc.eval()
    total_mmd = 0.0
    n = 0

    for raw_batch in loader:
        batch = SequenceBatch.from_loader_batch(raw_batch, device)
        mmd = divergence.fn(model_enc(batch.x), **divergence.kwargs).item()
        total_mmd += mmd * batch.batch_size
        n += batch.batch_size

    return total_mmd / n if n > 0 else 0.0


def train_eval_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainingConfig,
    val_loader=None,
    return_val_stability: bool = False,
) -> Tuple[float, Optional[float], Optional[float]] | Tuple[float, Optional[float], Optional[float], Optional[float]]:
    train_loss = train_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        config=config,
    )

    if val_loader is None:
        if return_val_stability:
            return train_loss, None, None, None
        return train_loss, None, None

    val_out = eval_epoch(
        loader=val_loader,
        model=model,
        device=device,
        config=config,
        return_stability=return_val_stability,
    )
    if return_val_stability:
        val_rec, val_mmd, val_stability = val_out  # type: ignore[misc]
        return train_loss, val_rec, val_mmd, val_stability

    val_rec, val_mmd = val_out  # type: ignore[misc]
    return train_loss, val_rec, val_mmd


def train_eval_epoch_enc_only(
    model_enc: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    divergence: DivergenceConfig,
    val_loader=None,
) -> Tuple[float, Optional[float]]:
    train_div = train_epoch_mmd_only(
        model_enc=model_enc,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        divergence=divergence,
    )
    if val_loader is None:
        return train_div, None

    val_div = eval_epoch_enc(
        loader=val_loader,
        model_enc=model_enc,
        device=device,
        divergence=divergence,
    )
    return train_div, val_div


__all__ = [
    "DivergenceConfig",
    "RegimeConfig",
    "SequenceBatch",
    "StabilityConfig",
    "TrainingConfig",
    "eval_epoch",
    "eval_epoch_enc",
    "eval_regime_batch_stats",
    "eval_stability_classification",
    "train_epoch",
    "train_epoch_mmd_only",
    "train_eval_epoch",
    "train_eval_epoch_enc_only",
]
