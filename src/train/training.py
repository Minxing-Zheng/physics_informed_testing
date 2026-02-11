"""
Training utilities for conditional GRU models.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from train.losses import mmd2_unif


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
    divergence_fn: Callable[[torch.Tensor], torch.Tensor] = mmd2_unif,
    divergence_kwargs: Optional[Dict] = None,
) -> float:
    """
    Train conditional model for one epoch with a configurable divergence.
    """
    if divergence_kwargs is None:
        divergence_kwargs = {}
    model.train()
    total_loss, n = 0.0, 0
    for x, o_hist_b, o_next_b in loader:
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)

        o_hat, v = model(x, o_hist_b)
        rec = F.mse_loss(o_hat, o_next_b)
        mmd = divergence_fn(v, **divergence_kwargs)
        loss = rec + lambda_mmd * mmd

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
    for x, _, _ in loader:
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
) -> Tuple[float, float]:
    if divergence_kwargs is None:
        divergence_kwargs = {}
    model.eval()
    total_rec, total_mmd, n = 0.0, 0.0, 0
    for x, o_hist_b, o_next_b in loader:
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)

        o_hat, v = model(x, o_hist_b)
        rec = F.mse_loss(o_hat, o_next_b, reduction="mean").item()
        mmd = divergence_fn(v, **divergence_kwargs).item()

        bs = x.shape[0]
        total_rec += rec * bs
        total_mmd += mmd * bs
        n += bs
    return total_rec / n, total_mmd / n


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
    for x, _, _ in loader:
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
) -> Tuple[float, Optional[float], Optional[float]]:
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
    )
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
