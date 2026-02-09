"""
Training utilities for conditional GRU models.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from train.losses import mmd2_unif


def train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    lambda_mmd: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, o_hist_b, o_next_b in loader:
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)

        o_hat, v = model(x, o_hist_b)
        rec = F.mse_loss(o_hat, o_next_b)
        mmd = mmd2_unif(v)
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
) -> float:
    model_enc.train()
    total_loss, n = 0.0, 0
    for x, _, _ in loader:
        x = x.to(device)
        v = model_enc(x)
        mmd = mmd2_unif(v)
        loss = lambda_mmd * mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


@torch.no_grad()
def eval_epoch(loader, model: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_rec, total_mmd, n = 0.0, 0.0, 0
    for x, o_hist_b, o_next_b in loader:
        x = x.to(device)
        o_hist_b = o_hist_b.to(device)
        o_next_b = o_next_b.to(device)

        o_hat, v = model(x, o_hist_b)
        rec = F.mse_loss(o_hat, o_next_b, reduction="mean").item()
        mmd = mmd2_unif(v).item()

        bs = x.shape[0]
        total_rec += rec * bs
        total_mmd += mmd * bs
        n += bs
    return total_rec / n, total_mmd / n


@torch.no_grad()
def eval_epoch_enc(loader, model_enc: torch.nn.Module, device: torch.device) -> float:
    model_enc.eval()
    total_mmd, n = 0.0, 0
    for x, _, _ in loader:
        x = x.to(device)
        v = model_enc(x)
        mmd = mmd2_unif(v).item()
        bs = x.shape[0]
        total_mmd += mmd * bs
        n += bs
    return total_mmd / n
