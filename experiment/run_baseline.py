#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.datasets import SpringDataset, make_sequence_pairs
from data.sample_x import sample_x
from data.solve_ode import SpringParams, simulate_spring
from data.utils import check_safety
from hypo_test.stats_tests import ks_2sample_pvalue
from train.eval import encode_to_latent
from train.losses import mmd2_unif
from train.models import EncoderOnly, build_cond_gru_model
from train.training_clean import (
    DivergenceConfig,
    StabilityConfig,
    TrainingConfig,
    eval_epoch,
    eval_stability_classification,
    train_eval_epoch,
    train_eval_epoch_enc_only,
)


def _fmt_float(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline-extended diagnostics for the physics-informed hypothesis test."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=5000, dest="n_samples")
    parser.add_argument("--test-n-samples", type=int, default=2000, dest="test_n_samples")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--perturb_type", type=str, default="mean", dest="perturb_type")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="stability_asymmetric_recon", # "stability_asymmetric_recon"
        dest="loss_type",
    )
    parser.add_argument(
        "--dist-mean",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("MU_X", "MU_Y"),
        dest="dist_mean",
    )
    parser.add_argument(
        "--dist-cov-diag",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        metavar=("VAR_X", "VAR_Y"),
        dest="dist_cov_diag",
    )
    parser.add_argument("--batch-train", type=int, default=1024, dest="batch_train")
    parser.add_argument("--batch-val", type=int, default=512, dest="batch_val")
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    parser.add_argument("--latent-batch-size", type=int, default=4096, dest="latent_batch_size")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5, dest="weight_decay")
    parser.add_argument("--lr-enc", type=float, default=3e-3, dest="lr_enc")
    parser.add_argument(
        "--weight-decay-enc",
        type=float,
        default=1e-5,
        dest="weight_decay_enc",
    )
    parser.add_argument("--lambda-mmd", type=float, default=1e-2, dest="lambda_mmd")
    parser.add_argument("--d-x", type=int, default=2, dest="d_x")
    parser.add_argument("--d-o", type=int, default=1, dest="d_o")
    parser.add_argument("--d-v", type=int, default=1, dest="d_v")
    parser.add_argument("--d-h", type=int, default=128, dest="d_h")
    parser.add_argument("--hidden-enc", type=int, default=512, dest="hidden_enc")
    parser.add_argument("--hidden-dec", type=int, default=128, dest="hidden_dec")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--asym-recon-delta", type=float, default=10.0, dest="asym_recon_delta")
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        default=[6.0, 10.0],
        metavar=("T_START", "T_END"),
        dest="time_window",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            "results/diagnostics_results_baselines/"
            f"seed_{args.seed}"
            f"_lr_{_fmt_float(args.lr)}"
            f"_lambda_mmd_{_fmt_float(args.lambda_mmd)}"
            f"_loss_type_{args.loss_type}"
            f"_tau_{_fmt_float(args.tau)}"
            f"_perturb_type_{args.perturb_type}"
            f"_epochs_{args.epochs}"
            f"_n_{args.n_samples}"
        )

    args.dist_mean = np.array(args.dist_mean, dtype=float)
    args.dist_cov = np.diag(np.array(args.dist_cov_diag, dtype=float))
    args.time_window = tuple(float(v) for v in args.time_window)
    return args


class MLPBinaryClassifier(nn.Module):
    def __init__(
        self,
        d_x: int = 2,
        hidden_size: int = 128,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers: list[nn.Module] = [nn.Linear(d_x, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _choose_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    return device


def _make_loader_kwargs(device: torch.device, num_workers: int) -> dict:
    kwargs = {
        "num_workers": max(int(num_workers), 0),
        "pin_memory": device.type == "cuda",
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def _make_sequence_train_config(
    *,
    lambda_mmd: float,
    loss_type: str,
    tau: float,
    time_window: tuple[float, float],
    sequence_dt: float,
    asym_recon_delta: float,
) -> TrainingConfig:
    div_fn = mmd2_unif
    div_kwargs = {"sigmas": None, "low_discrepancy": True}
    return TrainingConfig(
        divergence=DivergenceConfig(lambda_mmd=lambda_mmd, fn=div_fn, kwargs=div_kwargs),
        stability=StabilityConfig(
            loss_type=loss_type,
            threshold=tau,
            window=time_window,
            dt=sequence_dt,
            t0=sequence_dt,
            asym_recon_delta=asym_recon_delta,
        ),
    )


def _build_sequence_model(cfg: argparse.Namespace, device: torch.device) -> nn.Module:
    return build_cond_gru_model(
        d_x=cfg.d_x,
        d_o=cfg.d_o,
        d_v=cfg.d_v,
        d_h=cfg.d_h,
        enc_layers=2,
        dec_layers=1,
        hidden_enc=cfg.hidden_enc,
        hidden_dec=cfg.hidden_dec,
        output_activation="sigmoid",
        dropout_rnn=0.1,
    ).to(device)


def _scalar_or_none(value):
    if value is None:
        return None
    return float(value)


def evaluate_sequence_split_metrics(
    *,
    model: nn.Module,
    loader,
    split_name: str,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float | None]:
    rec, div = eval_epoch(
        loader=loader,
        model=model,
        device=device,
        config=config,
    )
    total = rec + float(config.divergence.lambda_mmd) * div
    metrics: dict[str, float | None] = {
        f"{split_name}_recon": float(rec),
        f"{split_name}_divergence": float(div),
        f"{split_name}_total_loss": float(total),
    }

    if config.stability.threshold is not None:
        cls = eval_stability_classification(
            loader=loader,
            model=model,
            device=device,
            stability=config.stability,
        )
        metrics.update(
            {
                f"{split_name}_safety_acc": _scalar_or_none(cls["overall_acc"]),
                f"{split_name}_safety_y1_acc": _scalar_or_none(cls["y1_correct_frac"]),
                f"{split_name}_safety_y0_acc": _scalar_or_none(cls["y0_correct_frac"]),
                f"{split_name}_frac_y1": _scalar_or_none(cls["frac_y1"]),
                f"{split_name}_frac_y0": _scalar_or_none(cls["frac_y0"]),
            }
        )
    return metrics


@torch.no_grad()
def build_reconstruction_snapshot(
    *,
    model: nn.Module,
    loader,
    split_name: str,
    device: torch.device,
    max_samples: int = 8,
) -> pd.DataFrame:
    batch = next(iter(loader))
    x_b, o_hist_b, o_next_b, y_b = batch
    non_blocking = device.type == "cuda"
    x_b = x_b[:max_samples].to(device, non_blocking=non_blocking)
    o_hist_b = o_hist_b[:max_samples].to(device, non_blocking=non_blocking)
    o_next_b = o_next_b[:max_samples].to(device, non_blocking=non_blocking)
    y_slice = None if y_b is None else y_b[:max_samples]

    model.eval()
    o_hat_b, _ = model(x_b, o_hist_b)
    per_sample_mse = torch.mean((o_hat_b - o_next_b) ** 2, dim=(1, 2))

    snapshot = pd.DataFrame(
        {
            "split": split_name,
            "sample_idx": np.arange(x_b.shape[0]),
            "recon_mse": per_sample_mse.detach().cpu().numpy(),
            "x": list(x_b.detach().cpu().numpy()),
            "o_hist": list(o_hist_b.detach().cpu().numpy()),
            "o_next_true": list(o_next_b.detach().cpu().numpy()),
            "o_next_pred": list(o_hat_b.detach().cpu().numpy()),
        }
    )
    if y_slice is not None:
        snapshot["y"] = list(np.asarray(y_slice[: x_b.shape[0]]))
    return snapshot


def train_sequence_model(
    *,
    model_name: str,
    cfg: argparse.Namespace,
    train_loader,
    val_loader,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    model = _build_sequence_model(cfg, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    use_val = val_loader is not None
    best_state = None
    best_metric = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, val_rec, val_div = train_eval_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            val_loader=val_loader if use_val else None,
        )

        train_metrics = evaluate_sequence_split_metrics(
            model=model,
            loader=train_loader,
            split_name="train",
            device=device,
            config=config,
        )
        if use_val:
            val_metrics = evaluate_sequence_split_metrics(
                model=model,
                loader=val_loader,
                split_name="val",
                device=device,
                config=config,
            )
        else:
            val_metrics = {
                "val_recon": None,
                "val_divergence": None,
                "val_total_loss": None,
                "val_safety_acc": None,
                "val_safety_y1_acc": None,
                "val_safety_y0_acc": None,
                "val_frac_y1": None,
                "val_frac_y0": None,
            }

        record = {
            "model_name": model_name,
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": (
                None
                if val_rec is None or val_div is None
                else float(val_rec + float(config.divergence.lambda_mmd) * val_div)
            ),
            "val_recon_raw": _scalar_or_none(val_rec),
            "val_divergence_raw": _scalar_or_none(val_div),
            **train_metrics,
            **val_metrics,
        }
        history.append(record)

        current_metric = (
            record["val_loss"]
            if use_val and record["val_loss"] is not None
            else train_metrics["train_total_loss"]
        )
        if current_metric is not None and float(current_metric) < best_metric:
            best_metric = float(current_metric)
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            msg = (
                f"[{model_name}][{epoch:02d}] "
                f"train-loss={train_loss:.6f} "
                f"train-recon={float(train_metrics['train_recon']):.6f} "
                f"train-div={float(train_metrics['train_divergence']):.6f}"
            )
            if train_metrics.get("train_safety_acc") is not None:
                msg += f" train-acc={float(train_metrics['train_safety_acc']):.4f}"
            if use_val:
                msg += (
                    f" val-recon={float(val_metrics['val_recon']):.6f}"
                    f" val-div={float(val_metrics['val_divergence']):.6f}"
                )
                if val_metrics.get("val_safety_acc") is not None:
                    msg += f" val-acc={float(val_metrics['val_safety_acc']):.4f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


def train_regularization_only_model(
    *,
    cfg: argparse.Namespace,
    train_loader,
    val_loader,
    device: torch.device,
    divergence: DivergenceConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    base_model = _build_sequence_model(cfg, device)
    model_enc = EncoderOnly(copy.deepcopy(base_model.enc)).to(device)
    optimizer = torch.optim.Adam(
        model_enc.parameters(),
        lr=cfg.lr_enc,
        weight_decay=cfg.weight_decay_enc,
    )

    use_val = val_loader is not None
    best_state = None
    best_metric = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_div, val_div = train_eval_epoch_enc_only(
            model_enc=model_enc,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            divergence=divergence,
            val_loader=val_loader if use_val else None,
        )

        record = {
            "model_name": "regularization_only",
            "epoch": epoch,
            "train_divergence": float(train_div),
            "val_divergence": _scalar_or_none(val_div),
        }
        history.append(record)

        current_metric = val_div if use_val and val_div is not None else train_div
        if float(current_metric) < best_metric:
            best_metric = float(current_metric)
            best_state = copy.deepcopy(model_enc.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            msg = f"[regularization_only][{epoch:02d}] train-div={train_div:.6f}"
            if use_val and val_div is not None:
                msg += f" val-div={val_div:.6f}"
            print(msg)

    if best_state is not None:
        model_enc.load_state_dict(best_state)

    return model_enc, pd.DataFrame(history)


def _unpack_xy_batch(raw_batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(raw_batch, (tuple, list)) or len(raw_batch) < 4:
        raise ValueError("Label baseline requires loader batches to include y labels")
    x = raw_batch[0].to(device)
    y = raw_batch[3].to(device).reshape(-1).float()
    return x, y


def evaluate_label_classifier(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    total_prob = 0.0

    with torch.inference_mode():
        for raw_batch in loader:
            x, y = _unpack_xy_batch(raw_batch, device)
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            probs = torch.sigmoid(logits)
            preds = probs >= 0.5
            total_loss += float(loss.item()) * int(y.numel())
            total_correct += int((preds == (y >= 0.5)).sum().item())
            total_prob += float(probs.sum().item())
            total += int(y.numel())

    if total == 0:
        raise ValueError("loader is empty")

    return {
        "loss": total_loss / total,
        "acc": total_correct / total,
        "mean_prob": total_prob / total,
    }


def train_label_classifier(
    *,
    cfg: argparse.Namespace,
    train_loader,
    val_loader,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame]:
    model = MLPBinaryClassifier(
        d_x=cfg.d_x,
        hidden_size=cfg.hidden_enc,
        num_hidden_layers=4,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr_enc,
        weight_decay=cfg.weight_decay_enc,
    )

    use_val = val_loader is not None
    best_state = None
    best_metric = float("inf")
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for raw_batch in train_loader:
            x, y = _unpack_xy_batch(raw_batch, device)
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            total_loss += float(loss.item()) * int(y.numel())
            total_correct += int(((probs >= 0.5) == (y >= 0.5)).sum().item())
            total += int(y.numel())

        train_metrics = {
            "train_loss": total_loss / total,
            "train_acc": total_correct / total,
        }
        val_metrics = (
            evaluate_label_classifier(model=model, loader=val_loader, device=device)
            if use_val
            else {"loss": None, "acc": None, "mean_prob": None}
        )

        history.append(
            {
                "model_name": "label_bce",
                "epoch": epoch,
                "train_loss": float(train_metrics["train_loss"]),
                "train_acc": float(train_metrics["train_acc"]),
                "val_loss": _scalar_or_none(val_metrics["loss"]),
                "val_acc": _scalar_or_none(val_metrics["acc"]),
                "val_mean_prob": _scalar_or_none(val_metrics["mean_prob"]),
            }
        )

        current_metric = val_metrics["loss"] if use_val and val_metrics["loss"] is not None else train_metrics["train_loss"]
        if float(current_metric) < best_metric:
            best_metric = float(current_metric)
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            msg = (
                f"[label_bce][{epoch:02d}] "
                f"train-loss={train_metrics['train_loss']:.6f} "
                f"train-acc={train_metrics['train_acc']:.4f}"
            )
            if use_val and val_metrics["loss"] is not None:
                msg += (
                    f" val-loss={float(val_metrics['loss']):.6f}"
                    f" val-acc={float(val_metrics['acc']):.4f}"
                )
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history)


@torch.inference_mode()
def predict_label_probabilities(
    model: nn.Module,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    probs = []
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        probs.append(torch.sigmoid(model(xb)).detach().cpu())
    return torch.cat(probs, dim=0).numpy()


def _latent_scalar(v_np: np.ndarray) -> np.ndarray:
    arr = np.asarray(v_np, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, 0]
    return arr.reshape(arr.shape[0], -1)[:, 0]


def _anderson_2sample_pvalue(x, y) -> float:
    x_arr = np.asarray(x, float).ravel()
    y_arr = np.asarray(y, float).ravel()
    try:
        res = stats.anderson_ksamp([x_arr, y_arr])
        return float(res.significance_level) / 100.0
    except ValueError:
        # Some baselines can collapse to nearly-constant latents, where
        # anderson_ksamp refuses to run because there are too few distinct values.
        # Fall back to the KS two-sample p-value so the sweep can continue.
        return float(ks_2sample_pvalue(x_arr, y_arr))


def _cvm_2sample_pvalue(x, y) -> float:
    x_arr = np.asarray(x, float).ravel()
    y_arr = np.asarray(y, float).ravel()
    try:
        res = stats.cramervonmises_2samp(x_arr, y_arr)
        return float(res.pvalue)
    except ValueError:
        return float(ks_2sample_pvalue(x_arr, y_arr))


def _uniform_reference_metrics(v_np, *, device: torch.device, ref_u_np: np.ndarray) -> dict[str, float]:
    v_np = np.clip(np.asarray(v_np, float).ravel(), 0.0, 1.0)
    ref_u_np = np.asarray(ref_u_np, float).ravel()

    ks_res = stats.kstest(v_np, "uniform", args=(0.0, 1.0))
    wass = stats.wasserstein_distance(v_np, ref_u_np)

    v_t = torch.as_tensor(v_np, device=device, dtype=torch.float32).view(-1, 1)
    ref_u_t = torch.as_tensor(ref_u_np, device=device, dtype=torch.float32).view(-1, 1)
    mmd = float(mmd2_unif(v_t, sigmas=None, ref_u=ref_u_t, use_median=False).item())

    return {
        "ks_stat": float(ks_res.statistic),
        "ks_pvalue": float(ks_res.pvalue),
        "mmd": mmd,
        "wasserstein": float(wass),
    }


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _label_probability_binom_pvalue(
    probabilities: np.ndarray,
    threshold: float = 0.9,
) -> float:
    probs = np.clip(np.asarray(probabilities, dtype=float).ravel(), 0.0, 1.0)
    n = int(probs.size)
    if n == 0:
        raise ValueError("Need at least one probability for the BCE binomial test")

    # Treat the summed predicted safe probabilities as an effective safe count
    # and test H0: mean probability >= threshold vs H1: mean probability < threshold.
    k_eff = int(np.rint(probs.sum()))
    return float(stats.binomtest(k_eff, n=n, p=float(threshold), alternative="less").pvalue)


def sweep_direction_parallel_twosample_ks(
    *,
    latent_models: dict[str, nn.Module],
    label_model: nn.Module,
    device: torch.device,
    latent_batch_size: int,
    label_null_threshold: float = 0.9,
    deltas,
    n_samples: int = 2000,
    mu_base=(0.0, 0.0),
    cov_base=np.diag([1.0, 1.0]),
    mean_direction=None,
    cov_direction=None,
    truncate_box=(-3.0, 3.0),
    seed: int = 1,
    n_reps: int = 10,
    n_jobs: int = -1,
    backend: str = "threading",
) -> pd.DataFrame:
    mu_base = np.asarray(mu_base, float).reshape(2)
    cov_base = np.asarray(cov_base, float).reshape(2, 2)

    if mean_direction is not None:
        mean_direction = np.asarray(mean_direction, float).reshape(2)
    if cov_direction is not None:
        cov_direction = np.asarray(cov_direction, float).reshape(2, 2)

    if device.type == "cuda" and n_jobs != 1:
        print(
            f"[sweep] device=cuda: forcing n_jobs=1 (requested {n_jobs}) to avoid GPU contention."
        )
        n_jobs = 1

    latent_specs = {
        "proposed": {"pval_col": "pval", "prefix": "prop"},
        "regularization_only": {"pval_col": "pval_base", "prefix": "base"},
        "reconstruction_only": {"pval_col": "pval_recon", "prefix": "recon"},
        "uniform_weight": {"pval_col": "pval_uniform", "prefix": "uniform"},
    }

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=len(deltas) * n_reps, dtype=np.uint32)

    def _build_shifted_params(delta):
        mu_shift = mu_base.copy()
        cov_shift = cov_base.copy()

        if mean_direction is not None:
            mu_shift = mu_shift + float(delta) * mean_direction

        if cov_direction is not None:
            cov_shift = cov_shift + float(delta) * cov_direction
            cov_shift = 0.5 * (cov_shift + cov_shift.T)
            if np.min(np.linalg.eigvalsh(cov_shift)) <= 0:
                return None, None

        return mu_shift, cov_shift

    def one(delta, rep_seed):
        rep_seed = int(rep_seed)
        mu_shift, cov_shift = _build_shifted_params(delta)
        if mu_shift is None:
            return {"delta": float(delta), "valid_cov": False}

        X0 = sample_x(
            n_samples=n_samples,
            dist_type="gaussian",
            params={"mean": mu_base, "cov": cov_base},
            seed=rep_seed,
            truncate_box=truncate_box,
        ).astype(np.float32)
        X1 = sample_x(
            n_samples=n_samples,
            dist_type="gaussian",
            params={"mean": mu_shift, "cov": cov_shift},
            seed=rep_seed + 99991,
            truncate_box=truncate_box,
        ).astype(np.float32)

        result = {
            "delta": float(delta),
            "valid_cov": True,
            "mu1": float(mu_shift[0]),
            "mu2": float(mu_shift[1]),
            "cov11": float(cov_shift[0, 0]),
            "cov12": float(cov_shift[0, 1]),
            "cov22": float(cov_shift[1, 1]),
        }

        for name, model in latent_models.items():
            spec = latent_specs[name]
            prefix = spec["prefix"]

            v0 = _latent_scalar(encode_to_latent(model, X0, device, batch_size=latent_batch_size))
            v1 = _latent_scalar(encode_to_latent(model, X1, device, batch_size=latent_batch_size))

            result[spec["pval_col"]] = float(ks_2sample_pvalue(v0, v1))
            result[f"ks_p_{prefix}"] = float(ks_2sample_pvalue(v0, v1))
            # Disabled for now: only keep the KS two-sample test in the sweep.
            # result[f"cvm_p_{prefix}"] = float(_cvm_2sample_pvalue(v0, v1))
            # result[f"ad_p_{prefix}"] = float(_anderson_2sample_pvalue(v0, v1))
            # uniform_metrics = _uniform_reference_metrics(v1, device=device, ref_u_np=ref_u_np)
            # result[f"ks_u_stat_{prefix}"] = uniform_metrics["ks_stat"]
            # result[f"ks_u_p_{prefix}"] = uniform_metrics["ks_pvalue"]
            # result[f"mmd_u_{prefix}"] = uniform_metrics["mmd"]
            # result[f"wass_u_{prefix}"] = uniform_metrics["wasserstein"]

        label_prob_base = predict_label_probabilities(
            label_model,
            X0,
            device=device,
            batch_size=latent_batch_size,
        )
        label_prob_shift = predict_label_probabilities(
            label_model,
            X1,
            device=device,
            batch_size=latent_batch_size,
        )
        result["pval_label"] = _label_probability_binom_pvalue(
            label_prob_shift,
            threshold=label_null_threshold,
        )
        result["label_prob_base"] = float(np.mean(label_prob_base))
        result["label_prob_shift"] = float(np.mean(label_prob_shift))
        return result

    tasks = [(delta, seeds[i * n_reps + j]) for i, delta in enumerate(deltas) for j in range(n_reps)]
    results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(one)(delta, s) for delta, s in tasks)

    rows = []
    for i, delta in enumerate(deltas):
        chunk = results[i * n_reps : (i + 1) * n_reps]
        valid_chunk = [r for r in chunk if r["valid_cov"]]

        if not valid_chunk:
            rows.append(
                {
                    "delta": float(delta),
                    "valid_cov": False,
                    "mu1": np.nan,
                    "mu2": np.nan,
                    "cov11": np.nan,
                    "cov12": np.nan,
                    "cov22": np.nan,
                }
            )
            continue

        r0 = valid_chunk[0]

        def arr(key):
            return np.array([r[key] for r in valid_chunk], dtype=float)

        row = {
            "delta": float(delta),
            "valid_cov": True,
            "mu1": r0["mu1"],
            "mu2": r0["mu2"],
            "cov11": r0["cov11"],
            "cov12": r0["cov12"],
            "cov22": r0["cov22"],
        }

        array_keys = [
            "pval",
            "pval_base",
            "pval_recon",
            "pval_uniform",
            "pval_label",
            "ks_p_prop",
            "ks_p_base",
            "ks_p_recon",
            "ks_p_uniform",
            # Disabled for now: only keep KS two-sample outputs.
            # "cvm_p_prop",
            # "cvm_p_base",
            # "cvm_p_recon",
            # "cvm_p_uniform",
            # "ad_p_prop",
            # "ad_p_base",
            # "ad_p_recon",
            # "ad_p_uniform",
            # "ks_u_stat_prop",
            # "ks_u_stat_base",
            # "ks_u_stat_recon",
            # "ks_u_stat_uniform",
            # "ks_u_p_prop",
            # "ks_u_p_base",
            # "ks_u_p_recon",
            # "ks_u_p_uniform",
            # "mmd_u_prop",
            # "mmd_u_base",
            # "mmd_u_recon",
            # "mmd_u_uniform",
            # "wass_u_prop",
            # "wass_u_base",
            # "wass_u_recon",
            # "wass_u_uniform",
            "label_prob_base",
            "label_prob_shift",
        ]

        for key in array_keys:
            values = arr(key)
            mean_val, std_val = _mean_std(values)
            row[key] = values
            row[f"{key}_mean"] = mean_val
            row[f"{key}_std"] = std_val

        rows.append(row)

    return pd.DataFrame(rows)


def power_by_delta(df: pd.DataFrame, method_to_col: dict[str, str], alpha: float) -> pd.DataFrame:
    rows = []
    for method_name, col in method_to_col.items():
        if col not in df.columns:
            continue
        for _, rec in df.iterrows():
            pvals = np.asarray(rec[col], dtype=float)
            rows.append(
                {
                    "delta": float(rec["delta"]),
                    "method": method_name,
                    "empirical_rejection_rate": float(np.mean(pvals <= alpha)),
                }
            )
    return pd.DataFrame(rows)


def build_scalar_summary_df(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for setting, df in dfs.items():
        scalar_cols = [
            c
            for c in df.columns
            if df[c].map(lambda x: np.isscalar(x) or x is None).all()
        ]
        summary = df[scalar_cols].copy()
        summary.insert(0, "setting", setting)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def build_compat_mean_summary(
    *,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    alpha: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_col = "pval_base"
    proposed_col = "pval"

    def _power_df(df: pd.DataFrame, setting: str) -> pd.DataFrame:
        tmp = df.copy()
        tmp[baseline_col] = tmp[baseline_col].apply(lambda p: np.mean(np.asarray(p) <= alpha))
        tmp[proposed_col] = tmp[proposed_col].apply(lambda p: np.mean(np.asarray(p) <= alpha))
        out = tmp[["delta", baseline_col, proposed_col]].copy()
        out["setting"] = setting
        return out

    power1 = _power_df(df1, "df1")
    power2 = _power_df(df2, "df2")
    power3 = _power_df(df3, "df3")

    gap_df = pd.concat([power1, power2, power3], ignore_index=True)
    gap_df["gap"] = gap_df[baseline_col] - gap_df[proposed_col]
    gap_table = gap_df.pivot(index="setting", columns="delta", values="gap")
    gap_table.columns = gap_table.columns.astype(float)

    mean_summary = pd.DataFrame(
        [
            {"metric": "df1_gap_mean", "value": float(gap_table.loc["df1"].mean())},
            {
                "metric": "df2_before_0.5_mean",
                "value": float(gap_table.loc["df2", gap_table.columns < 0.2].mean()),
            },
            {
                "metric": "df2_after_0.5_mean",
                "value": float(gap_table.loc["df2", gap_table.columns > 0.2].mean()),
            },
            {"metric": "df3_gap_mean", "value": float(gap_table.loc["df3"].mean())},
        ]
    )
    return gap_df, gap_table, mean_summary


def _make_output_dir(path_value: str) -> Path:
    output_dir = Path(path_value)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _serialize_args(cfg: argparse.Namespace) -> dict:
    data = vars(cfg).copy()
    data["dist_mean"] = cfg.dist_mean.tolist()
    data["dist_cov"] = cfg.dist_cov.tolist()
    data["time_window"] = list(cfg.time_window)
    return data


def _print_stage(message: str) -> None:
    print(f"\n[Stage] {message}", flush=True)


def main() -> None:
    cfg = parse_args()

    _print_stage("Parsing configuration")
    print("[Config]")
    print(f"  seed              : {cfg.seed}")
    print(f"  safety threshold  : {cfg.tau}")
    print(f"  loss_type         : {cfg.loss_type}")
    print(f"  perturb_type      : {cfg.perturb_type}")
    print(f"  lr                : {cfg.lr:g}")
    print(f"  lr_enc            : {cfg.lr_enc:g}")
    print(f"  lambda_mmd        : {cfg.lambda_mmd:g}")
    print(f"  asym_recon_delta  : {cfg.asym_recon_delta:g}")
    print(f"  epochs            : {cfg.epochs}")
    print(f"  n_samples         : {cfg.n_samples}")
    print(f"  test_n_samples    : {cfg.test_n_samples}")
    print(f"  num_workers       : {cfg.num_workers}")
    print(f"  latent_batch_size : {cfg.latent_batch_size}")
    print(f"  output_dir        : {cfg.output_dir}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = _choose_device()
    print("device:", device)

    _print_stage("Generating spring dataset and safety labels")
    z_0 = np.array([0.0, 0.0])
    params = SpringParams(mass=1.0, damping=1.0, stiffness=2.0)

    X = sample_x(
        n_samples=cfg.n_samples,
        dist_type="gaussian",
        params={"mean": cfg.dist_mean, "cov": cfg.dist_cov},
        seed=cfg.seed,
    )
    traj = simulate_spring(params, t_final=cfg.T, dt=cfg.dt, X=X, forcing_type="sin", config=z_0)
    s = traj["s"]
    assert X.shape[0] == s.shape[0]

    y = check_safety(
        traj=traj,
        threshold=cfg.tau,
        window=cfg.time_window,
    ).astype(np.float32)

    o_hist, o_next = make_sequence_pairs(s=s, stride=cfg.stride)
    x_np = X.astype(np.float32)

    print("Shapes -> X:", x_np.shape, "trajectory shape:", s.shape)

    _print_stage("Building train/validation loaders")
    N = x_np.shape[0]
    perm = np.random.RandomState(cfg.seed).permutation(N)
    n_train = int(0.8 * N)
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    train_ds = SpringDataset(x_np[tr_idx], o_hist[tr_idx], o_next[tr_idx], y=y[tr_idx])
    val_ds = SpringDataset(x_np[va_idx], o_hist[va_idx], o_next[va_idx], y=y[va_idx])

    loader_kwargs = _make_loader_kwargs(device, cfg.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_train,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_val,
        shuffle=False,
        **loader_kwargs,
    )

    sequence_dt = cfg.dt * cfg.stride
    proposed_cfg = _make_sequence_train_config(
        lambda_mmd=cfg.lambda_mmd,
        loss_type=cfg.loss_type,
        tau=cfg.tau,
        time_window=cfg.time_window,
        sequence_dt=sequence_dt,
        asym_recon_delta=cfg.asym_recon_delta,
    )
    uniform_weight_cfg = _make_sequence_train_config(
        lambda_mmd=cfg.lambda_mmd,
        loss_type="stability_asymmetric_recon",
        tau=cfg.tau,
        time_window=cfg.time_window,
        sequence_dt=sequence_dt,
        asym_recon_delta=1.0,
    )
    recon_only_cfg = _make_sequence_train_config(
        lambda_mmd=0.0,
        loss_type="stability_asymmetric_recon",
        tau=cfg.tau,
        time_window=cfg.time_window,
        sequence_dt=sequence_dt,
        asym_recon_delta=1.0,
    )
    reg_only_divergence = DivergenceConfig(
        lambda_mmd=cfg.lambda_mmd,
        fn=mmd2_unif,
        kwargs={"sigmas": None, "low_discrepancy": True},
    )

    _print_stage("Training proposed model")
    proposed_model, proposed_history = train_sequence_model(
        model_name="proposed",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=proposed_cfg,
    )
    _print_stage("Training uniform-weight baseline (asym_recon_delta=1)")
    uniform_model, uniform_history = train_sequence_model(
        model_name="uniform_weight",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=uniform_weight_cfg,
    )
    _print_stage("Training reconstruction-only baseline")
    recon_model, recon_history = train_sequence_model(
        model_name="reconstruction_only",
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=recon_only_cfg,
    )
    _print_stage("Training regularization-only baseline")
    reg_only_model, reg_only_history = train_regularization_only_model(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        divergence=reg_only_divergence,
    )
    _print_stage("Training BCE label predictor baseline")
    label_model, label_history = train_label_classifier(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    latent_models = {
        "proposed": proposed_model,
        "regularization_only": reg_only_model,
        "reconstruction_only": recon_model,
        "uniform_weight": uniform_model,
    }

    truncate_box = (-3.0, 3.0)
    n_reps = 50
    n_jobs = 1 if device.type == "cuda" else 15
    label_null_threshold = 0.9

    if cfg.perturb_type == "mean":
        _print_stage("Running mean-perturbation sweeps for df1/df2/df3")
        deltas = np.linspace(0, 1, 21)
        df1 = sweep_direction_parallel_twosample_ks(
            latent_models=latent_models,
            label_model=label_model,
            device=device,
            latent_batch_size=cfg.latent_batch_size,
            label_null_threshold=label_null_threshold,
            n_samples=cfg.test_n_samples,
            mu_base=(0, 0),
            mean_direction=(0, 1),
            deltas=deltas,
            truncate_box=truncate_box,
            seed=cfg.seed,
            n_reps=n_reps,
            n_jobs=n_jobs,
        )
        print("[Sweep] Finished df1 mean-direction=(0, 1)", flush=True)
        df2 = sweep_direction_parallel_twosample_ks(
            latent_models=latent_models,
            label_model=label_model,
            device=device,
            latent_batch_size=cfg.latent_batch_size,
            label_null_threshold=label_null_threshold,
            n_samples=cfg.test_n_samples,
            mu_base=(0, 0),
            mean_direction=(1, 0),
            deltas=deltas,
            truncate_box=truncate_box,
            seed=cfg.seed,
            n_reps=n_reps,
            n_jobs=n_jobs,
        )
        print("[Sweep] Finished df2 mean-direction=(1, 0)", flush=True)
        df3 = sweep_direction_parallel_twosample_ks(
            latent_models=latent_models,
            label_model=label_model,
            device=device,
            latent_batch_size=cfg.latent_batch_size,
            label_null_threshold=label_null_threshold,
            n_samples=cfg.test_n_samples,
            mu_base=(-1, 0),
            mean_direction=(0, 1),
            deltas=deltas,
            truncate_box=truncate_box,
            seed=cfg.seed,
            n_reps=n_reps,
            n_jobs=n_jobs,
        )
        print("[Sweep] Finished df3 mean-direction=(0, 1) from mu_base=(-1, 0)", flush=True)
    elif cfg.perturb_type == "cov":
        _print_stage("Running covariance-perturbation sweeps for df1/df2/df3")
        cov_ref = np.diag([1.0, 1.0])
        df1 = sweep_direction_parallel_twosample_ks(
            latent_models=latent_models,
            label_model=label_model,
            device=device,
            latent_batch_size=cfg.latent_batch_size,
            label_null_threshold=label_null_threshold,
            n_samples=cfg.test_n_samples,
            mu_base=(0, 0),
            cov_base=cov_ref,
            cov_direction=np.array([[-1.0, 0.0], [0.0, 0.0]]),
            deltas=np.linspace(0, 0.8, 21),
            truncate_box=truncate_box,
            seed=cfg.seed,
            n_reps=n_reps,
            n_jobs=n_jobs,
        )
        print("[Sweep] Finished df1 covariance shift: var-1 decrease", flush=True)
        df2 = sweep_direction_parallel_twosample_ks(
            latent_models=latent_models,
            label_model=label_model,
            device=device,
            latent_batch_size=cfg.latent_batch_size,
            label_null_threshold=label_null_threshold,
            n_samples=cfg.test_n_samples,
            mu_base=(0, 0),
            cov_base=cov_ref,
            cov_direction=np.array([[1.0, 0.0], [0.0, 0.0]]),
            deltas=np.linspace(0, 0.8, 21),
            truncate_box=truncate_box,
            seed=cfg.seed,
            n_reps=n_reps,
            n_jobs=n_jobs,
        )
        print("[Sweep] Finished df2 covariance shift: var-1 increase", flush=True)
        df3 = sweep_direction_parallel_twosample_ks(
            latent_models=latent_models,
            label_model=label_model,
            device=device,
            latent_batch_size=cfg.latent_batch_size,
            label_null_threshold=label_null_threshold,
            n_samples=cfg.test_n_samples,
            mu_base=(0, 0),
            cov_base=np.diag([1.6, 1.0]),
            cov_direction=np.array([[0.0, 1.0], [1.0, 0.0]]),
            deltas=np.linspace(0, 1, 21),
            truncate_box=truncate_box,
            seed=cfg.seed,
            n_reps=n_reps,
            n_jobs=n_jobs,
        )
        print("[Sweep] Finished df3 covariance shift: correlation increase", flush=True)
    else:
        raise ValueError(f"Unsupported perturb_type='{cfg.perturb_type}'")

    _print_stage("Building summary tables")
    dfs = {"df1": df1, "df2": df2, "df3": df3}
    scalar_summary_df = build_scalar_summary_df(dfs)

    method_to_pval_col = {
        "proposed": "pval",
        "regularization_only": "pval_base",
        "reconstruction_only": "pval_recon",
        "uniform_weight": "pval_uniform",
        "label_bce_binom": "pval_label",
    }
    rejection_rate_frames = []
    for setting, df in dfs.items():
        power_df = power_by_delta(df, method_to_pval_col, alpha=0.1)
        power_df.insert(0, "setting", setting)
        rejection_rate_frames.append(power_df)
    rejection_rate_summary = pd.concat(rejection_rate_frames, ignore_index=True)

    label_probability_summary = pd.concat(
        [
            df[["delta", "label_prob_base_mean", "label_prob_base_std", "label_prob_shift_mean", "label_prob_shift_std"]]
            .assign(setting=setting)
            for setting, df in dfs.items()
        ],
        ignore_index=True,
    )

    gap_df, gap_table, mean_summary = build_compat_mean_summary(df1=df1, df2=df2, df3=df3, alpha=0.1)

    _print_stage("Saving outputs")
    output_dir = _make_output_dir(cfg.output_dir)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_args(cfg), f, indent=2)

    for name, hist_df in {
        "proposed": proposed_history,
        "uniform_weight": uniform_history,
        "reconstruction_only": recon_history,
        "regularization_only": reg_only_history,
        "label_bce": label_history,
    }.items():
        hist_df.to_csv(output_dir / f"training_history_{name}.csv", index=False)

    for name, model in {
        "proposed": proposed_model,
        "uniform_weight": uniform_model,
        "reconstruction_only": recon_model,
    }.items():
        train_snapshot = build_reconstruction_snapshot(
            model=model,
            loader=train_loader,
            split_name="train",
            device=device,
        )
        val_snapshot = build_reconstruction_snapshot(
            model=model,
            loader=val_loader,
            split_name="val",
            device=device,
        )
        train_snapshot.to_pickle(output_dir / f"train_reconstruction_snapshot_{name}.pkl")
        val_snapshot.to_pickle(output_dir / f"val_reconstruction_snapshot_{name}.pkl")

    df1.to_pickle(output_dir / "df1.pkl")
    df2.to_pickle(output_dir / "df2.pkl")
    df3.to_pickle(output_dir / "df3.pkl")
    scalar_summary_df.to_csv(output_dir / "summary_df.csv", index=False)
    rejection_rate_summary.to_csv(output_dir / "rejection_rate_summary.csv", index=False)
    label_probability_summary.to_csv(output_dir / "label_probability_summary.csv", index=False)
    gap_df.to_csv(output_dir / "gap_df.csv", index=False)
    gap_table.to_csv(output_dir / "gap_table.csv")
    mean_summary.to_csv(output_dir / "mean_summary.csv", index=False)

    mean_summary_json = {
        rec["metric"]: float(rec["value"])
        for rec in mean_summary.to_dict(orient="records")
    }
    with open(output_dir / "mean_summary.json", "w", encoding="utf-8") as f:
        json.dump(mean_summary_json, f, indent=2)

    print(f"Saved results to {output_dir}")
    print(mean_summary.to_string(index=False))
    print("files saved to:", output_dir)


if __name__ == "__main__":
    main()
