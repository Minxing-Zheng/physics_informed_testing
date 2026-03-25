#!/usr/bin/env python
# coding: utf-8

# In[23]:


import argparse
import json
from pathlib import Path
import math,random,os,sys, importlib, time,copy
import pickle, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde, kstest, cramervonmises, chi2, ks_2samp
from scipy import stats

import torch
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
# Repo paths
PWD = Path.cwd().resolve()
ROOT = PWD.parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Local modules
import data.sample_x as sample_x_mod; importlib.reload(sample_x_mod)
import data.solve_ode as solve_ode_mod; importlib.reload(solve_ode_mod)
import data.utils as utils_mod; importlib.reload(utils_mod)
import train.training_clean as train_training; importlib.reload(train_training)

from data.sample_x import sample_x
from data.solve_ode import SpringParams, simulate_spring
from data.utils import plot_traj, check_safety, compute_mu_safety_map
from data.datasets import SpringDataset, make_sequence_pairs
from data.plotting import plot_x_safety_map, plot_mu_safety_map
from train.models import build_cond_gru_model, EncoderOnly

from train.training_clean import (
    train_eval_epoch,
    train_eval_epoch_enc_only,
    eval_epoch,
    eval_stability_classification,
    eval_regime_batch_stats
)
from train.eval import encode_to_latent

from train.losses import mmd2_unif, energy_distance_unif
from train.training_clean import (
    DivergenceConfig,
    StabilityConfig,
    RegimeConfig,
    TrainingConfig,
    train_eval_epoch,
    train_eval_epoch_enc_only,
    eval_epoch,
)
import hypo_test.stats_tests as test_mod; importlib.reload(test_mod)
from hypo_test.stats_tests import uniformity_pvalues_all,ecdf,ks_2sample_pvalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model diagnostics for the physics-informed hypothesis test."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=5000, dest="n_samples")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--perturb_type", type=str, default="mean", dest="perturb_type") # {"mean", "cov"}
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default='stability_asymmetric_recon', dest="loss_type")# {"bce", "weighted", "weighted_bce", "stability_asymmetric_recon"}
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
        default=0.0,
        dest="weight_decay_enc",
    )
    parser.add_argument("--lambda-mmd", type=float, default=1e-2, dest="lambda_mmd")
    parser.add_argument("--d-x", type=int, default=2, dest="d_x")
    parser.add_argument("--d-o", type=int, default=1, dest="d_o")
    parser.add_argument("--d-v", type=int, default=1, dest="d_v")
    parser.add_argument("--d-h", type=int, default=128, dest="d_h")
    parser.add_argument("--hidden-enc", type=int, default=512, dest="hidden_enc")
    parser.add_argument("--hidden-dec", type=int, default=128, dest="hidden_dec")
    parser.add_argument("--B-bins", type=int, default=20, dest="B_bins")
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
        "--lambda-stability",
        type=float,
        default=0.0,
        dest="lambda_stability",
    )
    parser.add_argument("--lambda-regime", type=float, default=0.0, dest="lambda_regime")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
    )

    args = parser.parse_args()

    def _fmt_float(x: float) -> str:
        return f"{x:g}".replace(".", "p")

    if args.output_dir is None:
        args.output_dir = (
            "results/diagnostics_results/"
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


Cfg = parse_args()

print("[Config]")
print(f"  seed              : {Cfg.seed}")
print(f"  Safety threshold tau  : {Cfg.tau}")
print(f"  loss_type         : {Cfg.loss_type}")
print(f"  perturb_type      : {Cfg.perturb_type}")
print(f"  lr                : {Cfg.lr:g}")
print(f"  lr_enc            : {Cfg.lr_enc:g}")
print(f"  lambda_mmd        : {Cfg.lambda_mmd:g}")
print(f"  lambda_stability  : {Cfg.lambda_stability:g}")
print(f"  lambda_regime     : {Cfg.lambda_regime:g}")
print(f"  asym_recon_delta  : {Cfg.asym_recon_delta:g}")
print(f"  seed              : {Cfg.seed}")
print(f"  epochs            : {Cfg.epochs}")
print(f"  n_samples         : {Cfg.n_samples}")
print(f"  num_workers       : {Cfg.num_workers}")
print(f"  latent_batch_size : {Cfg.latent_batch_size}")
print(f"  output_dir        : {Cfg.output_dir}")


# Reproducibility
torch.manual_seed(Cfg.seed)
np.random.seed(Cfg.seed)
random.seed(Cfg.seed)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# In[24]:


# 1. train two models: proposed model and encoder only model
# 2. visualize their latent variable uniformity
# 3. perturb data, change mean and variance, and check how latent shfits, say divergence between shifted latent to uniformity


# In[25]:


z_0 = np.array([0.0, 0.0]) # baselien config
params = SpringParams(mass=1.0, damping=1.0, stiffness=2.0) # paramters theta

X = sample_x(
    n_samples=Cfg.n_samples,
    dist_type="gaussian",
    params={"mean": Cfg.dist_mean, "cov": Cfg.dist_cov},
    seed=Cfg.seed,
)

traj = simulate_spring(params, t_final=Cfg.T, dt=Cfg.dt, X=X, forcing_type="sin", config=z_0)
s = traj["s"]                           # (N, T_full)
assert X.shape[0] == s.shape[0]


y = check_safety(
    traj=traj,
    threshold=Cfg.tau,
    window=Cfg.time_window,
).astype(np.float32)


o_hist,o_next = make_sequence_pairs( s=s,stride=Cfg.stride)
x_np   = X.astype(np.float32)

print("Shapes -> X:", x_np.shape, "trajectory shape:", s.shape)

N = x_np.shape[0]
perm = np.random.RandomState(Cfg.seed).permutation(N)
n_train = int(0.8 * N)
tr_idx, va_idx = perm[:n_train], perm[n_train:]

train_ds = SpringDataset(x_np[tr_idx], o_hist[tr_idx], o_next[tr_idx], y=y[tr_idx])
val_ds   = SpringDataset(x_np[va_idx], o_hist[va_idx], o_next[va_idx], y=y[va_idx])

loader_kwargs = {
    "num_workers": max(int(Cfg.num_workers), 0),
    "pin_memory": device.type == "cuda",
}
if loader_kwargs["num_workers"] > 0:
    loader_kwargs["persistent_workers"] = True

train_loader = DataLoader(
    train_ds,
    batch_size=Cfg.batch_train,
    shuffle=True,
    drop_last=True,
    **loader_kwargs,
)
val_loader = DataLoader(
    val_ds,
    batch_size=Cfg.batch_val,
    shuffle=False,
    **loader_kwargs,
)

# print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")



# In[ ]:


div_fn = mmd2_unif
div_kwargs = {"sigmas": None, "low_discrepancy": True}

# train_cfg = TrainingConfig(
#     divergence=DivergenceConfig(lambda_mmd=lambda_mmd, fn=mmd2_unif, kwargs=div_kwargs),
#     stability=StabilityConfig(
#         lambda_stability=Cfg.lambda_stability,
#         threshold=Cfg.tau,
#         window=Cfg.time_window,
#         dt=Cfg.dt,
#     ),
#     regime=RegimeConfig(
#         lambda_regime=Cfg.lambda_regime,
#         gamma=0.9,
#         delta_safe=0.05,
#         delta_unsafe=0.2,
#     ),
# )

train_cfg = TrainingConfig(
    divergence=DivergenceConfig(lambda_mmd=Cfg.lambda_mmd, fn=div_fn, kwargs=div_kwargs),
    stability=StabilityConfig(
        loss_type=Cfg.loss_type,
        threshold=Cfg.tau,
        asym_recon_delta=Cfg.asym_recon_delta,
    ),
)

baseline_model_cfg = DivergenceConfig(
    lambda_mmd=Cfg.lambda_mmd,
    fn=div_fn,
    kwargs=div_kwargs,
)

model = build_cond_gru_model(
    d_x=Cfg.d_x,
    d_o=Cfg.d_o,
    d_v=Cfg.d_v,
    d_h=Cfg.d_h,
    enc_layers=4,
    dec_layers=2,
    hidden_enc=Cfg.hidden_enc,
    hidden_dec=Cfg.hidden_dec,
    output_activation="sigmoid",
    dropout_rnn=0.2,
).to(device)

model_enc = EncoderOnly(copy.deepcopy(model.enc)).to(device)

opt = torch.optim.Adam(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
opt_enc = torch.optim.Adam(model_enc.parameters(), lr=Cfg.lr_enc, weight_decay=Cfg.weight_decay_enc)


def evaluate_split_metrics(loader, split_name):
    rec, div = eval_epoch(
        loader=loader,
        model=model,
        device=device,
        config=train_cfg,
    )
    total = rec + Cfg.lambda_mmd * div
    metrics = {
        f"{split_name}_recon": float(rec),
        f"{split_name}_divergence": float(div),
        f"{split_name}_total_loss": float(total),
    }
    if train_cfg.stability.threshold is not None:
        cls = eval_stability_classification(
            loader=loader,
            model=model,
            device=device,
            stability=train_cfg.stability,
        )
        metrics.update(
            {
                f"{split_name}_safety_acc": cls["overall_acc"],
                f"{split_name}_safety_y1_acc": cls["y1_correct_frac"],
                f"{split_name}_safety_y0_acc": cls["y0_correct_frac"],
                f"{split_name}_frac_y1": cls["frac_y1"],
                f"{split_name}_frac_y0": cls["frac_y0"],
            }
        )
    return metrics


@torch.no_grad()
def build_reconstruction_snapshot(loader, split_name, max_samples=8):
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


history = []
best_val_mmd = float("inf")
best_state = None
use_val = val_loader is not None

for epoch in range(1, Cfg.epochs + 1):
    tr_loss, val_rec, val_div = train_eval_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=opt,
        device=device,
        config=train_cfg,
        val_loader=val_loader if use_val else None,
    )

    tr_div_enc, val_div_enc = train_eval_epoch_enc_only(
        model_enc=model_enc,
        train_loader=train_loader,
        optimizer=opt_enc,
        device=device,
        divergence=baseline_model_cfg,
        val_loader=val_loader if use_val else None,
    )

    train_metrics = evaluate_split_metrics(train_loader, "train")
    val_metrics = evaluate_split_metrics(val_loader, "val") if use_val else {
        "val_recon": None,
        "val_divergence": None,
        "val_total_loss": None,
        "val_safety_acc": None,
        "val_safety_y1_acc": None,
        "val_safety_y0_acc": None,
        "val_frac_y1": None,
        "val_frac_y0": None,
    }

    history.append({
        "epoch": epoch,
        "train_loss": tr_loss,
        "val_loss": (None if val_rec is None or val_div is None else val_rec + Cfg.lambda_mmd * val_div),
        "val_recon_raw": val_rec,
        "val_divergence_raw": val_div,
        "train_div (enc-only)": tr_div_enc,
        "val_divergence (enc-only)": val_div_enc,
        **train_metrics,
        **val_metrics,
    })

    if use_val and val_div is not None and val_div < best_val_mmd:
        best_val_mmd = val_div
        best_state = model.state_dict()

    if epoch % 10 == 0 or epoch == 1:
        msg = (
            f"[{epoch:02d}] train-loss={tr_loss:.6f}"
            f" train-recon={train_metrics['train_recon']:.6f}"
            f" train-acc={train_metrics.get('train_safety_acc', float('nan')):.4f}"
        )
        if use_val:
            msg += (
                f"  val-recon={val_metrics['val_recon']:.6f}"
                f"  val-divergence={val_metrics['val_divergence']:.6f}"
                f"  val-acc={val_metrics.get('val_safety_acc', float('nan')):.4f}"
                f"  train-divergence (enc)={tr_div_enc:.6f}"
                f"  val-divergence (enc)={val_div_enc:.6f}"
            )
        print(msg)

if use_val and best_state is not None:
    model.load_state_dict(best_state)

# pd.DataFrame(history).tail()


def _anderson_2sample_pvalue(x, y):
    res = stats.anderson_ksamp([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    return float(res.significance_level) / 100.0


def _cvm_2sample_pvalue(x, y):
    res = stats.cramervonmises_2samp(np.asarray(x, float).ravel(), np.asarray(y, float).ravel())
    return float(res.pvalue)


def _uniform_reference_metrics(v_np, *, device, ref_u_np):
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


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


# def sweep_direction_parallel_twosample_ks(
#     mu_base,
#     direction,
#     deltas,
#     *,
#     n_samples=2000,
#     cov=np.diag([1., 1.]),
#     truncate_box=None,
#     seed=1,
#     n_reps=10,
#     n_jobs=-1,
#     backend="threading",
# ):
#     if device.type == "cuda" and n_jobs != 1:
#         print(
#             f"[sweep] device=cuda: forcing n_jobs=1 (requested {n_jobs}) to avoid GPU contention."
#         )
#         n_jobs = 1

#     rng = np.random.default_rng(seed)
#     seeds = rng.integers(0, 2**32 - 1, size=len(deltas) * n_reps, dtype=np.uint32)
#     cov = np.asarray(cov, float)

#     def one(delta, rep_seed):
#         rep_seed = int(rep_seed)
#         mu_shift = np.asarray(mu_base, float) + float(delta) * np.asarray(direction, float)

#         # baseline sample
#         X0 = sample_x(
#             n_samples=n_samples,
#             dist_type="gaussian",
#             params={"mean": np.asarray((0,0), float), "cov": cov},
#             seed=rep_seed,
#             truncate_box=truncate_box,
#         ).astype(np.float32)

#         # shifted sample (different seed to avoid identical draws)
#         X1 = sample_x(
#             n_samples=n_samples,
#             dist_type="gaussian",
#             params={"mean": mu_shift, "cov": cov},
#             seed=rep_seed + 99991,
#             truncate_box=truncate_box,
#         ).astype(np.float32)

#         # encode both samples for both models
#         v0_p = encode_to_latent(model,     X0, device, batch_size=Cfg.latent_batch_size)[:, 0]
#         v1_p = encode_to_latent(model,     X1, device, batch_size=Cfg.latent_batch_size)[:, 0]
#         v0_b = encode_to_latent(model_enc, X0, device, batch_size=Cfg.latent_batch_size)[:, 0]
#         v1_b = encode_to_latent(model_enc, X1, device, batch_size=Cfg.latent_batch_size)[:, 0]

#         # Two-sample p-values: baseline latent vs shifted latent.
#         ks_p_prop = ks_2sample_pvalue(v0_p, v1_p)
#         ks_p_base = ks_2sample_pvalue(v0_b, v1_b)
#         cvm_p_prop = _cvm_2sample_pvalue(v0_p, v1_p)
#         cvm_p_base = _cvm_2sample_pvalue(v0_b, v1_b)
#         ad_p_prop = _anderson_2sample_pvalue(v0_p, v1_p)
#         ad_p_base = _anderson_2sample_pvalue(v0_b, v1_b)

#         # Uniform-reference diagnostics on shifted latent only.
#         ref_rng = np.random.default_rng(rep_seed + 424242)
#         ref_u_np = ref_rng.uniform(0.0, 1.0, size=v1_p.shape[0]).astype(np.float32)
#         uniform_prop = _uniform_reference_metrics(v1_p, device=device, ref_u_np=ref_u_np)
#         uniform_base = _uniform_reference_metrics(v1_b, device=device, ref_u_np=ref_u_np)

#         return {
#             "delta": float(delta),
#             "mu1": float(mu_shift[0]),
#             "mu2": float(mu_shift[1]),
#             "ks_p_prop": float(ks_p_prop),
#             "ks_p_base": float(ks_p_base),
#             "cvm_p_prop": float(cvm_p_prop),
#             "cvm_p_base": float(cvm_p_base),
#             "ad_p_prop": float(ad_p_prop),
#             "ad_p_base": float(ad_p_base),
#             "ks_u_stat_prop": uniform_prop["ks_stat"],
#             "ks_u_stat_base": uniform_base["ks_stat"],
#             "ks_u_p_prop": uniform_prop["ks_pvalue"],
#             "ks_u_p_base": uniform_base["ks_pvalue"],
#             "mmd_u_prop": uniform_prop["mmd"],
#             "mmd_u_base": uniform_base["mmd"],
#             "wass_u_prop": uniform_prop["wasserstein"],
#             "wass_u_base": uniform_base["wasserstein"],
#         }

#     tasks = [(delta, seeds[i * n_reps + j]) for i, delta in enumerate(deltas) for j in range(n_reps)]
#     results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(one)(delta, s) for delta, s in tasks)

#     rows = []
#     for i, delta in enumerate(deltas):
#         chunk = results[i * n_reps : (i + 1) * n_reps]
#         mu1, mu2 = chunk[0]["mu1"], chunk[0]["mu2"]

#         ks_p_prop = np.array([r["ks_p_prop"] for r in chunk], dtype=float)
#         ks_p_base = np.array([r["ks_p_base"] for r in chunk], dtype=float)
#         cvm_p_prop = np.array([r["cvm_p_prop"] for r in chunk], dtype=float)
#         cvm_p_base = np.array([r["cvm_p_base"] for r in chunk], dtype=float)
#         ad_p_prop = np.array([r["ad_p_prop"] for r in chunk], dtype=float)
#         ad_p_base = np.array([r["ad_p_base"] for r in chunk], dtype=float)
#         ks_u_stat_prop = np.array([r["ks_u_stat_prop"] for r in chunk], dtype=float)
#         ks_u_stat_base = np.array([r["ks_u_stat_base"] for r in chunk], dtype=float)
#         ks_u_p_prop = np.array([r["ks_u_p_prop"] for r in chunk], dtype=float)
#         ks_u_p_base = np.array([r["ks_u_p_base"] for r in chunk], dtype=float)
#         mmd_u_prop = np.array([r["mmd_u_prop"] for r in chunk], dtype=float)
#         mmd_u_base = np.array([r["mmd_u_base"] for r in chunk], dtype=float)
#         wass_u_prop = np.array([r["wass_u_prop"] for r in chunk], dtype=float)
#         wass_u_base = np.array([r["wass_u_base"] for r in chunk], dtype=float)

#         ks_p_prop_mean, ks_p_prop_std = _mean_std(ks_p_prop)
#         ks_p_base_mean, ks_p_base_std = _mean_std(ks_p_base)
#         cvm_p_prop_mean, cvm_p_prop_std = _mean_std(cvm_p_prop)
#         cvm_p_base_mean, cvm_p_base_std = _mean_std(cvm_p_base)
#         ad_p_prop_mean, ad_p_prop_std = _mean_std(ad_p_prop)
#         ad_p_base_mean, ad_p_base_std = _mean_std(ad_p_base)
#         ks_u_stat_prop_mean, ks_u_stat_prop_std = _mean_std(ks_u_stat_prop)
#         ks_u_stat_base_mean, ks_u_stat_base_std = _mean_std(ks_u_stat_base)
#         mmd_u_prop_mean, mmd_u_prop_std = _mean_std(mmd_u_prop)
#         mmd_u_base_mean, mmd_u_base_std = _mean_std(mmd_u_base)
#         wass_u_prop_mean, wass_u_prop_std = _mean_std(wass_u_prop)
#         wass_u_base_mean, wass_u_base_std = _mean_std(wass_u_base)

#         rows.append({
#             "delta": float(delta),
#             "mu1": float(mu1),
#             "mu2": float(mu2),
#             "pval": ks_p_prop,
#             "pval_base": ks_p_base,
#             "ks_p_prop": ks_p_prop,
#             "ks_p_base": ks_p_base,
#             "cvm_p_prop": cvm_p_prop,
#             "cvm_p_base": cvm_p_base,
#             "ad_p_prop": ad_p_prop,
#             "ad_p_base": ad_p_base,
#             "ks_u_stat_prop": ks_u_stat_prop,
#             "ks_u_stat_base": ks_u_stat_base,
#             "ks_u_p_prop": ks_u_p_prop,
#             "ks_u_p_base": ks_u_p_base,
#             "mmd_u_prop": mmd_u_prop,
#             "mmd_u_base": mmd_u_base,
#             "wass_u_prop": wass_u_prop,
#             "wass_u_base": wass_u_base,
#             "ks_p_prop_mean": ks_p_prop_mean,
#             "ks_p_base_mean": ks_p_base_mean,
#             "ks_p_prop_std": ks_p_prop_std,
#             "ks_p_base_std": ks_p_base_std,
#             "cvm_p_prop_mean": cvm_p_prop_mean,
#             "cvm_p_base_mean": cvm_p_base_mean,
#             "cvm_p_prop_std": cvm_p_prop_std,
#             "cvm_p_base_std": cvm_p_base_std,
#             "ad_p_prop_mean": ad_p_prop_mean,
#             "ad_p_base_mean": ad_p_base_mean,
#             "ad_p_prop_std": ad_p_prop_std,
#             "ad_p_base_std": ad_p_base_std,
#             "ks_u_stat_prop_mean": ks_u_stat_prop_mean,
#             "ks_u_stat_base_mean": ks_u_stat_base_mean,
#             "ks_u_stat_prop_std": ks_u_stat_prop_std,
#             "ks_u_stat_base_std": ks_u_stat_base_std,
#             "mmd_u_prop_mean": mmd_u_prop_mean,
#             "mmd_u_base_mean": mmd_u_base_mean,
#             "mmd_u_prop_std": mmd_u_prop_std,
#             "mmd_u_base_std": mmd_u_base_std,
#             "wass_u_prop_mean": wass_u_prop_mean,
#             "wass_u_base_mean": wass_u_base_mean,
#             "wass_u_prop_std": wass_u_prop_std,
#             "wass_u_base_std": wass_u_base_std,
#         })

#     return pd.DataFrame(rows)
def sweep_direction_parallel_twosample_ks(
    *,
    model,
    model_enc,
    device,
    latent_batch_size,
    deltas,
    n_samples=2000,
    mu_base=(0.0, 0.0),
    cov_base=np.diag([1.0, 1.0]),
    mean_direction=None,
    cov_direction=None,
    truncate_box=(-3.0, 3.0),
    seed=1,
    n_reps=10,
    n_jobs=-1,
    backend="threading",
):
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    mu_base = np.asarray(mu_base, float).reshape(2)
    cov_base = np.asarray(cov_base, float).reshape(2, 2)

    if mean_direction is not None:
        mean_direction = np.asarray(mean_direction, float).reshape(2)
    if cov_direction is not None:
        cov_direction = np.asarray(cov_direction, float).reshape(2, 2)

    if device.type == "cuda" and n_jobs != 1:
        print(f"[sweep] device=cuda: forcing n_jobs=1 (requested {n_jobs}) to avoid GPU contention.")
        n_jobs = 1

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=len(deltas) * n_reps, dtype=np.uint32)

    def _mean_std(x):
        x = np.asarray(x, float)
        return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    def _build_shifted_params(delta):
        mu_shift = mu_base.copy()
        cov_shift = cov_base.copy()

        if mean_direction is not None:
            mu_shift = mu_shift + float(delta) * mean_direction

        if cov_direction is not None:
            cov_shift = cov_shift + float(delta) * cov_direction
            cov_shift = 0.5 * (cov_shift + cov_shift.T)

            eigvals = np.linalg.eigvalsh(cov_shift)
            if np.min(eigvals) <= 0:
                return None, None

        return mu_shift, cov_shift

    def one(delta, rep_seed):
        rep_seed = int(rep_seed)

        mu_shift, cov_shift = _build_shifted_params(delta)
        if mu_shift is None:
            return {
                "delta": float(delta),
                "valid_cov": False,
            }

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

        v0_p = encode_to_latent(model, X0, device, batch_size=latent_batch_size)[:, 0]
        v1_p = encode_to_latent(model, X1, device, batch_size=latent_batch_size)[:, 0]
        v0_b = encode_to_latent(model_enc, X0, device, batch_size=latent_batch_size)[:, 0]
        v1_b = encode_to_latent(model_enc, X1, device, batch_size=latent_batch_size)[:, 0]

        ks_p_prop = ks_2sample_pvalue(v0_p, v1_p)
        ks_p_base = ks_2sample_pvalue(v0_b, v1_b)
        cvm_p_prop = _cvm_2sample_pvalue(v0_p, v1_p)
        cvm_p_base = _cvm_2sample_pvalue(v0_b, v1_b)
        ad_p_prop = _anderson_2sample_pvalue(v0_p, v1_p)
        ad_p_base = _anderson_2sample_pvalue(v0_b, v1_b)

        ref_rng = np.random.default_rng(rep_seed + 424242)
        ref_u_np = ref_rng.uniform(0.0, 1.0, size=v1_p.shape[0]).astype(np.float32)
        uniform_prop = _uniform_reference_metrics(v1_p, device=device, ref_u_np=ref_u_np)
        uniform_base = _uniform_reference_metrics(v1_b, device=device, ref_u_np=ref_u_np)

        return {
            "delta": float(delta),
            "valid_cov": True,
            "mu1": float(mu_shift[0]),
            "mu2": float(mu_shift[1]),
            "cov11": float(cov_shift[0, 0]),
            "cov12": float(cov_shift[0, 1]),
            "cov22": float(cov_shift[1, 1]),
            "ks_p_prop": float(ks_p_prop),
            "ks_p_base": float(ks_p_base),
            "cvm_p_prop": float(cvm_p_prop),
            "cvm_p_base": float(cvm_p_base),
            "ad_p_prop": float(ad_p_prop),
            "ad_p_base": float(ad_p_base),
            "ks_u_stat_prop": uniform_prop["ks_stat"],
            "ks_u_stat_base": uniform_base["ks_stat"],
            "ks_u_p_prop": uniform_prop["ks_pvalue"],
            "ks_u_p_base": uniform_base["ks_pvalue"],
            "mmd_u_prop": uniform_prop["mmd"],
            "mmd_u_base": uniform_base["mmd"],
            "wass_u_prop": uniform_prop["wasserstein"],
            "wass_u_base": uniform_base["wasserstein"],
        }

    tasks = [(delta, seeds[i * n_reps + j]) for i, delta in enumerate(deltas) for j in range(n_reps)]
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(one)(delta, s) for delta, s in tasks
    )

    rows = []
    for i, delta in enumerate(deltas):
        chunk = results[i * n_reps:(i + 1) * n_reps]
        valid_chunk = [r for r in chunk if r["valid_cov"]]

        if not valid_chunk:
            rows.append({
                "delta": float(delta),
                "valid_cov": False,
                "mu1": np.nan,
                "mu2": np.nan,
                "cov11": np.nan,
                "cov12": np.nan,
                "cov22": np.nan,
            })
            continue

        r0 = valid_chunk[0]

        def arr(key):
            return np.array([r[key] for r in valid_chunk], dtype=float)

        ks_p_prop = arr("ks_p_prop")
        ks_p_base = arr("ks_p_base")
        cvm_p_prop = arr("cvm_p_prop")
        cvm_p_base = arr("cvm_p_base")
        ad_p_prop = arr("ad_p_prop")
        ad_p_base = arr("ad_p_base")
        ks_u_stat_prop = arr("ks_u_stat_prop")
        ks_u_stat_base = arr("ks_u_stat_base")
        ks_u_p_prop = arr("ks_u_p_prop")
        ks_u_p_base = arr("ks_u_p_base")
        mmd_u_prop = arr("mmd_u_prop")
        mmd_u_base = arr("mmd_u_base")
        wass_u_prop = arr("wass_u_prop")
        wass_u_base = arr("wass_u_base")

        rows.append({
            "delta": float(delta),
            "valid_cov": True,
            "mu1": r0["mu1"],
            "mu2": r0["mu2"],
            "cov11": r0["cov11"],
            "cov12": r0["cov12"],
            "cov22": r0["cov22"],
            "pval": ks_p_prop,
            "pval_base": ks_p_base,
            "ks_p_prop": ks_p_prop,
            "ks_p_base": ks_p_base,
            "cvm_p_prop": cvm_p_prop,
            "cvm_p_base": cvm_p_base,
            "ad_p_prop": ad_p_prop,
            "ad_p_base": ad_p_base,
            "ks_u_stat_prop": ks_u_stat_prop,
            "ks_u_stat_base": ks_u_stat_base,
            "ks_u_p_prop": ks_u_p_prop,
            "ks_u_p_base": ks_u_p_base,
            "mmd_u_prop": mmd_u_prop,
            "mmd_u_base": mmd_u_base,
            "wass_u_prop": wass_u_prop,
            "wass_u_base": wass_u_base,
            "ks_p_prop_mean": _mean_std(ks_p_prop)[0],
            "ks_p_base_mean": _mean_std(ks_p_base)[0],
            "ks_p_prop_std": _mean_std(ks_p_prop)[1],
            "ks_p_base_std": _mean_std(ks_p_base)[1],
            "cvm_p_prop_mean": _mean_std(cvm_p_prop)[0],
            "cvm_p_base_mean": _mean_std(cvm_p_base)[0],
            "cvm_p_prop_std": _mean_std(cvm_p_prop)[1],
            "cvm_p_base_std": _mean_std(cvm_p_base)[1],
            "ad_p_prop_mean": _mean_std(ad_p_prop)[0],
            "ad_p_base_mean": _mean_std(ad_p_base)[0],
            "ad_p_prop_std": _mean_std(ad_p_prop)[1],
            "ad_p_base_std": _mean_std(ad_p_base)[1],
            "ks_u_stat_prop_mean": _mean_std(ks_u_stat_prop)[0],
            "ks_u_stat_base_mean": _mean_std(ks_u_stat_base)[0],
            "ks_u_stat_prop_std": _mean_std(ks_u_stat_prop)[1],
            "ks_u_stat_base_std": _mean_std(ks_u_stat_base)[1],
            "mmd_u_prop_mean": _mean_std(mmd_u_prop)[0],
            "mmd_u_base_mean": _mean_std(mmd_u_base)[0],
            "mmd_u_prop_std": _mean_std(mmd_u_prop)[1],
            "mmd_u_base_std": _mean_std(mmd_u_base)[1],
            "wass_u_prop_mean": _mean_std(wass_u_prop)[0],
            "wass_u_base_mean": _mean_std(wass_u_base)[0],
            "wass_u_prop_std": _mean_std(wass_u_prop)[1],
            "wass_u_base_std": _mean_std(wass_u_base)[1],
        })

    return pd.DataFrame(rows)


# In[29]:


deltas = np.linspace(0, 1, 21)
truncate_box = (-3.0, 3.0)
n_reps = 50
n_jobs= 1 if device.type == "cuda" else 15

if Cfg.perturb_type == "mean":
    df1 = sweep_direction_parallel_twosample_ks(mu_base=(0,0),  mean_direction=(0,1), deltas=deltas,
                                    truncate_box=truncate_box, n_reps=n_reps,n_jobs=n_jobs)
    df2 = sweep_direction_parallel_twosample_ks(mu_base=(0,0),  mean_direction=(1,0), deltas=deltas,
                                    truncate_box=truncate_box, n_reps=n_reps,n_jobs=n_jobs)
    df3 = sweep_direction_parallel_twosample_ks(mu_base=(-1,0), mean_direction=(0,1), deltas=deltas,
                                    truncate_box=truncate_box, n_reps=n_reps,n_jobs=n_jobs)

if Cfg.perturb_type == "cov":
    cov_ref = np.diag([1.0, 1.0])
    # 1) var-1 decrease: Sigma(delta) = [[1 - delta, 0], [0, 1]]
    deltas = np.linspace(0, 0.8, 21)
    df1 = sweep_direction_parallel_twosample_ks(
        model=model,
        model_enc=model_enc,
        device=device,
        latent_batch_size=Cfg.latent_batch_size,
        mu_base=(0, 0),
        cov_base=cov_ref,
        cov_direction=np.array([[-1.0, 0.0], [0.0, 0.0]]),
        deltas=deltas,
        truncate_box=truncate_box,
        n_reps=n_reps,
        n_jobs=n_jobs,
    )

    # 2) var-1 increase: Sigma(delta) = [[1 + delta, 0], [0, 1]]
    deltas = np.linspace(0, 0.8, 21)
    df2 = sweep_direction_parallel_twosample_ks(
        model=model,
        model_enc=model_enc,
        device=device,
        latent_batch_size=Cfg.latent_batch_size,
        mu_base=(0, 0),
        cov_base=cov_ref,
        cov_direction=np.array([[1.0, 0.0], [0.0, 0.0]]),
        deltas=deltas,
        truncate_box=truncate_box,
        n_reps=n_reps,
        n_jobs=n_jobs,
    )

    # 3) correlation shift: Sigma(delta) = [[1, delta], [delta, 1]]
    deltas = np.linspace(0, 1, 21)
    cov_ref_3 = np.diag([1.6, 1.0])
    df3 = sweep_direction_parallel_twosample_ks(
        model=model,
        model_enc=model_enc,
        device=device,
        latent_batch_size=Cfg.latent_batch_size,
        mu_base=(0, 0),
        cov_base=cov_ref_3,
        cov_direction=np.array([[0.0, 1.0], [1.0, 0.0]]),
        deltas=deltas,
        truncate_box=truncate_box,
        n_reps=n_reps,
        n_jobs=n_jobs,
    )



# In[ ]:


dfs = {
    "df1": df1,
    "df2": df2,
    "df3": df3,
}

summary_frames = []

for name, df in dfs.items():
    # pick method columns = everything except metadata columns
    meta_cols = {"delta", "rep", "seed", "mu_base", "direction"}
    method_cols = [c for c in df.columns if c not in meta_cols]

    power_df = (
        df.groupby("delta")[method_cols]
        .mean()
        .reset_index()
    )
    power_df.insert(0, "setting", name)
    summary_frames.append(power_df)

summary_df = pd.concat(summary_frames, ignore_index=True)

alpha = 0.1
baseline_col = "pval_base"
proposed_col = "pval"

def power_by_delta(df):
    tmp = df.copy()
    tmp[baseline_col] = tmp[baseline_col].apply(lambda p: np.mean(np.asarray(p) <= alpha)).to_numpy()
    tmp[proposed_col] = tmp[proposed_col].apply(lambda p: np.mean(np.asarray(p) <= alpha)).to_numpy()
    # tmp[baseline_col] = (tmp[baseline_col] < alpha).astype(float)
    # tmp[proposed_col] = (tmp[proposed_col] < alpha).astype(float)
    return tmp.groupby("delta")[[baseline_col, proposed_col]].mean()

power1 = power_by_delta(df1).assign(setting="df1")
power2 = power_by_delta(df2).assign(setting="df2")
power3 = power_by_delta(df3).assign(setting="df3")

gap_df = pd.concat([power1, power2, power3]).reset_index()
gap_df["gap"] = gap_df[baseline_col] - gap_df[proposed_col] 
# baseline power - proposed power
# under the null, we want this to be positive and large, so we have better type-I error control
# under alternative, we want this to be negative and larger absolute difference better, meaning proposed has higher power than baseline
gap_table = gap_df.pivot(index="setting", columns="delta", values="gap")

# gap_table["row_mean"] = gap_table.mean(axis=1)
# display(gap_table)
# gap_table.mean(axis=1)


# In[53]:


gap_table.columns = gap_table.columns.astype(float)

df1_gap_mean = gap_table.loc["df1"].mean() # want this positive, 
df2_before_05_mean = gap_table.loc["df2", gap_table.columns < 0.2].mean() # want this positive 
df2_after_05_mean = gap_table.loc["df2", gap_table.columns > 0.2].mean() # want this negative 
df3_gap_mean = gap_table.loc["df3"].mean() # want this negative 

mean_summary = pd.DataFrame(
    [
        {"metric": "df1_gap_mean", "value": float(df1_gap_mean)},
        {"metric": "df2_before_0.5_mean", "value": float(df2_before_05_mean)},
        {"metric": "df2_after_0.5_mean", "value": float(df2_after_05_mean)},
        {"metric": "df3_gap_mean", "value": float(df3_gap_mean)},
    ]
)

output_dir = PWD / Cfg.output_dir
output_dir.mkdir(parents=True, exist_ok=True)

history_df = pd.DataFrame(history)
train_recon_snapshot = build_reconstruction_snapshot(train_loader, "train")
val_recon_snapshot = build_reconstruction_snapshot(val_loader, "val") if use_val else pd.DataFrame()

# Raw tables with array-valued columns are safest as pickle.
df1.to_pickle(output_dir / "df1.pkl")
df2.to_pickle(output_dir / "df2.pkl")
df3.to_pickle(output_dir / "df3.pkl")
train_recon_snapshot.to_pickle(output_dir / "train_reconstruction_snapshot.pkl")
val_recon_snapshot.to_pickle(output_dir / "val_reconstruction_snapshot.pkl")

# Save flat summaries as csv for quick inspection.
history_df.to_csv(output_dir / "training_history.csv", index=False)
summary_df.to_csv(output_dir / "summary_df.csv", index=False)
gap_df.to_csv(output_dir / "gap_df.csv", index=False)
gap_table.to_csv(output_dir / "gap_table.csv")
mean_summary.to_csv(output_dir / "mean_summary.csv", index=False)

with open(output_dir / "mean_summary.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "df1_gap_mean": float(df1_gap_mean),
            "df2_before_0.5_mean": float(df2_before_05_mean),
            "df2_after_0.5_mean": float(df2_after_05_mean),
            "df3_gap_mean": float(df3_gap_mean),
        },
        f,
        indent=2,
    )

print(f"Saved results to {output_dir}")
print(mean_summary.to_string(index=False))
print("files saved to:", output_dir)
