# Learning to Test (L2T): Physics-Informed Hypothesis Test

This `share_code` package is prepared for collaborators to run `experiment/run_baseline.py` on:

- built-in synthetic spring data, or
- personal data provided as a `.pkl` file.

## 1. Setup Conda Environment (`l2t`)

From the `share_code` directory:

```bash
cd share_code
conda env create -f environment.yml
conda activate l2t
```

If the environment already exists and you updated `environment.yml`:

```bash
conda env update -f environment.yml --prune
conda activate l2t
```

## 2. Toy Notebooks

Toy notebooks are in `notebook/`:

- `spring_example.ipynb`

It contains visualizations and a toy model training demonstration.

## 3. Main Experiment Script: `run_baseline.py`

## Data Modes

### A) No `--input-data` (default)

If `--input-data` is not provided, the script generates synthetic spring-mass-damper trajectories internally using:

- `--n-samples`
- `--dist-mean`
- `--dist-cov-diag`
- `--T`, `--dt`, `--stride`

### B) With `--input-data`

If `--input-data` is provided, the script loads your personal dataset from a pickle file.

Use a file path (absolute path recommended), e.g.:

- `--input-data /path/to/my_data.pkl`

`--input-data` must point to a pickle file containing a dict:

- `X`: numpy array of shape `(N, d_x)`, float
- `y`: numpy array of shape `(N,)`, binary values in `{0, 1}`
- `traj`: dict with key:
  - `traj["s"]`: numpy array of shape `(N, T)`

Constraints:

- `N` must match across `X`, `y`, and `traj["s"]`
- `y` must be binary (`0/1`)
- `time_window` must overlap the sequence time support; otherwise the script raises an error

### Minimal example to create a valid `.pkl`

```python
import pickle
import numpy as np

# Example placeholders
N, d_x, T = 1000, 2, 201
X = np.random.randn(N, d_x).astype(np.float32)
y = (np.random.rand(N) > 0.5).astype(np.float32)
s = np.random.randn(N, T).astype(np.float32)

payload = {
    "X": X,
    "y": y,
    "traj": {"s": s},
}

with open("my_data.pkl", "wb") as f:
    pickle.dump(payload, f)
```

## Main Configs (What Each Flag Is For)

### Model Selection and Run Mode

- `--models`: which methods to run. Valid presets: `all`, `proposed_only`, `baselines_only`. Valid custom names: `proposed,uniform_weight,reconstruction_only,regularization_only,label_bce`.
- `--input-data PATH`: path to one `.pkl`/`.pickle` file in the format described above. If omitted, synthetic spring data is generated.
- `--skip-sweeps`: if set, skips hypothesis testing and only runs training/evaluation.
- `--perturb_type`: sweep type when sweeps are enabled. Valid options: `mean`, `cov`, works for synthetic data.

### Synthetic Data Generation (Used Only When `--input-data` Is Omitted)

- `--n-samples`: number of training/validation samples generated for synthetic data.
- `--test-n-samples`: number of samples used per sweep comparison setting.
- `--dist-mean MU_X MU_Y`: Gaussian mean for synthetic context `X`.
- `--dist-cov-diag VAR_X VAR_Y`: diagonal Gaussian covariance entries for synthetic context `X`.
- `--T`: total simulation time horizon for spring trajectories.
- `--dt`: simulation time step.
- `--stride`: subsampling stride used when building sequence pairs.

### Loss and Stability Settings

- `--loss_type`: stability-loss mode. Use `stability_asymmetric_recon`.
- `--tau`: safety threshold used to derive/score stability labels.
- `--time-window T_START T_END`: time interval used by stability checks. Must lie inside the effective sequence time range, otherwise the script raises an error.
- `--asym-recon-delta`: stability asymmetric reconstruction weighting parameter (must be `>= 1.0`).
- `--lambda-mmd`: weight on divergence (MMD) regularization.

### Optimization and Capacity

- `--epochs`: number of training epochs.
- `--batch-train`, `--batch-val`: training and validation batch sizes.
- `--num-workers`: DataLoader worker count.
- `--lr`, `--weight-decay`: optimizer settings for sequence models.
- `--latent-batch-size`: batch size used for latent encoding during sweeps/evaluation.
- `--d-x`, `--d-o`, `--d-v`, `--d-h`, `--hidden-enc`, `--hidden-dec`: input data and model architecture dimensions. For external data, `d_x`/`d_o` are auto-adjusted from input arrays.

### Reproducibility and Outputs

- `--seed`: random seed used for sampling/splitting/training.
- `--output-dir`: output directory for configs, metrics, and artifacts.

## Example Commands

### 1 Standard synthetic run (all models)

```bash
python experiment/run_baseline.py \
  --models all \
  --epochs 100 \
  --n-samples 2000 \
  --test-n-samples 1000 \
  --num-workers 0
```

### 2 Quick smoke run (proposed only, no sweeps)

```bash
python experiment/run_baseline.py \
  --models proposed_only \
  --epochs 100 \
  --n-samples 200 \
  --test-n-samples 100 \
  --skip-sweeps \
  --num-workers 0
```

### 3 Baselines only

```bash
python experiment/run_baseline.py \
  --models baselines_only \
  --epochs 100 \
  --n-samples 2000 \
  --test-n-samples 1000
```

### 4 Personal data run (recommended: no sweeps)

```bash
python experiment/run_baseline.py \
  --input-data /path/to/my_data.pkl \
  --models proposed_only \
  --time-window 0.1 2.0 \
  --epochs 100 \
  --skip-sweeps \
  --output-dir results/my_personal_run
```



## 4. Output Files

Outputs are written to `--output-dir` (default under `results/diagnostics_results_baselines/...`).

Common files:

- `run_config.json`
- `training_history_<model>.csv` (for selected models only)
- `train_reconstruction_snapshot_<model>.pkl` / `val_reconstruction_snapshot_<model>.pkl`
  - generated for sequence models: `proposed`, `uniform_weight`, `reconstruction_only`

If sweeps are enabled, additional files include:

- `df1.pkl`, `df2.pkl`, `df3.pkl`
- `summary_df.csv`
- `rejection_rate_summary.csv`
- `label_probability_summary.csv` (if `label_bce` is selected)
- `gap_df.csv`, `gap_table.csv`, `mean_summary.csv`, `mean_summary.json`

If sweeps are skipped, summary files are still created but may be empty.

