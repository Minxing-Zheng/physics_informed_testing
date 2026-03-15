from pathlib import Path
import pandas as pd

base = Path("results/diagnostics_results")
rows = []

target_lr = 3e-3
target_lambdas = {1e-3,3e-3, 5e-3,7e-3,9e-3, 1e-2, 5e-2}
target_seeds = {1, 2, 3, 4, 5}

for path in sorted(base.glob("seed_*_lr_*_lambda_mmd_*/mean_summary.csv")):
    parts = path.parent.name.split("_")
    seed = int(parts[1])
    lr = float(parts[3].replace("p", "."))
    lambda_mmd = float(parts[6].replace("p", "."))

    if lr != target_lr:
        continue
    if lambda_mmd not in target_lambdas:
        continue
    if seed not in target_seeds:
        continue

    df = pd.read_csv(path)

    row = {
        "seed": seed,
        "lr": lr,
        "lambda_mmd": lambda_mmd,
        "run_dir": path.parent.name,
    }

    for metric in [
        "df1_gap_mean",
        "df2_before_0.5_mean",
        "df2_after_0.5_mean",
        "df3_gap_mean",
    ]:
        row[metric] = df.loc[df["metric"] == metric, "value"].iloc[0]

    rows.append(row)

out = (
    pd.DataFrame(rows)
    .sort_values(["lambda_mmd", "seed"])
    .reset_index(drop=True)
)
lambda_order = sorted(out["lambda_mmd"].unique())
lambda_to_color = {
    lam: ("background-color: white" if i % 2 == 0 else "background-color: #eeeeee")
    for i, lam in enumerate(lambda_order)
}

def shade_by_lambda(row):
    color = lambda_to_color[row["lambda_mmd"]]
    return [color] * len(row)

styled = out.style.apply(shade_by_lambda, axis=1)
styled
# print(out.to_string(index=False))
# html = styled.to_html()
# with open("lambda_grouped_table.html", "w", encoding="utf-8") as f:
#     f.write(html)
