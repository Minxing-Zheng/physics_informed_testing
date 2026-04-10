import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("tsa_test_ratio_experiment_v2_results_sweep_summary.csv")
df = df.sort_values(["model_type", "unstable_ratio"])

# 名字映射（和 Table 一致）
name_map = {
    "bce_classifier": "Binary-clf.",
    "context_mmd": "MMD-on-X",
    "mixed_piht": "L2T",
    "mmd_only": "Uniform-only",
    "trajectory_only": "Recon-only"
}

# 线型和 marker 映射
style_map = {
    "context_mmd":      {"linestyle": "-",  "marker": "o", "linewidth": 2.2, "markersize": 5},
    "bce_classifier":   {"linestyle": "--", "marker": "s", "linewidth": 2.2, "markersize": 5},
    "trajectory_only":  {"linestyle": "-.", "marker": "^", "linewidth": 2.2, "markersize": 5},
    "mmd_only":         {"linestyle": ":",  "marker": "D", "linewidth": 2.4, "markersize": 5},
    "mixed_piht":       {"linestyle": "-",  "marker": "o", "linewidth": 3.6, "markersize": 7},  # 高亮 L2T
}

plt.figure(figsize=(7.2, 5.2))

# 按固定顺序画，保证图例顺序稳定
plot_order = ["context_mmd", "bce_classifier", "trajectory_only", "mmd_only", "mixed_piht"]

for model in plot_order:
    sub = df[df["model_type"] == model]
    label = name_map[model]
    style = style_map[model]

    plt.plot(
        sub["unstable_ratio"],
        sub["reject_rate"],
        label=label,
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=style["linewidth"],
        markersize=style["markersize"],
    )

plt.xlabel("Unstable Ratio", fontsize=12)
plt.ylabel("Rejection Rate", fontsize=12)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.grid(alpha=0.25)

# 图例顺序和 Table 一致
legend_order = ["MMD-on-X", "Binary-clf.", "Recon-only", "Uniform-only", "L2T"]
handles, labels = plt.gca().get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))
plt.legend(
    [label_to_handle[l] for l in legend_order if l in label_to_handle],
    [l for l in legend_order if l in label_to_handle],
    fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.savefig("TSA_Sweep.png", dpi=300, bbox_inches="tight")
plt.show()