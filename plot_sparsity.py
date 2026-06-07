"""
從 results/comparison_table.csv 讀取現有結果，畫出 sparsity.png。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA = {
    "M0: Degree Stats": {
        "Sparse (20%)":  (0.7844, 0.0199),
        "Medium (50%)":  (0.8088, 0.0201),
        "Dense  (80%)":  (0.8456, 0.0428),
    },
    "M1: TF-IDF": {
        "Sparse (20%)":  (0.8056, 0.0029),
        "Medium (50%)":  (0.8972, 0.0049),
        "Dense  (80%)":  (0.9263, 0.0061),
    },
    "M11: LLM-LoRA+GNN": {
        "Sparse (20%)":  (0.7991, 0.0092),
        "Medium (50%)":  (0.8870, 0.0077),
        "Dense  (80%)":  (0.8721, 0.0523),
    },
    "Degree+LLM+GNN": {
        "Sparse (20%)":  (0.8236, 0.0056),
        "Medium (50%)":  (0.8905, 0.0005),
        "Dense  (80%)":  (0.9231, 0.0004),
    },
    "M7: LLM Pairwise (70b)": {
        "Sparse (20%)":  (0.8456, 0.0000),
        "Medium (50%)":  (0.8456, 0.0000),
        "Dense  (80%)":  (0.8456, 0.0000),
    },
}

COLORS = {
    "M0: Degree Stats":       "#6B7280",
    "M1: TF-IDF":             "#F59E0B",
    "M11: LLM-LoRA+GNN":      "#6366F1",
    "Degree+LLM+GNN":         "#D97706",
    "M7: LLM Pairwise (70b)": "#EF4444",
}

SPARSITY_LABELS = ["Sparse (20%)", "Medium (50%)", "Dense  (80%)"]
METHODS = list(DATA.keys())

n_sparse = len(SPARSITY_LABELS)
n_method = len(METHODS)
x        = np.arange(n_sparse)
width    = 0.15
offsets  = np.linspace(-(n_method - 1) / 2, (n_method - 1) / 2, n_method) * width

fig, ax = plt.subplots(figsize=(10, 5))

for i, method in enumerate(METHODS):
    means = [DATA[method][sp][0] for sp in SPARSITY_LABELS]
    stds  = [DATA[method][sp][1] for sp in SPARSITY_LABELS]
    bars  = ax.bar(
        x + offsets[i], means, width,
        yerr=stds, capsize=4,
        color=COLORS[method], alpha=0.88,
        label=method,
        error_kw={"elinewidth": 1.2, "ecolor": "#333"},
    )

ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title(
    "Link Prediction AUC by Feature Type & Graph Sparsity\n(GraphSAGE on ogbn-arxiv subset)",
    fontsize=13,
)
ax.set_xticks(x)
ax.set_xticklabels(SPARSITY_LABELS, fontsize=11)
ax.set_ylim(0.70, 1.00)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
path = "results/figures/sparsity.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"圖表已存至 {path}")
