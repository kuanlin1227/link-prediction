"""
eval/reporter.py
實驗結果匯整、輸出比較表、畫圖
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


FEATURE_LABELS = {
    "degree":       "M0: Degree Stats",
    "tfidf":        "M1: TF-IDF",
    "e5_small":     "M2: E5-Small (LLM)",
    "llm_keywords": "M6: LLM Keywords",
    "llm_pairwise": "M7: LLM Pairwise",
}

SPARSITY_LABELS = {
    0.2: "Sparse (20%)",
    0.5: "Medium (50%)",
    0.8: "Dense  (80%)",
}

COLORS = {
    "degree":       "#6B7280",
    "tfidf":        "#F59E0B",
    "e5_small":     "#3B82F6",
    "llm_keywords": "#EC4899",
    "llm_pairwise": "#EF4444",
}


class Reporter:
    def __init__(self):
        self.records = []

    def add(self, feature_name: str, sparsity: float, run_results: list):
        aucs = [r['auc'] for r in run_results]
        aps  = [r['ap']  for r in run_results]
        self.records.append({
            'feature':      feature_name,
            'sparsity':     sparsity,
            'auc_mean':     np.mean(aucs),
            'auc_std':      np.std(aucs),
            'ap_mean':      np.mean(aps),
            'ap_std':       np.std(aps),
        })

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({
                '特徵方法':   FEATURE_LABELS.get(r['feature'], r['feature']),
                '訓練邊比例': SPARSITY_LABELS.get(r['sparsity'], str(r['sparsity'])),
                'AUC':        f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
                'AP':         f"{r['ap_mean']:.4f} ± {r['ap_std']:.4f}",
            })
        return pd.DataFrame(rows)

    def print_table(self):
        df = self.to_dataframe()
        print("\n" + "="*70)
        print("實驗結果比較表（GraphSAGE on Cora）")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70 + "\n")

    def save_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False, encoding='utf-8-sig')
        print(f"[reporter] 結果已存至 {path}")

    def plot(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self._plot_bar_by_sparsity(out_dir)
        self._plot_llm_gain(out_dir)

    # ── 圖一：各稀疏度下的 AUC 比較 ──────────────────────────────────
    def _plot_bar_by_sparsity(self, out_dir):
        sparsities = sorted(set(r['sparsity'] for r in self.records))
        features   = list(dict.fromkeys(r['feature'] for r in self.records))
        n_feat     = len(features)
        n_sparse   = len(sparsities)

        fig, axes = plt.subplots(1, n_sparse, figsize=(5 * n_sparse, 5), sharey=True)
        if n_sparse == 1:
            axes = [axes]

        for ax, sp in zip(axes, sparsities):
            recs  = [r for r in self.records if r['sparsity'] == sp]
            rec_m = {r['feature']: r for r in recs}

            x     = np.arange(n_feat)
            bars  = ax.bar(
                x,
                [rec_m.get(f, {}).get('auc_mean', 0) for f in features],
                yerr=[rec_m.get(f, {}).get('auc_std', 0)  for f in features],
                color=[COLORS.get(f, '#999') for f in features],
                capsize=4, width=0.6, alpha=0.85
            )
            ax.set_title(SPARSITY_LABELS.get(sp, str(sp)), fontsize=12, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels([FEATURE_LABELS.get(f, f).split(":")[0]
                                 for f in features], rotation=30, ha='right')
            ax.set_ylim(0.5, 1.0)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.grid(axis='y', alpha=0.3)

        axes[0].set_ylabel("AUC", fontsize=12)
        fig.suptitle("Link Prediction AUC by Feature Type & Graph Sparsity\n(GraphSAGE on Cora)",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        path = os.path.join(out_dir, "auc_by_sparsity.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[reporter] 圖表已存至 {path}")

    # ── 圖二：LLM 特徵的邊際增益（vs TF-IDF baseline）────────────────
    def _plot_llm_gain(self, out_dir):
        sparsities = sorted(set(r['sparsity'] for r in self.records))
        llm_feats  = ["e5_small"]

        fig, ax = plt.subplots(figsize=(7, 4))

        for feat in llm_feats:
            gains, xs = [], []
            for sp in sparsities:
                tfidf_rec = next((r for r in self.records
                                  if r['feature'] == 'tfidf' and r['sparsity'] == sp), None)
                llm_rec   = next((r for r in self.records
                                  if r['feature'] == feat and r['sparsity'] == sp), None)
                if tfidf_rec and llm_rec:
                    gains.append(llm_rec['auc_mean'] - tfidf_rec['auc_mean'])
                    xs.append(sp)

            if gains:
                ax.plot(xs, gains, marker='o', label=FEATURE_LABELS[feat],
                        color=COLORS[feat], linewidth=2, markersize=7)

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Training Edge Ratio (Graph Density)", fontsize=11)
        ax.set_ylabel("AUC Gain over TF-IDF", fontsize=11)
        ax.set_title("LLM Feature Gain vs Graph Sparsity\n(positive = LLM helps)", fontsize=12)
        ax.set_xticks(sparsities)
        ax.set_xticklabels([SPARSITY_LABELS.get(s, str(s)) for s in sparsities])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(out_dir, "llm_gain_vs_sparsity.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[reporter] 圖表已存至 {path}")
