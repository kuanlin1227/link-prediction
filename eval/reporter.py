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
    "degree":                "M0: Degree Stats",
    "tfidf":                 "M1: TF-IDF",
    "e5_small":              "M2: E5-Small (LLM)",
    "llm_keywords":          "M6: LLM Keywords",
    "llm_embed":             "llm_embed+GNN",
    "llm_embed_lora":        "llm_embed_lora+GNN",
    "llm_graph_embed":       "llm+GAE+GNN",
    "gae_embed":             "GAE+GNN",
    "degree_llm_embed":      "Degree+LLM+GNN",
    "degree_llm_embed_lora": "Degree+LoRA-LLM+GNN",
    "llm_pairwise":          "M7: LLM Pairwise (70b)",
    "llm_pairwise_groq":     "LLM Pairwise (70b, Groq, sampled)",
    "llm_pairwise_majority": "llm_majority",
    "llm_lora":              "llm_lora",
}

SPARSITY_LABELS = {
    0.2: "Sparse (20%)",
    0.5: "Medium (50%)",
    0.8: "Dense  (80%)",
}

COLORS = {
    "degree":                "#6B7280",
    "tfidf":                 "#F59E0B",
    "e5_small":              "#3B82F6",
    "llm_keywords":          "#EC4899",
    "llm_embed":             "#0EA5E9",
    "llm_embed_lora":        "#6366F1",
    "llm_graph_embed":       "#F97316",
    "gae_embed":             "#84CC16",
    "degree_llm_embed":      "#D97706",
    "degree_llm_embed_lora": "#92400E",
    "llm_pairwise":          "#EF4444",
    "llm_pairwise_groq":     "#DC2626",
    "llm_pairwise_majority": "#7C3AED",
    "llm_lora":              "#10B981",
}


class Reporter:
    def __init__(self):
        self.records = []

    def add(self, feature_name: str, sparsity: float, run_results: list):
        def col(key):
            return [r[key] for r in run_results if key in r]

        rec = {
            'feature':  feature_name,
            'sparsity': sparsity,
            'auc_mean': np.mean(col('auc')), 'auc_std': np.std(col('auc')),
            'ap_mean':  np.mean(col('ap')),  'ap_std':  np.std(col('ap')),
        }
        # 衍生指標：mean ± std（pairwise 等沒提供時填 NaN）
        for key in ('accuracy', 'precision', 'recall', 'f1', 'kappa'):
            vals = col(key)
            rec[f'{key}_mean'] = np.mean(vals) if vals else np.nan
            rec[f'{key}_std']  = np.std(vals)  if vals else np.nan
        # 混淆矩陣計數：取各 run 平均
        for key in ('tp', 'fp', 'fn', 'tn'):
            vals = col(key)
            rec[key] = np.mean(vals) if vals else np.nan
        self.records.append(rec)

    @staticmethod
    def _fmt(mean, std):
        if mean is None or (isinstance(mean, float) and np.isnan(mean)):
            return "-"
        return f"{mean:.4f} ± {std:.4f}"

    @staticmethod
    def _fmt_count(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return f"{v:.1f}"

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            rows.append({
                '特徵方法':   FEATURE_LABELS.get(r['feature'], r['feature']),
                '訓練邊比例': SPARSITY_LABELS.get(r['sparsity'], str(r['sparsity'])),
                'AUC':        f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
                'AP':         f"{r['ap_mean']:.4f} ± {r['ap_std']:.4f}",
                'F1':         self._fmt(r.get('f1_mean'),        r.get('f1_std')),
                'Precision':  self._fmt(r.get('precision_mean'), r.get('precision_std')),
                'Recall':     self._fmt(r.get('recall_mean'),    r.get('recall_std')),
                'Accuracy':   self._fmt(r.get('accuracy_mean'),  r.get('accuracy_std')),
                'Kappa':      self._fmt(r.get('kappa_mean'),     r.get('kappa_std')),
                'TP':         self._fmt_count(r.get('tp')),
                'FP':         self._fmt_count(r.get('fp')),
                'FN':         self._fmt_count(r.get('fn')),
                'TN':         self._fmt_count(r.get('tn')),
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
        self._plot_sparsity_line(out_dir)

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

    # ── 圖二：AUC vs Sparsity 線圖 ───────────────────────────────────
    def _plot_sparsity_line(self, out_dir):
        """各方法的 AUC 隨稀疏度變化的折線圖，M7 以虛線呈現（與稀疏度無關）。"""
        sparsities = sorted(set(r['sparsity'] for r in self.records))
        features   = list(dict.fromkeys(r['feature'] for r in self.records))

        fig, ax = plt.subplots(figsize=(8, 5))

        for feat in features:
            aucs, stds, xs = [], [], []
            for sp in sparsities:
                rec = next((r for r in self.records
                            if r['feature'] == feat and r['sparsity'] == sp), None)
                if rec:
                    aucs.append(rec['auc_mean'])
                    stds.append(rec['auc_std'])
                    xs.append(sp)
            if not aucs:
                continue
            label  = FEATURE_LABELS.get(feat, feat)
            color  = COLORS.get(feat, '#999')
            is_m7  = feat == 'llm_pairwise'
            ax.errorbar(
                xs, aucs, yerr=stds,
                marker='s' if is_m7 else 'o',
                label=label, color=color,
                linewidth=2, markersize=7, capsize=4,
                linestyle='--' if is_m7 else '-',
                alpha=0.9,
            )

        ax.set_xlabel("Training Edge Ratio (Graph Sparsity)", fontsize=12)
        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_title(
            "Link Prediction AUC vs Graph Sparsity\n(GraphSAGE on ogbn-arxiv subset)",
            fontsize=13,
        )
        ax.set_xticks(sparsities)
        ax.set_xticklabels([SPARSITY_LABELS.get(s, str(s)) for s in sparsities])
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(alpha=0.3)
        fig.tight_layout()

        path = os.path.join(out_dir, "sparsity.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[reporter] 圖表已存至 {path}")

    # ── 圖三：LLM 特徵的邊際增益（vs TF-IDF baseline）────────────────
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
