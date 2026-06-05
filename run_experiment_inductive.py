"""
Cold-Start (Inductive) Link Prediction
研究問題：當「新節點」在訓練時完全沒出現過（degree=0、無任何鄰居），
          純圖結構特徵（M0）會失效，而文字 / LLM 特徵是否仍然有效？

與 run_experiment.py（transductive）的差別：
  - transductive：藏「邊」，所有節點訓練時都看得到 → RandomLinkSplit
  - inductive   ：藏「整個節點」，冷節點及其所有邊在訓練時被移除（本檔）

設計：
  - 隨機把節點分成 train / val / test(cold)
  - 訊息傳遞圖只保留「兩端都是 train」的邊；冷節點在訓練圖中是孤立點
  - 節點特徵在「訓練圖」上計算 → M0 對冷節點退化為 0；文字特徵與圖無關照常可用
  - 測試正邊 = 至少一端是冷節點的真實邊；測試負邊 = 等量隨機非邊
  - 結果寫入獨立資料夾 results_inductive/（不影響 transductive 結果）
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score

# 重用 run_experiment 的資料載入、模型、設定（import 時會套用其 torch.load patch）
from run_experiment import (load_dataset, DEVICE, EPOCHS, LR, HIDDEN, OUT_DIM,
                            RUNS, DATASET_TAG, USE_YEAR_FEATURE,
                            YEAR_MIN, YEAR_MAX)
from features.feature_factory import FeatureFactory
from models.graphsage import LinkSAGE, train_epoch, evaluate
from eval.reporter import FEATURE_LABELS, COLORS


# ── 設定 ──────────────────────────────────────────────────────────────
COLD_RATIOS  = [0.1, 0.2, 0.4]   # 掃描：冷節點（測試）佔比；訓練時整個移除
VAL_RATIO    = 0.1               # 驗證節點佔比（用於 early stopping）
RESULT_DIR   = "results_inductive"

# 走 GraphSAGE 的特徵（節點特徵 → GNN）
GNN_FEATURES = ["degree", "tfidf", "e5_small"]  # "llm_keywords" 已停用
# 邊級 LLM 推理（不經 GNN，與圖結構無關）
PAIRWISE_FEATURE = "llm_pairwise"

# llm_pairwise 在 cold test 邊上最多打分幾條（每類），控制 API 成本
MAX_EVAL_PAIRS_COLD = 200


# ── Inductive 切分 ────────────────────────────────────────────────────
def inductive_split(data, cold_ratio, val_ratio, seed):
    """切「節點」：train / val / cold(test)。回傳訊息傳遞圖與各 split 的監督邊。"""
    N   = data.num_nodes
    gen = torch.Generator().manual_seed(seed)

    perm    = torch.randperm(N, generator=gen)
    n_cold  = int(N * cold_ratio)
    n_val   = int(N * val_ratio)
    cold_nodes = perm[:n_cold]
    val_nodes  = perm[n_cold:n_cold + n_val]

    is_cold = torch.zeros(N, dtype=torch.bool); is_cold[cold_nodes] = True
    is_val  = torch.zeros(N, dtype=torch.bool); is_val[val_nodes]   = True
    is_train = ~(is_cold | is_val)

    ei   = data.edge_index
    s, d = ei[0], ei[1]

    # 三類正邊：train 內部 / 觸及 val（不觸 cold）/ 觸及 cold
    train_mask = is_train[s] & is_train[d]
    val_mask   = (is_val[s] | is_val[d]) & ~(is_cold[s] | is_cold[d])
    cold_mask  = is_cold[s] | is_cold[d]

    train_pos = ei[:, train_mask]
    val_pos   = ei[:, val_mask]
    cold_pos  = ei[:, cold_mask]

    # 訊息傳遞圖：對稱化的 train 邊（冷節點在此圖中為孤立點）
    mp_edge_index = torch.cat([train_pos, train_pos.flip(0)], dim=1)

    # 無向 existing set，供負邊取樣排除
    existing = set()
    sl, dl = s.tolist(), d.tolist()
    for a, b in zip(sl, dl):
        existing.add((a, b)); existing.add((b, a))

    def _sample_neg(k, kind):
        """取 k 條符合 kind 條件的非邊（無向去重）"""
        out, seen = [], set()
        guard = 0
        while len(out) < k and guard < 200:
            guard += 1
            us = torch.randint(0, N, (max(k * 4, 1000),), generator=gen).tolist()
            vs = torch.randint(0, N, (max(k * 4, 1000),), generator=gen).tolist()
            for u, v in zip(us, vs):
                if len(out) >= k:
                    break
                if u == v or (u, v) in existing or (u, v) in seen:
                    continue
                if kind == "train":
                    ok = bool(is_train[u]) and bool(is_train[v])
                elif kind == "val":
                    ok = (bool(is_val[u]) or bool(is_val[v])) and not (bool(is_cold[u]) or bool(is_cold[v]))
                else:  # cold
                    ok = bool(is_cold[u]) or bool(is_cold[v])
                if not ok:
                    continue
                seen.add((u, v)); seen.add((v, u))
                out.append((u, v))
        if not out:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(out, dtype=torch.long).t()

    def _make(pos, kind):
        neg = _sample_neg(pos.shape[1], kind)
        eli = torch.cat([pos, neg], dim=1)
        lab = torch.cat([torch.ones(pos.shape[1]), torch.zeros(neg.shape[1])])
        return eli, lab

    return {
        "num_nodes":     N,
        "mp_edge_index": mp_edge_index,
        "train":         _make(train_pos, "train"),
        "val":           _make(val_pos,   "val"),
        "cold":          _make(cold_pos,  "cold"),
    }


# ── 單次實驗：GNN 特徵 ─────────────────────────────────────────────────
def run_one_inductive(data_full, texts, feature_name, cold_ratio, run_id):
    sp = inductive_split(data_full, cold_ratio, VAL_RATIO, seed=run_id)
    N  = sp["num_nodes"]

    # 特徵在「訓練圖」上計算 → M0 對冷節點退化；文字特徵與圖無關（讀磁碟快取）
    train_graph = Data(edge_index=sp["mp_edge_index"], num_nodes=N)
    factory     = FeatureFactory(train_graph, DEVICE, texts=texts,
                                 dataset_tag=DATASET_TAG)
    x           = factory.get(feature_name)

    # 與主實驗一致：把發表年份當額外一維特徵併入（min-max 正規化到 0~1）
    if USE_YEAR_FEATURE and getattr(data_full, "node_year", None) is not None:
        span    = max(1, YEAR_MAX - YEAR_MIN)
        yr_norm = ((data_full.node_year.float() - YEAR_MIN) / span).clamp(0, 1)
        x       = torch.cat([x, yr_norm.to(x.device)], dim=1)

    in_channels = x.shape[1]

    def _data(key):
        eli, lab = sp[key]
        obj = Data()
        obj.x               = x
        obj.edge_index      = sp["mp_edge_index"]
        obj.edge_label_index = eli
        obj.edge_label       = lab
        return obj

    train_data, val_data, cold_data = _data("train"), _data("val"), _data("cold")

    model     = LinkSAGE(in_channels, HIDDEN, OUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, counter, best_state = 0.0, 0, None
    for _ in range(1, EPOCHS + 1):
        train_epoch(model, optimizer, train_data, DEVICE)
        vm = evaluate(model, train_data, val_data, DEVICE)
        if vm["auc"] > best_val:
            best_val   = vm["auc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
        if counter >= 20:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, train_data, cold_data, DEVICE)


# ── 單次實驗：LLM 邊級推理（M7）──────────────────────────────────────
def run_pairwise_inductive(data_full, texts, cold_ratio, run_id):
    from features.llm_pairwise import score_pairs

    sp       = inductive_split(data_full, cold_ratio, VAL_RATIO, seed=run_id)
    eli, lab = sp["cold"]

    gen      = torch.Generator().manual_seed(run_id)
    pos_cols = (lab == 1).nonzero(as_tuple=True)[0]
    neg_cols = (lab == 0).nonzero(as_tuple=True)[0]

    def _cap(cols):
        if len(cols) > MAX_EVAL_PAIRS_COLD:
            p = torch.randperm(len(cols), generator=gen)[:MAX_EVAL_PAIRS_COLD]
            return cols[p]
        return cols

    sel    = torch.cat([_cap(pos_cols), _cap(neg_cols)])
    pairs  = [(int(eli[0, i]), int(eli[1, i])) for i in sel]
    y      = lab[sel].numpy()
    years  = (data_full.node_year.view(-1).tolist()
              if getattr(data_full, "node_year", None) is not None else None)
    scores = score_pairs(pairs, texts, tag=DATASET_TAG, years=years)
    return {"auc": roc_auc_score(y, scores), "ap": average_precision_score(y, scores)}


# ── 結果輸出 ──────────────────────────────────────────────────────────
def save_and_plot(gnn_records, pairwise, out_dir):
    """
    gnn_records: dict[(feature, cold_ratio)] -> {auc_mean, auc_std, ap_mean, ap_std}
    pairwise:    {auc, ap} 或 None（M7，與冷節點比例無關，畫成水平線）
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    # CSV：列 = 特徵，欄 = 各冷節點比例的 AUC
    rows = []
    for feat in GNN_FEATURES:
        row = {"特徵方法": FEATURE_LABELS.get(feat, feat)}
        for cr in COLD_RATIOS:
            r = gnn_records.get((feat, cr))
            row[f"Cold {cr:.0%}"] = f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}" if r else "-"
        rows.append(row)
    if pairwise is not None:
        row = {"特徵方法": FEATURE_LABELS.get(PAIRWISE_FEATURE, PAIRWISE_FEATURE)}
        for cr in COLD_RATIOS:
            row[f"Cold {cr:.0%}"] = f"{pairwise['auc']:.4f} (圖無關)"
        rows.append(row)

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "comparison_table_inductive.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("Cold-Start (Inductive) AUC — 掃描冷節點比例")
    print("=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)
    print(f"[reporter] 結果已存至 {csv_path}")

    # 趨勢圖：x = 冷節點比例，每個 GNN 特徵一條線；M7 為水平線
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [cr for cr in COLD_RATIOS]

    for feat in GNN_FEATURES:
        ys   = [gnn_records[(feat, cr)]["auc_mean"] for cr in COLD_RATIOS
                if (feat, cr) in gnn_records]
        errs = [gnn_records[(feat, cr)]["auc_std"]  for cr in COLD_RATIOS
                if (feat, cr) in gnn_records]
        if ys:
            ax.errorbar(xs[:len(ys)], ys, yerr=errs, marker="o", capsize=3,
                        linewidth=2, markersize=6,
                        label=FEATURE_LABELS.get(feat, feat),
                        color=COLORS.get(feat, "#999"))

    if pairwise is not None:
        ax.axhline(pairwise["auc"], linestyle="--", linewidth=2,
                   color=COLORS.get(PAIRWISE_FEATURE, "#EF4444"),
                   label=FEATURE_LABELS.get(PAIRWISE_FEATURE) + " (圖無關)")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random (0.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{cr:.0%}" for cr in COLD_RATIOS])
    ax.set_xlabel("Cold-node ratio (unseen at training)", fontsize=11)
    ax.set_ylabel("AUC", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Cold-Start (Inductive) Link Prediction\n"
                 "(ogbn-arxiv) — higher cold ratio = more unseen nodes", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "figures", "auc_cold_start.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reporter] 圖表已存至 {fig_path}")


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")
    data_full, texts = load_dataset()
    print(f"ogbn-arxiv 子圖 — nodes: {data_full.num_nodes}, "
          f"edges: {data_full.edge_index.shape[1]}")
    print(f"Inductive 掃描：冷節點比例 {[f'{c:.0%}' for c in COLD_RATIOS]}\n")

    gnn_records = {}

    # GNN 特徵：每個冷節點比例 × RUNS 次取平均 ± std
    for feature_name in GNN_FEATURES:
        for cr in COLD_RATIOS:
            aucs, aps = [], []
            for run_id in range(RUNS):
                try:
                    m = run_one_inductive(data_full, texts, feature_name, cr, run_id)
                    aucs.append(m["auc"]); aps.append(m["ap"])
                    print(f"  [{feature_name}] cold={cr:.0%} run={run_id+1}/{RUNS}  "
                          f"AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
                except Exception as e:
                    print(f"  [{feature_name}] cold={cr:.0%} run={run_id+1} FAILED: {e}")
            if aucs:
                gnn_records[(feature_name, cr)] = {
                    "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                    "ap_mean":  float(np.mean(aps)),  "ap_std":  float(np.std(aps)),
                }

    # llm_pairwise：與圖結構無關 → 只評估一次（用中間的冷節點比例取一組冷邊），畫成水平線
    pairwise = None
    try:
        rep_ratio = COLD_RATIOS[len(COLD_RATIOS) // 2]
        m = run_pairwise_inductive(data_full, texts, rep_ratio, run_id=0)
        print(f"  [llm_pairwise] AUC={m['auc']:.4f}  AP={m['ap']:.4f}"
              f"（單次評估，與冷節點比例無關）")
        pairwise = m
    except Exception as e:
        print(f"  [llm_pairwise] FAILED: {e}")

    save_and_plot(gnn_records, pairwise, RESULT_DIR)


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    main()
