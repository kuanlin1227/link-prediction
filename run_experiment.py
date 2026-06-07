"""
LLM-Enhanced Link Prediction: Systematic Feature Comparison
研究問題：在 link prediction 任務上，LLM 語意特徵在什麼條件下最有效？

實驗設計：
  - 固定模型架構：GraphSAGE
  - 變因一：節點特徵（5種）
  - 變因二：圖稀疏程度（3種訓練邊比例）
  - 資料集：ogbn-arxiv（arXiv CS 論文引用圖，含真實標題+摘要）
"""

import gzip
import json
import os
import torch

# PyTorch 2.6 把 torch.load 預設改成 weights_only=True，
# 但 OGB 1.3.x 內部沒有跟進，這裡 patch 使其相容。
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.transforms import RandomLinkSplit

from features.feature_factory import FeatureFactory
from models.graphsage import LinkSAGE, train_epoch, evaluate
from eval.reporter import Reporter


# ── 設定 ──────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS      = 200
LR          = 0.01
HIDDEN      = 128
OUT_DIM     = 64
RUNS        = 3
SPARSITY_RATIOS = [0.2, 0.5, 0.8]

FEATURE_CONFIGS = [
    "degree",                # M0: 純圖統計（degree, clustering coeff）
    # "tfidf",               # M1: TF-IDF 文字向量（baseline 文字特徵）
    # "e5_small",            # M2: multilingual-e5-small（輕量 LLM）
    # "llm_struct",          # M5: Groq 結構化語意特徵（16 維，品質不佳，停用）
    # "llm_keywords",        # M6: Groq 關鍵詞 + TF-IDF（512 維，需 GROQ_API_KEY，已停用）
    "llm_embed",             # M10: Llama 節點嵌入 → GraphSAGE（圖聚合）
    "llm_embed_lora",        # M11: Llama + LoRA 節點嵌入 → GraphSAGE
    # "llm_graph_embed",     # M12: LLM(PCA) + GAE → GraphSAGE（待 gae_embed 確認後啟用）
    "gae_embed",             # M13: Graph Autoencoder 結構嵌入 → GraphSAGE（純圖，無文字）
    "degree_llm_embed",      # M14: M0（degree tile×32）+ LLM(PCA 128) → GraphSAGE
    "llm_pairwise",          # M7: LLM 邊級關係推理（greedy，temperature=0）
    "llm_pairwise_majority", # M8: LLM 多數決（n_votes 次，取正票比例）
    "llm_lora",              # M9: LoRA fine-tuned + 線性分類頭
]

# llm_pairwise 專用：每類（正/負）test 邊最多打分幾條，控制 API 成本
MAX_EVAL_PAIRS  = 300
# llm_lora 專用：每類訓練樣本數（正/負各取此數量做 fine-tune）
MAX_TRAIN_PAIRS = 400

# LLM 方法指標快取（避免已完成的方法重跑）
METRICS_CACHE_PATH = "results/method_metrics.json"


def _load_metrics_cache() -> dict:
    if os.path.exists(METRICS_CACHE_PATH):
        with open(METRICS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_metrics_cache(cache: dict):
    os.makedirs(os.path.dirname(METRICS_CACHE_PATH), exist_ok=True)
    with open(METRICS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

# ogbn-arxiv 子圖設定
YEAR_MIN    = 2010   # 取論文的起始年份（含）
YEAR_MAX    = 2020   # 取論文的結束年份（含）
MAX_NODES   = 12000  # 若該年份範圍論文數超過此值，則固定隨機取樣

# 是否把節點發表年份（min-max 正規化到 0~1）併入節點特徵向量。
# 設 False 即可做「有/無年份特徵」的 ablation 對照。
USE_YEAR_FEATURE = True

# 特徵/分數快取標籤：綁定年份範圍，避免讀到舊子圖（節點集合/順序不同）的快取
DATASET_TAG = f"arxiv_{YEAR_MIN}_{YEAR_MAX}"


# ── 資料集載入 ─────────────────────────────────────────────────────────
def load_dataset():
    """
    載入 ogbn-arxiv，篩選特定年份的子圖。
    回傳 (Data, texts)，texts 是長度 = num_nodes 的字串列表。
    """
    from ogb.nodeproppred import PygNodePropPredDataset

    print("載入 ogbn-arxiv 資料集（首次執行會自動下載）...")
    dataset  = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
    data_raw = dataset[0]

    # 篩選特定年份的節點
    years    = data_raw.node_year.squeeze()
    mask     = (years >= YEAR_MIN) & (years <= YEAR_MAX)
    node_idx = mask.nonzero(as_tuple=True)[0]

    # 以「邊」為中心取樣（避免跨年份隨機抽 node 造成子圖過度稀疏）：
    # 先取年份範圍內的完整誘導子圖，再隨機打亂邊、逐條累積端點，
    # 直到不重複節點數達 MAX_NODES，確保每個 node 至少有一條邊。
    if len(node_idx) > MAX_NODES:
        sub_edge_index, _ = pyg_subgraph(
            node_idx,
            data_raw.edge_index,
            relabel_nodes=False,
            num_nodes=data_raw.num_nodes,
        )
        g    = torch.Generator().manual_seed(42)
        perm = torch.randperm(sub_edge_index.shape[1], generator=g)
        seen = set()
        for e in perm.tolist():
            seen.add(int(sub_edge_index[0, e]))
            seen.add(int(sub_edge_index[1, e]))
            if len(seen) >= MAX_NODES:
                break
        node_idx = torch.tensor(sorted(seen), dtype=torch.long)

    # 誘導子圖（只保留兩端點都在 node_idx 內的邊）
    edge_index, _ = pyg_subgraph(
        node_idx,
        data_raw.edge_index,
        relabel_nodes=True,
        num_nodes=data_raw.num_nodes,
    )
 
    data = Data(edge_index=edge_index, num_nodes=len(node_idx))
    # 保留每個（重編號後）節點的發表年份，供節點特徵與 LLM prompt 使用
    data.node_year = data_raw.node_year[node_idx].view(-1, 1)

    # 載入對應節點的文字
    texts = _load_arxiv_texts(node_idx)

    return data, texts


def _load_arxiv_texts(node_idx):
    """從 titleabs.tsv.gz 讀取指定節點的「標題. 摘要」字串

    titleabs.tsv.gz 第一欄是 MAG paper ID，不是 PyG node index。
    需要先透過 nodeidx2paperid.csv.gz 做轉換。
    """
    import pandas as pd

    # Step 1: OGB node index → MAG paper ID
    mapping_path = 'data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz'
    with gzip.open(mapping_path, 'rt') as f:
        mapping_df = pd.read_csv(f)
    nodeidx2paperid = dict(zip(mapping_df['node idx'], mapping_df['paper id']))

    # Step 2: MAG paper ID → title + abstract
    path = 'data/ogbn_arxiv/mapping/titleabs.tsv.gz'
    url  = 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz'

    if not os.path.exists(path):
        import urllib.request
        print(f"下載論文文字檔案（約 100MB）：{url}")
        urllib.request.urlretrieve(url, path)
        print("下載完成")

    print(f"讀取論文文字：{path}")
    texts_dict = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            try:
                paper_id = int(parts[0])
                title    = parts[1].strip()
                abstr    = parts[2].strip()
                texts_dict[paper_id] = f"{title}. {abstr}" if abstr else title
            except ValueError:
                continue

    # Step 3: 用 node index → MAG paper ID → text 的兩段查表
    node_ids = node_idx.tolist()
    texts = []
    for i in node_ids:
        paper_id = nodeidx2paperid.get(i)
        text = texts_dict.get(paper_id, "unknown paper") if paper_id is not None else "unknown paper"
        texts.append(text)

    loaded = sum(1 for t in texts if t != "unknown paper")
    print(f"文字載入完成：{loaded}/{len(texts)} 筆")
    return texts


# ── 單次實驗 ──────────────────────────────────────────────────────────
def run_one(data_full, texts, feature_name, sparsity, _run_id):
    """單次實驗：一種特徵 × 一種稀疏度 × 一次 run"""

    val_ratio  = 0.1
    test_ratio = 1.0 - sparsity - val_ratio
    # 分成 train / val / test
    transform  = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data_full)

    factory     = FeatureFactory(data_full, DEVICE, texts=texts,
                                 dataset_tag=DATASET_TAG)
    x           = factory.get(feature_name)

    # 把發表年份當作額外一維特徵併入（min-max 正規化到 0~1）
    if USE_YEAR_FEATURE and getattr(data_full, "node_year", None) is not None:
        span    = max(1, YEAR_MAX - YEAR_MIN)
        yr_norm = ((data_full.node_year.float() - YEAR_MIN) / span).clamp(0, 1)
        x       = torch.cat([x, yr_norm.to(x.device)], dim=1)

    in_channels = x.shape[1]

    for split in [train_data, val_data, test_data]:
        split.x = x

    model     = LinkSAGE(in_channels, HIDDEN, OUT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc, patience, counter = 0.0, 20, 0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        loss        = train_epoch(model, optimizer, train_data, DEVICE)
        val_metrics = evaluate(model, train_data, val_data, DEVICE)
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            counter      = 0
        else:
            counter += 1
        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, train_data, test_data, DEVICE)
    return test_metrics


# ── 單次實驗（M7: LLM 邊級推理）────────────────────────────────────────
def run_pairwise_once(data_full, texts):
    """讓 LLM 直接對「正邊 + 負邊」打分算 AUC/AP（不經 GraphSAGE）。

    此方法只讀兩篇論文文字、完全不看圖結構，因此與圖稀疏度無關，
    只需評估「一次」。建立一組平衡測試集：正邊各 MAX_EVAL_PAIRS、
    等量隨機負邊，控制 API 成本。
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from torch_geometric.utils import negative_sampling
    from features.llm_pairwise import score_pairs

    g          = torch.Generator().manual_seed(42)
    edge_index = data_full.edge_index
    num_edges  = edge_index.shape[1]

    # 正邊：從真實邊隨機取 MAX_EVAL_PAIRS 條
    k    = min(MAX_EVAL_PAIRS, num_edges)
    perm = torch.randperm(num_edges, generator=g)[:k]
    pos  = edge_index[:, perm]

    # 負邊：隨機取等量「不存在的邊」（固定 seed 確保每次 pair 相同，快取可重用）
    torch.manual_seed(42)
    neg = negative_sampling(
        edge_index,
        num_nodes=data_full.num_nodes,
        num_neg_samples=k,
    )

    pairs = ([(int(pos[0, i]), int(pos[1, i])) for i in range(pos.shape[1])] +
             [(int(neg[0, i]), int(neg[1, i])) for i in range(neg.shape[1])])
    y     = np.array([1] * pos.shape[1] + [0] * neg.shape[1])

    years  = (data_full.node_year.view(-1).tolist()
              if getattr(data_full, "node_year", None) is not None else None)
    scores = score_pairs(pairs, texts, tag=DATASET_TAG, years=years)
    return {
        'auc': roc_auc_score(y, scores),
        'ap':  average_precision_score(y, scores),
    }


def run_lora_once(data_full, texts):
    """LoRA fine-tuned Llama 對正/負邊打分算 AUC/AP（不經 GraphSAGE）。

    訓練集：從完整圖採樣 MAX_TRAIN_PAIRS 正/負邊（seed=123，避免與測試集完全重疊）。
    測試集：與 pairwise 相同採樣（seed=42, MAX_EVAL_PAIRS 條）以確保公平比較。
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from torch_geometric.utils import negative_sampling
    from features.llm_lora import score_pairs_lora

    edge_index = data_full.edge_index
    num_edges  = edge_index.shape[1]
    years = (data_full.node_year.view(-1).tolist()
             if getattr(data_full, "node_year", None) is not None else None)

    # 訓練集（seed=123）
    g_tr    = torch.Generator().manual_seed(123)
    k_tr    = min(MAX_TRAIN_PAIRS, num_edges)
    pos_tr  = edge_index[:, torch.randperm(num_edges, generator=g_tr)[:k_tr]]
    torch.manual_seed(123)
    neg_tr  = negative_sampling(edge_index, num_nodes=data_full.num_nodes,
                                num_neg_samples=k_tr)
    train_pos = [(int(pos_tr[0, i]), int(pos_tr[1, i])) for i in range(pos_tr.shape[1])]
    train_neg = [(int(neg_tr[0, i]), int(neg_tr[1, i])) for i in range(neg_tr.shape[1])]

    # 測試集（seed=42，與 pairwise 一致）
    g_te   = torch.Generator().manual_seed(42)
    k_te   = min(MAX_EVAL_PAIRS, num_edges)
    pos_te = edge_index[:, torch.randperm(num_edges, generator=g_te)[:k_te]]
    torch.manual_seed(42)
    neg_te = negative_sampling(edge_index, num_nodes=data_full.num_nodes,
                               num_neg_samples=k_te)
    test_pairs = ([(int(pos_te[0, i]), int(pos_te[1, i])) for i in range(pos_te.shape[1])] +
                  [(int(neg_te[0, i]), int(neg_te[1, i])) for i in range(neg_te.shape[1])])
    y = np.array([1] * pos_te.shape[1] + [0] * neg_te.shape[1])

    scores = score_pairs_lora(
        test_pairs, texts, tag=DATASET_TAG, years=years,
        train_pairs_pos=train_pos, train_pairs_neg=train_neg,
    )
    return {
        'auc': roc_auc_score(y, scores),
        'ap':  average_precision_score(y, scores),
    }


def run_pairwise_majority_once(data_full, texts):
    """與 run_pairwise_once 相同的正/負邊採樣，但改用多數決打分。

    重複呼叫 LLM N_VOTES 次（temperature=VOTE_TEMPERATURE），
    每次分數 >= 0.5 算正票，最終回傳正票比例作為排名分數。
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from torch_geometric.utils import negative_sampling
    from features.llm_pairwise import score_pairs_majority, N_VOTES

    g          = torch.Generator().manual_seed(42)
    edge_index = data_full.edge_index
    num_edges  = edge_index.shape[1]

    k    = min(MAX_EVAL_PAIRS, num_edges)
    perm = torch.randperm(num_edges, generator=g)[:k]
    pos  = edge_index[:, perm]

    torch.manual_seed(42)
    neg = negative_sampling(
        edge_index,
        num_nodes=data_full.num_nodes,
        num_neg_samples=k,
    )

    pairs = ([(int(pos[0, i]), int(pos[1, i])) for i in range(pos.shape[1])] +
             [(int(neg[0, i]), int(neg[1, i])) for i in range(neg.shape[1])])
    y     = np.array([1] * pos.shape[1] + [0] * neg.shape[1])

    years  = (data_full.node_year.view(-1).tolist()
              if getattr(data_full, "node_year", None) is not None else None)
    scores = score_pairs_majority(pairs, texts, tag=DATASET_TAG, years=years,
                                  n_votes=N_VOTES)
    return {
        'auc': roc_auc_score(y, scores),
        'ap':  average_precision_score(y, scores),
    }


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    data_full, texts = load_dataset()
    print(f"ogbn-arxiv {YEAR_MIN}-{YEAR_MAX} 子圖 — "
          f"nodes: {data_full.num_nodes}, "
          f"edges: {data_full.edge_index.shape[1]}\n")

    reporter      = Reporter()
    metrics_cache = _load_metrics_cache()

    # LLM 方法執行器（key → callable）
    _llm_runners = {
        "llm_pairwise":          lambda: run_pairwise_once(data_full, texts),
        "llm_pairwise_majority": lambda: run_pairwise_majority_once(data_full, texts),
        "llm_lora":              lambda: run_lora_once(data_full, texts),
    }

    for feature_name in FEATURE_CONFIGS:
        if feature_name in _llm_runners:
            if feature_name in metrics_cache:
                m = metrics_cache[feature_name]
                print(f"  [{feature_name}] 從快取載入 "
                      f"AUC={m['auc']:.4f}  AP={m['ap']:.4f}")
            else:
                # llm_lora 前先釋放 pairwise 佔用的 VRAM
                if feature_name == "llm_lora":
                    from features.llm_pairwise import clear_model_cache as _clr
                    _clr()
                try:
                    m = _llm_runners[feature_name]()
                    print(f"  [{feature_name}] AUC={m['auc']:.4f}  AP={m['ap']:.4f}"
                          f"（與稀疏度無關，僅評估一次）")
                    metrics_cache[feature_name] = m
                    _save_metrics_cache(metrics_cache)
                except Exception as e:
                    print(f"  [{feature_name}] FAILED: {e}")
                    continue
            for sparsity in SPARSITY_RATIOS:
                reporter.add(feature_name, sparsity, [m])
            continue

        for sparsity in SPARSITY_RATIOS:
            run_results = []
            for run_id in range(RUNS):
                try:
                    metrics = run_one(data_full, texts, feature_name, sparsity, run_id)
                    run_results.append(metrics)
                    print(f"  [{feature_name}] sparsity={sparsity:.0%} "
                          f"run={run_id+1}/{RUNS}  "
                          f"AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  "
                          f"F1={metrics['f1']:.4f}")
                except Exception as e:
                    print(f"  [{feature_name}] sparsity={sparsity:.0%} "
                          f"run={run_id+1} FAILED: {e}")

            if run_results:
                reporter.add(feature_name, sparsity, run_results)

    reporter.print_table()
    reporter.save_csv("results/comparison_table.csv")
    reporter.plot("results/figures/")


if __name__ == "__main__":
    os.makedirs("results/figures", exist_ok=True)
    main()
