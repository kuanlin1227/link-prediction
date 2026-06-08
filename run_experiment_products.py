"""
LLM-Enhanced Link Prediction on ogbn-products (TAPE text subset)
使用 TAPE 提供的商品標題+描述文字，在 ogbn-products 的誘導子圖上重現與
run_experiment.py 相同的方法比較（M0 / TF-IDF / M11 / degree+LLM）。

差異：
  - 無 node_year（ogbn-products 沒有發表年份），不加年份特徵
  - 從 TAPE 的 54K 文字節點中取樣 12K 節點建子圖
  - 結果輸出到 results_products/
"""

import os
import json
import torch

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import builtins
builtins.input = lambda *a, **kw: "n"   # 不觸發 OGB 互動式提示

import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.transforms import RandomLinkSplit

from features.feature_factory import FeatureFactory
from models.graphsage import LinkSAGE, train_epoch, evaluate
from eval.reporter import Reporter


# ── 設定 ──────────────────────────────────────────────────────────────
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS          = 200
LR              = 0.01
HIDDEN          = 128
OUT_DIM         = 64
RUNS            = 3
SPARSITY_RATIOS = [0.2, 0.5, 0.8]
MAX_NODES       = 12000
DATASET_TAG     = "products_tape"

FEATURE_CONFIGS = [
    "degree",                # M0: 純圖統計
    "tfidf",                 # M1: TF-IDF（商品標題+描述）
    "llm_embed_lora",        # M11: Llama + LoRA 節點嵌入
    "degree_llm_embed_lora", # degree + LoRA-LLM embed
]

MAX_LORA_PAIRS  = 3000  # LoRA 訓練時正/負 pair 數上限
PAIRWISE_SAMPLE = 500   # Groq pairwise 每個 (sparsity, run) 各取樣的正/負 pair 數
RESULT_DIR = "results_products"


# ── 資料集載入 ─────────────────────────────────────────────────────────
def load_dataset():
    """
    載入 ogbn-products，只保留 TAPE CSV 中有文字的節點（~54K），
    再用 edge-density 取樣縮到 MAX_NODES，確保子圖密度合理。
    """
    from ogb.nodeproppred import PygNodePropPredDataset

    print("載入 ogbn-products...")
    dataset  = PygNodePropPredDataset(name='ogbn-products', root='data/')
    data_raw = dataset[0]

    # 載入 TAPE 文字，只取有 nid 對應的節點
    csv_path = "data/ogbn_products_subset/text.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} 不存在，請先下載：\n"
            "  python -c \"\n"
            "  import urllib.request\n"
            "  urllib.request.urlretrieve(\n"
            "    'https://raw.githubusercontent.com/XiaoxinHe/TAPE/main/dataset/ogbn_products_orig/ogbn-products_subset.csv',\n"
            "    'data/ogbn_products_subset/text.csv')\"\n"
        )

    print("載入 TAPE 文字...")
    df        = pd.read_csv(csv_path)
    # 以 nid（OGB 節點 index）為 key 建立文字查詢表
    nid2text  = {}
    for _, row in df.iterrows():
        title   = str(row['title']) if pd.notna(row['title']) else ""
        content = str(row['content']) if pd.notna(row['content']) else ""
        text    = f"Product: {title}; Description: {content}".strip()
        nid2text[int(row['nid'])] = text or "unknown product"

    # 只保留有文字的節點
    text_nodes = torch.tensor(sorted(nid2text.keys()), dtype=torch.long)
    print(f"有文字節點數：{len(text_nodes)}")

    # 從有文字的節點誘導子圖後，用 edge-density 取樣至 MAX_NODES
    sub_edge_index, _ = pyg_subgraph(
        text_nodes,
        data_raw.edge_index,
        relabel_nodes=False,
        num_nodes=data_raw.num_nodes,
    )

    if len(text_nodes) > MAX_NODES:
        g    = torch.Generator().manual_seed(42)
        perm = torch.randperm(sub_edge_index.shape[1], generator=g)
        seen = set()
        for e in perm.tolist():
            seen.add(int(sub_edge_index[0, e]))
            seen.add(int(sub_edge_index[1, e]))
            if len(seen) >= MAX_NODES:
                break
        node_idx = torch.tensor(sorted(seen), dtype=torch.long)
    else:
        node_idx = text_nodes

    # 最終誘導子圖（重編節點號）
    edge_index, _ = pyg_subgraph(
        node_idx,
        data_raw.edge_index,
        relabel_nodes=True,
        num_nodes=data_raw.num_nodes,
    )

    data = Data(edge_index=edge_index, num_nodes=len(node_idx))

    # 按重編後順序建立文字列表
    orig_ids = node_idx.tolist()
    texts    = [nid2text.get(i, "unknown product") for i in orig_ids]

    loaded = sum(1 for t in texts if t != "unknown product")
    print(f"子圖建立完成 — nodes: {data.num_nodes}, edges: {data.edge_index.shape[1]}")
    print(f"文字載入：{loaded}/{len(texts)} 筆\n")
    return data, texts


# ── LoRA 預訓練（若 adapter 不存在則自動觸發）────────────────────────
def pretrain_lora_adapter(data_full, texts):
    """在 products_tape 的邊上 fine-tune LoRA adapter，並存至 data/products_tape_lora_adapter/。
    若 adapter 已存在則直接跳過（斷點安全）。
    """
    import random
    from features.llm_lora import score_pairs_lora, clear_model_cache

    adapter_dir = f"data/{DATASET_TAG}_lora_adapter"
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        print(f"[llm_lora] adapter 已存在：{adapter_dir}，跳過訓練")
        return

    print("[llm_lora] 開始在 products_tape 上 fine-tune LoRA adapter...")

    # 用一個固定 split 的訓練邊來取樣訓練 pairs
    transform  = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, _, _ = transform(data_full)

    ei = train_data.edge_label_index
    lbl = train_data.edge_label
    pos_mask = lbl == 1
    neg_mask = lbl == 0

    all_pos = list(zip(ei[0, pos_mask].tolist(), ei[1, pos_mask].tolist()))
    all_neg = list(zip(ei[0, neg_mask].tolist(), ei[1, neg_mask].tolist()))

    rng      = random.Random(42)
    pairs_pos = rng.sample(all_pos, min(MAX_LORA_PAIRS, len(all_pos)))
    pairs_neg = rng.sample(all_neg, min(MAX_LORA_PAIRS, len(all_neg)))

    print(f"  訓練 pairs：{len(pairs_pos)} 正 + {len(pairs_neg)} 負")

    # 用 score_pairs_lora 觸發訓練（side-effect：自動存 adapter）
    score_pairs_lora(
        pairs=pairs_pos[:2],   # 只需 2 個 pair 來完成 score call
        texts=texts,
        tag=DATASET_TAG,
        years=None,
        train_pairs_pos=pairs_pos,
        train_pairs_neg=pairs_neg,
    )

    clear_model_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[llm_lora] LoRA adapter 訓練完成，GPU 快取已清除\n")


# ── Groq Pairwise 取樣評估 ────────────────────────────────────────────
def run_one_pairwise(data_full, texts, sparsity, run_id):
    """從 test set 取樣 PAIRWISE_SAMPLE 對正/負 pair，用 Groq 打分並回傳 metrics。"""
    import random
    from sklearn.metrics import roc_auc_score, average_precision_score
    from features.llm_pairwise import score_pairs_groq, PRODUCTS_SYSTEM_PROMPT

    val_ratio  = 0.1
    test_ratio = 1.0 - sparsity - val_ratio
    transform  = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=False,
    )
    _, _, test_data = transform(data_full)

    pos_mask = test_data.edge_label == 1
    neg_mask = test_data.edge_label == 0
    pos_idx  = pos_mask.nonzero(as_tuple=True)[0].tolist()
    neg_idx  = neg_mask.nonzero(as_tuple=True)[0].tolist()

    rng      = random.Random(run_id * 100 + int(sparsity * 100))
    n_sample = min(PAIRWISE_SAMPLE, len(pos_idx), len(neg_idx))
    pos_sel  = rng.sample(pos_idx, n_sample)
    neg_sel  = rng.sample(neg_idx, n_sample)

    ei     = test_data.edge_label_index
    pairs  = [(int(ei[0, i]), int(ei[1, i])) for i in pos_sel + neg_sel]
    labels = [1] * n_sample + [0] * n_sample

    scores = score_pairs_groq(
        pairs, texts, tag=DATASET_TAG,
        system_prompt=PRODUCTS_SYSTEM_PROMPT,
        log_prefix="[llm_pairwise_groq]",
    )

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    print(f"  [llm_pairwise_groq] sparsity={sparsity:.0%} run={run_id+1}  "
          f"AUC={auc:.4f}  AP={ap:.4f}  (N={n_sample*2} sampled pairs)")
    return {"auc": auc, "ap": ap}


# ── 單次實驗 ──────────────────────────────────────────────────────────
def run_one(data_full, texts, feature_name, sparsity, _run_id):
    val_ratio  = 0.1
    test_ratio = 1.0 - sparsity - val_ratio
    transform  = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data_full)

    factory = FeatureFactory(data_full, DEVICE, texts=texts, dataset_tag=DATASET_TAG)
    x       = factory.get(feature_name)

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
    return evaluate(model, train_data, test_data, DEVICE)


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "figures"), exist_ok=True)

    data_full, texts = load_dataset()
    pretrain_lora_adapter(data_full, texts)
    reporter = Reporter()

    for feature_name in FEATURE_CONFIGS:
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

    # Groq Pairwise 取樣評估（結果快取在 data/products_tape_groq_pairwise_scores.json）
    for sparsity in SPARSITY_RATIOS:
        pw_results = []
        for run_id in range(RUNS):
            try:
                metrics = run_one_pairwise(data_full, texts, sparsity, run_id)
                pw_results.append(metrics)
            except Exception as e:
                print(f"  [llm_pairwise_groq] sparsity={sparsity:.0%} "
                      f"run={run_id+1} FAILED: {e}")
        if pw_results:
            reporter.add("llm_pairwise_groq", sparsity, pw_results)

    reporter.print_table()
    reporter.save_csv(os.path.join(RESULT_DIR, "comparison_table.csv"))
    reporter.plot(os.path.join(RESULT_DIR, "figures"))


if __name__ == "__main__":
    main()
