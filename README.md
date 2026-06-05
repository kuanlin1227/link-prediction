# LLM-Enhanced Link Prediction
## When Does Text Help? LLM Node Features for Link Prediction under Varying Graph Sparsity

### 專案結構

```
link_prediction/
├── run_experiment.py          # 主程式，跑完所有實驗
├── requirements.txt
├── features/
│   └── feature_factory.py     # 五種節點特徵的統一介面
├── models/
│   └── graphsage.py           # GraphSAGE encoder + 內積 decoder
├── eval/
│   └── reporter.py            # 結果匯整、表格、圖表
├── data/                      # PyG 自動下載 Cora
└── results/
    ├── comparison_table.csv
    └── figures/
        ├── auc_by_sparsity.png
        └── llm_gain_vs_sparsity.png
```

---

### 安裝

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

---

### 執行

```bash
# 完整實驗（15 組 × 3 runs = 45 次訓練）
python run_experiment.py

# 只跑部分（修改 run_experiment.py 中的 FEATURE_CONFIGS）
# 例如只跑 M0, M1, M2：
# FEATURE_CONFIGS = ["degree", "tfidf", "e5_small"]
```

---

### 實驗設計

| 變因 | 設定 |
|---|---|
| 固定模型 | GraphSAGE（2層，hidden=128，out=64） |
| 節點特徵 | degree / tfidf / e5-small / llm_keywords / llm_pairwise |
| 圖稀疏程度 | 訓練邊 20% / 50% / 80% |
| 評估指標 | AUC、AP（3次平均 ± std） |
| 資料集 | Cora（2708 節點，5429 邊，論文引用圖） |

---

### 預期 finding

- LLM 特徵在**稀疏圖（20%）** 上的增益應大於稠密圖（80%）
  → 因為稀疏時純圖結構資訊不足，語意特徵補充效果明顯
- TF-IDF vs LLM 的差距說明「語意理解能力」而非「文字本身」的貢獻
- E5-small vs E5-base 的差距說明模型大小的影響

---

### 參考文獻

1. Grover & Leskovec (2016). node2vec. KDD. https://arxiv.org/abs/1607.00653
2. Hamilton et al. (2017). GraphSAGE. NeurIPS. https://arxiv.org/abs/1706.02216
3. Zhang & Chen (2018). SEAL. NeurIPS. https://arxiv.org/abs/1802.09691
4. He et al. (2024). TAPE. ICLR. https://arxiv.org/abs/2307.11709
5. arxiv.org/abs/2407.12860 — STAGE: LLM cascading for TAG
6. arxiv.org/abs/2502.18771 — LLM Graph Benchmark
