"""
features/feature_factory.py
五種節點特徵的統一介面

M0 degree   — 純圖統計（degree + clustering coefficient）
M1 tfidf    — TF-IDF on raw text（傳統文字 baseline）
M2 e5_small — intfloat/multilingual-e5-small（本地 LLM，輕量）
"""

import os
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx


class FeatureFactory:
    def __init__(self, data, device, texts=None, dataset_tag=None):
        """
        texts: 外部傳入的字串列表（長度 = num_nodes）。
               有 texts 時跳過 Cora 的文字讀取邏輯，直接使用傳入的文字。
        dataset_tag: 磁碟快取檔名前綴。未指定時依是否有 texts 推斷
               （"arxiv" / "cora"）。指定可將快取綁定到特定子圖設定，
               避免不同子圖（節點集合/順序不同）共用到錯誤的快取。
        """
        self.data         = data
        self.device       = device
        self._text_cache  = texts
        self._cache       = {}
        self._dataset_tag = dataset_tag or ("arxiv" if texts is not None else "cora")

    # ── 公開介面 ─────────────────────────────────────────────────────
    def get(self, name: str) -> torch.Tensor:
        if name not in self._cache:
            fn = getattr(self, f"_build_{name}")
            self._cache[name] = fn().to(self.device)
        return self._cache[name]

    # ── M0: 圖統計特徵 ───────────────────────────────────────────────
    def _build_degree(self) -> torch.Tensor:
        G = to_networkx(self.data, to_undirected=True)
        n = self.data.num_nodes
        feats = np.zeros((n, 4), dtype=np.float32)
        deg   = dict(G.degree())
        clust = nx.clustering(G)
        try:
            core = nx.core_number(G)
            pr   = nx.pagerank(G, max_iter=200)
        except Exception:
            core = {i: 0 for i in range(n)}
            pr   = {i: 1/n for i in range(n)}

        # 每個節點分別使用4個特徵
        for i in range(n):
            feats[i] = [deg.get(i, 0), clust.get(i, 0),
                        core.get(i, 0), pr.get(i, 0)]
        from sklearn.preprocessing import StandardScaler
        feats = StandardScaler().fit_transform(feats)
        return torch.tensor(feats, dtype=torch.float)

    # ── M1: TF-IDF ───────────────────────────────────────────────────
    def _build_tfidf(self) -> torch.Tensor:
        texts = self._load_texts()
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=768, sublinear_tf=True)
        X   = vec.fit_transform(texts).toarray().astype(np.float32)
        return torch.tensor(X, dtype=torch.float)

    # ── M2: multilingual-e5-small ────────────────────────────────────
    def _build_e5_small(self) -> torch.Tensor:
        return self._build_sentence_transformer("intfloat/multilingual-e5-small")

    # ── M5: LLM 關鍵詞 + TF-IDF ─────────────────────────────────────
    def _build_llm_keywords(self) -> torch.Tensor:
        """Groq 為每篇論文生成 10 個技術關鍵詞，再做 TF-IDF 向量化"""
        tag        = self._dataset_tag
        cache_path = f"data/{tag}_llm_keywords_emb.npy"
        kw_path    = f"data/{tag}_llm_keywords_raw.json"

        if os.path.exists(cache_path):
            print("[llm_keywords] 載入快取 embedding")
            return torch.tensor(np.load(cache_path), dtype=torch.float)

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[llm_keywords] GROQ_API_KEY 未設定，fallback 到 tfidf")
            return self._build_tfidf()

        from groq import Groq
        import json, time

        client = Groq(api_key=api_key)
        texts  = self._load_texts()
        n      = len(texts)

        # 支援斷點續跑：直接以關鍵詞 JSON 作為中間快取
        if os.path.exists(kw_path):
            with open(kw_path) as f:
                kw_list = json.load(f)
            print(f"[llm_keywords] 從 {len(kw_list)}/{n} 繼續")
        else:
            kw_list = []

        PROMPT = """\
Extract 5 technical keywords for each arXiv CS paper below.
Return a JSON array of arrays: one inner array of 5 keyword strings per paper.
Output ONLY the JSON array, no explanation.

Papers:
{papers}"""

        BATCH       = 5
        start_batch = len(kw_list) // BATCH
        n_batch     = (n + BATCH - 1) // BATCH

        for b in range(start_batch, n_batch):
            lo, hi   = b * BATCH, min((b + 1) * BATCH, n)
            batch_tx = texts[lo:hi]
            papers_s = "\n".join(f"[{i+1}] {t[:200]}" for i, t in enumerate(batch_tx))
            prompt   = PROMPT.format(papers=papers_s)

            batch_kws = [[] for _ in range(len(batch_tx))]
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=400,
                    )
                    raw = resp.choices[0].message.content.strip()
                    if b < 3:
                        print(f"  [llm_keywords] raw: {repr(raw[:300])}")
                    if raw.startswith("```"):
                        raw = raw.split("```")[1].lstrip("json").strip()
                    parsed = json.loads(raw)
                    batch_kws = [
                        [str(kw) for kw in item[:5]] if isinstance(item, list) else []
                        for item in parsed[:len(batch_tx)]
                    ]
                    filled  = sum(1 for kws in batch_kws if kws)
                    example = batch_kws[0][:2] if batch_kws else []
                    print(f"  [llm_keywords] batch {b:4d} | {filled}/{len(batch_tx)} 筆 | 範例: {example}")
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(4 * (attempt + 1))
                    else:
                        print(f"  [llm_keywords] batch {b:4d} | 失敗: {e}")

            kw_list.extend(batch_kws)
            time.sleep(2.5)

            if (b + 1) % 10 == 0 or hi == n:
                with open(kw_path, "w") as f:
                    json.dump(kw_list, f)

        with open(kw_path, "w") as f:
            json.dump(kw_list, f)

        # 把每個節點的關鍵詞合併成字串，再做 TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        kw_strings = [" ".join(kws) if kws else "unknown" for kws in kw_list]
        vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
        X   = vec.fit_transform(kw_strings).toarray().astype(np.float32)
        np.save(cache_path, X)
        print(f"[llm_keywords] 完成，{X.shape[1]} 維特徵儲存至 {cache_path}")
        return torch.tensor(X, dtype=torch.float)

    # ── M7: Groq 結構化特徵 ──────────────────────────────────────────
    def _build_llm_struct(self) -> torch.Tensor:
        """透過 Groq (llama-3.1-8b-instant) prompt 為每篇論文萃取 16 維結構化語意特徵"""
        tag        = self._dataset_tag
        cache_path = f"data/{tag}_llm_struct_emb.npy"
        part_path  = f"data/{tag}_llm_struct_partial.npy"
        prog_path  = f"data/{tag}_llm_struct_progress.json"

        if os.path.exists(cache_path):
            print("[llm_struct] 載入快取 embedding")
            return torch.tensor(np.load(cache_path), dtype=torch.float)

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[llm_struct] GROQ_API_KEY 未設定，fallback 到 tfidf")
            return self._build_tfidf()

        from groq import Groq
        import json, time

        client = Groq(api_key=api_key)
        texts  = self._load_texts()
        n      = len(texts)

        # 支援斷點續跑：若上次中途中斷，從上次進度繼續
        if os.path.exists(part_path) and os.path.exists(prog_path):
            feats = np.load(part_path)
            with open(prog_path) as f:
                start_batch = json.load(f)["next_batch"]
            print(f"[llm_struct] 從 batch {start_batch} 繼續（已完成 {start_batch * 5}/{n}）")
        else:
            feats       = np.zeros((n, 16), dtype=np.float32)
            start_batch = 0

        # 每篇輸出一行 16 個空格分隔的數字，避免 JSON 欄位名稱消耗大量 tokens
        PROMPT = """\
All papers below are computer science research papers from arXiv (2015).
For each paper output exactly 16 space-separated numbers on ONE line.

Columns (in order):
1  topic_ml       1=yes 0=no  (machine learning / deep learning)
2  topic_nlp      1=yes 0=no  (natural language processing)
3  topic_vision   1=yes 0=no  (computer vision / image)
4  topic_theory   1=yes 0=no  (theory / algorithms / complexity)
5  topic_systems  1=yes 0=no  (systems / distributed / hardware)
6  topic_other    1=yes 0=no  (other CS sub-field)
7  is_empirical   1=yes 0=no  (has experiments / evaluations)
8  is_theoretical 1=yes 0=no  (has proofs / formal analysis)
9  is_survey      1=yes 0=no  (survey or review paper)
10 is_benchmark   1=yes 0=no  (proposes a dataset or benchmark)
11 novelty        1-5  (1=incremental, 5=highly novel)
12 tech_depth     1-5  (1=shallow, 5=highly technical)
13 app_focus      1-5  (1=theory only, 5=strong application focus)
14 reproducibility 1-5 (1=hard to reproduce, 5=easy/open code)
15 interdisciplinary 1=yes 0=no (crosses multiple fields)
16 has_dataset    1=yes 0=no  (uses or releases a dataset)

Rules:
- Score fields (11-14) use 3 when uncertain, never 0.
- At least one of fields 1-6 must be 1 (all papers belong to some CS area).
- Output ONLY numbers, no labels, no explanation.

Example (2 papers):
[1] Attention Is All You Need. We propose the Transformer, a model based solely on attention.
[2] ImageNet Large Scale Visual Recognition Challenge. We describe the ILSVRC benchmark.
OUTPUT:
1 1 0 0 0 0 1 1 0 0 5 4 2 3 0 0
0 0 1 0 0 0 1 0 0 1 4 3 2 4 0 1

Now process these {n_papers} papers:
{papers}

Output ONLY {n_papers} lines of numbers."""

        def _parse_row_flat(vals):
            norm = lambda v: (max(1, min(5, int(round(v)))) - 1) / 4.0
            return [
                float(bool(round(vals[0]))),
                float(bool(round(vals[1]))),
                float(bool(round(vals[2]))),
                float(bool(round(vals[3]))),
                float(bool(round(vals[4]))),
                float(bool(round(vals[5]))),
                float(bool(round(vals[6]))),
                float(bool(round(vals[7]))),
                float(bool(round(vals[8]))),
                float(bool(round(vals[9]))),
                norm(vals[10]),
                norm(vals[11]),
                norm(vals[12]),
                norm(vals[13]),
                float(bool(round(vals[14]))),
                float(bool(round(vals[15]))),
            ]

        BATCH   = 10
        n_batch = (n + BATCH - 1) // BATCH

        for b in range(start_batch, n_batch):
            lo, hi   = b * BATCH, min((b + 1) * BATCH, n)
            batch_tx = texts[lo:hi]
            papers_s = "\n".join(
                f"[{i+1}] {t[:150]}" for i, t in enumerate(batch_tx)
            )
            prompt = PROMPT.format(papers=papers_s, n_papers=len(batch_tx))

            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=400,  # 10 篇 × 16 數字 ≈ 200 tokens
                    )
                    raw   = resp.choices[0].message.content.strip()
                    print(f"  [llm_struct] batch {b} raw: {repr(raw[:200])}")
                    # 只保留每個 token 都是數字的行（過濾掉說明文字行）
                    lines = []
                    for l in raw.split('\n'):
                        l = l.strip()
                        if not l:
                            continue
                        tokens = l.split()
                        try:
                            [float(x) for x in tokens]
                            lines.append(l)
                        except ValueError:
                            pass  # 有文字的行直接跳過
                    filled = 0
                    for j, line in enumerate(lines[:len(batch_tx)]):
                        vals = [float(x) for x in line.split()]
                        if len(vals) >= 16:
                            feats[lo + j] = _parse_row_flat(vals[:16])
                            filled += 1
                    status = f"batch {b:4d} | {filled}/{len(batch_tx)} 筆"
                    if filled == 0:
                        status += f" ← 全零！raw={repr(raw[:80])}"
                    print(f"  [llm_struct] {status}")
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(4 * (attempt + 1))
                    else:
                        print(f"  [llm_struct] batch {b} API 失敗: {e}")

            # Groq free tier 30 RPM → 每次請求至少間隔 2 秒
            time.sleep(2.5)

            # 每 10 批次儲存斷點
            if (b + 1) % 10 == 0 or hi == n:
                np.save(part_path, feats)
                with open(prog_path, "w") as f:
                    json.dump({"next_batch": b + 1}, f)

        filled_rows = int((feats.sum(axis=1) != 0).sum())
        print(f"[llm_struct] 成功填入 {filled_rows}/{n} 筆（{filled_rows/n:.1%}）")
        if filled_rows == 0:
            print("[llm_struct] 警告：全部為零向量，fallback 到 tfidf（不儲存 cache）")
            return self._build_tfidf()
        np.save(cache_path, feats)
        for p in [part_path, prog_path]:
            if os.path.exists(p):
                os.remove(p)
        print(f"[llm_struct] 完成，16 維特徵儲存至 {cache_path}")
        return torch.tensor(feats, dtype=torch.float)

    # ── 內部輔助 ─────────────────────────────────────────────────────
    def _build_sentence_transformer(self, model_name: str) -> torch.Tensor:
        tag        = self._dataset_tag
        cache_path = f"data/{tag}_{model_name.replace('/', '_')}_emb.npy"
        if os.path.exists(cache_path):
            print(f"[{model_name}] 載入快取 embedding")
            return torch.tensor(np.load(cache_path), dtype=torch.float)

        from sentence_transformers import SentenceTransformer
        print(f"[{model_name}] 載入模型...")
        model = SentenceTransformer(model_name)
        texts = self._load_texts()

        if "e5" in model_name:
            texts = [f"passage: {t}" for t in texts]

        print(f"[{model_name}] encoding {len(texts)} 個節點...")
        embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True)
        np.save(cache_path, embs)
        return torch.tensor(embs, dtype=torch.float)

    def _load_texts(self):
        """
        回傳節點文字列表。
        若建構時已傳入 texts，直接使用；否則 fallback 到 Cora BoW 轉字串。
        """
        if self._text_cache is not None:
            return self._text_cache

        # Cora fallback：把 BoW 特徵向量轉成詞袋字串
        print("[text] 未提供文字，使用 BoW 特徵轉字串作為 fallback")
        return self._bow_to_text()

    def _bow_to_text(self):
        x = self.data.x.numpy()
        n = x.shape[0]
        texts = []
        for i in range(n):
            indices = np.where(x[i] > 0)[0]
            texts.append(" ".join([f"word{j}" for j in indices[:50]]))
        return texts
