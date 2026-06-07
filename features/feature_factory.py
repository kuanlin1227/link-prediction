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

    # ── M10: Llama-3.1-8B-Instruct 節點嵌入 → GraphSAGE ─────────────
    def _build_llm_embed(self) -> torch.Tensor:
        from features.llm_embed import get_embeddings
        texts = self._load_texts()
        embs  = get_embeddings(texts, tag=self._dataset_tag)
        return torch.tensor(embs, dtype=torch.float)

    # ── M11: Llama + LoRA 節點嵌入 → GraphSAGE ───────────────────────
    def _build_llm_embed_lora(self) -> torch.Tensor:
        from features.llm_embed import get_embeddings_lora
        texts = self._load_texts()
        embs  = get_embeddings_lora(texts, tag=self._dataset_tag)
        return torch.tensor(embs, dtype=torch.float)

    # ── Spectral 圖結構嵌入（內部輔助，供 M12 使用）────────────────
    def _build_struct_embed(self) -> torch.Tensor:
        """正規化 Laplacian 的前 128 個非零特徵向量作為圖結構嵌入（Spectral Embedding）。
        只依賴 scipy，不需要 torch-cluster / pyg-lib 等需要編譯的套件。
        注意：使用完整圖（data_full）邊，屬 transductive 設定。
        """
        tag        = self._dataset_tag
        cache_path = f"data/{tag}_spectral_embed.npy"
        EMBED_DIM  = 128

        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            if arr.shape[0] == self.data.num_nodes:
                print(f"[struct_embed] 載入快取: {cache_path} {arr.shape}")
                return torch.tensor(arr, dtype=torch.float)

        from scipy.sparse import coo_matrix, diags, eye
        from scipy.sparse.linalg import eigsh
        from sklearn.preprocessing import normalize as sk_normalize

        n  = self.data.num_nodes
        ei = self.data.edge_index.cpu().numpy()

        # 無向圖 adjacency matrix（0/1，去除重複邊）
        src = np.concatenate([ei[0], ei[1]])
        dst = np.concatenate([ei[1], ei[0]])
        A   = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
        A   = (A > 0).astype(np.float32)

        # 正規化 Laplacian: L_sym = I − D^{-½} A D^{-½}
        deg        = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = diags(d_inv_sqrt)
        L_sym      = eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

        k = min(EMBED_DIM + 4, n - 1)   # 多求幾個以防前幾個是零特徵值
        print(f"[struct_embed] 計算 Laplacian 前 {k} 個特徵向量（{n} 節點）...")
        vals, vecs = eigsh(L_sym, k=k, which="SM", tol=1e-3, maxiter=3000)

        # 排序，跳過特徵值 ≈ 0（對應常數向量，不含結構資訊）
        order      = np.argsort(vals)
        nontrivial = [i for i in order if vals[i] > 1e-6][:EMBED_DIM]
        embs       = vecs[:, nontrivial].astype(np.float32)   # (N, ≤128)

        embs = sk_normalize(embs, norm="l2").astype(np.float32)
        np.save(cache_path, embs)
        print(f"[struct_embed] 完成：{embs.shape[1]} 維譜嵌入存至 {cache_path}")
        return torch.tensor(embs, dtype=torch.float)

    # ── M13: Graph Autoencoder 結構嵌入 → GraphSAGE ──────────────────
    def _build_gae_embed(self) -> torch.Tensor:
        """GCN 編碼器（常數初始特徵）+ 內積解碼器，以 link reconstruction 訓練。
        輸入為 all-ones，讓 GCN 純粹從圖拓樸推導結構嵌入，不依賴任何文字或手工特徵。
        注意：使用完整圖（data_full）邊，屬 transductive 設定。
        快取：data/{tag}_gae_embed.npy
        """
        tag        = self._dataset_tag
        cache_path = f"data/{tag}_gae_embed.npy"
        EMBED_DIM  = 128
        HIDDEN     = 256
        EPOCHS     = 150
        LR         = 0.01

        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            if arr.shape[0] == self.data.num_nodes:
                print(f"[gae_embed] 載入快取: {cache_path} {arr.shape}")
                return torch.tensor(arr, dtype=torch.float)

        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        from torch_geometric.utils import negative_sampling

        n          = self.data.num_nodes
        edge_index = self.data.edge_index.to(self.device)

        # 常數初始特徵：讓 GCN 純粹從鄰域聚合學習結構嵌入
        x = torch.ones(n, 1, dtype=torch.float, device=self.device)

        class _GCNEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(1, HIDDEN)
                self.conv2 = GCNConv(HIDDEN, EMBED_DIM)

            def forward(self, x, edge_index):
                h = F.relu(self.conv1(x, edge_index))
                return self.conv2(h, edge_index)

        encoder   = _GCNEncoder().to(self.device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)

        print(f"[gae_embed] 訓練 GAE（{n} 節點, {EMBED_DIM} 維, {EPOCHS} epochs）...")

        for epoch in range(1, EPOCHS + 1):
            encoder.train()
            optimizer.zero_grad()
            z = encoder(x, edge_index)

            pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
            neg_ei    = negative_sampling(
                edge_index, num_nodes=n,
                num_neg_samples=edge_index.shape[1],
            )
            neg_score = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)

            labels = torch.cat([
                torch.ones(pos_score.shape[0], device=self.device),
                torch.zeros(neg_score.shape[0], device=self.device),
            ])
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_score, neg_score]), labels
            )
            loss.backward()
            optimizer.step()

            if epoch % 30 == 0:
                print(f"[gae_embed] epoch {epoch:3d}/{EPOCHS}  loss={loss.item():.4f}")

        encoder.eval()
        with torch.no_grad():
            z = encoder(x, edge_index)

        embs  = z.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs  = embs / norms

        np.save(cache_path, embs)
        print(f"[gae_embed] 完成：{EMBED_DIM} 維 GAE 嵌入存至 {cache_path}")
        return torch.tensor(embs, dtype=torch.float)

    # ── M14: M0（degree stats）+ LLM embedding → GraphSAGE ───────────
    def _build_degree_llm_embed(self) -> torch.Tensor:
        """degree stats (4 dim, L2-norm → tile×32 → 128 dim)
        + Llama-3.1-8B PCA 嵌入 (4096 → 128 dim) 對等拼接 → 256 dim。
        兩個 block 都是 128 dim，GraphSAGE 第一層各獲得相同參數配額。
        快取：data/{tag}_llama_embed_pca128.npy（與 llm_graph_embed 共用）
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize as sk_normalize

        tag       = self._dataset_tag
        pca_cache = f"data/{tag}_llama_embed_pca128.npy"

        if os.path.exists(pca_cache):
            llm_pca = np.load(pca_cache)
            print(f"[degree_llm_embed] 載入 PCA-LLM 快取: {pca_cache} {llm_pca.shape}")
        else:
            llm_np  = self.get("llm_embed").cpu().numpy()
            print("[degree_llm_embed] PCA 4096 → 128 中...")
            pca     = PCA(n_components=128, random_state=42)
            llm_pca = pca.fit_transform(llm_np).astype(np.float32)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"[degree_llm_embed] PCA 完成，保留變異量 {var_exp:.1%}")
            llm_pca = sk_normalize(llm_pca, norm="l2").astype(np.float32)
            np.save(pca_cache, llm_pca)

        # degree block: 4 dim → L2-normalize → tile ×32 → 128 dim
        # isolated nodes（degree=0，cold-start 下的冷節點）的 degree block 強制歸零：
        # StandardScaler 會把 degree=0 轉成非零負值，L2-norm 後形成所有冷節點都相同
        # 的常數模式，佔用 50% feature space 卻不含節點區分資訊，稀釋 LLM 信號。
        # 歸零後冷節點退化為純 LLM 特徵，train 節點保留 degree + LLM 兩個 block。
        from torch_geometric.utils import degree as pyg_degree
        node_deg     = pyg_degree(self.data.edge_index[0],
                                  num_nodes=self.data.num_nodes).cpu().numpy()
        isolated     = node_deg == 0                                  # cold node mask

        degree_np    = self.get("degree").cpu().numpy()               # (N, 4)
        degree_l2    = sk_normalize(degree_np, norm="l2").astype(np.float32)
        degree_l2[isolated] = 0.0                                     # cold → zero block
        degree_tiled = np.tile(degree_l2, (1, 32))                    # (N, 128)

        combined = np.concatenate([llm_pca, degree_tiled], axis=1)    # (N, 256)
        return torch.tensor(combined, dtype=torch.float)

    # ── M12: LLM embedding + Spectral embedding → GraphSAGE ──────────
    def _build_llm_graph_embed(self) -> torch.Tensor:
        """凍結 Llama-3.1-8B 文字嵌入（PCA → 128 維）+ Laplacian 譜嵌入（128 維）
        維度對等後拼接，輸入 GraphSAGE，讓模型同時看到語意與拓樸資訊。
        快取：data/{tag}_llama_embed_pca128.npy
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize as sk_normalize

        tag       = self._dataset_tag
        pca_cache = f"data/{tag}_llama_embed_pca128.npy"

        if os.path.exists(pca_cache):
            llm_pca = np.load(pca_cache)
            print(f"[llm_graph_embed] 載入 PCA-LLM 快取: {pca_cache} {llm_pca.shape}")
        else:
            llm_np  = self.get("llm_embed").cpu().numpy()   # (N, 4096)
            print("[llm_graph_embed] PCA 4096 → 128 中...")
            pca     = PCA(n_components=128, random_state=42)
            llm_pca = pca.fit_transform(llm_np).astype(np.float32)
            var_exp = pca.explained_variance_ratio_.sum()
            print(f"[llm_graph_embed] PCA 完成，保留變異量 {var_exp:.1%}")
            llm_pca = sk_normalize(llm_pca, norm="l2").astype(np.float32)
            np.save(pca_cache, llm_pca)

        struct_np = self.get("gae_embed").cpu().numpy()       # (N, 128)
        combined  = np.concatenate([llm_pca, struct_np], axis=1)  # (N, 256)
        return torch.tensor(combined, dtype=torch.float)

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
