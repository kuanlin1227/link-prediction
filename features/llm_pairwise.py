"""
features/llm_pairwise.py
M7: LLM 邊級關係推理打分器

不同於 M0–M6（節點特徵 → GraphSAGE），本方法讓 LLM 直接對「候選邊」的
兩端論文做引用關係推理，輸出 0–1 機率，當成 link prediction 分數。

核心動機：E5 / TF-IDF 只能衡量「兩篇文字相不相似」，但引用關係 ≠ 相似
（方法論文常引用它要解決的問題論文，兩者文字未必像）。pairwise 推理正中
相似度方法的盲點，這是它有機會贏過 M1/M2/M3 的原因。

評估方式：對 test split 的正邊 + 負邊打分，直接算 AUC / AP。
快取：data/{tag}_llm_pairwise_scores.json，key = "min,max"，支援斷點續跑。
"""

import json
import os
import re
import time


# ── 設定 ──────────────────────────────────────────────────────────────
# Groq 模型：70b 推理品質遠勝 8b。若此模型在你的帳號不可用，
# 改成 "llama-3.1-8b-instant"（較快但判斷較粗）。
MODEL       = "llama-3.3-70b-versatile"
BATCH       = 12      # 一次打分幾個 pair（攤平 system prompt 成本、省 token）
TEMPERATURE = 0.0     # 打分要可重現、可校準
MAX_CHARS   = 180     # 每篇摘要截斷長度（省 token）


# ── Prompt（system）──────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert in computer-science citation analysis. For each PAIR of \
papers, output the probability (0.00-1.00) that a citation link exists between \
them in EITHER direction.

A link means one paper cites the other. It reflects intellectual DEPENDENCY, \
not textual similarity: similar-sounding papers may never cite each other \
(parallel work), and different-sounding papers may be linked (a method paper \
citing the problem or dataset it builds on). Judge the relationship, not word overlap.

Consider: same specific subfield/problem; whether one uses or extends the \
other's method; problem-solution or foundational dependency; shared named \
technique/dataset/benchmark. Generic overlap ("both are ML") is WEAK; specific \
shared concepts are STRONG.

Score guide: 0.90-1.00 clear specific dependency, same niche; 0.70-0.89 same \
subfield, shared technique, plausible link; 0.45-0.69 related but uncertain; \
0.20-0.44 same broad field only; 0.00-0.19 different subfields. Use the full \
range and fine values (e.g. 0.63); never round to 0 or 1. Most random pairs \
should score low.

Temporal prior: when a pair shows publication years, treat the time gap as a \
soft signal. A link is undirected (it exists if either paper cites the other), \
so direction does not matter. Papers published close together (same or adjacent \
year) sharing a specific topic are more likely linked, while a large gap (e.g. \
more than ~5 years) modestly LOWERS the probability. This is a weak prior — \
never let it override a strong specific topical dependency, and never treat it \
as a hard cutoff.

Examples:
[A] "Deep Residual Learning... ResNet for very deep CNNs on ImageNet." [B] \
"Identity Mappings in Deep Residual Networks... improved residual blocks." \
-> 0.96 (B directly extends A's exact architecture)
[A] "Spectral Methods for Community Detection in Graphs." [B] "BERT: \
pre-training bidirectional Transformers for NLP." -> 0.05 (different subfields)

Output ONLY a JSON array, one object per pair, preserving the given id:
[{"id":1,"score":0.74}, ...]
Output the JSON array and nothing else."""


# ── 對外介面 ──────────────────────────────────────────────────────────
def score_pairs(pairs, texts, tag, years=None, log_prefix="[llm_pairwise]"):
    """
    pairs: list[(u, v)]，整數節點對
    texts: list[str]，texts[i] = 節點 i 的文字
    tag:   資料集快取標籤（如 "arxiv" / "cora"）
    years: 可選 list[int]，years[i] = 節點 i 的發表年份；提供時會寫進 prompt
    回傳:  list[float]，與 pairs 對齊的 0-1 分數

    讀寫 data/{tag}_llm_pairwise_scores.json 做斷點續跑（key = "min,max"）。
    已快取的 pair 不重打，因此跨 run / sparsity 重複的 pair 成本為 0。
    """
    cache_path = f"data/{tag}_llm_pairwise_scores.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    # 找出還沒打分的 pair（去重）
    todo, seen = [], set()
    for u, v in pairs:
        k = _pair_key(u, v)
        if k not in cache and k not in seen:
            seen.add(k)
            todo.append((u, v))

    if todo:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("llm_pairwise 需要 GROQ_API_KEY")
        from groq import Groq
        client = Groq(api_key=api_key)

        n_batch = (len(todo) + BATCH - 1) // BATCH
        print(f"{log_prefix} 需打分 {len(todo)} 個新 pair，共 {n_batch} batch（model={MODEL}）")

        for b in range(n_batch):
            lo, hi = b * BATCH, min((b + 1) * BATCH, len(todo))
            batch  = todo[lo:hi]
            pairs_block = "\n\n".join(
                _format_pair(i, u, v, texts, years)
                for i, (u, v) in enumerate(batch)
            )
            scores = _call_batch(client, pairs_block, len(batch), log_prefix, b)
            for (u, v), s in zip(batch, scores):
                cache[_pair_key(u, v)] = s

            time.sleep(1.5)  # 控制 RPM
            if (b + 1) % 10 == 0 or hi == len(todo):
                _save(cache_path, cache)

        _save(cache_path, cache)

    return [cache[_pair_key(u, v)] for u, v in pairs]


# ── 內部工具 ──────────────────────────────────────────────────────────
def _format_pair(i, u, v, texts, years) -> str:
    """組出單一 pair 的 prompt 區塊；有 years 時附上各篇發表年份。"""
    def yr(n):
        return f" (year {int(years[n])})" if years is not None else ""
    return (f'PAIR {i+1}:\n'
            f'[A]{yr(u)} "{texts[u][:MAX_CHARS]}"\n'
            f'[B]{yr(v)} "{texts[v][:MAX_CHARS]}"')


def _pair_key(u, v) -> str:
    a, b = (u, v) if u <= v else (v, u)
    return f"{a},{b}"


def _save(path, cache):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _call_batch(client, pairs_block, n_pairs, log_prefix, b):
    """呼叫 Groq，回傳長度 = n_pairs 的分數（解析失敗的填 0.5 中性值）"""
    user_msg = (
        f"NOW SCORE THESE PAIRS:\n{pairs_block}\n\n"
        f"Output ONLY the JSON array of {n_pairs} objects."
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw    = resp.choices[0].message.content
            scores = _parse_scores(raw, n_pairs)
            filled = sum(1 for s in scores if s is not None)
            print(f"{log_prefix} batch {b:4d} | {filled}/{n_pairs} 筆")
            return [s if s is not None else 0.5 for s in scores]
        except Exception as e:
            if attempt < 2:
                time.sleep(4 * (attempt + 1))
            else:
                print(f"{log_prefix} batch {b:4d} | 失敗: {e}")
                return [0.5] * n_pairs


def _parse_scores(raw: str, n_pairs: int):
    """解析 LLM 回傳，回傳長度 = n_pairs 的 list（無法解析處為 None）"""
    text   = raw.strip()
    text   = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    scores = [None] * n_pairs

    # 優先：抓第一個 JSON 陣列，依 id 對位
    try:
        start = text.index("[")
        end   = text.rindex("]") + 1
        arr   = json.loads(text[start:end])
        for obj in arr:
            i = int(obj.get("id", 0)) - 1
            if 0 <= i < n_pairs and "score" in obj:
                scores[i] = _clip01(float(obj["score"]))
        if any(s is not None for s in scores):
            return scores
    except Exception:
        pass

    # 後備：依序抓所有 "score": x
    found = re.findall(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
    for i, val in enumerate(found[:n_pairs]):
        scores[i] = _clip01(float(val))
    return scores
