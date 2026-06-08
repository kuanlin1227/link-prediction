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

模型：本地以 HuggingFace transformers 載入 meta-llama/Llama-3.1-8B-Instruct，
優先跑 GPU（CUDA），偵測不到才退回 CPU。
注意：此模型為 gated repo，須先在 HF 網站接受授權並登入
（huggingface-cli login，或設環境變數 HF_TOKEN）。
"""

import json
import os
import re


# ── 設定 ──────────────────────────────────────────────────────────────
MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
BATCH       = 12      # 一次打分幾個 pair（攤平 system prompt 成本）
TEMPERATURE = 0.0     # 打分要可重現、可校準（temperature=0 → greedy decoding）
MAX_CHARS   = 180     # 每篇摘要截斷長度
MAX_NEW_TOKENS = 768  # 一個 batch 的 JSON 陣列輸出上限

# 多數決專用設定
N_VOTES          = 5    # 每個 pair 投票次數
VOTE_TEMPERATURE = 0.7  # 需要多樣性，改用非零 temperature
VOTE_THRESHOLD   = 0.5  # 單次分數 >= 此值視為正票


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


PRODUCTS_SYSTEM_PROMPT = """\
You are an expert in e-commerce product recommendation and consumer behavior analysis.
For each PAIR of products, output the probability (0.00-1.00) that these two products \
are frequently co-purchased together by the same customers.

A co-purchase link means customers who buy one product also tend to buy the other. \
This reflects COMPLEMENTARY use or RELATED needs, not mere similarity: \
different-category items can be strongly linked (e.g., phone + phone case), \
and similar items may rarely be co-purchased (competing alternatives).

Consider: complementary use (accessories, consumables, bundled items); same activity or hobby; \
problem-solution pairs; brand ecosystems; sequential use (buy item A → need item B).
Generic category overlap ("both are electronics") is WEAK; specific complementary utility is STRONG.

Score guide: 0.90-1.00 clear complementary pair, very commonly bought together; \
0.70-0.89 related use case, plausibly co-purchased; 0.45-0.69 related category but uncertain; \
0.20-0.44 same broad category only; 0.00-0.19 unrelated products.
Use the full range and fine values (e.g. 0.63); never round to 0 or 1. \
Most random pairs should score low.

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
        model, tokenizer, device = _get_model()

        n_batch = (len(todo) + BATCH - 1) // BATCH
        print(f"{log_prefix} 需打分 {len(todo)} 個新 pair，共 {n_batch} batch"
              f"（model={MODEL}, device={device}）")

        for b in range(n_batch):
            lo, hi = b * BATCH, min((b + 1) * BATCH, len(todo))
            batch  = todo[lo:hi]
            pairs_block = "\n\n".join(
                _format_pair(i, u, v, texts, years)
                for i, (u, v) in enumerate(batch)
            )
            scores = _call_batch(model, tokenizer, device,
                                 pairs_block, len(batch), log_prefix, b)
            for (u, v), s in zip(batch, scores):
                cache[_pair_key(u, v)] = s

            if (b + 1) % 10 == 0 or hi == len(todo):
                _save(cache_path, cache)

        _save(cache_path, cache)

    return [cache[_pair_key(u, v)] for u, v in pairs]


def score_pairs_majority(pairs, texts, tag, years=None, n_votes=N_VOTES,
                         log_prefix="[llm_majority]"):
    """
    對每個 pair 執行 n_votes 次推理（temperature=VOTE_TEMPERATURE），
    將每次分數二值化後取多數決，回傳正票比例（0~1）作為最終分數。

    快取：data/{tag}_llm_majority_votes.json，key = "min,max"，
    value = list[float]（每次推理的原始分數），長度達 n_votes 即完整。
    """
    cache_path = f"data/{tag}_llm_majority_votes.json"
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    # 找出還沒打滿 n_votes 票的 pair（去重）
    seen, todo = set(), []
    for u, v in pairs:
        k = _pair_key(u, v)
        if len(cache.get(k, [])) < n_votes and k not in seen:
            seen.add(k)
            todo.append((u, v))

    if todo:
        model, tokenizer, device = _get_model()

        # 最多需要補幾輪（每輪把所有不足的 pair 打一次）
        max_rounds = max(
            n_votes - len(cache.get(_pair_key(u, v), []))
            for u, v in todo
        )

        for rnd in range(max_rounds):
            # 本輪仍需補票的 pair
            round_todo = [(u, v) for u, v in todo
                          if len(cache.get(_pair_key(u, v), [])) < n_votes]
            if not round_todo:
                break

            n_batch = (len(round_todo) + BATCH - 1) // BATCH
            print(f"{log_prefix} round {rnd+1}/{max_rounds}，"
                  f"{len(round_todo)} 個 pair，{n_batch} batch"
                  f"（temperature={VOTE_TEMPERATURE}）")

            for b in range(n_batch):
                lo, hi = b * BATCH, min((b + 1) * BATCH, len(round_todo))
                batch = round_todo[lo:hi]
                pairs_block = "\n\n".join(
                    _format_pair(i, u, v, texts, years)
                    for i, (u, v) in enumerate(batch)
                )
                scores = _call_batch(model, tokenizer, device,
                                     pairs_block, len(batch), log_prefix, b,
                                     temperature=VOTE_TEMPERATURE)
                for (u, v), s in zip(batch, scores):
                    cache.setdefault(_pair_key(u, v), []).append(s)

                if (b + 1) % 10 == 0 or hi == len(round_todo):
                    _save(cache_path, cache)

        _save(cache_path, cache)

    # 多數決：正票比例 = 分數 >= VOTE_THRESHOLD 的次數 / n_votes
    results = []
    for u, v in pairs:
        votes = cache.get(_pair_key(u, v), [])[:n_votes]
        if votes:
            pos_frac = sum(1 for s in votes if s >= VOTE_THRESHOLD) / len(votes)
        else:
            pos_frac = 0.5
        results.append(pos_frac)
    return results


# ── 模型載入（單例，整個 process 只載一次）────────────────────────────
_MODEL_CACHE = {}


def _get_model():
    """延遲載入 Llama-3.1-8B-Instruct。優先 CUDA，否則退回 CPU。"""
    if "bundle" in _MODEL_CACHE:
        return _MODEL_CACHE["bundle"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token   = os.environ.get("HF_TOKEN")  # gated repo 需要；None 時靠 CLI 登入
    use_gpu = torch.cuda.is_available()
    device  = "cuda" if use_gpu else "cpu"
    dtype   = torch.bfloat16 if use_gpu else torch.float32
    if not use_gpu:
        print("[llm_pairwise] 警告：偵測不到 CUDA，將以 CPU 執行 8B 模型（會非常慢）。")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        token=token,
        torch_dtype=dtype,
        device_map="auto" if use_gpu else None,
    )
    if not use_gpu:
        model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    _MODEL_CACHE["bundle"] = (model, tokenizer, device)
    return _MODEL_CACHE["bundle"]


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


def _call_batch(model, tokenizer, device, pairs_block, n_pairs, log_prefix, b,
                temperature=None):
    """本地推理一個 batch，回傳長度 = n_pairs 的分數（解析失敗填 0.5 中性值）"""
    import torch

    if temperature is None:
        temperature = TEMPERATURE

    user_msg = (
        f"NOW SCORE THESE PAIRS:\n{pairs_block}\n\n"
        f"Output ONLY the JSON array of {n_pairs} objects."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        # 只取新生成的部分（去掉 prompt）
        gen = out[0][inputs.shape[1]:]
        raw = tokenizer.decode(gen, skip_special_tokens=True)

        scores = _parse_scores(raw, n_pairs)
        filled = sum(1 for s in scores if s is not None)
        print(f"{log_prefix} batch {b:4d} | {filled}/{n_pairs} 筆")
        return [s if s is not None else 0.5 for s in scores]
    except Exception as e:
        print(f"{log_prefix} batch {b:4d} | 失敗: {e}")
        return [0.5] * n_pairs


def score_pairs_groq(pairs, texts, tag, system_prompt=None,
                     model="llama-3.3-70b-versatile",
                     log_prefix="[llm_pairwise_groq]"):
    """用 Groq API 對 pairs 打分，結果快取至 data/{tag}_groq_pairwise_scores.json。"""
    import time
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(f"{log_prefix} 需要設定 GROQ_API_KEY 環境變數")

    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    client     = Groq(api_key=api_key)
    cache_path = f"data/{tag}_groq_pairwise_scores.json"
    cache      = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    todo, seen = [], set()
    for u, v in pairs:
        k = _pair_key(u, v)
        if k not in cache and k not in seen:
            seen.add(k)
            todo.append((u, v))

    if todo:
        n_batch = (len(todo) + BATCH - 1) // BATCH
        print(f"{log_prefix} 需打分 {len(todo)} 個新 pair，共 {n_batch} batch"
              f"（model={model}）")

        for b in range(n_batch):
            lo, hi = b * BATCH, min((b + 1) * BATCH, len(todo))
            batch  = todo[lo:hi]
            pairs_block = "\n\n".join(
                _format_pair(i, u, v, texts, None)
                for i, (u, v) in enumerate(batch)
            )
            user_msg = (
                f"NOW SCORE THESE PAIRS:\n{pairs_block}\n\n"
                f"Output ONLY the JSON array of {len(batch)} objects."
            )

            scores = [0.5] * len(batch)
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_msg},
                        ],
                        temperature=0,
                        max_tokens=512,
                    )
                    raw    = resp.choices[0].message.content.strip()
                    parsed = _parse_scores(raw, len(batch))
                    scores = [s if s is not None else 0.5 for s in parsed]
                    filled = sum(1 for s in parsed if s is not None)
                    print(f"{log_prefix} batch {b:4d} | {filled}/{len(batch)} 筆")
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(4 * (attempt + 1))
                    else:
                        print(f"{log_prefix} batch {b:4d} | 失敗: {e}")

            for (u, v), s in zip(batch, scores):
                cache[_pair_key(u, v)] = s

            if (b + 1) % 5 == 0 or hi == len(todo):
                _save(cache_path, cache)

            time.sleep(2.5)  # Groq rate limit: 30 RPM

        _save(cache_path, cache)

    return [cache[_pair_key(u, v)] for u, v in pairs]


def clear_model_cache():
    """釋放 GPU 記憶體，讓後續方法可重新載入不同模型。"""
    _MODEL_CACHE.clear()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
