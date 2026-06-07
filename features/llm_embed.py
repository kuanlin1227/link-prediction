"""
features/llm_embed.py
Llama-3.1-8B-Instruct 節點嵌入：取最後一層最後一個 token 的 hidden state。

get_embeddings()      — 凍結的 base model
get_embeddings_lora() — 載入 LoRA adapter（需先執行過 llm_lora 訓練）

快取：data/{tag}_llama_embed.npy / data/{tag}_llama_lora_embed.npy
支援斷點續跑：每 SAVE_EVERY 個節點儲存一次。
"""

import os
import numpy as np

MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
MAX_CHARS   = 400
MAX_SEQ_LEN = 256
BATCH_SIZE  = 8
SAVE_EVERY  = 400


# ── 對外介面 ──────────────────────────────────────────────────────────

def get_embeddings(texts, tag, log_prefix="[llm_embed]"):
    """凍結的 Llama-3.1-8B-Instruct 節點嵌入，回傳 (N, H) fp32 L2-normalized array。"""
    cache_path = f"data/{tag}_llama_embed.npy"
    return _get_or_encode(texts, cache_path, adapter_dir=None,
                          log_prefix=log_prefix)


def get_embeddings_lora(texts, tag, log_prefix="[llm_embed_lora]"):
    """LoRA fine-tuned Llama-3.1-8B-Instruct 節點嵌入。

    需要 data/{tag}_lora_adapter/ 已存在（即 llm_lora 至少跑過一次）。
    回傳 (N, H) fp32 L2-normalized array。
    """
    adapter_dir = f"data/{tag}_lora_adapter"
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        raise FileNotFoundError(
            f"[llm_embed_lora] 找不到 LoRA adapter：{adapter_dir}\n"
            "請先在 FEATURE_CONFIGS 中執行 llm_lora，或確認 adapter 路徑正確。"
        )
    cache_path = f"data/{tag}_llama_lora_embed.npy"
    return _get_or_encode(texts, cache_path, adapter_dir=adapter_dir,
                          log_prefix=log_prefix)


# ── 內部實作 ──────────────────────────────────────────────────────────

def _get_or_encode(texts, cache_path, adapter_dir, log_prefix):
    """共用 encode 邏輯；adapter_dir=None 時用 base model，否則載入 LoRA。"""
    if os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts) and arr.any():
            print(f"{log_prefix} 載入快取 embedding: {cache_path} {arr.shape}")
            return arr

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    token   = os.environ.get("HF_TOKEN")
    use_gpu = torch.cuda.is_available()
    dtype   = torch.bfloat16 if use_gpu else torch.float32
    in_dev  = "cuda:0" if use_gpu else "cpu"

    label = "Llama-3.1-8B + LoRA" if adapter_dir else "Llama-3.1-8B-Instruct"
    print(f"{log_prefix} 載入 {label}（device_map=auto）...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        MODEL, token=token, torch_dtype=dtype,
        device_map="auto" if use_gpu else None,
    )

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        model = base

    model.eval()

    n           = len(texts)
    hidden_size = base.config.hidden_size

    # 斷點續跑
    if os.path.exists(cache_path):
        embs = np.load(cache_path)
        if embs.shape == (n, hidden_size):
            done = int((embs.any(axis=1)).sum())
            print(f"{log_prefix} 繼續未完成的快取（{done}/{n}）")
        else:
            embs, done = np.zeros((n, hidden_size), dtype=np.float32), 0
    else:
        embs, done = np.zeros((n, hidden_size), dtype=np.float32), 0

    start = (done // BATCH_SIZE) * BATCH_SIZE

    for b_start in range(start, n, BATCH_SIZE):
        b_end       = min(b_start + BATCH_SIZE, n)
        batch_texts = [texts[i][:MAX_CHARS] for i in range(b_start, b_end)]

        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LEN,
        )
        enc = {k: v.to(in_dev) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        last_layer = out.hidden_states[-1]               # (B, T, H)
        seq_ends   = (enc["attention_mask"].sum(dim=1) - 1).to(last_layer.device)
        last_h     = last_layer[
            torch.arange(last_layer.shape[0], device=last_layer.device), seq_ends
        ].float().cpu().numpy()                          # (B, H)

        embs[b_start:b_end] = last_h

        processed = b_end
        if processed % SAVE_EVERY == 0 or processed == n:
            np.save(cache_path, embs)
            print(f"{log_prefix} {processed}/{n} 節點已 encode")

    # L2 正規化
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embs  = (embs / norms).astype(np.float32)

    np.save(cache_path, embs)
    print(f"{log_prefix} 完成：{hidden_size} 維 embedding 存至 {cache_path}")

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embs
