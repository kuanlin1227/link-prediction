"""
features/llm_lora.py
M9: LoRA fine-tuned Llama-3.1-8B + linear classification head

在 Llama-3.1-8B-Instruct 上加一個線性分類頭，用 LoRA 在固定訓練集上做
supervised fine-tuning，輸出每個 pair 屬於正邊的機率。

與 M7/M8 的差異：M7/M8 是 zero-shot prompt，本方法有監督訊號（訓練邊標籤）。
依賴：pip install peft
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
MAX_CHARS   = 180
MAX_SEQ_LEN = 512

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGET  = ["q_proj", "v_proj"]

TRAIN_EPOCHS = 3
TRAIN_LR     = 2e-4
TRAIN_BATCH  = 2      # 小 batch 以節省 activation 記憶體

_MODEL_CACHE = {}


# ── 分類頭 ────────────────────────────────────────────────────────────
class _LinkHead(nn.Module):
    """最後一個 token 的 hidden state → 二元分類 logit（float32）。"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, last_h):   # (B, H) float32
        return self.linear(last_h)


# ── Prompt 格式 ──────────────────────────────────────────────────────
def _format_input(u, v, texts, years) -> str:
    def yr(n):
        return f" (year {int(years[n])})" if years is not None else ""
    return (
        f'Paper A{yr(u)}: "{texts[u][:MAX_CHARS]}"\n'
        f'Paper B{yr(v)}: "{texts[v][:MAX_CHARS]}"\n'
        "Does a citation link exist between these papers?"
    )


# ── 取最後一個非 padding token 的 hidden state ──────────────────────
def _last_token_h(out, attention_mask):
    """
    out.hidden_states[-1]: (B, T, H)，在最後一層所在的 device 上。
    attention_mask: (B, T)，在 input device 上。
    回傳: (B, H) float32，device 與 last_layer 相同。
    """
    last_layer = out.hidden_states[-1]
    seq_ends   = (attention_mask.sum(dim=1) - 1).to(last_layer.device)
    return last_layer[torch.arange(last_layer.shape[0],
                                   device=last_layer.device),
                      seq_ends].float()


# ── 模型載入 / 訓練（單例）──────────────────────────────────────────
def _get_or_train(pairs_pos, pairs_neg, texts, years, tag):
    if "bundle" in _MODEL_CACHE:
        return _MODEL_CACHE["bundle"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    adapter_dir = f"data/{tag}_lora_adapter"
    token       = os.environ.get("HF_TOKEN")
    use_gpu     = torch.cuda.is_available()
    dtype       = torch.bfloat16 if use_gpu else torch.float32
    in_dev      = "cuda:0" if use_gpu else "cpu"   # 嵌入層 device，輸入放這裡

    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    base     = AutoModelForCausalLM.from_pretrained(
        MODEL, token=token, torch_dtype=dtype,
        device_map="auto" if use_gpu else None,
    )
    head_dev  = base.lm_head.weight.device   # 最後一層與 lm_head 同 device
    hidden_sz = base.config.hidden_size

    is_cached = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))

    if is_cached:
        print(f"[llm_lora] 載入已快取的 LoRA adapter：{adapter_dir}")
        backbone = PeftModel.from_pretrained(base, adapter_dir)
        head = _LinkHead(hidden_sz).to(head_dev)
        head.load_state_dict(torch.load(
            os.path.join(adapter_dir, "head.pt"),
            map_location=head_dev, weights_only=True,
        ))
    else:
        print(f"[llm_lora] 開始 LoRA fine-tuning"
              f"（{len(pairs_pos)} 正 + {len(pairs_neg)} 負，"
              f"r={LORA_R}, epochs={TRAIN_EPOCHS}）")
        lora_cfg = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            r              = LORA_R,
            lora_alpha     = LORA_ALPHA,
            lora_dropout   = LORA_DROPOUT,
            target_modules = LORA_TARGET,
        )
        backbone = get_peft_model(base, lora_cfg)
        backbone.print_trainable_parameters()
        head = _LinkHead(hidden_sz).to(head_dev)

        _train(backbone, head, tokenizer, in_dev, head_dev,
               pairs_pos, pairs_neg, texts, years)

        os.makedirs(adapter_dir, exist_ok=True)
        backbone.save_pretrained(adapter_dir)
        torch.save(head.state_dict(), os.path.join(adapter_dir, "head.pt"))
        print(f"[llm_lora] adapter + head 已存至 {adapter_dir}")

    backbone.eval()
    head.eval()
    _MODEL_CACHE["bundle"] = (backbone, head, tokenizer, in_dev, head_dev)
    return _MODEL_CACHE["bundle"]


def _train(backbone, head, tokenizer, in_dev, head_dev,
           pairs_pos, pairs_neg, texts, years):
    from torch.optim import AdamW

    all_pairs = pairs_pos + pairs_neg
    labels    = [1] * len(pairs_pos) + [0] * len(pairs_neg)

    g   = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(all_pairs), generator=g).tolist()
    all_pairs = [all_pairs[i] for i in idx]
    labels    = [labels[i]    for i in idx]

    lora_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer   = AdamW(lora_params + list(head.parameters()), lr=TRAIN_LR)
    n_batch     = (len(all_pairs) + TRAIN_BATCH - 1) // TRAIN_BATCH

    backbone.train()
    head.train()

    for epoch in range(TRAIN_EPOCHS):
        total_loss = 0.0
        for b in range(n_batch):
            lo, hi   = b * TRAIN_BATCH, min((b + 1) * TRAIN_BATCH, len(all_pairs))
            texts_in = [_format_input(u, v, texts, years) for u, v in all_pairs[lo:hi]]
            enc = tokenizer(texts_in, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_SEQ_LEN)
            enc = {k: v.to(in_dev) for k, v in enc.items()}
            y   = torch.tensor(labels[lo:hi], dtype=torch.long, device=head_dev)

            out    = backbone(**enc, output_hidden_states=True)
            last_h = _last_token_h(out, enc["attention_mask"])
            logits = head(last_h.to(head_dev))
            loss   = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                lora_params + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()

            if (b + 1) % 50 == 0:
                print(f"[llm_lora]   epoch {epoch+1} step {b+1}/{n_batch}"
                      f"  loss={loss.item():.4f}")

        print(f"[llm_lora] epoch {epoch+1}/{TRAIN_EPOCHS}"
              f"  avg_loss={total_loss / n_batch:.4f}")

    backbone.eval()
    head.eval()


# ── 對外介面 ─────────────────────────────────────────────────────────
def score_pairs_lora(pairs, texts, tag, years=None,
                     train_pairs_pos=None, train_pairs_neg=None,
                     log_prefix="[llm_lora]"):
    """
    pairs:           list[(u, v)]，要評分的 pair
    train_pairs_pos: 訓練用正邊（首次無快取時觸發 fine-tune）
    train_pairs_neg: 訓練用負邊
    回傳:            list[float]，與 pairs 對齊的正邊機率（0~1）
    """
    backbone, head, tokenizer, in_dev, head_dev = _get_or_train(
        train_pairs_pos or [], train_pairs_neg or [],
        texts, years, tag,
    )

    scores  = []
    n_batch = (len(pairs) + TRAIN_BATCH - 1) // TRAIN_BATCH

    for b in range(n_batch):
        lo, hi   = b * TRAIN_BATCH, min((b + 1) * TRAIN_BATCH, len(pairs))
        texts_in = [_format_input(u, v, texts, years) for u, v in pairs[lo:hi]]
        enc = tokenizer(texts_in, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SEQ_LEN)
        enc = {k: v.to(in_dev) for k, v in enc.items()}

        with torch.no_grad():
            out    = backbone(**enc, output_hidden_states=True)
            last_h = _last_token_h(out, enc["attention_mask"])
            probs  = (torch.softmax(head(last_h.to(head_dev)), dim=-1)[:, 1]
                      .cpu().tolist())
        scores.extend(probs)

        if (b + 1) % 20 == 0 or hi == len(pairs):
            print(f"{log_prefix} scored {hi}/{len(pairs)}")

    return scores


def clear_model_cache():
    """釋放 GPU 記憶體，避免與其他 LLM 方法同時佔用 VRAM。"""
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
