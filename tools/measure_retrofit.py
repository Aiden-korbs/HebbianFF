#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this harness in no-op retrofit mode. Force these even if the caller's
# shell has experimental settings from a previous run.
os.environ["USE_KV_CACHE"] = "1"
os.environ["USE_DRAFT_HEAD"] = "0"
os.environ["DRAFT_BLEND_BP"] = "0"
os.environ["INFER_MEMORY_TOKENS"] = "0"
os.environ["INFER_USE_ENGRAM"] = "0"
os.environ["CPU_CTX"] = "0"
os.environ["CPU_HASH_CTX"] = "0"
os.environ["USE_BITNET"] = "0"
os.environ.setdefault("NO_INIT_LOAD", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import chat_hf


EVAL_TEXTS = [
    "Write a Python function that returns the longest palindromic substring in a given string.",
    "A cache stores key and value tensors for each transformer layer. Explain why this saves decode compute.",
    "def add_numbers(a, b):\n    return a + b\n\nprint(add_numbers(2, 5))\n",
    "The quick brown fox jumps over the lazy dog. A language model predicts the next token from context.",
]

SAMPLE_PROMPTS = [
    "Write a Python function `is_prime(n)` and keep it concise.",
    "Explain in two sentences why KV caching speeds up autoregressive decoding.",
    "Given x = 17, y = 5, what are x // y and x % y?",
]

SPEED_PROMPT = (
    "You are reviewing a small Python program for correctness. "
    "Identify edge cases, explain the reasoning, and provide a concise fix. "
)

LONG_RECALL_KEY = "secret code"
LONG_RECALL_VALUE = "MANGO7429"


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_from_name(name: str, device: str) -> torch.dtype:
    name = (name or "auto").lower()
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"bad dtype: {name}")


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def reset_peak(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cuda_memory(device: str) -> Dict[str, int]:
    if device != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def parameter_bytes(model: torch.nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


def tensor_tree_bytes(obj: Any) -> int:
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(tensor_tree_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_bytes(v) for v in obj)
    if hasattr(obj, "to_legacy_cache"):
        try:
            return tensor_tree_bytes(obj.to_legacy_cache())
        except Exception:
            pass
    total = 0
    for attr in ("key_cache", "value_cache"):
        if hasattr(obj, attr):
            try:
                total += tensor_tree_bytes(getattr(obj, attr))
            except Exception:
                pass
    for attr in ("layers", "keys", "values"):
        if hasattr(obj, attr):
            try:
                total += tensor_tree_bytes(getattr(obj, attr))
            except Exception:
                pass
    return int(total)


def kv_layer_breakdown(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def tensor_bytes(t: Any) -> int:
        return int(t.numel() * t.element_size()) if torch.is_tensor(t) else 0

    if obj is None:
        return rows

    # FF_LLM cache format: {"ff": [(k, v), ...], "bp": [(k, v), ...], ...}
    if isinstance(obj, dict):
        if "ff" in obj or "bp" in obj:
            for group in ("ff", "bp"):
                for i, entry in enumerate(obj.get(group, []) or []):
                    if isinstance(entry, dict) and entry.get("format") == "int8":
                        k = entry.get("k")
                        v = entry.get("v")
                        k_scale = entry.get("k_scale")
                        v_scale = entry.get("v_scale")
                        kb = tensor_bytes(k)
                        vb = tensor_bytes(v)
                        ksb = tensor_bytes(k_scale)
                        vsb = tensor_bytes(v_scale)
                        rows.append({
                            "name": f"{group}.{i:02d}",
                            "format": "int8",
                            "key_bytes": kb,
                            "value_bytes": vb,
                            "key_scale_bytes": ksb,
                            "value_scale_bytes": vsb,
                            "total_bytes": kb + vb + ksb + vsb,
                            "key_shape": list(k.shape) if torch.is_tensor(k) else None,
                            "value_shape": list(v.shape) if torch.is_tensor(v) else None,
                            "key_scale_shape": list(k_scale.shape) if torch.is_tensor(k_scale) else None,
                            "value_scale_shape": list(v_scale.shape) if torch.is_tensor(v_scale) else None,
                            "dtype": str(k.dtype).replace("torch.", "") if torch.is_tensor(k) else None,
                            "scale_dtype": str(k_scale.dtype).replace("torch.", "") if torch.is_tensor(k_scale) else None,
                        })
                        continue
                    if isinstance(entry, dict) and entry.get("format") == "float":
                        k = entry.get("k")
                        v = entry.get("v")
                        pos = entry.get("pos")
                        kb = tensor_bytes(k)
                        vb = tensor_bytes(v)
                        pb = tensor_bytes(pos)
                        rows.append({
                            "name": f"{group}.{i:02d}",
                            "format": "float",
                            "key_bytes": kb,
                            "value_bytes": vb,
                            "position_bytes": pb,
                            "total_bytes": kb + vb + pb,
                            "key_shape": list(k.shape) if torch.is_tensor(k) else None,
                            "value_shape": list(v.shape) if torch.is_tensor(v) else None,
                            "position_shape": list(pos.shape) if torch.is_tensor(pos) else None,
                            "dtype": str(k.dtype).replace("torch.", "") if torch.is_tensor(k) else None,
                        })
                        continue
                    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                        continue
                    k, v = entry
                    kb = tensor_bytes(k)
                    vb = tensor_bytes(v)
                    rows.append({
                        "name": f"{group}.{i:02d}",
                        "format": "float",
                        "key_bytes": kb,
                        "value_bytes": vb,
                        "key_scale_bytes": 0,
                        "value_scale_bytes": 0,
                        "total_bytes": kb + vb,
                        "key_shape": list(k.shape) if torch.is_tensor(k) else None,
                        "value_shape": list(v.shape) if torch.is_tensor(v) else None,
                        "dtype": str(k.dtype).replace("torch.", "") if torch.is_tensor(k) else None,
                    })
            return rows
        for key, value in obj.items():
            rows.extend(kv_layer_breakdown(value, f"{prefix}{key}."))
        return rows

    # Legacy HF cache format: tuple/list of per-layer (k, v).
    if isinstance(obj, (tuple, list)):
        if obj and all(isinstance(x, (tuple, list)) and len(x) >= 2 and torch.is_tensor(x[0]) for x in obj):
            for i, pair in enumerate(obj):
                k, v = pair[0], pair[1]
                kb = tensor_bytes(k)
                vb = tensor_bytes(v)
                rows.append({
                    "name": f"{prefix}layer.{i:02d}",
                    "key_bytes": kb,
                    "value_bytes": vb,
                    "total_bytes": kb + vb,
                    "key_shape": list(k.shape),
                    "value_shape": list(v.shape),
                    "dtype": str(k.dtype).replace("torch.", ""),
                })
            return rows
        for i, value in enumerate(obj):
            rows.extend(kv_layer_breakdown(value, f"{prefix}{i}."))
        return rows

    # Current Transformers Cache object: cache.layers[*].keys / .values.
    if hasattr(obj, "layers"):
        for i, layer in enumerate(getattr(obj, "layers")):
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if torch.is_tensor(k) and torch.is_tensor(v):
                kb = tensor_bytes(k)
                vb = tensor_bytes(v)
                rows.append({
                    "name": f"{prefix}layer.{i:02d}",
                    "key_bytes": kb,
                    "value_bytes": vb,
                    "total_bytes": kb + vb,
                    "key_shape": list(k.shape),
                    "value_shape": list(v.shape),
                    "dtype": str(k.dtype).replace("torch.", ""),
                })
        return rows

    return rows


def trim_ff_cache(cache: Dict[str, Any], max_cache_len: Optional[int]) -> Dict[str, Any]:
    if max_cache_len is None or int(max_cache_len) <= 0:
        return cache
    keep = int(max_cache_len)
    for group in ("ff", "bp"):
        new_layers = []
        for entry in cache.get(group, []) or []:
            if entry is None:
                new_layers.append(None)
                continue
            if isinstance(entry, dict) and entry.get("format") == "int8":
                entry = dict(entry)
                for key in ("k", "v", "k_scale", "v_scale"):
                    t = entry.get(key)
                    if torch.is_tensor(t) and t.size(2) > keep:
                        entry[key] = t[:, :, -keep:, :].contiguous()
                pos = entry.get("pos")
                if torch.is_tensor(pos) and pos.numel() > keep:
                    entry["pos"] = pos[-keep:].contiguous()
                new_layers.append(entry)
                continue
            if isinstance(entry, dict) and entry.get("format") == "float":
                entry = dict(entry)
                for key in ("k", "v"):
                    t = entry.get(key)
                    if torch.is_tensor(t) and t.size(2) > keep:
                        entry[key] = t[:, :, -keep:, :].contiguous()
                pos = entry.get("pos")
                if torch.is_tensor(pos) and pos.numel() > keep:
                    entry["pos"] = pos[-keep:].contiguous()
                new_layers.append(entry)
                continue
            k, v = entry
            if torch.is_tensor(k) and k.size(2) > keep:
                k = k[:, :, -keep:, :].contiguous()
            if torch.is_tensor(v) and v.size(2) > keep:
                v = v[:, :, -keep:, :].contiguous()
            new_layers.append((k, v))
        if group in cache:
            cache[group] = new_layers
    cache["max_cache_len"] = keep
    return cache


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    return vals


def build_cache_policies(args, cfg_block_size: Optional[int] = None) -> List[Dict[str, Any]]:
    policies: List[Dict[str, Any]] = []
    requested = [p.strip().lower() for p in args.cache_policies.split(",") if p.strip()]
    for name in requested:
        if name == "full":
            policies.append({"name": "full", "kind": "full", "max_cache_len": cfg_block_size})
        elif name in ("bounded", "sliding"):
            for n in parse_int_list(args.bounded_cache_lens):
                policies.append({"name": f"bounded_{n}", "kind": "bounded", "max_cache_len": int(n)})
        elif name in ("sink", "sink_recent", "attention_sink"):
            for n in parse_int_list(args.bounded_cache_lens):
                sink = min(max(0, int(args.kv_cache_sink_tokens)), max(0, int(n) - 1))
                policies.append({
                    "name": f"sink_{sink}_recent_{int(n) - sink}",
                    "kind": "sink",
                    "max_cache_len": int(n),
                    "kv_cache_sink_tokens": sink,
                })
        elif name in ("int8", "kv_int8"):
            policies.append({
                "name": "int8",
                "kind": "int8",
                "max_cache_len": cfg_block_size,
                "kv_cache_int8": True,
            })
        else:
            raise ValueError(f"unknown cache policy: {name}")
    return policies


def set_ff_kv_cache_int8(raw_model, enabled: bool) -> None:
    raw_model.cfg.kv_cache_int8 = bool(enabled)
    for module in raw_model.modules():
        if hasattr(module, "kv_cache_int8"):
            module.kv_cache_int8 = bool(enabled)


def set_ff_kv_cache_sink_tokens(raw_model, sink_tokens: int) -> None:
    sink = max(0, int(sink_tokens or 0))
    raw_model.cfg.kv_cache_sink_tokens = sink
    for module in raw_model.modules():
        if hasattr(module, "kv_cache_sink_tokens"):
            module.kv_cache_sink_tokens = sink


def long_recall_policy_specs(args, cfg_block_size: int) -> List[Dict[str, Any]]:
    specs = [{"policy": "ff_full", "name": "ff_full", "kind": "full", "max_cache_len": cfg_block_size}]
    requested = {p.strip().lower() for p in args.cache_policies.split(",") if p.strip()}
    if "int8" in requested or "kv_int8" in requested:
        specs.append({
            "policy": "ff_int8",
            "name": "ff_int8",
            "kind": "int8",
            "max_cache_len": cfg_block_size,
            "kv_cache_int8": True,
        })
    if "sink" in requested or "sink_recent" in requested or "attention_sink" in requested:
        for n in parse_int_list(args.long_recall_cache_lens):
            sink = min(max(0, int(args.kv_cache_sink_tokens)), max(0, int(n) - 1))
            specs.append({
                "policy": f"sink_{sink}_recent_{int(n) - sink}",
                "name": f"sink_{sink}_recent_{int(n) - sink}",
                "kind": "sink",
                "max_cache_len": int(n),
                "kv_cache_sink_tokens": sink,
            })
    for n in parse_int_list(args.long_recall_cache_lens):
        specs.append({"policy": f"bounded_{n}", "name": f"bounded_{n}", "kind": "bounded", "max_cache_len": int(n)})
    return specs


def ensure_tokenizer(model_id_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def make_prompt_ids(tok, text: str, target_len: int, device: str) -> torch.Tensor:
    ids = tok(text, add_special_tokens=False).input_ids
    if not ids:
        ids = [tok.eos_token_id or 0]
    while len(ids) < target_len:
        ids.extend(ids)
    ids = ids[:target_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def make_eval_batches(tok, block_size: int, max_eval_tokens: int, device: str):
    batches = []
    max_len = max(8, min(block_size, max_eval_tokens))
    for text in EVAL_TEXTS:
        ids = tok(text, add_special_tokens=False).input_ids
        if tok.eos_token_id is not None:
            ids.append(int(tok.eos_token_id))
        if len(ids) < 3:
            continue
        ids = ids[: max_len + 1]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        batches.append((text, x, y))
    if not batches:
        raise RuntimeError("no usable eval texts")
    return batches


def build_long_recall_prompt(tok, target_tokens: int) -> Dict[str, Any]:
    header = (
        "Important memory instruction. The following sentence contains the only secret code.\n"
        f"The secret code is {LONG_RECALL_VALUE}.\n"
        "Remember the secret code exactly.\n\n"
    )
    question = (
        "\n\nFinal question: Reply with only the secret code. No explanation."
    )
    filler_unit = (
        "Filler note: this paragraph is unrelated bookkeeping about cache policy "
        "measurement, tensor shapes, throughput timing, and deterministic decoding. "
        "It intentionally contains no secret key or answer value.\n"
    )
    def render(content: str) -> str:
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return content

    content = header + question
    text = render(content)
    target_tokens = max(64, int(target_tokens))
    while len(tok(text, add_special_tokens=False).input_ids) < target_tokens:
        content = header + (filler_unit * (1 + content.count("Filler note:"))) + question
        text = render(content)
    ids = tok(text, add_special_tokens=False).input_ids[:target_tokens]
    # Ensure the final question remains present after trimming to the target.
    q_ids = tok(render(question), add_special_tokens=False).input_ids
    if len(ids) >= len(q_ids):
        ids[-len(q_ids):] = q_ids
    prompt = tok.decode(ids, skip_special_tokens=False)
    return {
        "key": LONG_RECALL_KEY,
        "answer": LONG_RECALL_VALUE,
        "target_tokens": target_tokens,
        "actual_tokens": len(ids),
        "prompt": prompt,
        "ids": ids,
    }


def normalize_recall_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def recall_hit_fields(expected_answer: str, generated_text: str) -> Dict[str, bool]:
    return {
        "exact_answer_appears": expected_answer in generated_text,
        "normalized_answer_appears": normalize_recall_text(expected_answer) in normalize_recall_text(generated_text),
    }


@torch.inference_mode()
def native_logits(model, x: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        return model(x, use_cache=False).logits.float()


@torch.inference_mode()
def ff_logits(raw_model, x: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    raw_model._working_mem = None
    raw_model._engram_state = None
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        if hasattr(raw_model, "forward_features_eval"):
            x_full = raw_model.forward_features_eval(x)
        else:
            x_full, _ = raw_model.forward_features(x, update_state=False)
        logits = raw_model._get_logits(x_full).float()
    raw_model._working_mem = None
    raw_model._engram_state = None
    return logits


@torch.inference_mode()
def ff_policy_logits(
    raw_model,
    x: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    max_cache_len: Optional[int],
    kv_cache_int8: bool = False,
    kv_cache_sink_tokens: int = 0,
) -> torch.Tensor:
    """Return per-position logits through the FF_LLM KV path under a cache policy."""
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    raw_model._working_mem = None
    raw_model._engram_state = None
    set_ff_kv_cache_int8(raw_model, kv_cache_int8)
    set_ff_kv_cache_sink_tokens(raw_model, kv_cache_sink_tokens)
    parts: List[torch.Tensor] = []
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        logits, cache = raw_model.prefill_kv(x[:, :1])
    cache = trim_ff_cache(cache, max_cache_len)
    parts.append(logits.float().unsqueeze(1))
    for pos in range(1, x.size(1)):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
            logits, cache = raw_model.decode_one_kv(x[:, pos:pos + 1], cache)
        cache = trim_ff_cache(cache, max_cache_len)
        parts.append(logits.float().unsqueeze(1))
    raw_model._working_mem = None
    raw_model._engram_state = None
    set_ff_kv_cache_int8(raw_model, False)
    set_ff_kv_cache_sink_tokens(raw_model, 0)
    return torch.cat(parts, dim=1)


def ce_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum")


def compare_logits(
    tok,
    native_cpu: List[torch.Tensor],
    other_cpu: List[torch.Tensor],
    labels_cpu: List[torch.Tensor],
    other_name: str = "ff",
) -> Dict[str, Any]:
    total_tokens = 0
    total_kl = 0.0
    top1_ok = 0
    native_top1_in_ff_top5 = 0
    ff_top1_in_native_top5 = 0
    top5_jaccard_sum = 0.0
    max_abs = 0.0
    mean_abs_num = 0.0
    ce_native = 0.0
    ce_ff = 0.0
    first_mismatch = None

    for bi, (na, fb, labels) in enumerate(zip(native_cpu, other_cpu, labels_cpu)):
        na = na.float()
        fb = fb.float()
        labels = labels.long()
        n_tok = labels.numel()
        total_tokens += n_tok

        ce_native += float(ce_from_logits(na, labels))
        ce_ff += float(ce_from_logits(fb, labels))

        n_logp = F.log_softmax(na, dim=-1)
        f_logp = F.log_softmax(fb, dim=-1)
        n_prob = n_logp.exp()
        kl = (n_prob * (n_logp - f_logp)).sum(dim=-1)
        total_kl += float(kl.sum())

        n_top5 = torch.topk(na, k=5, dim=-1).indices
        f_top5 = torch.topk(fb, k=5, dim=-1).indices
        n_top1 = n_top5[..., 0]
        f_top1 = f_top5[..., 0]

        same = n_top1 == f_top1
        top1_ok += int(same.sum())
        native_top1_in_ff_top5 += int((f_top5 == n_top1.unsqueeze(-1)).any(dim=-1).sum())
        ff_top1_in_native_top5 += int((n_top5 == f_top1.unsqueeze(-1)).any(dim=-1).sum())

        flat_n5 = n_top5.reshape(-1, 5)
        flat_f5 = f_top5.reshape(-1, 5)
        for a, b in zip(flat_n5.tolist(), flat_f5.tolist()):
            sa, sb = set(a), set(b)
            top5_jaccard_sum += len(sa & sb) / max(1, len(sa | sb))

        diff = (na - fb).abs()
        max_abs = max(max_abs, float(diff.max()))
        mean_abs_num += float(diff.sum())

        if first_mismatch is None and not bool(same.all()):
            pos = int((~same.reshape(-1)).nonzero()[0].item())
            nt = flat_n5[pos].tolist()
            ft = flat_f5[pos].tolist()
            first_mismatch = {
                "eval_text_index": bi,
                "token_position": pos,
                "label_token_id": int(labels.reshape(-1)[pos]),
                "label_token": tok.decode([int(labels.reshape(-1)[pos])]),
                "native_top5": [{"id": int(i), "text": tok.decode([int(i)])} for i in nt],
                "ff_top5": [{"id": int(i), "text": tok.decode([int(i)])} for i in ft],
                "native_top1_id": int(nt[0]),
                "ff_top1_id": int(ft[0]),
            }

    denom = max(1, total_tokens)
    return {
        "tokens": int(total_tokens),
        "native_cross_entropy": ce_native / denom,
        f"{other_name}_cross_entropy": ce_ff / denom,
        "native_perplexity": math.exp(min(50.0, ce_native / denom)),
        f"{other_name}_perplexity": math.exp(min(50.0, ce_ff / denom)),
        f"kl_native_to_{other_name}": total_kl / denom,
        # Backwards-compatible keys for the original FF_LLM parity path.
        "ff_cross_entropy": ce_ff / denom,
        "ff_perplexity": math.exp(min(50.0, ce_ff / denom)),
        "kl_native_to_ff": total_kl / denom,
        "top1_agreement": top1_ok / denom,
        "native_top1_in_ff_top5": native_top1_in_ff_top5 / denom,
        "ff_top1_in_native_top5": ff_top1_in_native_top5 / denom,
        "top5_jaccard": top5_jaccard_sum / denom,
        "max_abs_logit_diff": max_abs,
        "mean_abs_logit_diff": mean_abs_num / max(1, sum(t.numel() for t in native_cpu)),
        "first_mismatch": first_mismatch,
    }


@torch.inference_mode()
def native_speed_and_samples(model, tok, args, device: str, dtype: torch.dtype) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    model.eval()
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    prompt = make_prompt_ids(tok, SPEED_PROMPT, args.speed_prompt_tokens, device)

    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        out = model(prompt, use_cache=True)
    sync(device)
    prefill_sec = time.perf_counter() - t0
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    prefill_kv_bytes = tensor_tree_bytes(past)

    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    sync(device)
    t1 = time.perf_counter()
    for _ in range(args.decode_tokens):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    sync(device)
    decode_sec = time.perf_counter() - t1

    speed = {
        "prompt_tokens": int(prompt.size(1)),
        "decode_tokens": int(args.decode_tokens),
        "prefill_seconds": prefill_sec,
        "decode_seconds": decode_sec,
        "prefill_tokens_per_sec": float(prompt.size(1) / max(1e-9, prefill_sec)),
        "decode_tokens_per_sec": float(args.decode_tokens / max(1e-9, decode_sec)),
        "kv_cache_bytes_after_prefill": int(prefill_kv_bytes),
        "kv_cache_bytes_after_decode": int(tensor_tree_bytes(past)),
        "kv_cache_layers_after_decode": kv_layer_breakdown(past),
        **cuda_memory(device),
    }

    samples = []
    for prompt_text in SAMPLE_PROMPTS:
        ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        generated = ids
        past = None
        next_in = ids
        for _ in range(args.sample_tokens):
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
                out = model(next_in, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            next_in = next_tok
            if tok.eos_token_id is not None and int(next_tok.item()) == int(tok.eos_token_id):
                break
        samples.append({
            "prompt": prompt_text,
            "output": tok.decode(generated[0, ids.size(1):], skip_special_tokens=True),
        })

    return speed, samples


@torch.inference_mode()
def ff_speed_and_samples(
    raw_model,
    tok,
    cfg,
    args,
    device: str,
    dtype: torch.dtype,
    max_cache_len: Optional[int] = None,
    kv_cache_int8: bool = False,
    kv_cache_sink_tokens: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    raw_model.eval()
    set_ff_kv_cache_int8(raw_model, kv_cache_int8)
    set_ff_kv_cache_sink_tokens(raw_model, kv_cache_sink_tokens)
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    prompt = make_prompt_ids(tok, SPEED_PROMPT, min(args.speed_prompt_tokens, int(cfg.block_size)), device)

    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        logits, cache = raw_model.prefill_kv(prompt)
    prefill_kv_bytes_full = tensor_tree_bytes(cache)
    cache = trim_ff_cache(cache, max_cache_len)
    sync(device)
    prefill_sec = time.perf_counter() - t0
    prefill_kv_bytes_after_policy = tensor_tree_bytes(cache)

    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    sync(device)
    t1 = time.perf_counter()
    for _ in range(args.decode_tokens):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
            logits, cache = raw_model.decode_one_kv(next_tok, cache)
        cache = trim_ff_cache(cache, max_cache_len)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    sync(device)
    decode_sec = time.perf_counter() - t1

    speed = {
        "prompt_tokens": int(prompt.size(1)),
        "decode_tokens": int(args.decode_tokens),
        "prefill_seconds": prefill_sec,
        "decode_seconds": decode_sec,
        "prefill_tokens_per_sec": float(prompt.size(1) / max(1e-9, prefill_sec)),
        "decode_tokens_per_sec": float(args.decode_tokens / max(1e-9, decode_sec)),
        "kv_cache_bytes_after_prefill_full": int(prefill_kv_bytes_full),
        "kv_cache_bytes_after_prefill": int(prefill_kv_bytes_after_policy),
        "kv_cache_bytes_after_decode": int(tensor_tree_bytes(cache)),
        "kv_cache_layers_after_decode": kv_layer_breakdown(cache),
        "max_cache_len": int(max_cache_len) if max_cache_len is not None else int(cache.get("max_cache_len", cfg.block_size)),
        "kv_cache_int8": bool(kv_cache_int8),
        "kv_cache_sink_tokens": int(kv_cache_sink_tokens),
        **cuda_memory(device),
    }
    raw_model._working_mem = None
    raw_model._engram_state = None

    samples = []
    for prompt_text in SAMPLE_PROMPTS:
        ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        ids = ids[:, -int(cfg.block_size):]
        generated = ids
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
            logits, cache = raw_model.prefill_kv(ids)
        cache = trim_ff_cache(cache, max_cache_len)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        for _ in range(args.sample_tokens):
            generated = torch.cat([generated, next_tok], dim=1)
            if tok.eos_token_id is not None and int(next_tok.item()) == int(tok.eos_token_id):
                break
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
                logits, cache = raw_model.decode_one_kv(next_tok, cache)
            cache = trim_ff_cache(cache, max_cache_len)
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        samples.append({
            "prompt": prompt_text,
            "output": tok.decode(generated[0, ids.size(1):], skip_special_tokens=True),
        })
        raw_model._working_mem = None
        raw_model._engram_state = None

    set_ff_kv_cache_int8(raw_model, False)
    set_ff_kv_cache_sink_tokens(raw_model, 0)
    return speed, samples


@torch.inference_mode()
def native_recall_once(
    model,
    tok,
    prompt_ids: torch.Tensor,
    expected_answer: str,
    args,
    device: str,
    dtype: torch.dtype,
    prompt_tokens: int,
) -> Dict[str, Any]:
    model.eval()
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated: List[int] = []
    sync(device)
    decode_start = time.perf_counter()
    for _ in range(int(args.long_recall_new_tokens)):
        tid = int(next_tok.item())
        generated.append(tid)
        if tok.eos_token_id is not None and tid == int(tok.eos_token_id):
            break
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
            out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    sync(device)
    decode_sec = time.perf_counter() - decode_start
    generated_text = tok.decode(generated, skip_special_tokens=True)
    kv_bytes = int(tensor_tree_bytes(past))
    return {
        "policy": "native_full",
        "max_cache_len": int(prompt_tokens + len(generated)),
        "expected_answer": expected_answer,
        "generated_answer": generated_text,
        **recall_hit_fields(expected_answer, generated_text),
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "prompt_tokens": int(prompt_tokens),
        "kv_cache_bytes_after_decode": kv_bytes,
        "kv_cache_mib_after_decode": float(kv_bytes / 1024**2),
        "kv_cache_layers_after_decode": kv_layer_breakdown(past),
        "decode_seconds": decode_sec,
        "decode_tokens_per_sec": float(len(generated) / max(1e-9, decode_sec)),
        "total_seconds": time.perf_counter() - t0,
        **cuda_memory(device),
    }


@torch.inference_mode()
def ff_recall_once(
    raw_model,
    tok,
    cfg,
    prompt_ids: torch.Tensor,
    expected_answer: str,
    args,
    device: str,
    dtype: torch.dtype,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    raw_model.eval()
    kv_cache_int8 = bool(policy.get("kv_cache_int8", False))
    set_ff_kv_cache_int8(raw_model, kv_cache_int8)
    kv_cache_sink_tokens = int(policy.get("kv_cache_sink_tokens", 0) or 0)
    set_ff_kv_cache_sink_tokens(raw_model, kv_cache_sink_tokens)
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    max_cache_len = policy.get("max_cache_len")
    reset_peak(device)
    sync(device)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        logits, cache = raw_model.prefill_kv(prompt_ids)
    cache = trim_ff_cache(cache, max_cache_len)
    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    generated: List[int] = []
    sync(device)
    decode_start = time.perf_counter()
    for _ in range(int(args.long_recall_new_tokens)):
        tid = int(next_tok.item())
        generated.append(tid)
        if tok.eos_token_id is not None and tid == int(tok.eos_token_id):
            break
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
            logits, cache = raw_model.decode_one_kv(next_tok, cache)
        cache = trim_ff_cache(cache, max_cache_len)
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
    sync(device)
    decode_sec = time.perf_counter() - decode_start
    generated_text = tok.decode(generated, skip_special_tokens=True)
    kv_bytes = int(tensor_tree_bytes(cache))
    raw_model._working_mem = None
    raw_model._engram_state = None
    set_ff_kv_cache_int8(raw_model, False)
    set_ff_kv_cache_sink_tokens(raw_model, 0)
    return {
        "policy": policy["policy"],
        "max_cache_len": int(max_cache_len) if max_cache_len is not None else int(cfg.block_size),
        "kv_cache_int8": kv_cache_int8,
        "kv_cache_sink_tokens": kv_cache_sink_tokens,
        "expected_answer": expected_answer,
        "generated_answer": generated_text,
        **recall_hit_fields(expected_answer, generated_text),
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "prompt_tokens": int(prompt_ids.size(1)),
        "kv_cache_bytes_after_decode": kv_bytes,
        "kv_cache_mib_after_decode": float(kv_bytes / 1024**2),
        "kv_cache_layers_after_decode": kv_layer_breakdown(cache),
        "decode_seconds": decode_sec,
        "decode_tokens_per_sec": float(len(generated) / max(1e-9, decode_sec)),
        "total_seconds": time.perf_counter() - t0,
        **cuda_memory(device),
    }


@torch.inference_mode()
def run_long_context_recall(
    native_model,
    raw_model,
    tok,
    cfg,
    args,
    device: str,
    dtype: torch.dtype,
    policies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cases = []
    requested_lengths = [128] + parse_int_list(args.long_recall_lengths)
    seen = set()
    for length in requested_lengths:
        if length in seen:
            continue
        seen.add(length)
        prompt_len = min(int(length), int(cfg.block_size))
        prompt_info = build_long_recall_prompt(tok, prompt_len)
        prompt_ids = torch.tensor([prompt_info["ids"]], dtype=torch.long, device=device)
        expected = prompt_info["answer"]

        native_full = native_recall_once(native_model, tok, prompt_ids, expected, args, device, dtype, prompt_info["actual_tokens"])
        ff_full_policy = {"policy": "ff_full", "name": "ff_full", "kind": "full", "max_cache_len": int(cfg.block_size)}
        ff_full = ff_recall_once(raw_model, tok, cfg, prompt_ids, expected, args, device, dtype, ff_full_policy)
        controls_pass = bool(native_full["normalized_answer_appears"] and ff_full["normalized_answer_appears"])

        bounded = []
        for policy in policies:
            if policy["policy"] == "ff_full":
                continue
            item = ff_recall_once(raw_model, tok, cfg, prompt_ids, expected, args, device, dtype, policy)
            item["controls_passed_before_bounded_comparison"] = controls_pass
            bounded.append(item)

        cases.append({
            "prompt_tokens": prompt_info["actual_tokens"],
            "target_tokens": prompt_info["target_tokens"],
            "expected_answer": expected,
            "controls_passed": controls_pass,
            "native_full": native_full,
            "ff_full": ff_full,
            "bounded": bounded,
        })

    return {
        "note": "CE/PPL/KL from full or teacher-forced forward passes do not necessarily measure decode-time cache truncation. This recall test forces generation after prefill with each cache policy applied.",
        "key": LONG_RECALL_KEY,
        "expected_answer": LONG_RECALL_VALUE,
        "lengths": [case["prompt_tokens"] for case in cases],
        "question_distance_note": "The fact is near the beginning of each prompt and the final question is at the end; bounded caches shorter than the prompt can drop the fact before answer decoding.",
        "cases": cases,
    }


def default_checkpoint_for(model_id: str) -> Path:
    return PROJECT_ROOT / "models" / f"{Path(model_id).name}.pt"


def import_command(model_id: str, checkpoint: Path, block_size: int) -> str:
    return (
        f"python tools/import_hf.py {model_id} "
        f"--out {checkpoint} --block-size {block_size} --bp-layers 0"
    )


def write_json(outdir: Path, payload: Dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = outdir / f"retrofit_eval_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = outdir / "latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_summary(result: Dict[str, Any]) -> None:
    if result.get("status") == "missing_checkpoint":
        print("\nMissing imported FF_LLM checkpoint.")
        print("Run:")
        print(f"  {result['import_command']}")
        print(f"\nJSON: {result['json_path']}")
        return

    native = result["native"]
    ff = result["ff_llm"]
    cmp = result["comparison"]

    def gib(n: int) -> float:
        return float(n) / 1024**3

    print("\nSummary")
    print("-" * 104)
    print(f"{'model':<12} {'param GiB':>10} {'peak alloc':>11} {'peak reserv':>11} {'prefill tok/s':>14} {'decode tok/s':>13} {'KV MiB':>10} {'CE':>10} {'PPL':>10}")
    print("-" * 104)
    for name, item in [("native", native), ("ff_llm", ff)]:
        sp = item["speed"]
        ev = item["eval"]
        print(
            f"{name:<12} "
            f"{gib(item['parameter_bytes']):>10.3f} "
            f"{gib(sp['peak_allocated_bytes']):>11.3f} "
            f"{gib(sp['peak_reserved_bytes']):>11.3f} "
            f"{sp['prefill_tokens_per_sec']:>14.2f} "
            f"{sp['decode_tokens_per_sec']:>13.2f} "
            f"{sp['kv_cache_bytes_after_decode'] / 1024**2:>10.1f} "
            f"{ev['cross_entropy']:>10.4f} "
            f"{ev['perplexity']:>10.3f}"
        )
    print("-" * 104)
    print(
        f"KL(native||ff)={cmp['kl_native_to_ff']:.6g}  "
        f"top1={100*cmp['top1_agreement']:.2f}%  "
        f"native top1 in FF top5={100*cmp['native_top1_in_ff_top5']:.2f}%  "
        f"top5 jaccard={100*cmp['top5_jaccard']:.2f}%  "
        f"max|logit diff|={cmp['max_abs_logit_diff']:.6g}"
    )
    if result["parity"]["passed"]:
        print("Parity: PASS")
    else:
        print("Parity: FAIL")
        print(json.dumps(result["parity"]["mismatch"], indent=2))
    if result.get("cache_policies"):
        print("\nCache Policies")
        print("-" * 104)
        print(f"{'policy':<16} {'max_len':>8} {'prefill tok/s':>14} {'decode tok/s':>13} {'KV MiB':>10} {'CE':>10} {'PPL':>10} {'KL':>12} {'top1':>8}")
        print("-" * 104)
        for policy in result["cache_policies"]:
            if not policy.get("implemented", True):
                print(f"{policy['name']:<16} {'n/a':>8} {'not implemented':>56}  {policy.get('reason', '')[:32]}")
                continue
            sp = policy["speed"]
            ev = policy["eval"]
            cmp_pol = policy["comparison_to_native_baseline"]
            print(
                f"{policy['name']:<16} "
                f"{str(sp.get('max_cache_len', '')):>8} "
                f"{sp['prefill_tokens_per_sec']:>14.2f} "
                f"{sp['decode_tokens_per_sec']:>13.2f} "
                f"{sp['kv_cache_bytes_after_decode'] / 1024**2:>10.1f} "
                f"{ev['cross_entropy']:>10.4f} "
                f"{ev['perplexity']:>10.3f} "
                f"{cmp_pol['kl_native_to_ff']:>12.6g} "
                f"{100*cmp_pol['top1_agreement']:>7.2f}%"
            )
        if result.get("cache_policy_json_paths"):
            print("Policy JSON:")
            for name, path in result["cache_policy_json_paths"].items():
                print(f"  {name}: {path}")
    if result.get("long_context_recall"):
        recall = result["long_context_recall"]
        print("\nLong-Context Recall")
        print("-" * 104)
        print(f"expected_answer: {recall['expected_answer']}  lengths={recall['lengths']}")
        print(f"{'tokens':>6} {'policy':<16} {'max_cache_len':>13} {'exact':>5} {'norm':>5} {'decode_tokens_per_sec':>21} {'kv_cache_mib_after_decode':>26} {'peak_allocated_bytes':>22}  generated_answer")
        print("-" * 104)
        for case in recall["cases"]:
            rows = [case["native_full"], case["ff_full"]] + case["bounded"]
            for item in rows:
                print(
                    f"{case['prompt_tokens']:>6} "
                    f"{item['policy']:<16} "
                    f"{item['max_cache_len']:>13} "
                    f"{str(bool(item['exact_answer_appears'])):>5} "
                    f"{str(bool(item['normalized_answer_appears'])):>5} "
                    f"{item['decode_tokens_per_sec']:>21.2f} "
                    f"{item['kv_cache_mib_after_decode']:>26.1f} "
                    f"{item['peak_allocated_bytes']:>22}  "
                    f"{item['generated_answer'][:80]!r}"
                )
            if not case["controls_passed"]:
                print(f"{case['prompt_tokens']:>6} {'bounded comparison skipped: native_full and ff_full controls did not both pass':<96}")
    print(f"JSON: {result['json_path']}")


def parse_args():
    p = argparse.ArgumentParser(description="Measure native HF vs imported FF_LLM no-op parity and runtime.")
    p.add_argument("--native-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--tokenizer", default=None, help="Defaults to --native-model.")
    p.add_argument("--ff-checkpoint", default=None, help="Defaults to models/<model-name>.pt.")
    p.add_argument("--outdir", default="runs/retrofit_eval")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--max-eval-tokens", type=int, default=128)
    p.add_argument("--speed-prompt-tokens", type=int, default=512)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--sample-tokens", type=int, default=64)
    p.add_argument(
        "--cache-policies",
        default="full",
        help="Comma-separated FF_LLM cache policies to benchmark: full,bounded,sink,int8. Default keeps baseline parity path.",
    )
    p.add_argument(
        "--bounded-cache-lens",
        default="64,128,256,512",
        help="Comma-separated max_cache_len values used when --cache-policies includes bounded.",
    )
    p.add_argument(
        "--long-recall-cache-lens",
        default="64,128,256,512,1024,2048",
        help="Comma-separated bounded cache lengths for the long-context recall decode test.",
    )
    p.add_argument(
        "--kv-cache-sink-tokens",
        type=int,
        default=int(os.environ.get("KV_CACHE_SINK_TOKENS", "32")),
        help="Prefix tokens to keep for the sink cache policy, counted inside max_cache_len.",
    )
    p.add_argument("--long-recall-tokens", type=int, default=2048, help="Deprecated alias kept for compatibility; use --long-recall-lengths.")
    p.add_argument("--long-recall-lengths", default="512,1024,2048", help="Comma-separated prompt lengths for recall, plus an automatic 128-token sanity case.")
    p.add_argument("--long-recall-new-tokens", type=int, default=24)
    p.add_argument("--parity-kl-threshold", type=float, default=1e-3)
    p.add_argument("--parity-top1-threshold", type=float, default=0.99)
    p.add_argument("--parity-max-abs-threshold", type=float, default=1e-2)
    p.add_argument("--no-fail-exit", action="store_true", help="Do not return exit code 1 on parity failure.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or auto_device()
    dtype = dtype_from_name(args.dtype, device)
    tokenizer_id = args.tokenizer or args.native_model
    checkpoint = Path(args.ff_checkpoint) if args.ff_checkpoint else default_checkpoint_for(args.native_model)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    outdir = PROJECT_ROOT / args.outdir if not Path(args.outdir).is_absolute() else Path(args.outdir)

    base_meta = {
        "status": "started",
        "native_model": args.native_model,
        "tokenizer": tokenizer_id,
        "ff_checkpoint": str(checkpoint),
        "device": device,
        "dtype": dtype_name(dtype),
        "block_size": int(args.block_size),
        "no_op_env": {
            k: os.environ.get(k)
            for k in [
                "USE_KV_CACHE",
                "USE_DRAFT_HEAD",
                "DRAFT_BLEND_BP",
                "INFER_MEMORY_TOKENS",
                "INFER_USE_ENGRAM",
                "CPU_CTX",
                "CPU_HASH_CTX",
                "USE_BITNET",
            ]
        },
    }

    if not checkpoint.exists():
        payload = {
            **base_meta,
            "status": "missing_checkpoint",
            "import_command": import_command(args.native_model, checkpoint, args.block_size),
        }
        path = write_json(outdir, payload)
        payload["json_path"] = str(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (outdir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print_summary(payload)
        return 2

    tok = ensure_tokenizer(tokenizer_id)
    eval_batches = make_eval_batches(tok, args.block_size, args.max_eval_tokens, device)

    print(f"[LOAD] native HF: {args.native_model}", flush=True)
    native = AutoModelForCausalLM.from_pretrained(
        args.native_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    native.eval()
    native_param_bytes = parameter_bytes(native)

    native_logits_cpu: List[torch.Tensor] = []
    labels_cpu: List[torch.Tensor] = []
    native_ce_sum = 0.0
    native_tokens = 0
    for _, x, y in eval_batches:
        logits = native_logits(native, x, device, dtype)
        native_ce_sum += float(ce_from_logits(logits, y))
        native_tokens += int(y.numel())
        native_logits_cpu.append(logits.cpu())
        labels_cpu.append(y.cpu())

    native_speed, native_samples = native_speed_and_samples(native, tok, args, device, dtype)
    native_eval = {
        "tokens": native_tokens,
        "cross_entropy": native_ce_sum / max(1, native_tokens),
        "perplexity": math.exp(min(50.0, native_ce_sum / max(1, native_tokens))),
    }

    del native
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[LOAD] FF_LLM checkpoint: {checkpoint}", flush=True)
    load_args = SimpleNamespace(
        checkpoint=str(checkpoint),
        tokenizer=tokenizer_id,
        device=device,
        dtype=args.dtype,
        block_size=args.block_size,
    )
    ff_model, raw_ff, ff_tok, cfg, ff_device, ff_dtype = chat_hf.load_model(load_args)
    if ff_tok.get_vocab() != tok.get_vocab():
        raise RuntimeError("tokenizer vocab mismatch between native tokenizer and FF_LLM tokenizer")

    ff_param_bytes = parameter_bytes(raw_ff)
    ff_logits_cpu: List[torch.Tensor] = []
    ff_ce_sum = 0.0
    ff_tokens = 0
    for _, x, y in eval_batches:
        logits = ff_logits(raw_ff, x, device, dtype)
        ff_ce_sum += float(ce_from_logits(logits, y))
        ff_tokens += int(y.numel())
        ff_logits_cpu.append(logits.cpu())

    ff_speed, ff_samples = ff_speed_and_samples(raw_ff, tok, cfg, args, device, dtype)
    ff_eval = {
        "tokens": ff_tokens,
        "cross_entropy": ff_ce_sum / max(1, ff_tokens),
        "perplexity": math.exp(min(50.0, ff_ce_sum / max(1, ff_tokens))),
    }

    comparison = compare_logits(tok, native_logits_cpu, ff_logits_cpu, labels_cpu)
    cache_policy_results: List[Dict[str, Any]] = []
    policies = build_cache_policies(args, int(cfg.block_size))
    for policy in policies:
        if not policy.get("implemented", True):
            cache_policy_results.append(policy)
            continue

        max_cache_len = policy.get("max_cache_len")
        kv_cache_int8 = bool(policy.get("kv_cache_int8", False))
        kv_cache_sink_tokens = int(policy.get("kv_cache_sink_tokens", 0) or 0)
        policy_logits_cpu: List[torch.Tensor] = []
        policy_ce_sum = 0.0
        policy_tokens = 0
        for _, x, y in eval_batches:
            logits = ff_policy_logits(
                raw_ff,
                x,
                device,
                dtype,
                max_cache_len,
                kv_cache_int8=kv_cache_int8,
                kv_cache_sink_tokens=kv_cache_sink_tokens,
            )
            policy_ce_sum += float(ce_from_logits(logits, y))
            policy_tokens += int(y.numel())
            policy_logits_cpu.append(logits.cpu())

        policy_speed, policy_samples = ff_speed_and_samples(
            raw_ff,
            tok,
            cfg,
            args,
            device,
            dtype,
            max_cache_len=max_cache_len,
            kv_cache_int8=kv_cache_int8,
            kv_cache_sink_tokens=kv_cache_sink_tokens,
        )
        policy_cmp = compare_logits(tok, native_logits_cpu, policy_logits_cpu, labels_cpu, other_name=policy["name"])
        cache_policy_results.append({
            **policy,
            "implemented": True,
            "eval": {
                "tokens": policy_tokens,
                "cross_entropy": policy_ce_sum / max(1, policy_tokens),
                "perplexity": math.exp(min(50.0, policy_ce_sum / max(1, policy_tokens))),
            },
            "comparison_to_native_baseline": policy_cmp,
            "speed": policy_speed,
            "samples": policy_samples,
        })

    recall_specs = long_recall_policy_specs(args, int(cfg.block_size))
    print(f"[LOAD] native HF recall controls: {args.native_model}", flush=True)
    native_recall = AutoModelForCausalLM.from_pretrained(
        args.native_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    native_recall.eval()
    long_context_recall = run_long_context_recall(native_recall, raw_ff, tok, cfg, args, device, dtype, recall_specs)

    del native_recall
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    mismatch = comparison["first_mismatch"]
    parity_passed = (
        comparison["kl_native_to_ff"] <= args.parity_kl_threshold
        and comparison["top1_agreement"] >= args.parity_top1_threshold
        and comparison["max_abs_logit_diff"] <= args.parity_max_abs_threshold
        and mismatch is None
    )

    samples = []
    for na, fb in zip(native_samples, ff_samples):
        samples.append({
            "prompt": na["prompt"],
            "native_output": na["output"],
            "ff_output": fb["output"],
            "exact_match": na["output"] == fb["output"],
        })

    result = {
        **base_meta,
        "status": "ok" if parity_passed else "parity_failed",
        "native": {
            "parameter_bytes": native_param_bytes,
            "eval": native_eval,
            "speed": native_speed,
            "samples": native_samples,
        },
        "ff_llm": {
            "parameter_bytes": ff_param_bytes,
            "eval": ff_eval,
            "speed": ff_speed,
            "samples": ff_samples,
        },
        "comparison": comparison,
        "cache_policies": cache_policy_results,
        "long_context_recall": long_context_recall,
        "int8_kv_hook": {
            "implemented": True,
            "off_switch": "KV_CACHE_INT8=0 or omit --cache-policies int8",
            "hook_point": "HebbianFF.blocks.RevGQACausalAttention.forward_kv stores int8 K/V plus fp16 per-token scales and dequantizes before SDPA.",
        },
        "sample_comparison": samples,
        "parity": {
            "passed": bool(parity_passed),
            "thresholds": {
                "kl_native_to_ff": args.parity_kl_threshold,
                "top1_agreement": args.parity_top1_threshold,
                "max_abs_logit_diff": args.parity_max_abs_threshold,
            },
            "mismatch": mismatch,
        },
    }
    path = write_json(outdir, result)
    result["json_path"] = str(path)
    policy_paths = {}
    for policy in cache_policy_results:
        name = policy.get("name", "policy")
        policy_path = path.with_name(f"{path.stem}_{name}.json")
        policy_payload = {
            **base_meta,
            "json_path": str(policy_path),
            "native_baseline": {
                "parameter_bytes": native_param_bytes,
                "eval": native_eval,
                "speed": native_speed,
            },
            "policy": policy,
        }
        policy_path.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")
        policy_paths[name] = str(policy_path)
    result["cache_policy_json_paths"] = policy_paths
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outdir / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result)

    if not parity_passed and not args.no_fail_exit:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
