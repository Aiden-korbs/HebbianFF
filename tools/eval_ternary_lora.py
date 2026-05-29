#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ffbp_ema_cpu_ssm.ternary_runtime import PackedTernaryLoRALinear, current_ternary_runtime, resolved_runtime_config
from tools.repair_ternary_lora import load_model
from tools.sensitivity_compression_scan import get_child, set_child, topk_agreement


def cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mib": 0.0, "reserved_mib": 0.0, "peak_mib": 0.0}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def load_token_batches(path: str, tokenizer_name: str, tokens: int, batch: int, max_records: int) -> list[torch.Tensor]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows: list[torch.Tensor] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_records > 0 and len(rows) >= max_records:
                break
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tok(text.strip(), add_special_tokens=False).input_ids
            if len(ids) < 2:
                continue
            ids = ids[:tokens] if len(ids) >= tokens else ids + [tok.pad_token_id] * (tokens - len(ids))
            rows.append(torch.tensor(ids, dtype=torch.long))
    batches = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        if len(chunk) == batch:
            batches.append(torch.stack(chunk, dim=0))
    if not batches:
        raise ValueError("no eval batches built")
    return batches


def apply_adapter(model: torch.nn.Module, adapter_path: str, dtype: torch.dtype, runtime: str | None = None) -> dict:
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    modules = adapter["modules"]
    cfg = resolved_runtime_config()
    dense_modules = set(cfg["selective_dense_modules"]) if cfg["selective_dense_cache"] else set()
    extra_mib = 0.0
    for name in dense_modules:
        state = modules.get(name)
        if state is not None:
            extra_mib += int(state["in_features"]) * int(state["out_features"]) * torch.tensor([], dtype=dtype).element_size() / 1024**2
    print(
        "[TERNARY] "
        f"preset={cfg['preset']} runtime={runtime or cfg['runtime']} "
        f"prefill={cfg['prefill_runtime']} decode={cfg['decode_runtime']} "
        f"auto_prefill={cfg['auto_prefill']} dense_cache={'yes' if cfg['selective_dense_cache'] else 'no'} "
        f"dense_modules={len(dense_modules)} extra_dense_cache_mib={extra_mib:.1f} "
        f"adapter_modules={len(modules)} profile={cfg['selective_dense_profile'] or 'none'}",
        flush=True,
    )
    for name, state in modules.items():
        old = get_child(model, name)
        if not isinstance(old, torch.nn.Linear):
            raise TypeError(f"{name} is {type(old).__name__}, expected nn.Linear")
        set_child(model, name, PackedTernaryLoRALinear(state, device="cuda", dtype=dtype, runtime=runtime, module_name=name).eval())
    return adapter


@torch.no_grad()
def logits_for(model: nn.Module, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
        x, _ = model.forward_features(idx, update_state=False)
        return model._get_logits(x)


@torch.no_grad()
def eval_pair(native: nn.Module, compressed: nn.Module, batches: list[torch.Tensor], dtype: torch.dtype, topk: int, warmup: int) -> dict:
    totals = {
        "native_ce_sum": 0.0,
        "compressed_ce_sum": 0.0,
        "mean_err_sum": 0.0,
        "max_err": 0.0,
        "topk_sum": 0.0,
        "tokens": 0,
        "batches": 0,
        "native_time": 0.0,
        "compressed_time": 0.0,
    }
    for i, cpu_idx in enumerate(batches):
        idx = cpu_idx.to(device="cuda", non_blocking=True)
        labels = idx[:, 1:].contiguous()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        native_logits = logits_for(native, idx, dtype)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        compressed_logits = logits_for(compressed, idx, dtype)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        n_next = native_logits[:, :-1, :].contiguous()
        c_next = compressed_logits[:, :-1, :].contiguous()
        flat_y = labels.reshape(-1)
        flat_n = n_next.reshape(-1, n_next.size(-1)).float()
        flat_c = c_next.reshape(-1, c_next.size(-1)).float()
        tok_count = int(flat_y.numel())
        diff = (flat_n - flat_c).abs()
        if i >= warmup:
            totals["native_ce_sum"] += float(F.cross_entropy(flat_n, flat_y, reduction="sum").item())
            totals["compressed_ce_sum"] += float(F.cross_entropy(flat_c, flat_y, reduction="sum").item())
            totals["mean_err_sum"] += float(diff.mean().item()) * tok_count
            totals["max_err"] = max(totals["max_err"], float(diff.max().item()))
            totals["topk_sum"] += float(topk_agreement(n_next, c_next, topk)) * tok_count
            totals["tokens"] += tok_count
            totals["batches"] += 1
            totals["native_time"] += t1 - t0
            totals["compressed_time"] += t2 - t1
    tokens = max(1, totals["tokens"])
    return {
        "eval_batches": totals["batches"],
        "eval_tokens": totals["tokens"],
        "native_ce": totals["native_ce_sum"] / tokens,
        "compressed_ce": totals["compressed_ce_sum"] / tokens,
        "ce_delta": (totals["compressed_ce_sum"] - totals["native_ce_sum"]) / tokens,
        "native_ppl": float(torch.exp(torch.tensor(totals["native_ce_sum"] / tokens)).item()),
        "compressed_ppl": float(torch.exp(torch.tensor(totals["compressed_ce_sum"] / tokens)).item()),
        "mean_logit_error": totals["mean_err_sum"] / tokens,
        "max_logit_error": totals["max_err"],
        f"top{topk}_agreement": totals["topk_sum"] / tokens,
        "native_forwards_per_sec": totals["batches"] / max(totals["native_time"], 1e-9),
        "compressed_forwards_per_sec": totals["batches"] / max(totals["compressed_time"], 1e-9),
        "speed_ratio": totals["native_time"] / max(totals["compressed_time"], 1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--calibration-jsonl", required=True)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--runtime", default=None, help="Overrides TERNARY_RUNTIME for this eval.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    batches = load_token_batches(args.calibration_jsonl, args.tokenizer, args.tokens, args.batch, args.max_records)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    native, _, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    native_only_mem = cuda_mem()
    del native
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    compressed, _, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    runtime = args.runtime or current_ternary_runtime()
    adapter = apply_adapter(compressed, args.adapter, dtype, runtime=runtime)
    torch.cuda.empty_cache()
    compressed_only_mem = cuda_mem()
    del compressed
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    native, _, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    compressed, _, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    apply_adapter(compressed, args.adapter, dtype, runtime=runtime)
    torch.cuda.empty_cache()
    pair_loaded_mem = cuda_mem()
    metrics = eval_pair(native.eval(), compressed.eval(), batches, dtype, args.topk, args.warmup)
    pair_peak_mem = cuda_mem()

    result = {
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "runtime": runtime,
        "adapter_format": adapter.get("format"),
        "adapter_metrics": adapter.get("metrics"),
        "adapter_module_count": len(adapter.get("modules", {})),
        "native_only_mem": native_only_mem,
        "compressed_only_mem": compressed_only_mem,
        "pair_loaded_mem": pair_loaded_mem,
        "pair_peak_mem": pair_peak_mem,
        **metrics,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
