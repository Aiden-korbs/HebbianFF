#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from HebbianFF.ternary_runtime import TERNARY_RUNTIME_MODES, reset_auto_dense_budget
from tools.eval_ternary_lora import apply_adapter, load_token_batches, logits_for
from tools.repair_ternary_lora import load_model
from tools.sensitivity_compression_scan import topk_agreement


def cuda_mem() -> dict[str, float]:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


@torch.no_grad()
def bench_forwards(model, idx: torch.Tensor, dtype: torch.dtype, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        logits_for(model, idx, dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        logits_for(model, idx, dtype)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "forwards_per_sec": iters / max(elapsed, 1e-9),
        "ms_per_forward": 1000.0 * elapsed / max(iters, 1),
        "peak_forward_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def adapter_flop_overhead(adapter_path: str) -> dict[str, float | int]:
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    base = 0
    lora = 0
    ranks = {}
    for state in adapter.get("modules", {}).values():
        in_features = int(state["in_features"])
        out_features = int(state["out_features"])
        rank = int(state["rank"])
        base += 2 * in_features * out_features
        lora += 2 * (in_features * rank + rank * out_features)
        ranks[rank] = ranks.get(rank, 0) + 1
    return {
        "module_count": int(sum(ranks.values())),
        "ranks": ranks,
        "base_flops_per_token": int(base),
        "lora_flops_per_token": int(lora),
        "lora_overhead_vs_base": float(lora / base) if base else 0.0,
    }


def load_one(args, mode: str | None, env: dict[str, str] | None = None):
    old_env = {}
    if env:
        for key, value in env.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
    reset_auto_dense_budget()
    model, _, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    adapter = None
    if mode is not None:
        adapter = apply_adapter(model, args.adapter, dtype, runtime=mode)
    if env:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return model.eval(), dtype, adapter


@torch.no_grad()
def compare_to_native(args, mode: str, idx: torch.Tensor, native_logits: torch.Tensor, native_ce: float, dtype: torch.dtype, env: dict[str, str] | None = None) -> dict[str, float]:
    torch.cuda.empty_cache()
    model, _, _ = load_one(args, mode, env)
    logits = logits_for(model, idx, dtype)
    labels = idx[:, 1:].contiguous().reshape(-1)
    n = native_logits[:, :-1, :].contiguous()
    c = logits[:, :-1, :].contiguous()
    diff = (n.float() - c.float()).abs()
    ce = float(F.cross_entropy(c.reshape(-1, c.size(-1)).float(), labels, reduction="mean").item())
    out = {
        "mean_logit_error": float(diff.mean().item()),
        "max_logit_error": float(diff.max().item()),
        f"top{args.topk}_agreement": float(topk_agreement(n, c, args.topk)),
        "ce": ce,
        "ce_delta_vs_native": ce - native_ce,
    }
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--calibration-jsonl", required=True)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--prefill-tokens", type=int, default=64)
    ap.add_argument("--decode-tokens", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.decode_tokens != 1:
        raise ValueError("--decode-tokens currently must be 1")

    prefill_idx = load_token_batches(args.calibration_jsonl, args.tokenizer, args.prefill_tokens, args.batch, args.max_records)[0].cuda()
    decode_idx = prefill_idx[:, :1].contiguous()

    results = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    native, dtype, _ = load_one(args, None)
    native_mem = cuda_mem()
    native_prefill = bench_forwards(native, prefill_idx, dtype, args.warmup, args.iters)
    native_decode = bench_forwards(native, decode_idx, dtype, args.warmup, args.iters)
    native_logits = logits_for(native, prefill_idx, dtype)
    native_ce = float(F.cross_entropy(native_logits[:, :-1, :].reshape(-1, native_logits.size(-1)).float(), prefill_idx[:, 1:].reshape(-1), reduction="mean").item())
    results["native_bf16"] = {
        "load_mem": native_mem,
        "prefill": native_prefill,
        "decode": native_decode,
        "ce": native_ce,
    }
    del native
    torch.cuda.empty_cache()

    cases = [
        ("dense_debug", "dense_debug", {}),
        ("dense_debug_merge_lora", "dense_debug", {"TERNARY_DENSE_MERGE_LORA": "1"}),
        ("packed_fallback", "packed_fallback", {}),
        ("triton_gemv", "triton_gemv", {}),
        (
            "hybrid_temp_dense_triton",
            "hybrid",
            {"TERNARY_PREFILL_RUNTIME": "temp_dense", "TERNARY_DECODE_RUNTIME": "triton_gemv"},
        ),
        (
            "hybrid_dense_debug_triton",
            "hybrid",
            {"TERNARY_PREFILL_RUNTIME": "dense_debug", "TERNARY_DECODE_RUNTIME": "triton_gemv"},
        ),
        (
            "hybrid_packed_triton",
            "hybrid",
            {"TERNARY_PREFILL_RUNTIME": "packed_fallback", "TERNARY_DECODE_RUNTIME": "triton_gemv"},
        ),
        (
            "auto",
            "auto",
            {"TERNARY_AUTO_PREFILL": "dense_if_possible", "TERNARY_DENSE_MERGE_LORA": "1"},
        ),
    ]

    for label, mode, env in cases:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model, dtype, adapter = load_one(args, mode, env)
            load_mem = cuda_mem()
            prefill = bench_forwards(model, prefill_idx, dtype, args.warmup, args.iters)
            decode = bench_forwards(model, decode_idx, dtype, args.warmup, args.iters)
            del model
            torch.cuda.empty_cache()
            compare = compare_to_native(args, mode, prefill_idx, native_logits, native_ce, dtype, env)
            results[label] = {
                "runtime": mode,
                "env": env,
                "load_mem": load_mem,
                "prefill": prefill,
                "decode": decode,
                "prefill_speed_ratio_vs_native": prefill["forwards_per_sec"] / results["native_bf16"]["prefill"]["forwards_per_sec"],
                "decode_speed_ratio_vs_native": decode["forwards_per_sec"] / results["native_bf16"]["decode"]["forwards_per_sec"],
                "adapter_format": None if adapter is None else adapter.get("format"),
                **compare,
            }
        except Exception as exc:
            results[label] = {"runtime": mode, "env": env, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            torch.cuda.empty_cache()

    result = {
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "available_modes": sorted(TERNARY_RUNTIME_MODES),
        "adapter_flops": adapter_flop_overhead(args.adapter),
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "batch": args.batch,
        "iters": args.iters,
        "results": results,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
