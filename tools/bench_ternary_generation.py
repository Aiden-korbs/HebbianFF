#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from HebbianFF.ternary_runtime import (
    reset_auto_dense_budget,
    reset_ternary_profile,
    save_profile_json,
    ternary_profile_summary,
)
from tools.eval_ternary_lora import apply_adapter
from tools.repair_ternary_lora import load_model
from tools.sensitivity_compression_scan import topk_agreement


def cuda_mem() -> dict[str, float]:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mib": torch.cuda.memory_reserved() / 1024**2,
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


class EnvPatch:
    def __init__(self, values: dict[str, str]):
        self.values = values
        self.old: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.old[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


def make_prompt(vocab_size: int, prompt_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + prompt_len)
    return torch.randint(0, vocab_size, (1, prompt_len), device="cuda", dtype=torch.long, generator=gen)


def load_case(args, runtime: str | None, env: dict[str, str]):
    with EnvPatch(env):
        reset_auto_dense_budget()
        model, _cfg, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
        adapter = None
        if runtime is not None or env.get("TERNARY_PRESET") in {"low_vram", "balanced", "speed", "manual"}:
            adapter = apply_adapter(model, args.adapter, dtype, runtime=runtime)
    return model.eval(), dtype, adapter


@torch.inference_mode()
def run_generation(
    model,
    prompt: torch.Tensor,
    new_tokens: int,
    dtype: torch.dtype,
    forced_tokens: list[torch.Tensor] | None = None,
) -> tuple[dict[str, float], list[torch.Tensor], list[torch.Tensor]]:
    logits_history: list[torch.Tensor] = []
    generated: list[torch.Tensor] = []
    autocast = torch.amp.autocast("cuda", dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with autocast:
        t0 = time.perf_counter()
        logits, cache = model.prefill_kv(prompt)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits_history.append(logits.detach())
        next_token = forced_tokens[0] if forced_tokens else torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token.detach())
        for step in range(new_tokens):
            logits, cache = model.decode_one_kv(next_token, cache)
            logits_history.append(logits.detach())
            if forced_tokens is not None and step + 1 < len(forced_tokens):
                next_token = forced_tokens[step + 1]
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if step + 1 < new_tokens:
                generated.append(next_token.detach())
        torch.cuda.synchronize()
        t2 = time.perf_counter()
    prefill_s = t1 - t0
    decode_s = t2 - t1
    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": t2 - t0,
        "decode_tokens_per_sec": new_tokens / max(decode_s, 1e-9),
        "total_tokens_per_sec": (int(prompt.numel()) + new_tokens) / max(t2 - t0, 1e-9),
        "peak_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }, logits_history, generated


def compare_logits(native: list[torch.Tensor], other: list[torch.Tensor], steps: int, topk: int) -> dict[str, float]:
    count = min(len(native), len(other), max(1, steps))
    diffs = []
    tops = []
    for i in range(count):
        n = native[i]
        c = other[i]
        diffs.append((n.float() - c.float()).abs())
        tops.append(float(topk_agreement(n, c, topk)))
    diff = torch.cat([d.reshape(-1) for d in diffs])
    return {
        "quality_steps": count,
        "mean_logit_error": float(diff.mean().item()),
        "max_logit_error": float(diff.max().item()),
        f"top{topk}_agreement": float(sum(tops) / max(len(tops), 1)),
    }


def run_case(
    args,
    label: str,
    runtime: str | None,
    env: dict[str, str],
    prompts: list[torch.Tensor],
    native_logits: dict[int, list[torch.Tensor]] | None,
    native_tokens: dict[int, list[torch.Tensor]] | None = None,
    profile: bool = False,
) -> dict:
    case_env = dict(env)
    if profile:
        case_env["TERNARY_PROFILE_MODULES"] = "1"
        reset_ternary_profile()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, dtype, adapter = load_case(args, runtime, case_env)
    load_mem = cuda_mem()
    if prompts and args.warmup_new_tokens > 0:
        run_generation(model, prompts[0], min(args.new_tokens, args.warmup_new_tokens), dtype)
    if profile and prompts:
        run_generation(model, prompts[0], min(args.new_tokens, 4), dtype)
        reset_ternary_profile()
    prompt_results = {}
    profile_rows = []
    for prompt in prompts:
        forced = None if native_tokens is None else native_tokens.get(int(prompt.numel()))
        metrics, logits, _generated = run_generation(model, prompt, args.new_tokens, dtype, forced_tokens=forced)
        if native_logits is not None:
            metrics.update(compare_logits(native_logits[int(prompt.numel())], logits, args.quality_steps, args.topk))
        prompt_results[str(int(prompt.numel()))] = metrics
    if profile:
        profile_rows = ternary_profile_summary(20)
    out = {
        "runtime": runtime or env.get("TERNARY_PRESET", "native_bf16"),
        "env": env,
        "adapter_format": None if adapter is None else adapter.get("format"),
        "load_mem": load_mem,
        "prompts": prompt_results,
    }
    if profile_rows:
        out["profile_top20"] = profile_rows
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--prompt-lens", default="64,256")
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--warmup-new-tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--quality-steps", type=int, default=8)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--preset", choices=["low_vram", "balanced", "speed", "all"], default="all")
    ap.add_argument("--profile-json", default="ternary_repair_runs/ternary_runtime_profile.json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    prompt_lens = [int(x.strip()) for x in args.prompt_lens.split(",") if x.strip()]

    native_tmp, _cfg, _dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    vocab_size = int(native_tmp.vocab_size)
    del native_tmp
    torch.cuda.empty_cache()
    prompts = [make_prompt(vocab_size, n, args.seed) for n in prompt_lens]

    results = {}
    native = run_case(args, "native_bf16", None, {}, prompts, None)
    results["native_bf16"] = native
    native_logits = {}
    native_tokens = {}
    # Re-run native once with logits retained after the timing run has been recorded.
    model, dtype, _adapter = load_case(args, None, {})
    for prompt in prompts:
        _metrics, logits, generated = run_generation(model, prompt, args.new_tokens, dtype)
        native_logits[int(prompt.numel())] = logits
        native_tokens[int(prompt.numel())] = generated
    del model
    torch.cuda.empty_cache()

    profile_path = Path(args.profile_json)
    need_profile = args.preset in {"balanced", "speed", "all"} and not profile_path.exists()
    if need_profile:
        profile_env = {"TERNARY_SELECTIVE_DENSE_PROFILE": str(profile_path)}
        profile_result = run_case(args, "profile_seed_triton_gemv", "triton_gemv", profile_env, prompts, native_logits, native_tokens, profile=True)
        save_profile_json(
            str(profile_path),
            {
                "checkpoint": args.checkpoint,
                "adapter": args.adapter,
                "prompt_lens": prompt_lens,
                "new_tokens": args.new_tokens,
                "profile_seed": profile_result.get("profile_top20", []),
            },
        )

    all_cases = [
        ("low_vram", None, {"TERNARY_PRESET": "low_vram"}),
        ("balanced", None, {"TERNARY_PRESET": "balanced", "TERNARY_SELECTIVE_DENSE_PROFILE": str(profile_path)}),
        ("speed", None, {"TERNARY_PRESET": "speed", "TERNARY_SELECTIVE_DENSE_PROFILE": str(profile_path)}),
    ]
    if args.preset == "all":
        cases = all_cases
    else:
        cases = [case for case in all_cases if case[0] == args.preset]
    for label, runtime, env in cases:
        results[label] = run_case(args, label, runtime, env, prompts, native_logits, native_tokens, profile=(label == "triton_gemv"))

    result = {
        "checkpoint": args.checkpoint,
        "adapter": args.adapter,
        "prompt_lens": prompt_lens,
        "new_tokens": args.new_tokens,
        "preset": args.preset,
        "profile_json": str(profile_path),
        "quality_steps": args.quality_steps,
        "results": results,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
