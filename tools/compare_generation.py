#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat_hf import load_model, run_prompt


DEFAULT_PROMPTS = [
    "hi",
    "What is 2+2?",
    "Write a Python function that adds two numbers.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke compare dense and packed generation through scripts/inference/chat_hf.py.")
    p.add_argument("--dense-checkpoint", default=None)
    p.add_argument("--packed-checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompts", action="append", default=[])
    p.add_argument("--device", default=os.environ.get("DEVICE", None))
    p.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    p.add_argument("--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", "256")))
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--repeat-penalty", type=float, default=1.05)
    p.add_argument("--repeat-window", type=int, default=256)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--system", default=os.environ.get("SYSTEM_PROMPT", "You are a helpful coding assistant. Be concise and accurate."))
    p.add_argument("--no-stop-eos", action="store_true")
    return p.parse_args()


def chat_args(args: argparse.Namespace, checkpoint: str) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=checkpoint,
        tokenizer=args.tokenizer,
        prompt=None,
        system=args.system,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        kv_cache_max_len=int(os.environ.get("KV_CACHE_MAX_LEN", "0")),
        kv_cache_sink_tokens=int(os.environ.get("KV_CACHE_SINK_TOKENS", "0")),
        max_new=args.max_new,
        temp=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        repeat_window=args.repeat_window,
        no_stop_eos=args.no_stop_eos,
        extra_stop_tokens=[],
        no_chat_template=args.no_chat_template,
        suppress_im_start=True,
        hide_reasoning=False,
        show_prompt=False,
        stream=False,
        verbose_keys=False,
        raw=args.raw,
        history=False,
        max_turns=1,
    )


def run_checkpoint(label: str, checkpoint: str, args: argparse.Namespace, prompts: list[str]) -> dict[str, str]:
    print(f"\n===== {label}: {checkpoint} =====", flush=True)
    cargs = chat_args(args, checkpoint)
    os.environ["KV_CACHE_MAX_LEN"] = str(cargs.kv_cache_max_len)
    os.environ["KV_CACHE_SINK_TOKENS"] = str(cargs.kv_cache_sink_tokens)
    model, raw_model, tok, cfg, device, dtype = load_model(cargs)
    outputs: dict[str, str] = {}
    for prompt in prompts:
        print(f"\n--- prompt ---\n{prompt}\n--- output ---", flush=True)
        text = run_prompt(cargs, model, raw_model, tok, cfg, device, dtype, prompt, [])
        outputs[prompt] = text
        print(text, flush=True)
    del model, raw_model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> int:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    dense_outputs = None
    if args.dense_checkpoint:
        dense_outputs = run_checkpoint("dense", args.dense_checkpoint, args, prompts)
    packed_outputs = run_checkpoint("packed", args.packed_checkpoint, args, prompts)

    if dense_outputs is not None:
        print("\n===== side-by-side =====", flush=True)
        for prompt in prompts:
            print(f"\nPROMPT: {prompt}")
            print(f"DENSE : {dense_outputs[prompt]}")
            print(f"PACKED: {packed_outputs[prompt]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
