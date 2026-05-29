#!/usr/bin/env python3
"""Seed a base checkpoint with randomly-initialised optional feature modules.

The base DeepSeek-R1-Distill-Qwen-1.5B checkpoint has no weights for:
  - ff_draft_head  (FFDraftHead)
  - mem_compressor (ChunkMemoryCompressor)
  - engram_*       (EngramMemoryBank + projections)
  - cpu_ctx_*      (CPU hash context projection)

Loading with strict=False skips missing keys; the modules stay at their
default init. This script does that explicitly and saves the result so
training / eval can start from a known seed.

Usage:
  python seed_features.py \
    --checkpoint models/DeepSeek-R1-Distill-Qwen-1.5B-bs8192-bf16.pt \
    --features draft,memory,engram,cpuctx \
    --out models/seeded_draft_memory_engram_cpuctx.pt \
    --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import chat_hf
from types import SimpleNamespace


def apply_features_to_env(features: set[str]):
    os.environ["USE_DRAFT_HEAD"] = "1" if "draft" in features else "0"
    os.environ["DRAFT_BLEND_BP"] = "1" if "draft" in features else "0"
    os.environ["INFER_MEMORY_TOKENS"] = "64" if "memory" in features else "0"
    os.environ["INFER_USE_ENGRAM"] = "1" if "engram" in features else "0"
    os.environ["CPU_HASH_CTX"] = "1" if "cpuctx" in features else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--features", default="draft",
                    help="Comma-separated: draft,memory,engram,cpuctx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--block-size", type=int, default=2048)
    args = ap.parse_args()

    features = {f.strip() for f in args.features.split(",") if f.strip()}
    if not features:
        raise SystemExit("No features specified.")

    apply_features_to_env(features)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    load_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        prompt=None, system="", max_new=1, temp=0.0, top_k=0, top_p=1.0,
        repeat_penalty=1.0, no_stop_eos=True, raw=True,
        history=False, max_turns=0,
        verbose_keys=False,
    )

    model, raw_model, tok, cfg, device, dtype = chat_hf.load_model(load_args)

    # Report what got random-init'd (missing from ckpt, now seeded).
    print(f"[SEED] features={','.join(sorted(features))}  seed={args.seed}")
    if "draft" in features and raw_model.ff_draft_head is not None:
        n = sum(p.numel() for p in raw_model.ff_draft_head.parameters())
        print(f"[SEED] ff_draft_head: {n/1e6:.1f}M params  blend={float(raw_model.ff_draft_head.blend[0]):.4f}")
    if "memory" in features and raw_model.mem_compressor is not None:
        n = sum(p.numel() for p in raw_model.mem_compressor.parameters())
        print(f"[SEED] mem_compressor: {n/1e6:.1f}M params")
    if "engram" in features and raw_model.use_engram:
        engram_params = [p for n, p in raw_model.named_parameters() if n.startswith("engram_")]
        n = sum(p.numel() for p in engram_params)
        print(f"[SEED] engram: {n/1e6:.1f}M params  gate={raw_model.engram_gate_val:.4f}")
    if "cpuctx" in features and raw_model.cpu_ctx_proj is not None:
        n = sum(p.numel() for p in [raw_model.cpu_ctx_proj, raw_model.cpu_ctx_gate])
        print(f"[SEED] cpu_ctx: {n/1e6:.1f}M params  gate={float(torch.sigmoid(raw_model.cpu_ctx_gate[0])):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict, is_dataclass
    cfg_obj = asdict(raw_model.cfg) if is_dataclass(raw_model.cfg) else dict(vars(raw_model.cfg))

    payload = {
        "model": raw_model.state_dict(),
        "cfg": cfg_obj,
        "step": 0,
        "loss": float("nan"),
        "seed_features": sorted(features),
        "seed": args.seed,
    }
    torch.save(payload, out_path)
    print(f"[SEED] saved -> {out_path}")


if __name__ == "__main__":
    main()