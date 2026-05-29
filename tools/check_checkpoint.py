#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import torch

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python tools/check_checkpoint.py models/model.pt")
    p = Path(sys.argv[1]).expanduser()
    if not p.exists():
        raise SystemExit(f"Checkpoint not found: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    required = ["tok_emb.weight", "out_proj.weight", "final_ln.weight", "final_proj.weight"]
    missing = [k for k in required if k not in state]
    if missing:
        raise SystemExit(f"[BAD] Missing required keys: {missing}")
    vocab, d = state["tok_emb.weight"].shape
    print(f"[OK] {p}")
    print(f"vocab_size={vocab} n_embd={d}")
    cfg = ckpt.get("cfg", {})
    if isinstance(cfg, dict):
        for k in ["n_embd","n_head","n_kv_head","ff_n_layer","bp_n_layer","block_size","use_draft_head","memory_tokens","use_engram"]:
            if k in cfg: print(f"{k}={cfg[k]}")
    print("has_draft_head_weights=", any(k.startswith("ff_draft_head.") for k in state))

if __name__ == "__main__":
    main()
