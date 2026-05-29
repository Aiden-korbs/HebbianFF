#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from chat_hf import cfg_from_ckpt, no_init_weights
from HebbianFF.config import CFG
from HebbianFF.model import FF_LLM
from HebbianFF.packed import is_packed_entry, replace_packed_linears


def dtype_from_name(name: str, device: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    raise ValueError(name)


def load_dense_model(checkpoint: str, device: str, dtype: torch.dtype, block_size: int | None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(CFG(), ckpt, block_size)
    cfg.use_bitnet = False
    state = ckpt.get("model", ckpt)
    vocab = int(state["tok_emb.weight"].shape[0])
    with no_init_weights():
        model = FF_LLM(vocab, cfg).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, missing, unexpected


def load_packed_model(checkpoint: str, device: str, dtype: torch.dtype, block_size: int | None):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "ff_llm_packed_1bit_transfer_v1":
        raise ValueError("packed checkpoint must have format=ff_llm_packed_1bit_transfer_v1")
    cfg = cfg_from_ckpt(CFG(), ckpt, block_size)
    cfg.use_bitnet = False
    state: Dict[str, Any] = ckpt["model"]
    tok_emb = state.get("tok_emb.weight")
    if not torch.is_tensor(tok_emb):
        raise KeyError("packed checkpoint must keep tok_emb.weight dense")
    vocab = int(tok_emb.shape[0])

    with no_init_weights():
        model = FF_LLM(vocab, cfg)

    packed_module_names = {
        key[: -len(".weight")]
        for key, value in state.items()
        if key.endswith(".weight") and is_packed_entry(value)
    }
    consumed_biases = replace_packed_linears(model, state)

    dense_state = {
        key: value
        for key, value in state.items()
        if torch.is_tensor(value) and key not in consumed_biases
    }
    missing, unexpected = model.load_state_dict(dense_state, strict=False)
    expected_packed_missing = {
        f"{name}.{suffix}"
        for name in packed_module_names
        for suffix in ("packed", "scale", "bias")
    }
    missing = [key for key in missing if key not in expected_packed_missing]
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg, missing, unexpected, ckpt.get("manifest", {})


@torch.inference_mode()
def logits_for(model, ids: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    ids = ids.to(device)
    use_autocast = device == "cuda" and dtype in {torch.float16, torch.bfloat16}
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
        h = model.forward_features_eval(ids)
        return model._get_logits_base(h).float().cpu()


def compare(ref: torch.Tensor, out: torch.Tensor) -> Dict[str, Any]:
    diff = (ref - out).abs()
    ref_logp = F.log_softmax(ref, dim=-1)
    out_logp = F.log_softmax(out, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - out_logp)).sum(dim=-1)
    ref_top = ref.argmax(dim=-1)
    out_top = out.argmax(dim=-1)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "kl_ref_to_packed": float(kl.mean()),
        "top1_agreement": float((ref_top == out_top).float().mean()),
        "ref_last_top5": torch.topk(ref[0, -1], 5).indices.tolist(),
        "packed_last_top5": torch.topk(out[0, -1], 5).indices.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a packed 1-bit transfer checkpoint against a dense FF_LLM checkpoint.")
    p.add_argument("--dense-checkpoint", required=True)
    p.add_argument("--packed-checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="The answer to 2 + 2 is")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--skip-dense", action="store_true", help="Only load/run packed model.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("NO_INIT_LOAD", "1")
    os.environ.setdefault("USE_DRAFT_HEAD", "0")
    os.environ.setdefault("DRAFT_BLEND_BP", "0")
    os.environ.setdefault("INFER_MEMORY_TOKENS", "0")
    os.environ.setdefault("INFER_USE_ENGRAM", "0")

    device = args.device
    dtype = dtype_from_name(args.dtype, device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids[:, : int(args.tokens)]
    print(f"[eval] token_ids={ids.tolist()}", flush=True)

    ref = None
    if not args.skip_dense:
        print("[eval] loading dense reference", flush=True)
        t0 = time.perf_counter()
        dense, _, missing, unexpected = load_dense_model(args.dense_checkpoint, device, dtype, args.block_size)
        print(f"[eval] dense loaded in {time.perf_counter() - t0:.2f}s missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        t1 = time.perf_counter()
        ref = logits_for(dense, ids, device, dtype)
        print(f"[eval] dense forward {time.perf_counter() - t1:.2f}s", flush=True)
        del dense
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print("[eval] loading packed model", flush=True)
    t2 = time.perf_counter()
    packed, _, missing, unexpected, manifest = load_packed_model(args.packed_checkpoint, device, dtype, args.block_size)
    print(
        f"[eval] packed loaded in {time.perf_counter() - t2:.2f}s "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"compression={manifest.get('compression_ratio_tensor_bytes')}",
        flush=True,
    )
    t3 = time.perf_counter()
    out = logits_for(packed, ids, device, dtype)
    print(f"[eval] packed forward {time.perf_counter() - t3:.2f}s", flush=True)
    if device == "cuda":
        print(f"[eval] cuda_peak_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f}", flush=True)

    if ref is not None:
        print("RESULT", compare(ref, out), flush=True)
    else:
        print("RESULT packed_last_top5", torch.topk(out[0, -1], 5).indices.tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
