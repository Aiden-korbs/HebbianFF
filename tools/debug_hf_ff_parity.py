#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

import chat_hf


def dtype_from_name(name: str, device: str) -> torch.dtype:
    name = name.lower()
    if name in ("auto", "bf16", "bfloat16"):
        return torch.bfloat16 if device == "cuda" else torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def metrics(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    d = (af - bf).abs()
    cos = F.cosine_similarity(af, bf, dim=0).item() if af.numel() else float("nan")
    return float(d.max().item()), float(d.mean().item()), float(cos)


def print_row(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    mx, mean, cos = metrics(a, b)
    print(f"{name:<24} max_abs={mx:>12.6g} mean_abs={mean:>12.6g} cos={cos:>10.7f}")


@torch.inference_mode()
def hf_trace(model, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    trace: Dict[str, torch.Tensor] = {}
    h = model.model.embed_tokens(input_ids)
    trace["embedding"] = h

    position_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    past_key_values = DynamicCache(config=model.config)
    mask_kwargs = {
        "config": model.config,
        "inputs_embeds": h,
        "attention_mask": None,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    causal_mask = create_causal_mask(**mask_kwargs)
    position_embeddings = model.model.rotary_emb(h, position_ids)

    for i, layer in enumerate(model.model.layers):
        residual = h
        n = layer.input_layernorm(h)
        attn_out, _ = layer.self_attn(
            hidden_states=n,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        h = residual + attn_out
        trace[f"layer_{i:02d}_after_attn"] = h

        residual = h
        m = layer.post_attention_layernorm(h)
        h = residual + layer.mlp(m)
        trace[f"layer_{i:02d}_after_mlp"] = h

    h_norm = model.model.norm(h)
    trace["final_norm"] = h_norm
    trace["final_logits"] = model.lm_head(h_norm)
    return trace


@torch.inference_mode()
def ff_trace(raw_model, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    trace: Dict[str, torch.Tensor] = {}
    h_emb = raw_model.tok_emb(input_ids)
    trace["embedding"] = h_emb
    h = raw_model.pre_ff_norm(h_emb) if raw_model.pre_ff_norm is not None else h_emb
    if raw_model.pre_ff_norm is not None:
        trace["after_pre_ff_norm"] = h

    for i, blk in enumerate(raw_model.ff_blocks):
        h = h + blk.attn(h, mem=None, pos_offset=0)
        trace[f"layer_{i:02d}_after_attn"] = h
        h = h + blk.mlp(h)
        trace[f"layer_{i:02d}_after_mlp"] = h

    if raw_model.post_ff_norm is not None:
        h = raw_model.post_ff_norm(h)
        trace["after_post_ff_norm"] = h

    for i, blk in enumerate(raw_model.bp_blocks):
        h = blk(h, mem=None, pos_offset=0)
        trace[f"bp_{i:02d}_after_mlp"] = h

    h_norm = raw_model.final_ln(h)
    trace["final_norm"] = h_norm
    h_proj = raw_model.final_proj(h_norm)
    trace["final_proj"] = h_proj
    trace["final_logits"] = raw_model.out_proj(h_proj) * raw_model.head_scale
    return trace


def parse_args():
    p = argparse.ArgumentParser(description="Layer-by-layer native HF vs FF_LLM parity diagnostic.")
    p.add_argument("--native-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p.add_argument("--ff-checkpoint", default="models/Qwen2.5-Coder-0.5B-Instruct.pt")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--prompt", default="def add(a, b):")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="bf16")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = args.device or auto_device()
    dtype = dtype_from_name(args.dtype, device)
    tokenizer_id = args.tokenizer or args.native_model

    tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    print(f"prompt={args.prompt!r} ids={ids[0].tolist()} device={device} dtype={dtype}")

    hf = AutoModelForCausalLM.from_pretrained(
        args.native_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    hf.eval()

    load_args = SimpleNamespace(
        checkpoint=args.ff_checkpoint,
        tokenizer=tokenizer_id,
        device=device,
        dtype=args.dtype,
        block_size=1024,
    )
    _model, raw_ff, _tok2, cfg, _device, _dtype = chat_hf.load_model(load_args)
    raw_ff.eval()

    print(
        "ff_cfg "
        f"qk_norm={cfg.use_qk_norm} pre_ff_norm={cfg.use_pre_ff_norm} "
        f"post_ff_norm={cfg.use_post_ff_norm} local_window={cfg.local_window} "
        f"head_scale={raw_ff.head_scale}"
    )

    hf_t = hf_trace(hf, ids)
    ff_t = ff_trace(raw_ff, ids)

    print(f"{'stage':<24} {'max_abs':>21} {'mean_abs':>21} {'cos':>14}")
    print("-" * 84)
    ordered: List[str] = ["embedding"]
    n_layers = min(len(hf.model.layers), len(raw_ff.ff_blocks))
    for i in range(n_layers):
        ordered.append(f"layer_{i:02d}_after_attn")
        ordered.append(f"layer_{i:02d}_after_mlp")
    ordered.extend(["final_norm", "final_logits"])

    for key in ordered:
        if key in hf_t and key in ff_t:
            print_row(key, hf_t[key], ff_t[key])

    if "after_pre_ff_norm" in ff_t:
        print("\nFF-only extra stages:")
        print_row("embedding_vs_pre_ff", hf_t["embedding"], ff_t["after_pre_ff_norm"])
    if "after_post_ff_norm" in ff_t:
        last_key = f"layer_{n_layers - 1:02d}_after_mlp"
        print_row("last_mlp_vs_post_ff", hf_t[last_key], ff_t["after_post_ff_norm"])
    if "final_proj" in ff_t:
        print_row("final_norm_vs_proj", hf_t["final_norm"], ff_t["final_proj"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
