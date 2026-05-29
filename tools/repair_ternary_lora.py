#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from chat_hf import cfg_from_ckpt, dtype_from_name, no_init_weights
from ffbp_ema_cpu_ssm.config import CFG
from ffbp_ema_cpu_ssm.model import FF_LLM
from tools.sensitivity_compression_scan import (
    block_index,
    get_child,
    set_child,
    ternary_scan_linear,
    topk_agreement,
)


class TrainableTernaryLoRALinear(nn.Module):
    def __init__(self, source: nn.Linear, base_weight: torch.Tensor, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.rank = int(rank)
        self.alpha = nn.Parameter(torch.tensor(float(alpha), device=base_weight.device, dtype=torch.float32))
        self.register_buffer("base_weight", base_weight.detach().contiguous(), persistent=True)
        self.a = nn.Parameter(torch.zeros(source.out_features, rank, device=base_weight.device, dtype=source.weight.dtype))
        self.b = nn.Parameter(torch.zeros(rank, source.in_features, device=base_weight.device, dtype=source.weight.dtype))
        nn.init.normal_(self.b, std=1.0 / math.sqrt(source.in_features))
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone(), persistent=True)

    @torch.no_grad()
    def init_from_residual(self, target_weight: torch.Tensor) -> None:
        residual = (target_weight.detach() - self.base_weight).float()
        rank = min(self.rank, residual.size(0), residual.size(1))
        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        self.a.zero_()
        self.b.zero_()
        self.a[:, :rank].copy_((u[:, :rank] * s[:rank].view(1, -1)).to(dtype=self.a.dtype))
        self.b[:rank, :].copy_(vh[:rank, :].to(dtype=self.b.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight.to(dtype=x.dtype), self.bias)
        residual = F.linear(F.linear(x, self.b.to(dtype=x.dtype)), self.a.to(dtype=x.dtype))
        return base + residual * self.alpha.to(dtype=x.dtype)


def load_model(checkpoint: str, block_size: int, dtype_name: str):
    device = "cuda"
    dtype = dtype_from_name(dtype_name, device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(CFG(), ckpt, block_size)
    cfg.batch_size = 1
    cfg.use_ff_ema_bp = False
    cfg.memory_tokens = 0
    state = ckpt.get("model", ckpt)
    vocab_size = int(state["tok_emb.weight"].shape[0])
    with no_init_weights():
        model = FF_LLM(vocab_size, cfg).to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, dtype


def candidate_names(model: nn.Module, cfg: CFG, args) -> list[str]:
    parts = set(x.strip() for x in args.module_parts.split(",") if x.strip())
    out = []
    for name, module in model.named_modules():
        if type(module) is not nn.Linear:
            continue
        if not any(name.endswith(f".mlp.{part}") for part in parts):
            continue
        idx = block_index(name)
        if idx is None:
            continue
        family, i = idx
        n = int(cfg.ff_n_layer if family == "ff_blocks" else cfg.bp_n_layer)
        if i < args.keep_edge_blocks or i >= n - args.keep_edge_blocks:
            continue
        if module.weight.numel() < args.min_numel:
            continue
        out.append(name)
    return out[: args.max_modules] if args.max_modules > 0 else out


@torch.no_grad()
def make_batch(vocab_size: int, batch: int, tokens: int, seed: int, step: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed + step)
    return torch.randint(0, vocab_size, (batch, tokens), device="cuda", dtype=torch.long, generator=gen)


def load_calibration_batches(args, vocab_size: int) -> list[torch.Tensor]:
    if not args.calibration_jsonl:
        return [make_batch(vocab_size, args.batch, args.tokens, args.seed, i) for i in range(max(args.steps, 1))]
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    texts = []
    with open(args.calibration_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if args.max_records > 0 and len(texts) >= args.max_records:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    if not texts:
        raise ValueError(f"no calibration text found in {args.calibration_jsonl}")

    ids: list[torch.Tensor] = []
    for text in texts:
        toks = tok(text, add_special_tokens=False).input_ids
        if len(toks) < 2:
            continue
        if len(toks) >= args.tokens:
            toks = toks[: args.tokens]
        else:
            toks = toks + [tok.pad_token_id] * (args.tokens - len(toks))
        ids.append(torch.tensor(toks, dtype=torch.long))
    if not ids:
        raise ValueError("calibration records produced no token batches")

    batches = []
    for i in range(0, len(ids), args.batch):
        chunk = ids[i : i + args.batch]
        if len(chunk) < args.batch:
            break
        batches.append(torch.stack(chunk, dim=0).to(device="cuda"))
    if not batches:
        raise ValueError("not enough tokenized calibration records for one batch")
    return batches


@torch.no_grad()
def logits_for(model: FF_LLM, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
        x, _ = model.forward_features(idx, update_state=False)
        return model._get_logits(x).detach()


def train_logits_for(model: FF_LLM, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype in (torch.float16, torch.bfloat16)):
        x, _ = model.forward_features(idx, update_state=False)
        return model._get_logits(x)


def install_lora_modules(student: nn.Module, names: list[str], rank: int, base_mode: str, init_svd: bool) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for name in names:
        source = get_child(student, name)
        if not isinstance(source, nn.Linear):
            raise TypeError(name)
        base = ternary_scan_linear(source, base_mode).weight
        repl = TrainableTernaryLoRALinear(source, base, rank=rank)
        if init_svd:
            repl.init_from_residual(source.weight)
        set_child(student, name, repl)
        params.extend([repl.a, repl.b, repl.alpha])
    return params


def pack_ternary_codes(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = (codes.to(torch.int16) + 1).clamp_(0, 2).flatten().to(torch.uint8)
    pad = (-flat.numel()) % 4
    if pad:
        flat = F.pad(flat, (0, pad), value=1)
    packed = (
        flat[0::4]
        | (flat[1::4] << 2)
        | (flat[2::4] << 4)
        | (flat[3::4] << 6)
    )
    return packed.contiguous(), pad


def compact_ternary_base(base_weight: torch.Tensor, base_mode: str) -> dict:
    weight = base_weight.detach().cpu()
    codes = torch.sign(weight).to(torch.int8)
    m = re.search(r"_g(\d+)", base_mode)
    group_size = int(m.group(1)) if m else int(weight.shape[1])
    cols = int(weight.shape[1])
    pad_cols = (-cols) % group_size
    padded = F.pad(weight.float(), (0, pad_cols)) if pad_cols else weight.float()
    grouped = padded.view(weight.shape[0], -1, group_size)
    scales = grouped.abs().amax(dim=-1).to(torch.bfloat16)
    packed, code_pad = pack_ternary_codes(codes)
    return {
        "encoding": "ternary_2bit_signed_group_scale_v1",
        "packed_codes": packed,
        "scales": scales,
        "shape": tuple(weight.shape),
        "group_size": group_size,
        "pad_cols": pad_cols,
        "code_pad": code_pad,
    }


def collect_lora_state(student: nn.Module, names: list[str], base_mode: str) -> dict:
    out = {}
    for name in names:
        module = get_child(student, name)
        if not isinstance(module, TrainableTernaryLoRALinear):
            continue
        out[name] = {
            "base": compact_ternary_base(module.base_weight, base_mode),
            "a": module.a.detach().cpu(),
            "b": module.b.detach().cpu(),
            "alpha": module.alpha.detach().cpu(),
            "bias": None if module.bias is None else module.bias.detach().cpu(),
            "rank": module.rank,
            "in_features": module.in_features,
            "out_features": module.out_features,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--base-mode", default="ternary_twn_g64")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-modules", type=int, default=2)
    ap.add_argument("--min-numel", type=int, default=262144)
    ap.add_argument("--keep-edge-blocks", type=int, default=2)
    ap.add_argument("--module-parts", default="gate,up",
                    help="Comma-separated MLP linear parts to repair. Default: gate,up")
    ap.add_argument("--calibration-jsonl", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-records", type=int, default=128)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=None, help="Optional .pt path for repaired ternary LoRA adapter state.")
    ap.add_argument("--no-svd-init", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.calibration_jsonl and not args.tokenizer:
        raise ValueError("--tokenizer is required with --calibration-jsonl")

    teacher, cfg, dtype = load_model(args.checkpoint, args.block_size, args.dtype)
    student = copy.deepcopy(teacher).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in student.parameters():
        p.requires_grad_(False)

    names = candidate_names(student, cfg, args)
    params = install_lora_modules(student, names, args.rank, args.base_mode, init_svd=not args.no_svd_init)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    batches = load_calibration_batches(args, student.vocab_size)

    eval_idx = batches[0]
    teacher_eval = logits_for(teacher, eval_idx, dtype)
    before = logits_for(student, eval_idx, dtype)
    before_diff = (teacher_eval.float() - before.float()).abs()
    before_topk = topk_agreement(teacher_eval, before, args.topk)

    student.train()
    for step in range(args.steps):
        idx = batches[step % len(batches)]
        with torch.no_grad():
            teacher_logits = logits_for(teacher, idx, dtype)
        student_logits = train_logits_for(student, idx, dtype)
        temp = float(args.temperature)
        loss = F.kl_div(
            F.log_softmax(student_logits.float() / temp, dim=-1),
            F.softmax(teacher_logits.float() / temp, dim=-1),
            reduction="batchmean",
        ) * (temp * temp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) == args.steps:
            print(f"[repair] step={step+1}/{args.steps} loss={float(loss.detach()):.6f}", flush=True)

    student.eval()
    after = logits_for(student, eval_idx, dtype)
    after_diff = (teacher_eval.float() - after.float()).abs()
    after_topk = topk_agreement(teacher_eval, after, args.topk)

    result = {
        "modules": names,
        "calibration_jsonl": args.calibration_jsonl,
        "module_parts": args.module_parts,
        "rank": args.rank,
        "base_mode": args.base_mode,
        "before_mean_error": float(before_diff.mean().item()),
        "before_max_error": float(before_diff.max().item()),
        "before_topk": before_topk,
        "after_mean_error": float(after_diff.mean().item()),
        "after_max_error": float(after_diff.max().item()),
        "after_topk": after_topk,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "ternary_lora_repair_v1",
                "cfg": {
                    "rank": args.rank,
                    "base_mode": args.base_mode,
                    "module_parts": args.module_parts,
                    "keep_edge_blocks": args.keep_edge_blocks,
                    "block_size": args.block_size,
                    "tokens": args.tokens,
                },
                "metrics": result,
                "modules": collect_lora_state(student, names, args.base_mode),
            },
            out_path,
        )
        print(f"[repair] saved adapter state to {out_path}", flush=True)


if __name__ == "__main__":
    main()
