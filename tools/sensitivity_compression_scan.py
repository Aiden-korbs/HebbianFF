#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from chat_hf import cfg_from_ckpt, dtype_from_name, no_init_weights
from ffbp_ema_cpu_ssm.bitnet import BitLinear
from ffbp_ema_cpu_ssm.config import CFG
from ffbp_ema_cpu_ssm.model import FF_LLM
from ffbp_ema_cpu_ssm.packed_1bit import Packed1BitLinear


def mib(n: float) -> float:
    return float(n) / 1048576.0


def split_fragments(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def mode_group_size(mode: str, default: int) -> int:
    match = re.fullmatch(r"[24]bit_g(\d+)", mode)
    if match:
        return int(match.group(1))
    return default


def is_four_bit_mode(mode: str) -> bool:
    return mode == "4bit_groupwise" or re.fullmatch(r"4bit_g\d+", mode) is not None


def is_two_bit_mode(mode: str) -> bool:
    return mode == "2bit_groupwise" or re.fullmatch(r"2bit_g\d+", mode) is not None


def is_ternary_mode(mode: str) -> bool:
    return (
        mode in {"ternary", "ternary_row", "ternary_old", "ternary_twn", "ternary_twn_row"}
        or re.fullmatch(r"ternary_g\d+", mode) is not None
        or re.fullmatch(r"ternary_twn_g\d+", mode) is not None
        or re.fullmatch(r"ternary_twn_g\d+_r\d+", mode) is not None
        or re.fullmatch(r"ternary_twn_row_r\d+", mode) is not None
        or re.fullmatch(r"ternary_awq_g\d+", mode) is not None
        or re.fullmatch(r"ternary_twn_g\d+_lora\d+", mode) is not None
        or re.fullmatch(r"ternary_twn_g\d+_outlier\d+", mode) is not None
    )


def is_double_ternary_mode(mode: str) -> bool:
    return re.fullmatch(r"ternary2_twn_g\d+", mode) is not None or mode == "ternary2_twn_row"


def get_child(root: nn.Module, name: str) -> nn.Module:
    child = root
    for part in name.split("."):
        child = child[int(part)] if part.isdigit() else getattr(child, part)
    return child


def set_child(root: nn.Module, name: str, child: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = child
    else:
        setattr(parent, last, child)


def block_index(name: str) -> Optional[tuple[str, int]]:
    match = re.search(r"\b(ff_blocks|bp_blocks)\.(\d+)\.", name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def protected_reason(name: str, module: nn.Linear, cfg: CFG, args) -> str:
    if name in {"out_proj", "final_proj"}:
        return "output_or_final_projection"
    if name.startswith(("tok_emb", "final_ln")):
        return "embedding_or_norm"
    if module.bias is not None and not args.include_bias:
        return "bias"
    if ".attn." in name and not args.include_attention:
        return "attention_default_full_precision"
    idx = block_index(name)
    if idx is not None and not args.include_edge_layers:
        family, i = idx
        n = int(cfg.ff_n_layer if family == "ff_blocks" else cfg.bp_n_layer)
        if i == 0:
            return "first_layer_default_full_precision"
        if i == n - 1:
            return "last_layer_default_full_precision"
    if module.weight.numel() < args.min_numel:
        return "small"
    if any(fragment in name for fragment in args.exclude):
        return "excluded"
    return ""


class DenseWeightLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight: torch.Tensor):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class LowRankCorrectedLinear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        base_weight: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ):
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("base_weight", base_weight.detach().contiguous(), persistent=False)
        self.register_buffer("left", left.detach().contiguous(), persistent=False)
        self.register_buffer("right", right.detach().contiguous(), persistent=False)
        if source.bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runtime shape this is modeling:
        #   base path: quantized/packed weight matmul
        #   residual: two small dense matmuls, x @ right.T @ left.T
        # The scan keeps base_weight dense to isolate quality impact. A real
        # packed runtime should store base_weight packed, not dense.
        y = F.linear(x, self.base_weight.to(dtype=x.dtype), self.bias)
        residual = F.linear(F.linear(x, self.right.to(dtype=x.dtype)), self.left.to(dtype=x.dtype))
        return y + residual


@torch.no_grad()
def one_bit_dense_weight(source: nn.Linear) -> torch.Tensor:
    w = source.weight.detach()
    scale = w.abs().mean(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.where(w >= 0, scale, -scale).to(dtype=w.dtype)


@torch.no_grad()
def ternary_scan_linear(source: nn.Linear, mode: str) -> DenseWeightLinear:
    old_quant_act = BitLinear._QUANT_ACT
    old_apply_scale = BitLinear._APPLY_SCALE
    old_scale_mode = BitLinear._SCALE_MODE
    old_group_size = BitLinear._GROUP_SIZE
    old_scale_selected = BitLinear._SCALE_SELECTED
    try:
        BitLinear._QUANT_ACT = False
        BitLinear._APPLY_SCALE = mode != "ternary_old"
        BitLinear._SCALE_SELECTED = mode.startswith("ternary_twn")
        if mode in {"ternary_row", "ternary_twn_row"}:
            BitLinear._SCALE_MODE = "row"
        elif mode.startswith("ternary_g") or mode.startswith("ternary_twn_g"):
            BitLinear._SCALE_MODE = "group"
            BitLinear._GROUP_SIZE = int(mode.rsplit("g", 1)[1])
        else:
            BitLinear._SCALE_MODE = "tensor"
        bit = BitLinear(source.in_features, source.out_features, bias=source.bias is not None).to(
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        bit.weight.copy_(source.weight)
        if source.bias is not None:
            bit.bias.copy_(source.bias)
        w_q, _ = bit._quantised_weight()
        return DenseWeightLinear(source, w_q)
    finally:
        BitLinear._QUANT_ACT = old_quant_act
        BitLinear._APPLY_SCALE = old_apply_scale
        BitLinear._SCALE_MODE = old_scale_mode
        BitLinear._GROUP_SIZE = old_group_size
        BitLinear._SCALE_SELECTED = old_scale_selected


def ternary_residual_parts(mode: str) -> Optional[tuple[str, int]]:
    group_match = re.fullmatch(r"(ternary_twn_g\d+)_r(\d+)", mode)
    if group_match:
        return group_match.group(1), int(group_match.group(2))
    row_match = re.fullmatch(r"(ternary_twn_row)_r(\d+)", mode)
    if row_match:
        return row_match.group(1), int(row_match.group(2))
    return None


def ternary_lora_parts(mode: str) -> Optional[tuple[str, int]]:
    match = re.fullmatch(r"(ternary_twn_g\d+)_lora(\d+)", mode)
    if match:
        return match.group(1), int(match.group(2))
    return None


def ternary_outlier_parts(mode: str) -> Optional[tuple[str, int]]:
    match = re.fullmatch(r"(ternary_twn_g\d+)_outlier(\d+)", mode)
    if match:
        return match.group(1), int(match.group(2))
    return None


@torch.no_grad()
def ternary_residual_scan_linear(source: nn.Linear, mode: str) -> DenseWeightLinear:
    base_mode, residual_percent = ternary_residual_parts(mode) or ("ternary_twn_g64", 0)
    base = ternary_scan_linear(source, base_mode).weight
    w = source.weight.detach()
    residual = (w - base).float()
    k = max(1, int(round(w.size(1) * float(residual_percent) / 100.0)))
    k = min(k, w.size(1))
    values, indices = torch.topk(residual.abs(), k=k, dim=1)
    signed_values = residual.gather(1, indices)
    corrected = base.float()
    corrected.scatter_add_(1, indices, signed_values)
    return DenseWeightLinear(source, corrected.to(dtype=w.dtype))


@torch.no_grad()
def ternary_lora_scan_linear(source: nn.Linear, mode: str) -> LowRankCorrectedLinear:
    base_mode, rank = ternary_lora_parts(mode) or ("ternary_twn_g64", 8)
    base = ternary_scan_linear(source, base_mode).weight
    w = source.weight.detach()
    residual = (w - base).float()
    out_features, in_features = residual.shape
    rank = max(1, min(rank, out_features, in_features))
    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    left = (u[:, :rank] * s[:rank].view(1, -1)).to(dtype=w.dtype)
    right = vh[:rank, :].to(dtype=w.dtype)
    return LowRankCorrectedLinear(source, base, left, right)


@torch.no_grad()
def ternary_outlier_scan_linear(source: nn.Linear, mode: str, x_sample: torch.Tensor) -> DenseWeightLinear:
    base_mode, percent = ternary_outlier_parts(mode) or ("ternary_twn_g64", 1)
    base = ternary_scan_linear(source, base_mode).weight.float()
    w = source.weight.detach().float()
    in_features = w.size(1)
    k = max(1, int(round(in_features * float(percent) / 100.0)))
    k = min(k, in_features)
    x = x_sample.detach().reshape(-1, in_features).float()
    score = x.abs().mean(dim=0) * w.abs().mean(dim=0)
    cols = torch.topk(score, k=k).indices
    corrected = base
    corrected[:, cols] = w[:, cols]
    return DenseWeightLinear(source, corrected.to(dtype=source.weight.dtype))


@torch.no_grad()
def ternary_awq_scan_linear(source: nn.Linear, mode: str, x_sample: torch.Tensor) -> DenseWeightLinear:
    w = source.weight.detach()
    out_features, in_features = w.shape
    group_size = int(mode.rsplit("g", 1)[1])
    groups = (in_features + group_size - 1) // group_size
    x = x_sample.detach().reshape(-1, in_features).float()
    w_out = torch.empty_like(w, dtype=torch.float32)
    threshold = float(BitLinear._THRESHOLD)

    for group_idx in range(groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        wg = w[:, start:end].float()
        xg = x[:, start:end]
        base_scale = wg.abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        keep = wg.abs() >= (threshold * base_scale)
        q = torch.where(keep, wg.sign(), torch.zeros_like(wg))
        basis = xg @ q.T
        target = xg @ wg.T
        numer = (basis * target).sum(dim=0)
        denom = basis.square().sum(dim=0).clamp_min(1e-8)
        scale = (numer / denom).view(-1, 1)
        w_out[:, start:end] = q * scale

    return DenseWeightLinear(source, w_out.to(dtype=w.dtype))


@torch.no_grad()
def double_ternary_scan_linear(source: nn.Linear, mode: str) -> DenseWeightLinear:
    first_mode = mode.replace("ternary2", "ternary", 1)
    first = ternary_scan_linear(source, first_mode).weight
    residual = (source.weight.detach() - first).to(dtype=source.weight.dtype)
    scratch = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None).to(
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    scratch.weight.copy_(residual)
    if source.bias is not None:
        scratch.bias.copy_(source.bias)
    second = ternary_scan_linear(scratch, first_mode).weight
    return DenseWeightLinear(source, (first + second).to(dtype=source.weight.dtype))


@torch.no_grad()
def two_bit_groupwise_scan_linear(source: nn.Linear, group_size: int) -> DenseWeightLinear:
    w = source.weight.detach()
    out_features, in_features = w.shape
    groups = (in_features + group_size - 1) // group_size
    pad = groups * group_size - in_features
    if pad:
        w_work = F.pad(w, (0, pad))
    else:
        w_work = w
    wg = w_work.view(out_features, groups, group_size)
    scale = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-6) / 1.5
    q = (wg / scale).round().clamp(-2, 1)
    w_q = (q * scale).view(out_features, groups * group_size)[:, :in_features].to(dtype=w.dtype)
    return DenseWeightLinear(source, w_q)


@torch.no_grad()
def four_bit_groupwise_scan_linear(source: nn.Linear, group_size: int) -> DenseWeightLinear:
    w = source.weight.detach()
    out_features, in_features = w.shape
    groups = (in_features + group_size - 1) // group_size
    pad = groups * group_size - in_features
    if pad:
        w_work = F.pad(w, (0, pad))
    else:
        w_work = w
    wg = w_work.view(out_features, groups, group_size)
    scale = wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-6) / 7.0
    q = (wg / scale).round().clamp(-8, 7)
    w_q = (q * scale).view(out_features, groups * group_size)[:, :in_features].to(dtype=w.dtype)
    return DenseWeightLinear(source, w_q)


@torch.no_grad()
def lowrank_residual_1bit_scan_linear(source: nn.Linear, rank: int, oversample: int, niter: int) -> LowRankCorrectedLinear:
    w = source.weight.detach()
    base = one_bit_dense_weight(source)
    residual = (w - base).float()
    out_features, in_features = residual.shape
    rank = max(1, min(rank, out_features, in_features))
    sketch_cols = max(rank, min(rank + oversample, out_features, in_features))
    omega = torch.randn(in_features, sketch_cols, device=w.device, dtype=torch.float32)
    y = residual @ omega
    for _ in range(max(0, niter)):
        y = residual @ (residual.T @ y)
    q, _ = torch.linalg.qr(y, mode="reduced")
    b = q.T @ residual
    u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_hat[:, :rank]
    left = (u * s[:rank].view(1, -1)).to(dtype=w.dtype)
    right = vh[:rank, :].to(dtype=w.dtype)
    return LowRankCorrectedLinear(source, base, left, right)


def saving_estimate_bytes(module: nn.Linear, mode: str, group_size: int) -> int:
    params = module.weight.numel()
    original = params * module.weight.element_size()
    out_features, in_features = module.weight.shape
    if mode == "packed_1bit":
        packed = out_features * ((in_features + 7) // 8)
        scales = out_features * 4
    elif mode.startswith("1bit_lr"):
        try:
            rank = int(mode.removeprefix("1bit_lr"))
        except ValueError:
            rank = 16
        packed = out_features * ((in_features + 7) // 8)
        scales = out_features * 4
        lowrank = (out_features + in_features) * rank * module.weight.element_size()
        packed += lowrank
    elif is_ternary_mode(mode):
        packed = (params * 2 + 7) // 8
        residual = ternary_residual_parts(mode)
        lora = ternary_lora_parts(mode)
        outlier = ternary_outlier_parts(mode)
        scale_mode = (residual or lora or outlier or (mode, 0))[0]
        if scale_mode in {"ternary_row", "ternary_twn_row"}:
            scales = out_features * 4
        elif (
            scale_mode.startswith("ternary_g")
            or scale_mode.startswith("ternary_twn_g")
            or scale_mode.startswith("ternary_awq_g")
        ):
            gs = int(scale_mode.rsplit("g", 1)[1])
            scales = out_features * ((in_features + gs - 1) // gs) * 4
        else:
            scales = 4
        if residual:
            _, residual_percent = residual
            k = max(1, int(round(in_features * float(residual_percent) / 100.0)))
            k = min(k, in_features)
            index_bytes = 2 if in_features <= 65535 else 4
            packed += out_features * k * (module.weight.element_size() + index_bytes)
        if lora:
            _, rank = lora
            packed += (out_features + in_features) * rank * module.weight.element_size()
        if outlier:
            _, percent = outlier
            k = max(1, int(round(in_features * float(percent) / 100.0)))
            k = min(k, in_features)
            index_bytes = 2 if in_features <= 65535 else 4
            packed += out_features * k * module.weight.element_size() + k * index_bytes
    elif is_double_ternary_mode(mode):
        packed = (params * 4 + 7) // 8
        if mode == "ternary2_twn_row":
            scales = out_features * 2 * 4
        else:
            gs = int(mode.rsplit("g", 1)[1])
            scales = out_features * ((in_features + gs - 1) // gs) * 2 * 4
    elif is_two_bit_mode(mode):
        packed = (params * 2 + 7) // 8
        gs = mode_group_size(mode, group_size)
        scales = out_features * ((in_features + gs - 1) // gs) * 4
    elif is_four_bit_mode(mode):
        packed = (params * 4 + 7) // 8
        gs = mode_group_size(mode, group_size)
        scales = out_features * ((in_features + gs - 1) // gs) * 4
    else:
        return 0
    bias = 0 if module.bias is None else module.bias.numel() * module.bias.element_size()
    return max(0, original + bias - packed - scales - bias)


def make_replacement(name: str, module: nn.Linear, mode: str, args) -> nn.Module:
    if mode == "packed_1bit":
        return Packed1BitLinear.from_linear(module).eval()
    if mode.startswith("1bit_lr"):
        try:
            rank = int(mode.removeprefix("1bit_lr"))
        except ValueError:
            rank = args.lowrank
        return lowrank_residual_1bit_scan_linear(module, rank, args.lowrank_oversample, args.lowrank_niter).eval()
    if is_ternary_mode(mode):
        if mode.startswith("ternary_awq_g"):
            x_sample = args.activation_cache.get(name)
            if x_sample is None:
                raise RuntimeError(f"missing activation sample for {name}")
            return ternary_awq_scan_linear(module, mode, x_sample).eval()
        if ternary_lora_parts(mode):
            return ternary_lora_scan_linear(module, mode).eval()
        if ternary_outlier_parts(mode):
            x_sample = args.activation_cache.get(name)
            if x_sample is None:
                raise RuntimeError(f"missing activation sample for {name}")
            return ternary_outlier_scan_linear(module, mode, x_sample).eval()
        if ternary_residual_parts(mode):
            return ternary_residual_scan_linear(module, mode).eval()
        return ternary_scan_linear(module, mode).eval()
    if is_double_ternary_mode(mode):
        return double_ternary_scan_linear(module, mode).eval()
    if is_two_bit_mode(mode):
        return two_bit_groupwise_scan_linear(module, mode_group_size(mode, args.group_size)).eval()
    if is_four_bit_mode(mode):
        return four_bit_groupwise_scan_linear(module, mode_group_size(mode, args.group_size)).eval()
    raise ValueError(mode)


def load_model(args):
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device != "cuda":
        raise RuntimeError("sensitivity scan currently requires CUDA")
    dtype = dtype_from_name(args.dtype, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(CFG(), ckpt, args.block_size)
    cfg.batch_size = 1
    cfg.use_ff_ema_bp = False
    cfg.memory_tokens = 0
    state = ckpt.get("model", ckpt)
    vocab_size = int(state["tok_emb.weight"].shape[0])
    with no_init_weights():
        model = FF_LLM(vocab_size, cfg).to(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[scan WARN] missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"[scan WARN] unexpected keys: {len(unexpected)}", flush=True)
    model.eval()
    return model, cfg, dtype


def make_input(model: FF_LLM, args) -> torch.Tensor:
    if args.prompt:
        from transformers import AutoTokenizer

        if not args.tokenizer:
            raise ValueError("--tokenizer is required when --prompt is used")
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        ids = tok(args.prompt, add_special_tokens=False).input_ids
        if not ids:
            raise ValueError("prompt produced no tokens")
        ids = ids[-args.tokens :]
        return torch.tensor([ids], dtype=torch.long, device="cuda")
    gen = torch.Generator(device="cuda")
    gen.manual_seed(args.seed)
    return torch.randint(0, model.vocab_size, (1, args.tokens), device="cuda", dtype=torch.long, generator=gen)


@torch.inference_mode()
def logits_for(model: FF_LLM, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    autocast_enabled = dtype in (torch.float16, torch.bfloat16)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        x_full, _ = model.forward_features(idx, update_state=False)
        return model._get_logits(x_full).detach()


@torch.inference_mode()
def time_logits(model: FF_LLM, idx: torch.Tensor, dtype: torch.dtype, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        _ = logits_for(model, idx, dtype)
    torch.cuda.synchronize()
    start = time.perf_counter()
    logits = None
    for _ in range(iters):
        logits = logits_for(model, idx, dtype)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return logits, iters / max(elapsed, 1e-9)


def topk_agreement(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    if k <= 0:
        return float("nan")
    k = min(k, a.size(-1))
    a_idx = torch.topk(a.float(), k, dim=-1).indices
    b_idx = torch.topk(b.float(), k, dim=-1).indices
    hits = (a_idx.unsqueeze(-1) == b_idx.unsqueeze(-2)).any(dim=-1).float().mean()
    return float(hits.item())


def recommendation(mean_rel_error: float, topk: float, speed_ratio: float, args) -> bool:
    return (
        mean_rel_error <= args.max_mean_rel_error
        and (not torch.isfinite(torch.tensor(topk)) or topk >= args.min_topk_agreement)
        and speed_ratio >= args.min_speed_ratio
    )


def classify_module(mode_results: list[dict], args) -> tuple[str, str]:
    by_mode = {row["mode"]: row for row in mode_results}
    one = by_mode.get("packed_1bit")
    if one and recommendation(one["mean_rel_logit_error"], one["topk_agreement"], one["speed_ratio"], args):
        return "safe_to_1bit", "packed_1bit"
    passing = [
        row
        for row in mode_results
        if row["mode"] != "packed_1bit"
        and recommendation(row["mean_rel_logit_error"], row["topk_agreement"], row["speed_ratio"], args)
    ]
    if passing:
        best = max(passing, key=lambda row: (row["vram_saving_mib"], row["topk_agreement"], -row["mean_rel_logit_error"]))
        return "safer_as_ternary_or_2bit", best["mode"]
    return "keep_full_precision", "full_precision"


def iter_mlp_block_prefixes(model: nn.Module, cfg: CFG, args) -> list[str]:
    prefixes = []
    seen = set()
    for name, module in model.named_modules():
        if type(module) is not nn.Linear or not name.endswith(".mlp.gate"):
            continue
        prefix = name.removesuffix(".gate")
        idx = block_index(name)
        if idx is None:
            continue
        family, i = idx
        n = int(cfg.ff_n_layer if family == "ff_blocks" else cfg.bp_n_layer)
        if i < args.block_keep_edge or i >= n - args.block_keep_edge:
            continue
        if prefix not in seen:
            seen.add(prefix)
            prefixes.append(prefix)
    return prefixes[: args.max_blocks] if args.max_blocks > 0 else prefixes


def scan_mlp_blocks(model, cfg, idx, dtype, baseline_logits, baseline_fps, baseline_abs_mean, args) -> list[dict]:
    combos = [
        ("gate_full_up_ternary_down_full", {"up": args.block_ternary_mode}),
        ("gate_ternary_up_ternary_down_full", {"gate": args.block_ternary_mode, "up": args.block_ternary_mode}),
        ("gate_ternary_up_ternary_down_4bit", {"gate": args.block_ternary_mode, "up": args.block_ternary_mode, "down": args.block_down_mode}),
        ("gate_lora_up_lora_down_full", {"gate": args.block_lora_mode, "up": args.block_lora_mode}),
    ]
    rows = []
    for prefix in iter_mlp_block_prefixes(model, cfg, args):
        names = {part: f"{prefix}.{part}" for part in ("gate", "up", "down")}
        originals = {part: get_child(model, name) for part, name in names.items()}
        print(f"[block_scan] {prefix}", flush=True)
        for combo_name, policy in combos:
            replacements = {}
            saving = 0.0
            try:
                for part, mode in policy.items():
                    module = originals[part]
                    repl = make_replacement(names[part], module, mode, args)
                    replacements[part] = repl
                    saving += mib(saving_estimate_bytes(module, mode, args.group_size))
                    set_child(model, names[part], repl)
                logits, fps = time_logits(model, idx, dtype, args.warmup, args.iters)
                diff = (baseline_logits.float() - logits.float()).abs()
                mean_err = float(diff.mean().item())
                max_err = float(diff.max().item())
                mean_rel = float((diff.mean() / baseline_abs_mean).item())
                agree = topk_agreement(baseline_logits, logits, args.topk)
                speed_ratio = fps / max(baseline_fps, 1e-9)
                row = {
                    "block": prefix,
                    "combo": combo_name,
                    "policy": ",".join(f"{k}:{v}" for k, v in policy.items()),
                    "vram_saving_mib": saving,
                    "mean_logit_error": mean_err,
                    "max_logit_error": max_err,
                    "mean_rel_logit_error": mean_rel,
                    "topk_agreement": agree,
                    "forward_per_sec": fps,
                    "speed_ratio": speed_ratio,
                }
                rows.append(row)
                print(
                    f"  {combo_name}: save={saving:.2f}MiB rel={mean_rel:.4g} "
                    f"max={max_err:.4g} top{args.topk}={agree:.4f} speed={speed_ratio:.3f}x",
                    flush=True,
                )
            finally:
                for part, original in originals.items():
                    set_child(model, names[part], original)
                del replacements
                gc.collect()
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min-numel", type=int, default=262144)
    ap.add_argument("--max-modules", type=int, default=0)
    ap.add_argument("--modes", default="packed_1bit,1bit_lr16,4bit_groupwise,2bit_groupwise,ternary")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--lowrank", type=int, default=16)
    ap.add_argument("--lowrank-oversample", type=int, default=8)
    ap.add_argument("--lowrank-niter", type=int, default=1)
    ap.add_argument("--include-attention", action="store_true")
    ap.add_argument("--include-edge-layers", action="store_true")
    ap.add_argument("--include-bias", action="store_true")
    ap.add_argument("--exclude", type=split_fragments, default=split_fragments("ff_draft_head,mem_compressor,engram,cpu_ctx"))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--max-mean-rel-error", type=float, default=0.02)
    ap.add_argument("--min-topk-agreement", type=float, default=0.95)
    ap.add_argument("--min-speed-ratio", type=float, default=0.70)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--print-limit", type=int, default=20)
    ap.add_argument("--scan-blocks", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--block-keep-edge", type=int, default=2)
    ap.add_argument("--block-ternary-mode", default="ternary_twn_g64")
    ap.add_argument("--block-lora-mode", default="ternary_twn_g64_lora8")
    ap.add_argument("--block-down-mode", default="4bit_g64")
    args = ap.parse_args()

    modes = tuple(x.strip() for x in args.modes.split(",") if x.strip())
    allowed = {
        "packed_1bit",
        "ternary",
        "ternary_row",
        "ternary_old",
        "ternary_twn",
        "ternary_twn_row",
        "2bit_groupwise",
        "4bit_groupwise",
    }
    bad = sorted(
        mode
        for mode in modes
        if mode not in allowed
        and not mode.startswith("1bit_lr")
        and not re.fullmatch(r"[24]bit_g\d+", mode)
        and not re.fullmatch(r"ternary_g\d+", mode)
        and not re.fullmatch(r"ternary_twn_g\d+", mode)
        and not re.fullmatch(r"ternary_twn_g\d+_r\d+", mode)
        and not re.fullmatch(r"ternary_twn_g\d+_lora\d+", mode)
        and not re.fullmatch(r"ternary_twn_g\d+_outlier\d+", mode)
        and not re.fullmatch(r"ternary_twn_row_r\d+", mode)
        and not re.fullmatch(r"ternary2_twn_g\d+", mode)
        and not re.fullmatch(r"ternary_awq_g\d+", mode)
    )
    if bad:
        raise ValueError(f"unknown modes: {bad}")

    model, cfg, dtype = load_model(args)
    idx = make_input(model, args)
    baseline_logits, baseline_fps = time_logits(model, idx, dtype, args.warmup, args.iters)
    baseline_abs_mean = baseline_logits.float().abs().mean().clamp_min(1e-6)
    print(
        f"[scan] baseline forward_per_sec={baseline_fps:.4f} tokens={idx.size(1)} "
        f"dtype={dtype} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    candidates: list[tuple[str, nn.Linear, str]] = []
    protected: list[tuple[str, str]] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or type(module) is not nn.Linear:
            continue
        reason = protected_reason(name, module, cfg, args)
        if reason:
            protected.append((name, reason))
            continue
        candidates.append((name, module, ""))
    if args.max_modules > 0:
        candidates = candidates[: args.max_modules]

    activation_cache: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module, _ in candidates:
        def _capture(mod, inputs, output, module_name=name):
            if module_name not in activation_cache:
                activation_cache[module_name] = inputs[0].detach()
        hooks.append(module.register_forward_hook(_capture))
    _ = logits_for(model, idx, dtype)
    for hook in hooks:
        hook.remove()
    args.activation_cache = activation_cache

    print(f"[scan] candidates={len(candidates)} protected={len(protected)} modes={','.join(modes)}", flush=True)

    rows: list[dict] = []
    plan: dict[str, list[str]] = {
        "safe_to_1bit": [],
        "safer_as_ternary_or_2bit": [],
        "keep_full_precision": [],
    }
    recommended_mode_by_module: dict[str, str] = {}

    for idx_mod, (name, module, _) in enumerate(candidates, start=1):
        original = get_child(model, name)
        module_rows: list[dict] = []
        shape = tuple(module.weight.shape)
        params = module.weight.numel()
        print(f"[scan] {idx_mod}/{len(candidates)} {name} shape={shape}", flush=True)
        for mode in modes:
            replacement = make_replacement(name, module, mode, args)
            set_child(model, name, replacement)
            torch.cuda.empty_cache()
            try:
                logits, fps = time_logits(model, idx, dtype, args.warmup, args.iters)
                diff = (baseline_logits.float() - logits.float()).abs()
                mean_err = float(diff.mean().item())
                max_err = float(diff.max().item())
                mean_rel = float((diff.mean() / baseline_abs_mean).item())
                agree = topk_agreement(baseline_logits, logits, args.topk)
                speed_ratio = fps / max(baseline_fps, 1e-9)
                row = {
                    "module": name,
                    "mode": mode,
                    "shape": "x".join(str(x) for x in shape),
                    "params": params,
                    "vram_saving_mib": mib(saving_estimate_bytes(module, mode, args.group_size)),
                    "mean_logit_error": mean_err,
                    "max_logit_error": max_err,
                    "mean_rel_logit_error": mean_rel,
                    "topk_agreement": agree,
                    "forward_per_sec": fps,
                    "speed_ratio": speed_ratio,
                }
                rows.append(row)
                module_rows.append(row)
                print(
                    f"  {mode}: save={row['vram_saving_mib']:.2f}MiB "
                    f"mean={mean_err:.4g} rel={mean_rel:.4g} max={max_err:.4g} "
                    f"top{args.topk}={agree:.4f} speed={speed_ratio:.3f}x",
                    flush=True,
                )
            finally:
                set_child(model, name, original)
                del replacement
                gc.collect()
                torch.cuda.empty_cache()
        bucket, selected_mode = classify_module(module_rows, args)
        plan[bucket].append(name)
        recommended_mode_by_module[name] = selected_mode

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"[scan] wrote CSV {args.csv}", flush=True)
    if args.json:
        payload = {
            "baseline_forward_per_sec": baseline_fps,
            "tokens": int(idx.size(1)),
            "criteria": {
                "max_mean_rel_error": args.max_mean_rel_error,
                "min_topk_agreement": args.min_topk_agreement,
                "min_speed_ratio": args.min_speed_ratio,
            },
            "protected": [{"module": name, "reason": reason} for name, reason in protected],
            "results": rows,
            "recommended_plan": plan,
            "recommended_mode_by_module": recommended_mode_by_module,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[scan] wrote JSON {args.json}", flush=True)

    print("[recommended_plan]")
    for key, names in plan.items():
        print(f"  {key}: {len(names)}")
        for name in names[: args.print_limit]:
            print(f"    {name} -> {recommended_mode_by_module.get(name, 'unknown')}")

    if args.scan_blocks:
        block_rows = scan_mlp_blocks(
            model,
            cfg,
            idx,
            dtype,
            baseline_logits,
            baseline_fps,
            baseline_abs_mean,
            args,
        )
        if block_rows:
            best = sorted(
                block_rows,
                key=lambda row: (
                    row["mean_rel_logit_error"] > args.max_mean_rel_error,
                    row["topk_agreement"] < args.min_topk_agreement,
                    -row["vram_saving_mib"],
                ),
            )[0]
            print(f"[block_scan_best] {json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
