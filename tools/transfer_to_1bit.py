#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_HOLDOUT_SUFFIXES = (
    "tok_emb.weight",
    "out_proj.weight",
    "final_proj.weight",
)

DEFAULT_HOLDOUT_CONTAINS = (
    "ln.weight",
    "norm.weight",
    "c_norm.weight",
    "final_ln.weight",
)

ATTENTION_CONTAINS = (
    ".attn.",
    ".attention.",
    ".self_attn.",
)

ATTENTION_PROJ_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".c_proj.weight",
)

MLP_CONTAINS = (
    ".mlp.",
    ".ffn.",
    ".feed_forward.",
    ".feedforward.",
)

MLP_PROJ_SUFFIXES = (
    ".gate.weight",
    ".up.weight",
    ".down.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".fc1.weight",
    ".fc2.weight",
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
)


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def is_quantizable_weight(name: str, t: torch.Tensor, quantize_head: bool = False) -> bool:
    if not torch.is_tensor(t) or t.ndim != 2 or not name.endswith(".weight"):
        return False
    if (not quantize_head) and any(name.endswith(s) for s in DEFAULT_HOLDOUT_SUFFIXES):
        return False
    if any(part in name for part in DEFAULT_HOLDOUT_CONTAINS):
        return False
    return t.is_floating_point()


def compile_regexes(patterns: Iterable[str] | None) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in (patterns or [])]


def matches_any_regex(name: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.search(name) for p in patterns)


def matches_any_contains(name: str, needles: Iterable[str] | None) -> bool:
    return any(s in name for s in (needles or []))


def is_attention_weight(name: str) -> bool:
    return matches_any_contains(name, ATTENTION_CONTAINS) or any(name.endswith(s) for s in ATTENTION_PROJ_SUFFIXES)


def is_mlp_weight(name: str) -> bool:
    return matches_any_contains(name, MLP_CONTAINS) or any(name.endswith(s) for s in MLP_PROJ_SUFFIXES)


def is_vo_weight(name: str) -> bool:
    return name.endswith(".v_proj.weight") or name.endswith(".o_proj.weight") or name.endswith(".c_proj.weight")


def selection_decision(
    name: str,
    value: Any,
    args: argparse.Namespace,
    only_regex: list[re.Pattern[str]],
    skip_regex: list[re.Pattern[str]],
) -> tuple[bool, str]:
    if not torch.is_tensor(value):
        return False, "not a tensor"
    if value.ndim != 2 or not name.endswith(".weight"):
        return False, "not a 2D .weight tensor"
    if not value.is_floating_point():
        return False, f"non-floating dtype {value.dtype}"
    if (not args.quantize_head) and any(name.endswith(s) for s in DEFAULT_HOLDOUT_SUFFIXES):
        return False, "default head/embedding holdout"
    if any(part in name for part in DEFAULT_HOLDOUT_CONTAINS):
        return False, "default norm holdout"

    preset = args.preset
    if preset == "mlp-only":
        if is_attention_weight(name):
            return False, "preset mlp-only keeps attention dense"
        if not is_mlp_weight(name):
            return False, "preset mlp-only only quantizes MLP/feed-forward weights"
    elif preset == "mlp-plus-vo":
        if name.endswith(".q_proj.weight") or name.endswith(".k_proj.weight"):
            return False, "preset mlp-plus-vo keeps q_proj/k_proj dense"
        if is_attention_weight(name) and not is_vo_weight(name):
            return False, "preset mlp-plus-vo keeps this attention weight dense"
        if not (is_mlp_weight(name) or is_vo_weight(name)):
            return False, "preset mlp-plus-vo only quantizes MLP plus v_proj/o_proj"
    elif preset == "attention-experiment":
        pass
    elif preset == "all-current":
        pass
    else:
        raise ValueError(f"unknown preset: {preset}")

    if args.only_contains and not matches_any_contains(name, args.only_contains):
        return False, "does not match --only-contains"
    if args.skip_contains and matches_any_contains(name, args.skip_contains):
        return False, "matched --skip-contains"
    if only_regex and not matches_any_regex(name, only_regex):
        return False, "does not match --only-regex"
    if skip_regex and matches_any_regex(name, skip_regex):
        return False, "matched --skip-regex"

    return True, f"selected by preset {preset}"


def pack_bits(mask: torch.Tensor) -> torch.Tensor:
    flat = mask.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.view(-1, 8)
    shifts = torch.arange(8, dtype=torch.uint8)
    return (flat << shifts).sum(dim=1).to(torch.uint8).contiguous()


def pack_2bit(vals: torch.Tensor) -> torch.Tensor:
    flat = vals.reshape(-1).to(torch.uint8)
    pad = (-flat.numel()) % 4
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.uint8)])
    flat = flat.view(-1, 4)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)
    return (flat << shifts).sum(dim=1).to(torch.uint8).contiguous()


def unpack_bits(packed: torch.Tensor, n: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed.reshape(-1, 1) >> shifts) & 1).reshape(-1)
    return bits[:n].to(torch.bool)


def unpack_2bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    vals = ((packed.reshape(-1, 1) >> shifts) & 3).reshape(-1)
    return vals[:n].to(torch.uint8)


def group_view(w: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    flat = w.reshape(-1).float()
    if group_size <= 0:
        group_size = flat.numel()
    pad = (-flat.numel()) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    return flat.view(-1, group_size), pad


def quantize_binary(w: torch.Tensor, group_size: int) -> Dict[str, torch.Tensor | int | list[int] | str]:
    groups, pad = group_view(w, group_size)
    sign = groups >= 0
    q = sign.float().mul(2.0).sub(1.0)
    scale = (groups * q).mean(dim=1).clamp_min(1e-8).to(torch.float16).cpu()
    return {
        "format": "binary_sign_scale",
        "shape": list(w.shape),
        "numel": int(w.numel()),
        "group_size": int(group_size if group_size > 0 else w.numel()),
        "pad": int(pad),
        "scale": scale,
        "packed": pack_bits(sign.cpu()),
    }


def quantize_ternary(
    w: torch.Tensor,
    group_size: int,
    threshold: float,
) -> Dict[str, torch.Tensor | int | float | list[int] | str]:
    groups, pad = group_view(w, group_size)
    abs_g = groups.abs()
    # Threshold is relative to each group's mean absolute weight. The default
    # 0.7 is a practical BitNet-style starting point; lower values keep more
    # non-zero weights, higher values compress signal harder.
    mean_abs = abs_g.mean(dim=1, keepdim=True).clamp_min(1e-8)
    keep = abs_g >= (float(threshold) * mean_abs)
    sign_pos = groups >= 0
    q = torch.where(keep, torch.where(sign_pos, torch.ones_like(groups), -torch.ones_like(groups)), torch.zeros_like(groups))
    denom = q.pow(2).sum(dim=1).clamp_min(1.0)
    scale = ((groups * q).sum(dim=1) / denom).clamp_min(1e-8).to(torch.float16).cpu()
    # Store symbols as 0 => zero, 1 => negative, 2 => positive. Symbol 3 unused.
    vals = torch.zeros_like(groups, dtype=torch.uint8)
    vals[keep & ~sign_pos] = 1
    vals[keep & sign_pos] = 2
    return {
        "format": "ternary_2bit_scale",
        "shape": list(w.shape),
        "numel": int(w.numel()),
        "group_size": int(group_size if group_size > 0 else w.numel()),
        "pad": int(pad),
        "threshold": float(threshold),
        "scale": scale,
        "packed": pack_2bit(vals.cpu()),
    }


def dequantize_binary(entry: Dict) -> torch.Tensor:
    shape = tuple(int(x) for x in entry["shape"])
    n = int(entry["numel"])
    group_size = int(entry["group_size"])
    bits = unpack_bits(entry["packed"], n + int(entry.get("pad", 0))).float()
    q = bits.mul(2.0).sub(1.0).view(-1, group_size)
    scale = entry["scale"].float().view(-1, 1)
    return (q * scale).reshape(-1)[:n].view(shape)


def dequantize_ternary(entry: Dict) -> torch.Tensor:
    shape = tuple(int(x) for x in entry["shape"])
    n = int(entry["numel"])
    group_size = int(entry["group_size"])
    vals = unpack_2bit(entry["packed"], n + int(entry.get("pad", 0))).view(-1, group_size)
    q = torch.zeros(vals.shape, dtype=torch.float32)
    q[vals == 1] = -1.0
    q[vals == 2] = 1.0
    scale = entry["scale"].float().view(-1, 1)
    return (q * scale).reshape(-1)[:n].view(shape)


def dequantize_entry(entry: Dict) -> torch.Tensor:
    fmt = entry.get("format")
    if fmt == "binary_sign_scale":
        return dequantize_binary(entry)
    if fmt == "ternary_2bit_scale":
        return dequantize_ternary(entry)
    raise ValueError(f"unknown quantized format: {fmt}")


def mse_for_entry(original: torch.Tensor, entry: Dict) -> Tuple[float, float]:
    recon = dequantize_entry(entry).to(dtype=torch.float32)
    ref = original.float().cpu()
    diff = recon - ref
    mse = float(diff.pow(2).mean())
    rel = float(diff.norm() / ref.norm().clamp_min(1e-8))
    return mse, rel


def symbol_stats(entry: Dict) -> tuple[float, float, float]:
    n = int(entry["numel"])
    if entry.get("format") == "ternary_2bit_scale":
        vals = unpack_2bit(entry["packed"], n + int(entry.get("pad", 0)))[:n]
        if n == 0:
            return 0.0, 0.0, 0.0
        zero = float((vals == 0).float().mean())
        neg = float((vals == 1).float().mean())
        pos = float((vals == 2).float().mean())
        return zero, pos, neg
    if entry.get("format") == "binary_sign_scale":
        bits = unpack_bits(entry["packed"], n + int(entry.get("pad", 0)))[:n]
        if n == 0:
            return 0.0, 0.0, 0.0
        pos = float(bits.float().mean())
        return 0.0, pos, 1.0 - pos
    return 0.0, 0.0, 0.0


def tensor_report_base(name: str, value: torch.Tensor, quantized: bool, reason: str) -> Dict[str, Any]:
    source_bytes = tensor_bytes(value)
    return {
        "name": name,
        "shape": list(value.shape),
        "source_dtype": str(value.dtype).replace("torch.", ""),
        "quantized": bool(quantized),
        "reason": reason,
        "source_bytes": source_bytes,
        "packed_bytes": source_bytes,
        "compression_ratio": 1.0,
        "mse": None,
        "relative_l2_error": None,
        "ternary_zero_fraction": None,
        "positive_fraction": None,
        "negative_fraction": None,
        "scale_mean": None,
        "scale_min": None,
        "scale_max": None,
        "group_size": None,
        "threshold": None,
    }


def fill_quantized_report(report: Dict[str, Any], entry: Dict, mse: float, rel: float) -> None:
    scale = entry["scale"].float()
    packed_bytes = tensor_bytes(entry["packed"]) + tensor_bytes(entry["scale"])
    zero, pos, neg = symbol_stats(entry)
    report.update(
        {
            "packed_bytes": packed_bytes,
            "compression_ratio": float(report["source_bytes"] / max(1, packed_bytes)),
            "mse": float(mse),
            "relative_l2_error": float(rel),
            "ternary_zero_fraction": zero,
            "positive_fraction": pos,
            "negative_fraction": neg,
            "scale_mean": float(scale.mean()) if scale.numel() else 0.0,
            "scale_min": float(scale.min()) if scale.numel() else 0.0,
            "scale_max": float(scale.max()) if scale.numel() else 0.0,
            "group_size": int(entry["group_size"]),
            "threshold": entry.get("threshold", None),
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer an FF_LLM checkpoint into packed 1-bit/ternary weight format.")
    p.add_argument("--checkpoint", required=True, help="Source .pt checkpoint.")
    p.add_argument("--out", required=True, help="Output checkpoint.")
    p.add_argument("--mode", choices=["binary", "ternary"], default="ternary")
    p.add_argument(
        "--preset",
        choices=["all-current", "mlp-only", "mlp-plus-vo", "attention-experiment"],
        default="mlp-only",
        help="Tensor selection preset. Default keeps attention dense.",
    )
    p.add_argument("--group-size", type=int, default=256, help="Scale group size. 0 means one scale per tensor.")
    p.add_argument("--threshold", type=float, default=0.7, help="Ternary threshold relative to group mean abs.")
    p.add_argument("--quantize-head", action="store_true", help="Also quantize tok_emb/out_proj if they are 2D float tensors.")
    p.add_argument("--only-contains", nargs="*", default=[], help="Only quantize tensors whose name contains one of these strings.")
    p.add_argument("--skip-contains", nargs="*", default=[], help="Hold out tensors whose name contains one of these strings.")
    p.add_argument("--only-regex", nargs="*", default=[], help="Only quantize tensors whose name matches one of these regexes.")
    p.add_argument("--skip-regex", nargs="*", default=[], help="Hold out tensors whose name matches one of these regexes.")
    p.add_argument("--list-tensors", action="store_true", help="List tensor names/shapes/dtypes and exit.")
    p.add_argument("--list-quantizable", action="store_true", help="List tensors selected for quantization under current filters and exit.")
    p.add_argument("--max-error-tensors", type=int, default=16, help="Number of largest relative-error tensors to print.")
    p.add_argument("--max-relative-l2", type=float, default=None, help="Warn/fail/hold out tensors above this relative L2 error.")
    p.add_argument("--fail-on-error", action="store_true", help="Abort if --max-relative-l2 is exceeded.")
    p.add_argument("--auto-holdout-bad-tensors", action="store_true", help="Keep tensors dense when they exceed --max-relative-l2.")
    p.add_argument("--dequantize", action="store_true", help="Read a packed transfer checkpoint and write a normal dense checkpoint.")
    p.add_argument("--inspect", action="store_true", help="Print manifest for a packed transfer checkpoint and exit.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.checkpoint)
    out = Path(args.out)
    t0 = time.time()
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if args.inspect:
        manifest = ckpt.get("manifest", {})
        print(json.dumps(manifest, indent=2)[:12000])
        return 0

    if args.dequantize:
        if ckpt.get("format") != "ff_llm_packed_1bit_transfer_v1":
            raise ValueError("--dequantize expects format=ff_llm_packed_1bit_transfer_v1")
        dense_state = {}
        for name, value in ckpt["model"].items():
            if isinstance(value, dict) and value.get("format") in {"binary_sign_scale", "ternary_2bit_scale"}:
                dense_state[name] = dequantize_entry(value).to(torch.bfloat16)
            else:
                dense_state[name] = value
        out_ckpt = {
            "model": dense_state,
            "cfg": ckpt.get("cfg", {}),
            "source": ckpt.get("source", None),
            "dequantized_from": str(src),
            "transfer_manifest": ckpt.get("manifest", {}),
        }
        if args.dry_run:
            print(f"[dequantize] --dry-run: would write dense checkpoint to {out}")
            return 0
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_ckpt, out)
        print(f"[dequantize] wrote {out} ({out.stat().st_size / 1024**3:.3f} GiB) in {time.time() - t0:.1f}s")
        return 0

    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dict")

    only_regex = compile_regexes(args.only_regex)
    skip_regex = compile_regexes(args.skip_regex)

    if args.list_tensors:
        for name, value in state.items():
            if torch.is_tensor(value):
                print(f"{name}\tshape={list(value.shape)}\tdtype={value.dtype}\tbytes={tensor_bytes(value)}")
        return 0

    if args.preset == "attention-experiment":
        print(
            "[transfer WARNING] --preset attention-experiment may quantize attention projections "
            "(q/k/v/o). This is expected to be destructive; use only for controlled tests.",
            file=sys.stderr,
            flush=True,
        )

    out_state: Dict[str, object] = {}
    manifest = {
        "source_checkpoint": str(src),
        "mode": args.mode,
        "preset": args.preset,
        "group_size": int(args.group_size),
        "threshold": float(args.threshold),
        "quantize_head": bool(args.quantize_head),
        "only_contains": list(args.only_contains),
        "skip_contains": list(args.skip_contains),
        "only_regex": list(args.only_regex),
        "skip_regex": list(args.skip_regex),
        "created_by": "tools/transfer_to_1bit.py",
        "created_at_unix": time.time(),
        "quantized_tensors": 0,
        "holdout_tensors": 0,
        "source_tensor_bytes": 0,
        "packed_tensor_bytes": 0,
        "quality_gate": {
            "max_relative_l2": args.max_relative_l2,
            "fail_on_error": bool(args.fail_on_error),
            "auto_holdout_bad_tensors": bool(args.auto_holdout_bad_tensors),
        },
        "tensors": [],
        "estimated_runtime_note": (
            "Packed linear entries stay compressed in VRAM and are replaced before CUDA materialization."
        ),
    }
    errors = []
    bad_tensors = []

    if args.list_quantizable:
        for name, value in state.items():
            if not torch.is_tensor(value):
                continue
            selected, reason = selection_decision(name, value, args, only_regex, skip_regex)
            if selected:
                print(f"{name}\tshape={list(value.shape)}\tdtype={value.dtype}\treason={reason}")
        return 0

    for name, value in state.items():
        if not torch.is_tensor(value):
            out_state[name] = value
            continue
        manifest["source_tensor_bytes"] += tensor_bytes(value)
        selected, reason = selection_decision(name, value, args, only_regex, skip_regex)
        report = tensor_report_base(name, value, selected, reason)
        if selected:
            if args.mode == "binary":
                entry = quantize_binary(value, args.group_size)
            else:
                entry = quantize_ternary(value, args.group_size, args.threshold)
            mse, rel = mse_for_entry(value, entry)
            entry["mse"] = float(mse)
            entry["relative_l2_error"] = float(rel)
            fill_quantized_report(report, entry, mse, rel)

            over_gate = args.max_relative_l2 is not None and rel > float(args.max_relative_l2)
            if over_gate:
                msg = (
                    f"[transfer WARN] {name} relative_l2_error={rel:.6f} "
                    f"exceeds --max-relative-l2={float(args.max_relative_l2):.6f}"
                )
                print(msg, file=sys.stderr, flush=True)
                bad_tensors.append(report.copy())
                if args.fail_on_error:
                    raise RuntimeError(msg)
                if args.auto_holdout_bad_tensors:
                    kept = value.detach().cpu()
                    out_state[name] = kept
                    report["quantized"] = False
                    report["reason"] = f"auto holdout: relative_l2_error {rel:.6f} exceeded gate"
                    report["packed_bytes"] = tensor_bytes(kept)
                    report["compression_ratio"] = 1.0
                    manifest["packed_tensor_bytes"] += tensor_bytes(kept)
                    manifest["holdout_tensors"] += 1
                    manifest["tensors"].append(report)
                    continue

            packed_bytes = tensor_bytes(entry["packed"]) + tensor_bytes(entry["scale"])
            manifest["packed_tensor_bytes"] += packed_bytes
            manifest["quantized_tensors"] += 1
            out_state[name] = entry
            errors.append((rel, mse, name, list(value.shape), packed_bytes, tensor_bytes(value)))
        else:
            kept = value.detach().cpu()
            out_state[name] = kept
            manifest["packed_tensor_bytes"] += tensor_bytes(kept)
            manifest["holdout_tensors"] += 1
        manifest["tensors"].append(report)

    errors.sort(reverse=True, key=lambda x: x[0])
    manifest["largest_relative_l2_errors"] = [
        {
            "name": name,
            "shape": shape,
            "relative_l2_error": rel,
            "mse": mse,
            "packed_bytes": packed_bytes,
            "source_bytes": source_bytes,
        }
        for rel, mse, name, shape, packed_bytes, source_bytes in errors[: max(0, int(args.max_error_tensors))]
    ]
    manifest["quality_gate_failures"] = bad_tensors
    src_bytes = int(manifest["source_tensor_bytes"])
    dst_bytes = int(manifest["packed_tensor_bytes"])
    manifest["compression_ratio_tensor_bytes"] = float(src_bytes / max(1, dst_bytes))

    out_ckpt = {
        "format": "ff_llm_packed_1bit_transfer_v1",
        "model": out_state,
        "cfg": ckpt.get("cfg", {}),
        "source": ckpt.get("source", None),
        "manifest": manifest,
    }

    print(json.dumps(manifest, indent=2)[:6000])
    print(f"[transfer] source tensor bytes: {src_bytes / 1024**3:.3f} GiB")
    print(f"[transfer] packed tensor bytes: {dst_bytes / 1024**3:.3f} GiB")
    print(f"[transfer] tensor compression ratio: {manifest['compression_ratio_tensor_bytes']:.2f}x")
    if args.dry_run:
        print("[transfer] dry-run tensor decisions:")
        for item in manifest["tensors"]:
            action = "QUANTIZE" if item["quantized"] else "HOLDOUT "
            rel = item["relative_l2_error"]
            rel_s = "n/a" if rel is None else f"{rel:.6f}"
            print(
                f"  {action} {item['name']} shape={item['shape']} dtype={item['source_dtype']} "
                f"reason={item['reason']} rel_l2={rel_s} ratio={item['compression_ratio']:.2f}x"
            )
        print("[transfer] --dry-run: not writing output")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out)
    print(f"[transfer] wrote {out} ({out.stat().st_size / 1024**3:.3f} GiB) in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
