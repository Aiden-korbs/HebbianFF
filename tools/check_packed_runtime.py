#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ffbp_ema_cpu_ssm.packed import is_packed_entry
from tools.ternary_linear import TernaryLinear


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare packed ternary Triton runtime against PyTorch fallback.")
    p.add_argument("--checkpoint", required=True, help="Packed transfer checkpoint.")
    p.add_argument("--max-tensors", type=int, default=8, help="Number of packed tensors to test.")
    p.add_argument("--batch", type=int, default=8, help="Random input rows per tensor.")
    p.add_argument("--threshold", type=float, default=0.02, help="Maximum allowed relative L2 error.")
    p.add_argument("--benchmark-iters", type=int, default=0, help="Also time Triton and fallback forwards.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm()
    denom = b.float().norm().clamp_min(1e-8)
    return float(diff / denom)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("check_packed_runtime requires CUDA")
    if not TernaryLinear.has_triton:
        raise RuntimeError("Triton is required for this check but is unavailable")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dict")

    entries = [
        (name, value)
        for name, value in state.items()
        if name.endswith(".weight") and is_packed_entry(value) and value.get("format") == "ternary_2bit_scale"
    ]
    if not entries:
        raise RuntimeError("no packed ternary .weight entries found")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    failed = False
    tested = 0

    for name, entry in entries[: max(1, int(args.max_tensors))]:
        out_features, in_features = (int(x) for x in entry["shape"])
        x = torch.randn(int(args.batch), in_features, device="cuda", dtype=torch.bfloat16)
        triton_layer = TernaryLinear.from_entry(entry, device="cuda", use_triton=True)
        fallback_layer = TernaryLinear.from_entry(entry, device="cuda", use_triton=False)

        with torch.inference_mode():
            y_triton = triton_layer(x)
            y_fallback = fallback_layer(x)

        diff = (y_triton.float() - y_fallback.float()).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        rel = rel_l2(y_triton, y_fallback)
        status = "PASS" if rel <= float(args.threshold) else "FAIL"
        bench = ""
        if int(args.benchmark_iters) > 0:
            iters = int(args.benchmark_iters)
            with torch.inference_mode():
                for _ in range(3):
                    _ = triton_layer(x)
                    _ = fallback_layer(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = triton_layer(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for _ in range(iters):
                    _ = fallback_layer(x)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
            triton_ms = (t1 - t0) * 1000.0 / iters
            fallback_ms = (t2 - t1) * 1000.0 / iters
            bench = f" triton_ms={triton_ms:.3f} fallback_ms={fallback_ms:.3f} speedup_vs_fallback={fallback_ms / max(1e-9, triton_ms):.2f}x"
        print(
            f"{status} {name} shape=({out_features},{in_features}) "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_l2={rel:.6g}{bench}"
        )
        tested += 1
        failed = failed or rel > float(args.threshold)

    print(f"[check_packed_runtime] tested={tested} threshold={float(args.threshold):.6g}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
