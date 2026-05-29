#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CANDIDATES = (
    ("mlp-only", 64, 0.3),
    ("mlp-only", 128, 0.3),
    ("mlp-only", 256, 0.5),
    ("mlp-plus-vo", 128, 0.3),
)


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate preset packed ternary checkpoint candidates.")
    p.add_argument("--checkpoint", required=True, help="Dense source checkpoint.")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "models" / "ternary_candidates"))
    p.add_argument("--mode", choices=["ternary"], default="ternary")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed through to transfer_to_1bit.py.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_name = Path(args.checkpoint).stem
    transfer = PROJECT_ROOT / "tools" / "transfer_to_1bit.py"

    for preset, group_size, threshold in CANDIDATES:
        tmp = out_dir / f".{src_name}-{preset}-g{group_size}-th{threshold_tag(threshold)}.tmp.pt"
        cmd = [
            sys.executable,
            str(transfer),
            "--checkpoint",
            str(args.checkpoint),
            "--out",
            str(tmp),
            "--mode",
            "ternary",
            "--preset",
            preset,
            "--group-size",
            str(group_size),
            "--threshold",
            str(threshold),
            *args.extra_arg,
        ]
        print("[candidate] running", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

        ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
        ratio = float(ckpt.get("manifest", {}).get("compression_ratio_tensor_bytes", 0.0))
        final = out_dir / (
            f"{src_name}-{preset}-g{group_size}-th{threshold_tag(threshold)}-cr{ratio:.2f}x.pt"
        )
        if final.exists():
            final.unlink()
        tmp.rename(final)
        print(f"[candidate] wrote {final}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
