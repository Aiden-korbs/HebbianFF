from __future__ import annotations

from contextlib import nullcontext
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn

from .liger import HAS_LIGER, LigerRMSNorm

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The old fallback did x.float() and .to(x.dtype) on every call.
        # In bf16 AMP that shows up as many aten::_to_copy / aten::copy_
        # events. Default to same-dtype RMSNorm for speed/copy reduction.
        # Set RMSNORM_FP32=1 to restore the older fp32-cast behaviour.
        if os.environ.get("RMSNORM_FP32", "0") == "1":
            xf = x.float()
            rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (xf * rms).to(x.dtype) * self.weight

        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def setup_amp(device: str):
    use_bf16 = (device == "cuda") and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    def autocast_ctx(dtype):
        if device != "cuda": return nullcontext()
        return torch.amp.autocast("cuda", dtype=dtype)
    def make_scaler(enabled: bool):
        if device != "cuda" or amp_dtype == torch.bfloat16: return None
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return amp_dtype, autocast_ctx, make_scaler

def make_rmsnorm(dim: int, use_liger: bool) -> nn.Module:
    allow_liger_rmsnorm = os.environ.get("USE_LIGER_RMSNORM", "0") == "1"
    if allow_liger_rmsnorm and use_liger and HAS_LIGER and LigerRMSNorm is not None:
        return LigerRMSNorm(dim)
    return RMSNorm(dim)

def get_curriculum_len(step: int, cfg) -> int:
    if not cfg.use_curriculum or step >= cfg.curriculum_ramp_steps:
        return cfg.block_size
    frac = step / max(1, cfg.curriculum_ramp_steps)
    raw = cfg.curriculum_start_len + frac * (cfg.block_size - cfg.curriculum_start_len)
    return max(cfg.curriculum_start_len, min(cfg.block_size, int(raw // 64) * 64))

def build_scheduler(opt, cfg, remaining_steps: int):
    warmup = max(1, min(cfg.warmup_update_steps, remaining_steps // 10))
    total = max(warmup + 1, remaining_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, total - warmup, cfg.eta_min),
        ],
        milestones=[warmup],
    )

def apply_warm_restart(optimizers, cfg) -> None:
    for opt in optimizers:
        for pg in opt.param_groups:
            base = pg.get("initial_lr", pg["lr"])
            pg["lr"] = base * cfg.warm_restart_lr_frac
    print(f"[RESTART] LR reset to {cfg.warm_restart_lr_frac:.0%} of peak.", flush=True)

def get_grad_clip(cfg, current_lr: float) -> float:
    tail = current_lr / max(1e-12, cfg.lr)
    return float(cfg.grad_clip) if tail > 0.10 else float(cfg.grad_clip) * 2.0
