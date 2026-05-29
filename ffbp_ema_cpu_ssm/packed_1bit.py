from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def packed_1bit_active() -> bool:
    return os.environ.get("PACKED_1BIT_ACTIVE", "0") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _split_fragments(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


@dataclass
class Packed1BitStats:
    modules: int = 0
    packed_params: int = 0
    original_bytes: int = 0
    packed_bytes: int = 0
    skipped: int = 0

    @property
    def saved_bytes(self) -> int:
        return max(0, self.original_bytes - self.packed_bytes)


class Packed1BitLinear(nn.Module):
    """
    Inference-only packed sign linear.

    Persistent storage is uint8-packed sign bits plus one scale per output
    channel. Forward unpacks only this layer's weight on the current CUDA
    device, runs F.linear, then lets the temporary dense weight die. A future
    partial-active path can hook into _dequant_weight by selecting output rows
    or input-column byte ranges before unpacking.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: Optional[torch.Tensor],
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        weight_dtype: torch.dtype,
    ):
        super().__init__()
        if packed_weight.dtype != torch.uint8:
            raise TypeError("packed_weight must be uint8")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_dtype = weight_dtype
        self.register_buffer("packed_weight", packed_weight.contiguous(), persistent=True)
        self.register_buffer("scale", scale.contiguous(), persistent=True)

        packed_cols = self.packed_weight.size(1)
        byte_offsets = torch.arange(self.in_features, device=packed_weight.device, dtype=torch.long).div(
            8, rounding_mode="floor"
        )
        bit_offsets = (torch.arange(self.in_features, device=packed_weight.device, dtype=torch.long) & 7).to(torch.uint8)
        bit_shifts = (1 << bit_offsets).to(torch.uint8)
        self.register_buffer("byte_offsets", byte_offsets.clamp_max(max(0, packed_cols - 1)), persistent=False)
        self.register_buffer("bit_shifts", bit_shifts, persistent=False)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "Packed1BitLinear":
        if not linear.weight.is_cuda:
            raise ValueError("Packed1BitLinear conversion requires CUDA weights")
        weight = linear.weight.detach()
        out_features, in_features = weight.shape
        scale = weight.abs().mean(dim=1).clamp_min(1e-6).to(dtype=torch.float32)

        signs = (weight >= 0).to(torch.uint8)
        packed_cols = (in_features + 7) // 8
        padded_cols = packed_cols * 8
        if padded_cols != in_features:
            pad = torch.zeros((out_features, padded_cols - in_features), device=weight.device, dtype=torch.uint8)
            signs = torch.cat([signs, pad], dim=1)

        shifts = (1 << torch.arange(8, device=weight.device, dtype=torch.uint8)).view(1, 1, 8)
        packed = (signs.view(out_features, packed_cols, 8) * shifts).sum(dim=2).to(torch.uint8)
        bias = linear.bias.detach() if linear.bias is not None else None
        return cls(
            in_features,
            out_features,
            bias=bias,
            packed_weight=packed,
            scale=scale,
            weight_dtype=weight.dtype,
        )

    def _dequant_weight(self, dtype: torch.dtype) -> torch.Tensor:
        if not self.packed_weight.is_cuda:
            raise RuntimeError("Packed1BitLinear hot path requires CUDA packed weights")
        bytes_for_cols = self.packed_weight.index_select(1, self.byte_offsets)
        bits = bytes_for_cols.bitwise_and(self.bit_shifts.view(1, -1)).ne(0)
        sign = bits.to(dtype=dtype).mul_(2.0).add_(-1.0)
        return sign.mul_(self.scale.to(device=sign.device, dtype=dtype).view(-1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("Packed1BitLinear forward requires CUDA input")
        weight = self._dequant_weight(x.dtype)
        return F.linear(x, weight, self.bias)

    def dense_weight(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self._dequant_weight(dtype or self.weight_dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        weight = self.dense_weight(self.weight_dtype)
        destination[prefix + "weight"] = weight if keep_vars else weight.detach()
        if self.bias is not None:
            destination[prefix + "bias"] = self.bias if keep_vars else self.bias.detach()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant=packed_sign_1bit"
        )


def _default_exclude_fragments() -> tuple[str, ...]:
    return (
        "ff_draft_head",
        "mem_compressor",
        "engram",
        "cpu_ctx",
    )


def _should_pack(name: str, module: nn.Linear, min_numel: int, exclude: Iterable[str]) -> bool:
    if type(module) is not nn.Linear:
        return False
    if name in {"out_proj", "final_proj"}:
        return False
    if ".attn." in name and os.environ.get("PACKED_1BIT_INCLUDE_ATTENTION", "0") != "1":
        return False
    if module.bias is not None:
        return False
    if module.weight.numel() < min_numel:
        return False
    return not any(fragment and fragment in name for fragment in exclude)


def _set_child(root: nn.Module, name: str, child: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = child
    else:
        setattr(parent, last, child)


@torch.no_grad()
def convert_model_to_packed_1bit(model: nn.Module) -> Packed1BitStats:
    stats = Packed1BitStats()
    if not packed_1bit_active():
        return stats
    if not torch.cuda.is_available():
        print("[PACKED_1BIT WARN] CUDA unavailable; packed mode disabled.", flush=True)
        return stats

    min_numel = _env_int("PACKED_1BIT_MIN_NUMEL", 262144)
    debug = os.environ.get("PACKED_1BIT_DEBUG", "0") == "1"
    force_verify = os.environ.get("PACKED_1BIT_FORCE_VERIFY", "0") == "1"
    user_exclude = _split_fragments(os.environ.get("PACKED_1BIT_EXCLUDE", ""))
    exclude = _default_exclude_fragments() + user_exclude

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _should_pack(name, module, min_numel, exclude):
            stats.skipped += 1
            continue
        if not module.weight.is_cuda:
            print(f"[PACKED_1BIT WARN] {name}: weight is not CUDA; skipping.", flush=True)
            stats.skipped += 1
            continue
        packed = Packed1BitLinear.from_linear(module)
        if force_verify:
            sample = torch.randn(2, 3, module.in_features, device=module.weight.device, dtype=module.weight.dtype)
            ref = module(sample)
            got = packed(sample)
            max_err = (ref - got).abs().max().float().item()
            mean_err = (ref - got).abs().mean().float().item()
            print(f"[PACKED_1BIT VERIFY] {name}: max_err={max_err:.6g} mean_err={mean_err:.6g}", flush=True)
        original_bytes = module.weight.numel() * module.weight.element_size()
        packed_bytes = packed.packed_weight.numel() * packed.packed_weight.element_size()
        packed_bytes += packed.scale.numel() * packed.scale.element_size()
        if packed.bias is not None:
            packed_bytes += packed.bias.numel() * packed.bias.element_size()
        _set_child(model, name, packed)
        stats.modules += 1
        stats.packed_params += module.weight.numel()
        stats.original_bytes += original_bytes
        stats.packed_bytes += packed_bytes
        if debug:
            print(
                f"[PACKED_1BIT] packed {name}: shape={tuple(module.weight.shape)} "
                f"persistent={original_bytes/1048576:.2f}MiB->{packed_bytes/1048576:.2f}MiB",
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[PACKED_1BIT] active modules={stats.modules} params={stats.packed_params/1e6:.2f}M "
        f"persistent_saved={stats.saved_bytes/1048576:.2f}MiB min_numel={min_numel}",
        flush=True,
    )
    return stats
