from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unpack_bits(packed: torch.Tensor, n: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed.reshape(-1, 1) >> shifts) & 1).reshape(-1)
    return bits[:n].to(torch.bool)


def _unpack_2bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    vals = ((packed.reshape(-1, 1) >> shifts) & 3).reshape(-1)
    return vals[:n].to(torch.uint8)


class PackedLinear(nn.Module):
    """
    Inference-only packed binary/ternary Linear.

    This module stores transferred packed weights and dequantizes only this
    layer's dense weight during forward. It is a correctness bridge for packed
    checkpoints, not an optimized bit-serial GEMM kernel.
    """

    def __init__(
        self,
        entry: Dict[str, Any],
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.format = str(entry["format"])
        self.out_features = int(entry["shape"][0])
        self.in_features = int(entry["shape"][1])
        self.numel = int(entry["numel"])
        self.group_size = int(entry["group_size"])
        self.pad = int(entry.get("pad", 0))
        self.register_buffer("packed", entry["packed"].detach().cpu().to(torch.uint8), persistent=True)
        self.register_buffer("scale", entry["scale"].detach().cpu().to(torch.float16), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().cpu(), persistent=True)

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantize(dtype=self.scale.dtype, device=self.scale.device)

    def dequantize(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        packed = self.packed.to(device=device)
        scale = self.scale.to(device=device).float().view(-1, 1)
        n = self.numel + self.pad
        if self.format == "binary_sign_scale":
            bits = _unpack_bits(packed, n).float()
            q = bits.mul(2.0).sub(1.0).view(-1, self.group_size)
        elif self.format == "ternary_2bit_scale":
            vals = _unpack_2bit(packed, n).view(-1, self.group_size)
            q = torch.zeros(vals.shape, dtype=torch.float32, device=vals.device)
            q[vals == 1] = -1.0
            q[vals == 2] = 1.0
        else:
            raise ValueError(f"unknown packed linear format: {self.format}")
        w = (q * scale).reshape(-1)[: self.numel].view(self.out_features, self.in_features)
        return w.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize(dtype=x.dtype, device=x.device)
        b = self.bias
        if b is not None and (b.device != x.device or b.dtype != x.dtype):
            b = b.to(device=x.device, dtype=x.dtype)
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"format={self.format}, group={self.group_size}, "
            f"bias={self.bias is not None}"
        )


class PackedEmbedding(nn.Module):
    """
    Inference-only packed embedding.

    The transfer format is the same packed 2D tensor format used for Linear
    weights. For row-aligned group sizes, forward dequantizes only the token
    rows requested by the input instead of expanding the full vocab table.
    """

    def __init__(self, entry: Dict[str, Any], padding_idx: Optional[int] = None):
        super().__init__()
        self.format = str(entry["format"])
        self.num_embeddings = int(entry["shape"][0])
        self.embedding_dim = int(entry["shape"][1])
        self.numel = int(entry["numel"])
        self.group_size = int(entry["group_size"])
        self.pad = int(entry.get("pad", 0))
        self.padding_idx = padding_idx
        self.register_buffer("packed", entry["packed"].detach().cpu().to(torch.uint8), persistent=True)
        self.register_buffer("scale", entry["scale"].detach().cpu().to(torch.float16), persistent=True)

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantize(dtype=self.scale.dtype, device=self.scale.device)

    def dequantize(self, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return PackedLinear(
            {
                "format": self.format,
                "shape": [self.num_embeddings, self.embedding_dim],
                "numel": self.numel,
                "group_size": self.group_size,
                "pad": self.pad,
                "packed": self.packed,
                "scale": self.scale,
            }
        ).dequantize(dtype=dtype, device=device)

    def _dequantize_rows(self, ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if self.embedding_dim % self.group_size != 0:
            return self.dequantize(dtype=dtype, device=ids.device).index_select(0, ids)

        rows = ids.to(device=self.packed.device, dtype=torch.long)
        groups_per_row = self.embedding_dim // self.group_size
        scale_idx = rows[:, None] * groups_per_row + torch.arange(groups_per_row, device=rows.device)
        scales = self.scale.index_select(0, scale_idx.reshape(-1)).to(device=ids.device, dtype=dtype)
        scales = scales.view(rows.numel(), groups_per_row, 1)

        if self.format == "binary_sign_scale":
            if self.embedding_dim % 8 != 0:
                return self.dequantize(dtype=dtype, device=ids.device).index_select(0, ids)
            bytes_per_row = self.embedding_dim // 8
            byte_idx = rows[:, None] * bytes_per_row + torch.arange(bytes_per_row, device=rows.device)
            packed_rows = self.packed.index_select(0, byte_idx.reshape(-1)).to(device=ids.device)
            bits = _unpack_bits(packed_rows, rows.numel() * self.embedding_dim).view(rows.numel(), self.embedding_dim)
            q = bits.to(dtype=dtype).mul_(2.0).sub_(1.0).view(rows.numel(), groups_per_row, self.group_size)
        elif self.format == "ternary_2bit_scale":
            if self.embedding_dim % 4 != 0:
                return self.dequantize(dtype=dtype, device=ids.device).index_select(0, ids)
            bytes_per_row = self.embedding_dim // 4
            byte_idx = rows[:, None] * bytes_per_row + torch.arange(bytes_per_row, device=rows.device)
            packed_rows = self.packed.index_select(0, byte_idx.reshape(-1)).to(device=ids.device)
            vals = _unpack_2bit(packed_rows, rows.numel() * self.embedding_dim).view(rows.numel(), self.embedding_dim)
            q = torch.zeros(vals.shape, dtype=dtype, device=ids.device)
            q[vals == 1] = -1.0
            q[vals == 2] = 1.0
            q = q.view(rows.numel(), groups_per_row, self.group_size)
        else:
            raise ValueError(f"unknown packed embedding format: {self.format}")

        return (q * scales).reshape(rows.numel(), self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1)
        emb = self._dequantize_rows(flat, dtype=self.scale.dtype)
        return emb.to(device=x.device).view(*x.shape, self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"format={self.format}, group={self.group_size}, padding_idx={self.padding_idx}"
        )


class CpuOffloadedLinear(nn.Module):
    """
    Linear weight kept on CPU across model.to(cuda). Intended for exact output
    head offload in memory-constrained inference.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().cpu(), persistent=True)
        if bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", bias.detach().cpu(), persistent=True)
        self.out_features = int(weight.shape[0])
        self.in_features = int(weight.shape[1])

    def _apply(self, fn):
        # Keep buffers on CPU even when the parent model is moved to CUDA.
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU offload path: CPU F.linear requires activation/weight dtype to match.
        x_cpu = x.detach().to(device="cpu", dtype=self.weight.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != self.weight.dtype:
            bias = bias.to(dtype=self.weight.dtype)
        y = F.linear(x_cpu, self.weight, bias)
        return y.to(device=x.device, dtype=x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, device=cpu"


class CpuOffloadedEmbedding(nn.Module):
    """Embedding weight kept on CPU across model.to(cuda)."""

    def __init__(self, weight: torch.Tensor, padding_idx: Optional[int] = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().cpu(), persistent=True)
        self.padding_idx = padding_idx
        self.num_embeddings = int(weight.shape[0])
        self.embedding_dim = int(weight.shape[1])

    def _apply(self, fn):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.embedding(x.detach().cpu(), self.weight, padding_idx=self.padding_idx)
        return y.to(device=x.device)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, device=cpu"


def is_packed_entry(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("format") in {"binary_sign_scale", "ternary_2bit_scale"}


def resolve_module(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def set_module(root: nn.Module, dotted: str, module: nn.Module) -> None:
    parent, name = resolve_module(root, dotted)
    if name.isdigit():
        parent[int(name)] = module  # type: ignore[index]
    else:
        setattr(parent, name, module)


def make_runtime_linear(entry: Dict[str, Any], bias: Optional[torch.Tensor] = None) -> nn.Module:
    if entry.get("format") == "ternary_2bit_scale" and os.environ.get("PACKED_TERNARY_LINEAR", "1") != "0":
        try:
            from tools.ternary_linear import TernaryLinear

            use_triton = os.environ.get(
                "PACKED_TERNARY_TRITON",
                "1" if torch.cuda.is_available() else "0",
            ) == "1"
            require_triton = os.environ.get("PACKED_REQUIRE_TRITON", "0") == "1"
            if require_triton and (not use_triton or not getattr(TernaryLinear, "has_triton", False)):
                raise RuntimeError("PACKED_REQUIRE_TRITON=1 but Triton ternary runtime is unavailable or disabled")
            return TernaryLinear.from_entry(entry, device="cpu", use_triton=use_triton, bias=bias)
        except Exception as exc:
            if os.environ.get("PACKED_STRICT_TERNARY_LINEAR", "0") == "1" or os.environ.get("PACKED_REQUIRE_TRITON", "0") == "1":
                raise
            print(
                f"[PACKED WARN] TernaryLinear unavailable ({type(exc).__name__}: {exc}); "
                "falling back to PackedLinear.",
                flush=True,
            )
    return PackedLinear(entry, bias)


def replace_packed_linears(model: nn.Module, state: Dict[str, Any]) -> set[str]:
    consumed_biases: set[str] = set()
    for key, entry in list(state.items()):
        if not key.endswith(".weight") or not is_packed_entry(entry):
            continue
        module_name = key[: -len(".weight")]
        old_parent, old_leaf = resolve_module(model, module_name)
        old_module = old_parent[int(old_leaf)] if old_leaf.isdigit() else getattr(old_parent, old_leaf)
        if isinstance(old_module, nn.Embedding):
            set_module(model, module_name, PackedEmbedding(entry, padding_idx=getattr(old_module, "padding_idx", None)))
            continue
        bias_key = module_name + ".bias"
        bias = state.get(bias_key)
        if torch.is_tensor(bias):
            consumed_biases.add(bias_key)
        set_module(model, module_name, make_runtime_linear(entry, bias if torch.is_tensor(bias) else None))
    return consumed_biases
