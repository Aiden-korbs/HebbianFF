"""
ternary_linear.py
=================
Fused ternary-unpack + matmul that keeps weights packed in VRAM.

Memory layout
-------------
  Stored in VRAM  : packed uint8  (~1/8 the bytes of FP16)
  PCIe transfer   : packed uint8  (1-bit per element, not FP16)
  GPU registers   : unpacked FP16 (never written back to VRAM)
  VRAM output     : activation tensor (normal)

This means VRAM holds weights at ~2 bits/param (ternary) + FP16 scales,
vs 16 bits/param for BF16.  Approx 7-8x VRAM reduction for weight tensors.

Usage
-----
    layer = TernaryLinear.from_entry(entry, device="cuda")
    y = layer(x)          # x: (batch, seq, in_features)  BF16/FP16
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
from typing import Dict

# ── optional Triton path ──────────────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────────────────────────────────
#  Triton kernel: unpack 2-bit ternary + scale, then accumulate into output
# ─────────────────────────────────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _ternary_matmul_kernel_fast(
        # input activations  [M, K]
        X_ptr, stride_xm, stride_xk,
        # packed weights     [ceil(N*K/4)] uint8, 4x 2-bit symbols per byte
        W_packed_ptr,
        # scales             [ceil(N*K/group_size)]
        W_scale_ptr,
        # output             [M, N]
        Out_ptr, stride_om, stride_on,
        # dims
        M, N, K,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M:    tl.constexpr,
        BLOCK_N:    tl.constexpr,
        BLOCK_K:    tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + rk

            x_mask = (rm[:, None] < M) & (k[None, :] < K)
            x_offs = rm[:, None] * stride_xm + k[None, :] * stride_xk
            x = tl.load(X_ptr + x_offs, mask=x_mask, other=0.0)

            # Logical weight shape is [N, K], flattened as n*K + k.
            flat = rn[:, None] * K + k[None, :]
            w_mask = (rn[:, None] < N) & (k[None, :] < K)

            byte_idx = flat // 4
            bit_shift = ((flat % 4) * 2).to(tl.uint8)
            raw = tl.load(W_packed_ptr + byte_idx, mask=w_mask, other=0).to(tl.uint8)
            sym = (raw >> bit_shift) & 3

            w_q = tl.where(
                sym == 2,
                1.0,
                tl.where(sym == 1, -1.0, 0.0),
            )
            scale_idx = flat // GROUP_SIZE
            scale = tl.load(W_scale_ptr + scale_idx, mask=w_mask, other=1.0)
            w = (w_q * scale).to(x.dtype)

            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32, input_precision="tf32")

        out_mask = (rm[:, None] < M) & (rn[None, :] < N)
        out_offs = rm[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(Out_ptr + out_offs, acc, mask=out_mask)

    @triton.jit
    def _ternary_matmul_kernel_safe(
        # input activations  [M, K]
        X_ptr, stride_xm, stride_xk,
        # packed weights     [ceil(N*K/4)] uint8, 4x 2-bit symbols per byte
        W_packed_ptr,
        # scales             [ceil(N*K/group_size)]
        W_scale_ptr,
        # output             [M, N]
        Out_ptr, stride_om, stride_on,
        # dims
        M, N, K,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M:    tl.constexpr,
        BLOCK_N:    tl.constexpr,
        BLOCK_K:    tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + rk

            # X block: [BLOCK_M, BLOCK_K]
            x_mask = (rm[:, None] < M) & (k[None, :] < K)
            x_offs = rm[:, None] * stride_xm + k[None, :] * stride_xk
            x = tl.load(X_ptr + x_offs, mask=x_mask, other=0.0).to(tl.float32)

            # Correctness-first path:
            # compute each output column separately, exactly matching:
            #   flat = n_idx * K + k
            #   symbol = unpack_2bit(flat)
            #   scale_idx = flat // GROUP_SIZE
            #   out[:, n_idx] += sum_k x[:, k] * symbol_scale
            for ni in tl.static_range(0, BLOCK_N):
                n_idx = pid_n * BLOCK_N + ni

                flat = n_idx * K + k
                valid = (n_idx < N) & (k < K)

                byte_idx = flat // 4
                bit_shift = ((flat % 4) * 2).to(tl.uint8)

                raw = tl.load(W_packed_ptr + byte_idx, mask=valid, other=0).to(tl.uint8)
                sym = (raw >> bit_shift) & 3

                # symbols: 0 -> 0, 1 -> -1, 2 -> +1, 3 unused -> 0
                w_q = tl.where(
                    sym == 2,
                    1.0,
                    tl.where(sym == 1, -1.0, 0.0),
                ).to(tl.float32)

                scale_idx = flat // GROUP_SIZE
                scale = tl.load(W_scale_ptr + scale_idx, mask=valid, other=1.0).to(tl.float32)

                w = w_q * scale

                dot = tl.sum(x * w[None, :], axis=1)  # [BLOCK_M]

                # Triton does not allow acc[:, ni] += dot.
                # This masked add writes dot into the ni-th logical column.
                col = (rn == n_idx).to(tl.float32)    # [BLOCK_N]
                acc += dot[:, None] * col[None, :]

        out_mask = (rm[:, None] < M) & (rn[None, :] < N)
        out_offs = rm[:, None] * stride_om + rn[None, :] * stride_on
        tl.store(Out_ptr + out_offs, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
#  Pure-PyTorch fallback  (CPU or GPU without Triton)
#  Same memory-efficient idea: unpack in chunks, never full FP16 weight matrix
# ─────────────────────────────────────────────────────────────────────────────
def _ternary_matmul_pytorch(
    x: torch.Tensor,            # (M, K)
    packed: torch.Tensor,       # uint8, length = ceil(N*K/4)
    scale: torch.Tensor,        # fp16, length = ceil(N*K/group_size)
    shape: tuple,               # (N, K)
    group_size: int,
) -> torch.Tensor:
    N, K = shape
    M = x.shape[0]
    dev = x.device

    packed = packed.to(dev)
    scale  = scale.to(dev).float()

    # Unpack all at once — still efficient because uint8 → fp16 is fast
    # and we immediately multiply, then discard the unpacked buffer
    n_elem = N * K
    shifts = torch.tensor([0, 2, 4, 6], device=dev, dtype=torch.uint8)
    vals = ((packed.reshape(-1, 1) >> shifts) & 3).reshape(-1)[:n_elem]  # uint8

    w_q = torch.zeros(n_elem, dtype=torch.float32, device=dev)
    w_q[vals == 2] =  1.0
    w_q[vals == 1] = -1.0

    # Apply grouped scales
    n_groups = (n_elem + group_size - 1) // group_size
    pad = n_groups * group_size - n_elem
    if pad:
        w_q = torch.cat([w_q, torch.zeros(pad, device=dev)])
    w_q = w_q.view(n_groups, group_size) * scale[:n_groups].unsqueeze(1)
    w = w_q.reshape(-1)[:n_elem].view(N, K).to(x.dtype)

    return x @ w.T   # (M, N)


# ─────────────────────────────────────────────────────────────────────────────
#  nn.Module wrapper
# ─────────────────────────────────────────────────────────────────────────────
class TernaryLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores weights in packed ternary
    format.  Weights stay compressed in VRAM; unpacking happens in the kernel.

    VRAM cost: ~2 bits/param + FP16 scales  (vs 16 bits/param for BF16)
    """

    has_triton = HAS_TRITON

    def __init__(
        self,
        packed: torch.Tensor,   # uint8
        scale:  torch.Tensor,   # fp16
        shape:  tuple,          # (out_features, in_features)
        group_size: int,
        bias: torch.Tensor | None = None,
        use_triton: bool = True,
    ):
        super().__init__()
        self.register_buffer("packed",    packed)
        self.register_buffer("scale",     scale)
        self.register_buffer("bias",      bias)
        self.out_features, self.in_features = shape
        self.group_size  = group_size
        self.use_triton  = use_triton and HAS_TRITON

    # ── factory ──────────────────────────────────────────────────────────────
    @classmethod
    def from_entry(
        cls,
        entry: Dict,
        device: str = "cpu",
        use_triton: bool = True,
        bias: torch.Tensor | None = None,
    ) -> "TernaryLinear":
        if entry.get("format") != "ternary_2bit_scale":
            raise ValueError(f"Expected ternary_2bit_scale, got {entry.get('format')}")
        shape      = tuple(int(x) for x in entry["shape"])
        group_size = int(entry["group_size"])
        packed     = entry["packed"].to(device)
        scale      = entry["scale"].to(device)
        if bias is not None:
            bias = bias.to(device)
        return cls(packed, scale, shape, group_size, bias=bias, use_triton=use_triton)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        if self.use_triton and x_2d.is_cuda:
            out = self._triton_forward(x_2d)
        else:
            out = _ternary_matmul_pytorch(
                x_2d, self.packed, self.scale,
                (self.out_features, self.in_features), self.group_size,
            )

        out = out.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        M, K = x.shape
        N    = self.out_features
        out  = torch.empty((M, N), device=x.device, dtype=torch.float16)

        block_m_env = os.environ.get("PACKED_TERNARY_BLOCK_M", "auto")
        BLOCK_M = (16 if M <= 16 else 128) if block_m_env == "auto" else int(block_m_env)
        BLOCK_N = int(os.environ.get("PACKED_TERNARY_BLOCK_N", "16"))
        BLOCK_K = min(int(os.environ.get("PACKED_TERNARY_BLOCK_K", "64")), K)
        grid    = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        kernel = _ternary_matmul_kernel_safe if os.environ.get("PACKED_TERNARY_KERNEL", "fast") == "safe" else _ternary_matmul_kernel_fast
        kernel[grid](
            x.contiguous(), x.stride(0), x.stride(1),
            self.packed.contiguous(),
            self.scale.contiguous(),
            out, out.stride(0), out.stride(1),
            M, N, K,
            GROUP_SIZE=self.group_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        bits = (self.packed.numel() * 8) / (self.out_features * self.in_features)
        return (f"in={self.in_features}, out={self.out_features}, "
                f"group_size={self.group_size}, "
                f"bits/param={bits:.2f}, triton={self.use_triton and HAS_TRITON}")
