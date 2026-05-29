"""
bitnet.py  –  BitNet b1.58 primitives.

Implements ternary weight quantisation {-1, 0, +1} with a per-tensor
abs-mean scale and per-token int8 activation quantisation, following
Ma et al. (2024) "The Era of 1-bit LLMs".

Performance design
------------------
The naive implementation recomputes weight quantisation on every forward
pass, which adds ~2s of elementwise overhead per step with no benefit —
the ternary weights only change when the optimiser runs (once per N
grad-accum steps).

This version caches the quantised weight tensor and its scale, keyed on
PyTorch's internal ._version counter for the weight tensor.  The cache
is invalidated automatically when the optimiser writes to the weight
in-place (which increments ._version).  Within a grad-accum loop and
across the RevNet backward recomputation passes, the cache is hit every
time.

With grad_accum=4 and 6 BP blocks each recomputing attn in backward,
each BitLinear weight is quantised once and reused:
  4 (grad accum) x (1 forward + 1 backward recompute) = 8 reuses

Activation quantisation cannot be cached (activations change every
forward) but can be disabled for training via USE_BITNET_QUANT_ACT=0,
which leaves only weight quantisation active.  This is useful for
establishing a training baseline before enabling full int8 activations.

Precision holdout strategy
--------------------------
The following layers are kept in full precision regardless of use_bitnet:

    tok_emb   (nn.Embedding)          – vocabulary embeddings
    out_proj  (lm_head, nn.Linear)    – tied to tok_emb; must stay float
    ChunkMemoryCompressor              – kv_proj + out_proj
    FFDraftHead                        – proj + out
    engram_*_proj                      – minor, disabled by default
    cpu_ctx_proj                       – minor utility projection
"""
from __future__ import annotations

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _STEBitLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, weight_q: torch.Tensor, bias: Optional[torch.Tensor]):
        ctx.save_for_backward(x, weight_q)
        ctx.has_bias = bias is not None
        return F.linear(x, weight_q, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight_q = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_out.matmul(weight_q)
        if ctx.needs_input_grad[1]:
            grad_flat = grad_out.reshape(-1, grad_out.size(-1))
            x_flat = x.reshape(-1, x.size(-1)).to(dtype=grad_flat.dtype)
            grad_weight = grad_flat.transpose(0, 1).matmul(x_flat)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            grad_bias = grad_out.reshape(-1, grad_out.size(-1)).sum(dim=0)

        return grad_x, grad_weight, None, grad_bias


class BitLinear(nn.Linear):
    """
    Drop-in replacement for nn.Linear with 1.58-bit (ternary) weights.

    Forward pass (train and eval):
      1. Weights  → round(W / mean|W|).clamp(-1, 1)  {-1, 0, +1}  [cached]
      2. Inputs   → round(X / max|X|_token * 127)    int8 per-token
      3. STE on both steps so latent float weights get real gradients.

    Weight quantisation is cached between optimiser steps using
    self.weight._version as a staleness key.  The cache is valid for the
    full grad-accum loop and RevNet recomputation passes; it is
    invalidated when the optimiser writes the weight in-place.

    Activation quantisation can be disabled at runtime via:
        USE_BITNET_QUANT_ACT=0  (env var, read once at import time)
    This is useful for speed benchmarking or ablations.
    """

    # Read once at import time so there is no per-forward env-var overhead.
    _QUANT_ACT: bool = os.environ.get("USE_BITNET_QUANT_ACT", "1") == "1"
    _CUSTOM_STE: bool = os.environ.get("BITNET_CUSTOM_STE", "0") == "1"
    _BYPASS_TRAIN: bool = os.environ.get("BITNET_BYPASS_TRAIN", "0") == "1"
    _TRAIN_CACHE: bool = os.environ.get("BITNET_CACHE_TRAIN", "0") == "1"
    try:
        _TRAIN_CACHE_BUDGET_BYTES: int = max(0, int(float(os.environ.get("BITNET_CACHE_TRAIN_MIB", "0")) * 1024 * 1024))
    except ValueError:
        _TRAIN_CACHE_BUDGET_BYTES = 0
    _TRAIN_CACHE_USED_BYTES: int = 0
    _TRAIN_CACHE_PEAK_BYTES: int = 0

    @staticmethod
    def _train_cache_budget_bytes() -> int:
        return BitLinear._TRAIN_CACHE_BUDGET_BYTES

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        # Weight quantisation cache – invalidated by ._version changes.
        self._w_version:    int                    = -1
        self._w_q_cache:    Optional[torch.Tensor] = None
        self._w_scale_cache: Optional[torch.Tensor] = None

    def clear_weight_cache(self) -> None:
        self._w_version = -1
        self._w_q_cache = None
        self._w_scale_cache = None

    def _cache_bytes(self, w_q: torch.Tensor, w_scale: torch.Tensor) -> int:
        return int(w_q.numel() * w_q.element_size() + w_scale.numel() * w_scale.element_size())

    def _try_store_weight_cache(self, w_q: torch.Tensor, w_scale: torch.Tensor, version: int, training: bool) -> bool:
        if not training:
            self._w_q_cache = w_q.detach()
            self._w_scale_cache = w_scale.detach()
            self._w_version = version
            return True

        budget = self._train_cache_budget_bytes()
        if budget <= 0:
            return False
        need = self._cache_bytes(w_q, w_scale)
        if BitLinear._TRAIN_CACHE_USED_BYTES + need > budget:
            return False
        self._w_q_cache = w_q.detach()
        self._w_scale_cache = w_scale.detach()
        self._w_version = version
        BitLinear._TRAIN_CACHE_USED_BYTES += need
        BitLinear._TRAIN_CACHE_PEAK_BYTES = max(
            BitLinear._TRAIN_CACHE_PEAK_BYTES,
            BitLinear._TRAIN_CACHE_USED_BYTES,
        )
        return True

    # ------------------------------------------------------------------
    # Weight quantisation cache
    # ------------------------------------------------------------------

    def _quantised_weight(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (w_q, w_scale) from cache if self.weight has not been
        modified since the last call, otherwise recompute and cache.

        ._version is a monotonically increasing integer maintained by
        PyTorch that is incremented on any in-place write to the tensor
        (optimizer step, zero_grad, etc.).  Reading it is a pure Python
        attribute lookup – no CUDA synchronisation.
        """
        # Important:
        # Some optimizers, especially bitsandbytes AdamW8bit, may update
        # parameters without reliably bumping PyTorch's Tensor._version.
        # If we trust _version during training, BitLinear can keep using a
        # stale ternary cache while the latent/master weights drift. That
        # makes live validation look good but saved/reloaded checkpoints
        # catastrophically disagree.
        #
        # Therefore: always recompute ternary weights during training.
        # Cache is only used for eval/inference unless BITNET_CACHE_TRAIN=1.
        is_training_grad = self.training and torch.is_grad_enabled()
        train_cache = self._TRAIN_CACHE
        allow_train_cache = is_training_grad and train_cache
        force_recompute = is_training_grad and not train_cache

        v = self.weight._version
        if force_recompute or v != self._w_version or self._w_q_cache is None:
            with torch.no_grad():
                w = self.weight
                w_scale = w.abs().mean().clamp(min=1e-5)
                w_q = torch.empty_like(w)
                torch.div(w, w_scale, out=w_q)
                w_q.round_().clamp_(-1.0, 1.0)

            if force_recompute or (allow_train_cache and not self._try_store_weight_cache(w_q, w_scale, v, training=True)):
                # Do not store training caches by default. This prevents stale
                # caches across optimizer steps, grad-accum boundaries, and
                # the next eval pass if the optimizer did not bump _version.
                self._w_q_cache = None
                self._w_scale_cache = None
                self._w_version = -1
                return w_q, w_scale

            # Eval/inference cache.
            self._try_store_weight_cache(w_q, w_scale, v, training=False)
        return self._w_q_cache, self._w_scale_cache   # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w      = self.weight
        if self._BYPASS_TRAIN and self.training and torch.is_grad_enabled():
            return F.linear(x, w, self.bias)
        w_q, _ = self._quantised_weight()

        # Training path: STE for weights.
        # Inference/eval path: use cached quantised weight directly to avoid
        # creating a full-size w_ste temporary every token.
        if self._QUANT_ACT:
            # Per-token int8 activation quantisation.
            x_scale = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5).div_(127.0)
            x_q     = x.div(x_scale).round_().clamp_(-128.0, 127.0)

            if self.training and torch.is_grad_enabled():
                x_eff = x + (x_q.mul_(x_scale) - x).detach()
            else:
                x_eff = x_q.mul_(x_scale)
        else:
            x_eff = x

        if self.training and torch.is_grad_enabled():
            if self._CUSTOM_STE:
                return _STEBitLinearFn.apply(x_eff, w, w_q, self.bias)
            w_eff = w + (w_q - w).detach()
        else:
            w_eff = w_q

        return F.linear(x_eff, w_eff, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"quant={'train_dense_bypass' if self._BYPASS_TRAIN else 'ternary+' + ('int8' if self._QUANT_ACT else 'fp_acts')}, "
                f"w_cache={'hit' if self._w_q_cache is not None else 'cold'}")


# ─────────────────────────────────────────────────────────────────────────────

def make_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    use_bitnet: bool = False,
) -> nn.Linear:
    """Return BitLinear if use_bitnet else plain nn.Linear."""
    cls = BitLinear if use_bitnet else nn.Linear
    return cls(in_features, out_features, bias=bias)


def count_bitnet_params(model: nn.Module) -> tuple[int, int]:
    """
    Returns (bitnet_params, total_params).
    Call after model construction to confirm quantisation coverage.
    """
    total = sum(p.numel() for p in model.parameters())
    bit   = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, BitLinear)
        for p in m.parameters()
    )
    return bit, total


def clear_bitnet_weight_caches(model: nn.Module) -> None:
    """Drop cached ternary weights after an optimizer step."""
    BitLinear._TRAIN_CACHE_USED_BYTES = 0
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.clear_weight_cache()


def bitnet_train_cache_stats() -> tuple[float, float]:
    """Return current train-cache usage and budget in MiB."""
    used = BitLinear._TRAIN_CACHE_USED_BYTES / 1024**2
    budget = BitLinear._train_cache_budget_bytes() / 1024**2
    return float(used), float(budget)


def bitnet_train_cache_peak_mib() -> float:
    """Return peak train-cache usage for this process in MiB."""
    return float(BitLinear._TRAIN_CACHE_PEAK_BYTES / 1024**2)
