from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_rmsnorm
from .bitnet import make_linear

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Llama/TinyLlama RoPE rotates the first half against the second half.
    # This is NOT GPT-J/even-odd RoPE.
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    hd = x.size(-1)
    if hd % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {hd}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=x.dtype).view(1, 1, -1, hd)
    sin = emb.sin().to(dtype=x.dtype).view(1, 1, -1, hd)
    return (x * cos) + (rotate_half(x) * sin)

class RevGQACausalAttention(nn.Module):
    def __init__(self, dim: int, cfg, local_window: int):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        if cfg.n_head % cfg.n_kv_head != 0:
            raise ValueError(f"n_head ({cfg.n_head}) must be divisible by n_kv_head ({cfg.n_kv_head})")
        if dim % cfg.n_head != 0:
            raise ValueError(f"Attention dim ({dim}) must be divisible by n_head ({cfg.n_head})")
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.hd = dim // cfg.n_head
        self.qk_norm_scale = float(math.sqrt(self.hd))
        self.dropout = cfg.dropout
        self.local_window = local_window
        self.use_qk_norm = cfg.use_qk_norm
        self.use_rope = cfg.use_rope
        self.rope_theta = cfg.rope_theta
        self.kv_cache_int8 = bool(getattr(cfg, "kv_cache_int8", False))
        self.kv_cache_sink_tokens = max(0, int(getattr(cfg, "kv_cache_sink_tokens", 0) or 0))
        # With memory prepended to K/V, a custom bool attn_mask forces PyTorch
        # onto the slow math SDPA backend. The flash-prefix path makes Q/K/V
        # square by adding memory-prefix Q rows, calls SDPA with is_causal=True,
        # then slices those prefix outputs away. Set FLASH_PREFIX_MEMORY=0 to
        # force the exact old mask fallback.
        self.flash_prefix_memory = bool(getattr(cfg, "flash_prefix_memory", True))
        if self.hd % 2 != 0 and self.use_rope:
            raise ValueError(f"RoPE requires even attention head_dim, got {self.hd}")
        _bit = getattr(cfg, 'use_bitnet', False)
        self.q_proj = make_linear(dim, cfg.n_head * self.hd, bias=bool(getattr(cfg, "attn_qkv_bias", False)), use_bitnet=_bit)
        self.k_proj = make_linear(dim, cfg.n_kv_head * self.hd, bias=bool(getattr(cfg, "attn_qkv_bias", False)), use_bitnet=_bit)
        self.v_proj = make_linear(dim, cfg.n_kv_head * self.hd, bias=bool(getattr(cfg, "attn_qkv_bias", False)), use_bitnet=_bit)
        self.c_proj = make_linear(cfg.n_head * self.hd, dim, bias=False, use_bitnet=_bit)
        self.ln = make_rmsnorm(dim, cfg.use_liger_rmsnorm)
        # BitNet: c_proj input (attention output) has no upstream norm
        self.c_norm = make_rmsnorm(cfg.n_head * self.hd, cfg.use_liger_rmsnorm) if _bit else None
        if self.use_rope:
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.hd, 2).float() / self.hd))
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("rope_inv_freq", torch.empty(0), persistent=False)

    def _apply_rope_cached(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Llama/TinyLlama RoPE layout:
        #   freqs duplicated as [freqs, freqs]
        #   rotate first-half/second-half, not even/odd channels.
        pos = positions.to(device=x.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(pos, self.rope_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().view(1, 1, -1, self.hd)
        sin = emb.sin().view(1, 1, -1, self.hd)
        if cos.dtype != x.dtype:
            cos = cos.to(dtype=x.dtype)
            sin = sin.to(dtype=x.dtype)
        return (x * cos) + (rotate_half(x) * sin)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        # Avoid expand(...).reshape(...), which creates a non-contiguous
        # expanded view and then forces a large materialising layout copy.
        # repeat_interleave makes the required GQA materialisation explicit
        # and tends to compile to a single, cleaner copy kernel.
        return x.repeat_interleave(n_rep, dim=1)

    def _mem_causal_mask(self, T: int, M: int, pos_offset: int, device) -> torch.Tensor:
        q_pos = torch.arange(pos_offset, pos_offset + T, device=device)
        m_start = max(0, pos_offset - M)
        m_pos = torch.arange(m_start, m_start + M, device=device)
        kv_pos = torch.cat([m_pos, q_pos])
        q = q_pos.view(1, 1, -1, 1)
        k = kv_pos.view(1, 1, 1, -1)
        causal = k <= q
        is_mem = torch.zeros(kv_pos.size(0), dtype=torch.bool, device=device)
        is_mem[:M] = True
        local = (q - k) < self.local_window
        return causal & (is_mem.view(1, 1, 1, -1) | local)

    def _rope_positions(self, T: int, M: int, pos_offset: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        q_pos = torch.arange(pos_offset, pos_offset + T, device=device)
        if M == 0: return q_pos, q_pos
        m_start = max(0, pos_offset - M)
        m_pos = torch.arange(m_start, m_start + M, device=device)
        return q_pos, torch.cat([m_pos, q_pos])

    @staticmethod
    def _quantize_kv_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xf = x.detach().float().contiguous()
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        q = (xf / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale.to(dtype=torch.float16)

    @staticmethod
    def _dequantize_kv_tensor(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (q.float() * scale.float()).to(dtype=dtype)

    def _trim_kv_cache_tensors(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        max_cache_len: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if max_cache_len is None or max_cache_len <= 0 or k.size(2) <= max_cache_len:
            return k, v, positions

        keep = int(max_cache_len)
        sink = min(int(getattr(self, "kv_cache_sink_tokens", 0)), max(0, keep - 1))
        recent = keep - sink
        if sink <= 0:
            return (
                k[:, :, -keep:, :].contiguous(),
                v[:, :, -keep:, :].contiguous(),
                positions[-keep:].contiguous(),
            )
        if recent <= 0:
            return (
                k[:, :, :sink, :].contiguous(),
                v[:, :, :sink, :].contiguous(),
                positions[:sink].contiguous(),
            )
        k = torch.cat([k[:, :, :sink, :], k[:, :, -recent:, :]], dim=2).contiguous()
        v = torch.cat([v[:, :, :sink, :], v[:, :, -recent:, :]], dim=2).contiguous()
        positions = torch.cat([positions[:sink], positions[-recent:]], dim=0).contiguous()
        return k, v, positions

    def _pack_kv_cache(self, k: torch.Tensor, v: torch.Tensor, positions: Optional[torch.Tensor] = None):
        pos = positions.detach().to(device=k.device, dtype=torch.long).contiguous() if positions is not None else None
        if not bool(getattr(self, "kv_cache_int8", False)):
            if pos is None and int(getattr(self, "kv_cache_sink_tokens", 0)) <= 0:
                return (k, v)
            return {"format": "float", "k": k, "v": v, "pos": pos}
        k_q, k_scale = self._quantize_kv_tensor(k)
        v_q, v_scale = self._quantize_kv_tensor(v)
        return {
            "format": "int8",
            "k": k_q,
            "k_scale": k_scale,
            "v": v_q,
            "v_scale": v_scale,
            "pos": pos,
        }

    def _unpack_kv_cache(self, cache, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        k, v, _ = self._unpack_kv_cache_with_pos(cache, dtype)
        return k, v

    def _unpack_kv_cache_with_pos(self, cache, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(cache, dict) and cache.get("format") == "int8":
            k = self._dequantize_kv_tensor(cache["k"], cache["k_scale"], dtype)
            v = self._dequantize_kv_tensor(cache["v"], cache["v_scale"], dtype)
            return k, v, cache.get("pos")
        if isinstance(cache, dict) and cache.get("format") == "float":
            return cache["k"], cache["v"], cache.get("pos")
        old_k, old_v = cache
        return old_k, old_v, None

    def forward_kv(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard-attention inference KV-cache path.

        Cache tensors store already-RoPE-applied K/V:
          k: [B, n_kv_head, S, head_dim]
          v: [B, n_kv_head, S, head_dim]
        """
        h = self.ln(x)
        B, T, C = h.shape

        q = self.q_proj(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k_new = self.k_proj(h).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v_new = self.v_proj(h).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(pos_offset, pos_offset + T, device=x.device)
            q = self._apply_rope_cached(q, pos)
            k_new = self._apply_rope_cached(k_new, pos)

        if self.use_qk_norm:
            q = F.normalize(q, p=2, dim=-1) * self.qk_norm_scale
            k_new = F.normalize(k_new, p=2, dim=-1) * self.qk_norm_scale

        new_pos = torch.arange(pos_offset, pos_offset + T, device=x.device, dtype=torch.long)

        if cache is None:
            k = k_new
            v = v_new
            cache_pos = new_pos
            is_causal = T > 1
            attn_mask = None
        else:
            old_k, old_v, old_pos = self._unpack_kv_cache_with_pos(cache, dtype=k_new.dtype)
            old_len = old_k.size(2)
            if old_pos is None:
                old_pos = torch.arange(pos_offset - old_len, pos_offset, device=x.device, dtype=torch.long)
            else:
                old_pos = old_pos.to(device=x.device, dtype=torch.long)
            k = torch.cat([old_k, k_new], dim=2)
            v = torch.cat([old_v, v_new], dim=2)
            cache_pos = torch.cat([old_pos, new_pos], dim=0)
            is_causal = False
            if T > 1:
                q_pos = new_pos.view(1, 1, T, 1)
                k_pos = cache_pos.view(1, 1, 1, cache_pos.numel())
                attn_mask = k_pos <= q_pos
                if max_cache_len is not None and max_cache_len > 0:
                    keep = int(max_cache_len)
                    sink = min(int(getattr(self, "kv_cache_sink_tokens", 0)), max(0, keep - 1))
                    recent = keep - sink
                    key_idx = torch.arange(cache_pos.numel(), device=x.device).view(1, 1, 1, -1)
                    sink_key = key_idx < sink
                    # Match decode_one_kv exactly: each token attends the
                    # cache as it existed before that token is appended, plus
                    # its own K/V row, then the cache is trimmed afterwards.
                    recent_key = k_pos >= (q_pos - recent)
                    attn_mask = attn_mask & (sink_key | recent_key)
            else:
                attn_mask = None

        k_cache, v_cache, cache_pos = self._trim_kv_cache_tensors(k, v, cache_pos, max_cache_len)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout if self.training else 0.0),
            is_causal=is_causal,
            enable_gqa=(self.n_rep != 1),
        )

        out = y.permute(0, 2, 1, 3).reshape(B, T, self.n_head * self.hd)
        if self.c_norm is not None:
            out = self.c_norm(out)
        return self.c_proj(out), self._pack_kv_cache(k_cache, v_cache, cache_pos)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        h = self.ln(x)
        B, T, C = h.shape
        # Treat empty memory as no memory. Passing an empty [B,0,C] tensor
        # previously forced torch.cat([empty, h]) every block, which copies h
        # while changing nothing semantically.
        M = 0 if mem is None else int(mem.size(1))
        if M == 0:
            _mem = None
            kv = h
        else:
            _mem = mem[..., :C] if mem.size(-1) > C else mem
            if _mem.size(-1) < C:
                _mem = F.pad(_mem, (0, C - _mem.size(-1)))
            if _mem.device != h.device or _mem.dtype != h.dtype:
                _mem = _mem.to(device=h.device, dtype=h.dtype)
            kv = torch.cat([_mem, h], dim=1)
        S = kv.size(1)
        k = self.k_proj(kv).view(B, S, self.n_kv_head, self.hd).transpose(1, 2)
        v = self.v_proj(kv).view(B, S, self.n_kv_head, self.hd).transpose(1, 2)

        # Do NOT manually expand/repeat K/V for GQA. PyTorch SDPA supports
        # native grouped-query attention via enable_gqa=True, avoiding the
        # large K/V materialisation and the profiler-visible copy kernels from
        # expand(...).reshape(...) / repeat_interleave(...).
        enable_gqa = self.n_rep != 1
        dp = self.dropout if self.training else 0.0

        if M > 0 and self.flash_prefix_memory:
            # Flash Attention in this PyTorch build rejects causal non-square
            # attention where seqlen_q != seqlen_k. Make the call square by
            # also projecting Q for the memory prefix, then discard the prefix
            # query outputs after SDPA. For current-token outputs this is
            # equivalent to a prefix-memory causal layout: query M+i can attend
            # keys 0..M+i, i.e. all memory tokens plus current causal history.
            q_full = self.q_proj(kv).view(B, S, self.n_head, self.hd).transpose(1, 2)
            if self.use_rope:
                _, k_pos = self._rope_positions(T, M, pos_offset, x.device)
                q_full = self._apply_rope_cached(q_full, k_pos)
                k = self._apply_rope_cached(k, k_pos)
            if self.use_qk_norm:
                q_full = F.normalize(q_full, p=2, dim=-1) * self.qk_norm_scale
                k = F.normalize(k, p=2, dim=-1) * self.qk_norm_scale
            y_full = F.scaled_dot_product_attention(
                q_full, k, v, dropout_p=dp, is_causal=True, enable_gqa=enable_gqa
            )
            y = y_full[:, :, M:, :]
        else:
            q = self.q_proj(h).view(B, T, self.n_head, self.hd).transpose(1, 2)
            if self.use_rope:
                q_pos, k_pos = self._rope_positions(T, M, pos_offset, x.device)
                q = self._apply_rope_cached(q, q_pos)
                k = self._apply_rope_cached(k, k_pos)
            if self.use_qk_norm:
                q = F.normalize(q, p=2, dim=-1) * self.qk_norm_scale
                k = F.normalize(k, p=2, dim=-1) * self.qk_norm_scale
            if M == 0:
                y = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=dp, is_causal=True, enable_gqa=enable_gqa
                )
            else:
                # Exact fallback: preserves the old local-window + memory mask,
                # but this forces math SDPA because Flash Attention does not
                # support a non-null attn_mask in this PyTorch path.
                mask = self._mem_causal_mask(T, M, pos_offset, x.device)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dp, enable_gqa=enable_gqa
                )
        # Same logical layout as transpose(1, 2).contiguous().view(...),
        # but expressed as one reshape path so Inductor has less layout
        # bookkeeping to split into separate ops.
        out = y.permute(0, 2, 1, 3).reshape(B, T, self.n_head * self.hd)
        if self.c_norm is not None:
            out = self.c_norm(out)
        return self.c_proj(out)

class RevSwiGLUMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, cfg):
        super().__init__()
        self.ln = make_rmsnorm(dim, cfg.use_liger_rmsnorm)
        h = (int(2 / 3 * mlp_ratio * dim) + 63) // 64 * 64
        _bit = getattr(cfg, 'use_bitnet', False)
        self.fused_gate_up = bool(getattr(cfg, "use_fused_swiglu", False))
        if self.fused_gate_up:
            self.gate_up = make_linear(dim, 2 * h, bias=False, use_bitnet=_bit)
            self.gate = None
            self.up = None
        else:
            self.gate = make_linear(dim, h, bias=False, use_bitnet=_bit)
            self.up   = make_linear(dim, h, bias=False, use_bitnet=_bit)
            self.gate_up = None
        self.down = make_linear(h, dim, bias=False, use_bitnet=_bit)
        # BitNet: down input (gated activation) has no upstream norm
        self.down_norm = make_rmsnorm(h, cfg.use_liger_rmsnorm) if _bit else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.ln(x)
        if self.fused_gate_up:
            gate, up = self.gate_up(n).chunk(2, dim=-1)
            gated = F.silu(gate) * up
        else:
            gated = F.silu(self.gate(n)) * self.up(n)
        if self.down_norm is not None:
            gated = self.down_norm(gated)
        return self.down(gated)

class DepthwiseConvMixer(nn.Module):
    """
    Cheap causal token mixer for CPU-efficient FF layers.

    This preserves left-to-right LM causality by padding only on the left. The
    residual gate starts small so replacing an attention block begins close to
    an identity path and can learn local mixing as training progresses.
    """
    def __init__(self, dim: int, kernel_size: int, cfg):
        super().__init__()
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k += 1
        self.kernel_size = k
        self.ln = make_rmsnorm(dim, cfg.use_liger_rmsnorm)
        self.conv = nn.Conv1d(dim, dim, kernel_size=k, groups=dim, bias=False)
        self.gate = nn.Parameter(torch.tensor([-2.0]))
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.ln(x).transpose(1, 2)
        n = F.pad(n, (self.kernel_size - 1, 0))
        y = self.conv(n).transpose(1, 2)
        return y * torch.sigmoid(self.gate[0]).to(dtype=y.dtype)

class ResidualBlock(nn.Module):
    """
    Full-width Transformer residual block.

    This intentionally replaces the old half-width reversible/coupling block:
      - attention and MLP both operate on cfg.n_embd, not cfg.n_embd // 2
      - shapes now match standard pretrained decoder blocks closely enough for
        direct Q/K/V/O + MLP weight transfer
      - BP activation VRAM is controlled from model.py with torch checkpointing
    """
    def __init__(self, cfg, is_ff: bool, attn_enabled: bool = True, mixer: str = "none"):
        super().__init__()
        dim = cfg.n_embd
        window = cfg.local_window if is_ff else cfg.block_size
        ratio = cfg.mlp_ratio if is_ff else cfg.bp_mlp_ratio
        self.attn_enabled = bool(attn_enabled)
        self.attn = RevGQACausalAttention(dim, cfg, window) if self.attn_enabled else None
        mixer = mixer.strip().lower()
        if mixer not in {"none", "depthwise_conv"}:
            raise ValueError(f"Unsupported mixer {mixer!r}")
        self.mixer = DepthwiseConvMixer(dim, getattr(cfg, "local_mixer_kernel", 5), cfg) if mixer == "depthwise_conv" else None
        self.mlp = RevSwiGLUMLP(dim, ratio, cfg)

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        _mem = mem if (mem is not None and mem.size(1) > 0) else None
        if self.attn is not None:
            x = x + self.attn(x, _mem, pos_offset)
        elif self.mixer is not None:
            x = x + self.mixer(x)
        x = x + self.mlp(x)
        return x

    def forward_skip_mlp(self, x: torch.Tensor, mem: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
        _mem = mem if (mem is not None and mem.size(1) > 0) else None
        if self.attn is not None:
            x = x + self.attn(x, _mem, pos_offset)
        elif self.mixer is not None:
            x = x + self.mixer(x)
        return x

    def forward_kv(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.attn is not None:
            attn_out, cache = self.attn.forward_kv(
                x,
                cache=cache,
                pos_offset=pos_offset,
                max_cache_len=max_cache_len,
            )
            x = x + attn_out
        elif self.mixer is not None:
            x = x + self.mixer(x)
        x = x + self.mlp(x)
        return x, cache

    def forward_kv_skip_mlp(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        max_cache_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.attn is not None:
            attn_out, cache = self.attn.forward_kv(
                x,
                cache=cache,
                pos_offset=pos_offset,
                max_cache_len=max_cache_len,
            )
            x = x + attn_out
        elif self.mixer is not None:
            x = x + self.mixer(x)
        return x, cache


# Backwards-compatible import name for the rest of the codebase.
# This is no longer reversible and no longer half-width.
RevBlock = ResidualBlock
