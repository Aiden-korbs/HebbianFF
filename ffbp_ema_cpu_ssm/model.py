from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .config import CFG
from .utils import make_rmsnorm
from .liger import HAS_LIGER, LigerFusedLinearCrossEntropyLoss
from .blocks import RevBlock
from .bitnet import make_linear, count_bitnet_params
from .memory import FFDraftHead, ChunkMemoryCompressor, EngramMemoryBank

class FF_LLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.n_embd = cfg.n_embd
        self.block_size = cfg.block_size
        _head_scale = float(getattr(cfg, "head_scale", float("nan")))
        self.head_scale = _head_scale if math.isfinite(_head_scale) else float(1.0 / math.sqrt(cfg.n_embd))
        self.engram_attn_scale = float(1.0 / math.sqrt(max(1, cfg.engram_key_dim)))

        self.tok_emb = nn.Embedding(vocab_size, cfg.n_embd)
        nn.init.normal_(self.tok_emb.weight, std=cfg.emb_init_std)

        self.ff_attention_layers = self._build_ff_attention_schedule(cfg)
        self.ff_blocks = nn.ModuleList([
            RevBlock(
                cfg,
                is_ff=True,
                attn_enabled=(i in self.ff_attention_layers),
                mixer="none" if (i in self.ff_attention_layers) else str(getattr(cfg, "ff_mixer", "depthwise_conv")),
            )
            for i in range(cfg.ff_n_layer)
        ])
        self.pre_ff_norm = make_rmsnorm(cfg.n_embd, cfg.use_liger_rmsnorm) if cfg.use_pre_ff_norm else None
        self.post_ff_norm = make_rmsnorm(cfg.n_embd, cfg.use_liger_rmsnorm) if cfg.use_post_ff_norm else None
        self.ff_draft_head = FFDraftHead(cfg.n_embd, vocab_size, cfg.use_liger_rmsnorm) if cfg.use_draft_head else None

        self.bp_blocks = nn.ModuleList([RevBlock(cfg, is_ff=False) for _ in range(cfg.bp_n_layer)])
        for blk in self.bp_blocks:
            nn.init.zeros_(blk.attn.c_proj.weight)
            nn.init.zeros_(blk.mlp.down.weight)

        self.final_ln = make_rmsnorm(cfg.n_embd, cfg.use_liger_rmsnorm)
        self.final_proj = make_linear(cfg.n_embd, cfg.n_embd, bias=False, use_bitnet=cfg.use_bitnet)
        # Apply BitNet EMA warmup override
        if cfg.use_bitnet:
            cfg.ff_ema_warmup_steps = cfg.ff_ema_warmup_steps_bitnet
        self.out_proj = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)

        # CPU_CTX v2 projection. Hash context emits low-dim CPU memory; project
        # to n_embd on GPU with a tiny trainable projection and gated scale.
        if getattr(cfg, "use_cpu_hash_context", False):
            self.cpu_ctx_proj = nn.Linear(int(cfg.cpu_context_dim), cfg.n_embd, bias=False)
            nn.init.normal_(self.cpu_ctx_proj.weight, std=1.0 / math.sqrt(int(cfg.cpu_context_dim)))
            self.cpu_ctx_gate = nn.Parameter(torch.tensor([float(cfg.cpu_context_gate_init)]))
        else:
            self.cpu_ctx_proj = None
            self.cpu_ctx_gate = None
        if getattr(cfg, "tie_token_embeddings", False):
            self.out_proj.weight = self.tok_emb.weight

        self.mem_compressor = ChunkMemoryCompressor(cfg.n_embd, cfg.memory_tokens, cfg.memory_gate, cfg.use_liger_rmsnorm) if cfg.memory_tokens > 0 else None

        self.use_engram = cfg.use_engram and cfg.engram_bank_size > 0
        if self.use_engram:
            self.engram_bank = EngramMemoryBank(cfg)
            self.engram_key_proj = nn.Linear(cfg.n_embd, cfg.engram_key_dim, bias=False)
            self.engram_val_proj = nn.Linear(cfg.n_embd, cfg.engram_key_dim, bias=False)
            self.engram_val_up = nn.Linear(cfg.engram_key_dim, cfg.n_embd, bias=False)
            gate_logit = math.log(max(1e-4, cfg.engram_gate_init) / max(1e-4, 1.0 - cfg.engram_gate_init))
            self.engram_gate = nn.Parameter(torch.tensor([gate_logit]))

        self.use_liger_ce = cfg.use_liger_ce and HAS_LIGER
        self.liger_fused_ce = LigerFusedLinearCrossEntropyLoss() if self.use_liger_ce and LigerFusedLinearCrossEntropyLoss is not None else None

        # EMA circuit-breaker buffers for FF layers only.
        self.register_buffer("ff_ema_mu", torch.zeros(cfg.ff_n_layer))
        self.register_buffer("ff_ema_var", torch.ones(cfg.ff_n_layer))
        self.register_buffer("ff_ema_seen", torch.zeros(cfg.ff_n_layer))

        self._current_step = 0
        self._last_final_ce = float("nan")
        self._last_draft_ce = float("nan")
        self._last_grad_norm = float("nan")
        self._working_mem = None
        self._engram_state = None
        self._last_draft_logits: Optional[torch.Tensor] = None
        self._draft_debug_printed = False

        self._diag: Dict[str, float] = {
            "engram_write_rate": float("nan"),
            "engram_valid_frac": float("nan"),
            "engram_retrieval_sim": float("nan"),
            "ff_act_norm_mean": float("nan"),
            "ff_act_norm_std": float("nan"),
            "ff_internal_norm": float("nan"),
            "ff_ema_trips": 0.0,
            "ff_ema_max_z": float("nan"),
            "ff_ema_mean_z": float("nan"),
            "ff_ema_goodness": float("nan"),
            "ff_ema_ce": float("nan"),
            "draft_ce_gap": float("nan"),
            "emb_clamp_frac": float("nan"),
            "emb_norm_mean": float("nan"),
            "emb_norm_std": float("nan"),
            "fullres_block_delta": float("nan"),
            "cpu_ctx_tokens": float("nan"),
            "cpu_ctx_gate": float("nan"),
        }

    @staticmethod
    def _build_ff_attention_schedule(cfg: CFG) -> set[int]:
        n = int(cfg.ff_n_layer)
        if n <= 0:
            return set()
        if not bool(getattr(cfg, "cpu_efficient_ff", False)):
            return set(range(n))

        every = max(1, int(getattr(cfg, "ff_attn_every", 1)))
        force_last = max(0, int(getattr(cfg, "ff_force_attn_last", 0)))
        layers = {i for i in range(n) if i % every == 0}
        if force_last > 0:
            layers.update(range(max(0, n - force_last), n))
        return layers

    @property
    def blend_gate(self) -> float:
        if self.ff_draft_head is None:
            return 1.0
        return float(torch.sigmoid(self.ff_draft_head.blend).item())

    @property
    def engram_gate_val(self) -> float:
        if not self.use_engram: return 0.0
        floor = self.cfg.engram_gate_floor
        return float((floor + (1.0 - floor) * torch.sigmoid(self.engram_gate[0])).item())

    def _floored_engram_gate(self, dtype) -> torch.Tensor:
        floor = self.cfg.engram_gate_floor
        return (floor + (1.0 - floor) * torch.sigmoid(self.engram_gate[0])).to(dtype=dtype)

    def _init_state(self, B: int, device, dtype):
        dtype = torch.bfloat16
        if self._working_mem is None or (self.mem_compressor and self._working_mem.size(0) != B):
            self._working_mem = self.mem_compressor.init_memory(B, device, dtype) if self.mem_compressor else None
        if self.use_engram and (self._engram_state is None or self._engram_state["keys"].size(0) != B):
            self._engram_state = self.engram_bank.init_state(B, device, dtype)
        return self._working_mem, self._engram_state

    def _draft_weight(self, current_lr: Optional[float] = None) -> float:
        if self.ff_draft_head is None or self.cfg.draft_weight <= 0:
            return 0.0
        warmup_factor = min(1.0, self._current_step / max(1, self.cfg.draft_warmup_steps))
        lr_factor = 1.0
        if current_lr is not None:
            lr_factor = min(1.0, (current_lr / max(1e-12, self.cfg.lr)) / 0.10)
        ce_factor = 1.0
        if (self._current_step > self.cfg.draft_warmup_steps and not math.isnan(self._last_draft_ce)
                and not math.isnan(self._last_final_ce) and self._last_draft_ce > self._last_final_ce + self.cfg.draft_ce_clamp_margin):
            ce_factor = 0.5
        return float(self.cfg.draft_weight * warmup_factor * lr_factor * ce_factor)

    def _use_draft_logit_blend(self) -> bool:
        return self.ff_draft_head is not None and os.environ.get("DRAFT_BLEND_BP", "1") != "0"

    def _use_draft_runtime(self) -> bool:
        return self.ff_draft_head is not None and (
            os.environ.get("DRAFT_BLEND_BP", "1") != "0"
            or os.environ.get("SPEC_DRAFT", "0") == "1"
        )

    def _draft_blend_alpha(self) -> float:
        try:
            return float(os.environ.get("DRAFT_BLEND_ALPHA", "0.0"))
        except ValueError:
            return 0.0

    def _ff_skip_enabled(self) -> bool:
        return (
            os.environ.get("USE_FF_SKIP", "0") == "1"
            or os.environ.get("USE_FF_DRAFT_SKIP", "0") == "1"
        )

    def _ff_skip_mode(self) -> str:
        mode = os.environ.get("FF_SKIP_MODE", "block").strip().lower()
        if mode in {"mlp", "ffn", "feedforward", "feed_forward"}:
            return "mlp"
        return "block"

    def _ff_skip_indices(self) -> set[int]:
        if not self._ff_skip_enabled():
            return set()
        raw = os.environ.get("FF_SKIP_LAYERS", "").strip()
        if not raw:
            return set()
        n_layers = len(self.ff_blocks)
        if raw.lower() == "all":
            return set(range(n_layers))
        out: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part)
            except ValueError:
                continue
            if idx < 0:
                idx = n_layers + idx
            if 0 <= idx < n_layers:
                out.add(idx)
        return out

    def _should_skip_ff_layer(self, layer_idx: int, skip_indices: Optional[set[int]] = None) -> bool:
        if skip_indices is None:
            skip_indices = self._ff_skip_indices()
        return int(layer_idx) in skip_indices

    def _run_ff_block_with_skip(self, blk, x: torch.Tensor, mem, pos_offset: int, skip: bool) -> torch.Tensor:
        if not skip:
            return blk(x, mem=mem, pos_offset=pos_offset)
        if self._ff_skip_mode() == "mlp" and hasattr(blk, "forward_skip_mlp"):
            return blk.forward_skip_mlp(x, mem=mem, pos_offset=pos_offset)
        return x

    def _run_ff_block_kv_with_skip(
        self,
        blk,
        x: torch.Tensor,
        cache,
        pos_offset: int,
        max_cache_len: int,
        skip: bool,
    ):
        if not skip:
            return blk.forward_kv(x, cache=cache, pos_offset=pos_offset, max_cache_len=max_cache_len)
        if self._ff_skip_mode() == "mlp" and hasattr(blk, "forward_kv_skip_mlp"):
            return blk.forward_kv_skip_mlp(x, cache=cache, pos_offset=pos_offset, max_cache_len=max_cache_len)
        return x, None

    def _runtime_kv_cache_len(self) -> int:
        raw = os.environ.get("KV_CACHE_MAX_LEN", None)
        if raw is None:
            value = int(getattr(self.cfg, "kv_cache_max_len", 0) or 0)
        else:
            try:
                value = int(raw)
            except ValueError:
                value = 0
        if value <= 0:
            return int(self.cfg.block_size)
        return min(int(value), int(self.cfg.block_size))

    def _normalize_draft_logits(self, base_logits: torch.Tensor, draft_logits: torch.Tensor) -> torch.Tensor:
        draft_f = draft_logits.float()
        if os.environ.get("DRAFT_BLEND_NORM", "1") != "0":
            base_std = base_logits.float().std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            draft_mean = draft_f.mean(dim=-1, keepdim=True)
            draft_std = draft_f.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            delta = (draft_f - draft_mean) / draft_std * base_std
        else:
            delta = draft_f

        try:
            clamp = float(os.environ.get("DRAFT_BLEND_CLAMP", "3.0"))
        except ValueError:
            clamp = 3.0
        if clamp > 0:
            delta = delta.clamp(min=-clamp, max=clamp)
        return delta.to(dtype=base_logits.dtype)

    def _format_topk(self, logits: torch.Tensor, k: int = 10) -> str:
        row = logits.detach().float().reshape(-1, logits.size(-1))[-1]
        vals, ids = torch.topk(row, k=min(k, row.numel()))
        tok = getattr(self, "_debug_tokenizer", None)
        parts = []
        for tid, val in zip(ids.tolist(), vals.tolist()):
            if tok is not None:
                try:
                    text = repr(tok.decode([int(tid)], skip_special_tokens=False))
                except Exception:
                    text = str(tid)
            else:
                text = str(tid)
            parts.append(f"{text}:{val:.3f}")
        return ", ".join(parts)

    def _debug_draft_logits(
        self,
        base_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_delta: torch.Tensor,
        final_logits: torch.Tensor,
    ) -> None:
        if os.environ.get("DRAFT_DEBUG_TOPK", "0") != "1" or self._draft_debug_printed:
            return
        self._draft_debug_printed = True

        def std(x: torch.Tensor) -> float:
            return float(x.detach().float().reshape(-1, x.size(-1))[-1].std(unbiased=False))

        print("[DRAFT DEBUG] base top10:", self._format_topk(base_logits), flush=True)
        print("[DRAFT DEBUG] draft raw top10:", self._format_topk(draft_logits), flush=True)
        print("[DRAFT DEBUG] draft normalized top10:", self._format_topk(draft_delta), flush=True)
        print("[DRAFT DEBUG] final blended top10:", self._format_topk(final_logits), flush=True)
        print(
            f"[DRAFT DEBUG] std base={std(base_logits):.6f} "
            f"draft={std(draft_logits):.6f} final={std(final_logits):.6f}",
            flush=True,
        )

    def _blend_draft_logits(
        self,
        base_logits: torch.Tensor,
        draft_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._use_draft_logit_blend():
            return base_logits
        if draft_logits is None:
            draft_logits = self._last_draft_logits
        if draft_logits is None:
            return base_logits

        alpha = self._draft_blend_alpha()
        debug = os.environ.get("DRAFT_DEBUG_TOPK", "0") == "1"
        if alpha == 0.0 and not debug:
            return base_logits

        if draft_logits.dim() == base_logits.dim() + 1 and base_logits.dim() == 2:
            draft_logits = draft_logits[:, -1, :]
        if draft_logits.shape[:-1] != base_logits.shape[:-1]:
            if draft_logits.dim() == base_logits.dim() == 3:
                draft_logits = draft_logits[..., -base_logits.size(1):, :]
            else:
                return base_logits
        draft_delta = self._normalize_draft_logits(base_logits, draft_logits)
        final_logits = base_logits + float(alpha) * draft_delta
        self._debug_draft_logits(base_logits, draft_logits, draft_delta, final_logits)
        return final_logits

    def _draft_ce(self, draft_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(draft_logits.reshape(-1, self.vocab_size), y.reshape(-1).long())

    def _final_ce(self, h_flat: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
        E = self.out_proj.weight
        scale = self.head_scale
        if self.use_liger_ce and self.liger_fused_ce is not None:
            try:
                out = self.liger_fused_ce(E, h_flat * scale, y_flat.long())
                loss = out.loss if hasattr(out, "loss") else (out[0] if isinstance(out, tuple) else out)
                if torch.is_tensor(loss): return loss
            except Exception:
                pass
        ce_sum = h_flat.new_zeros(())
        for t0 in range(0, h_flat.size(0), self.cfg.head_ce_chunk):
            t1 = min(t0 + self.cfg.head_ce_chunk, h_flat.size(0))
            logits = (h_flat[t0:t1] @ E.T) * scale
            ce_sum = ce_sum + F.cross_entropy(logits, y_flat[t0:t1].long(), reduction="sum")
        return ce_sum / max(1, h_flat.size(0))

    @staticmethod
    def _draft_entropy_scores(draft_logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = torch.softmax(draft_logits.detach().float(), dim=-1)
            entropy = -(probs * probs.clamp(min=1e-9).log()).sum(-1)
            scores = entropy.mean(dim=1, keepdim=True)
            return (scores / scores.mean().clamp(min=1e-6)).clamp(0.0, 2.0)

    def _ff_ema_check(self, layer_idx: int, x_out: torch.Tensor):
        """Compatibility wrapper for single-layer EMA checks."""
        trips, goodness, z = self._ff_ema_check_many(
            x_out.detach().pow(2).mean().sqrt().view(1),
            layer_indices=torch.tensor([layer_idx], device=self.ff_ema_mu.device),
        )
        return bool(trips.detach().cpu()[0]), float(goodness.detach().float().cpu()[0]), float(z.detach().float().cpu()[0])

    @torch.no_grad()
    def _ff_ema_check_many(
        self,
        goodnesses: torch.Tensor,
        layer_indices: Optional[torch.Tensor] = None,
    ):
        """
        Vectorised FF EMA trip check.

        The previous hot path called _ff_ema_check once per FF layer and used
        .item()/bool()/float() in each call. On CUDA that drains the pipeline per
        layer per chunk. This updates all touched layers with one tensor op and
        performs a single host transfer for the trip decisions/diagnostics.
        """
        if layer_indices is None:
            layer_indices = torch.arange(goodnesses.numel(), device=self.ff_ema_mu.device)
        else:
            layer_indices = layer_indices.to(device=self.ff_ema_mu.device, dtype=torch.long)

        g = goodnesses.detach().to(device=self.ff_ema_mu.device, dtype=self.ff_ema_mu.dtype).flatten()
        mu = self.ff_ema_mu.index_select(0, layer_indices)
        var = self.ff_ema_var.index_select(0, layer_indices)
        seen = self.ff_ema_seen.index_select(0, layer_indices)
        first = seen < 1

        alpha = float(self.cfg.ff_ema_alpha)
        new_mu_raw = (1.0 - alpha) * mu + alpha * g
        new_var_raw = (1.0 - alpha) * var + alpha * (g - mu) * (g - new_mu_raw)
        new_mu = torch.where(first, g, new_mu_raw)
        new_var = torch.where(first, torch.ones_like(new_var_raw), new_var_raw.clamp_min(1e-8))

        self.ff_ema_mu.index_copy_(0, layer_indices, new_mu)
        self.ff_ema_var.index_copy_(0, layer_indices, new_var)
        self.ff_ema_seen.index_copy_(0, layer_indices, seen + 1)

        delta = (g - new_mu).abs()
        z = delta / new_var.sqrt().clamp_min(1e-6)
        if self._current_step < self.cfg.ff_ema_warmup_steps:
            trips = torch.zeros_like(z, dtype=torch.bool)
        else:
            trips = (~first) & (delta > float(self.cfg.ff_ema_min_abs_delta)) & (z > float(self.cfg.ff_ema_std_mult))

        return trips.detach(), g.detach(), z.detach()

    def _ff_block_forward_standard(
        self,
        blk,
        x_in: torch.Tensor,
        mem: Optional[torch.Tensor],
        pos_offset: int,
    ) -> torch.Tensor:
        """Full-width single-block forward used only for EMA CE strikes."""
        x_local = x_in.detach()

        if mem is None or mem.size(1) == 0:
            mem_local = None
        else:
            mem_local = mem.detach()
            if mem_local.device != x_local.device or mem_local.dtype != x_local.dtype:
                mem_local = mem_local.to(device=x_local.device, dtype=x_local.dtype)

        return blk(x_local, mem=mem_local, pos_offset=pos_offset)

    def _ff_layer_ce_strike(
        self,
        blk,
        x_in: torch.Tensor,
        y_chunk: torch.Tensor,
        mem: Optional[torch.Tensor],
        pos_offset: int,
        loss_scale: float,
    ) -> float:
        """
        Surgical CE strike for one FF layer.

        Important:
          - recomputes the tripped FF block using normal autograd
          - detaches x_in and mem, so gradients touch only this block's params
          - detaches final/head weights, so the CE probe does not train the head
        """
        x_probe = self._ff_block_forward_standard(
            blk=blk,
            x_in=x_in,
            mem=mem,
            pos_offset=pos_offset,
        )

        h = x_probe.reshape(-1, self.n_embd)
        if hasattr(self.final_ln, "weight"):
            h = F.rms_norm(
                h,
                (self.n_embd,),
                self.final_ln.weight.detach(),
                eps=getattr(self.final_ln, "eps", 1e-6),
            )
        else:
            h = F.rms_norm(h, (self.n_embd,), eps=1e-6)

        logits = F.linear(h, self.out_proj.weight.detach()) * self.head_scale
        layer_loss = F.cross_entropy(logits, y_chunk.reshape(-1).long())

        (layer_loss * loss_scale * float(self.cfg.ff_ema_bp_weight)).backward()
        return float(layer_loss.detach())

    @torch.no_grad()
    def check_fullres_block_delta(self, x_sample: torch.Tensor) -> float:
        """Lightweight sanity diagnostic for the full-width residual FF stack."""
        self.eval()
        deltas: List[float] = []
        x = x_sample.detach()
        if self.pre_ff_norm is not None:
            x = self.pre_ff_norm(x)
        for blk in self.ff_blocks:
            out = blk(x, None, 0)
            denom = x.norm().clamp(min=1e-6)
            deltas.append(float(((out - x).norm() / denom).item()))
            x = out
        self.train()
        return float(np.mean(deltas)) if deltas else float("nan")

    # Compatibility name for old train.py / scripts. This is no longer a
    # reversible reconstruction check after RevNet removal.
    check_revnet_reconstruction = check_fullres_block_delta

    def forward_features(self, xb: torch.Tensor, yb: Optional[torch.Tensor] = None, update_state: Optional[bool] = None,
                         current_lr: Optional[float] = None, external_mem: Optional[torch.Tensor] = None):
        B, T = xb.shape
        device = xb.device
        dtype = self.tok_emb.scale.dtype if hasattr(self.tok_emb, "scale") else self.tok_emb.weight.dtype
        _update = self.training if update_state is None else update_state
        is_training = yb is not None and self.training and torch.is_grad_enabled()
        collect_norm_diag = (not is_training) or os.environ.get("COLLECT_TRAIN_NORM_DIAG", "0") == "1"
        collect_ema_diag = (not is_training) or os.environ.get("COLLECT_TRAIN_EMA_DIAG", "0") == "1"
        retain_graph_steps = self.cfg.retain_graph_steps
        self._last_draft_logits = None

        x_emb = self.tok_emb(xb)
        # Hoist pre-FF norm out of the chunk loop. This removes one norm dispatch
        # per chunk and keeps curriculum/chunking from multiplying tiny kernels.
        x_ff_input = self.pre_ff_norm(x_emb) if self.pre_ff_norm is not None else x_emb
        working_mem, engram_state = self._init_state(B, device, dtype)
        ff_draft_hiddens: List[torch.Tensor] = []
        draft_ce_accum = x_emb.new_zeros(())
        n_chunks = 0
        _ff_act_norms: List[float] = []
        trip_count = 0
        z_values: List[float] = []
        goodness_values: List[float] = []
        strike_ce_values: List[float] = []

        if external_mem is not None:
            self._diag["cpu_ctx_tokens"] = float(external_mem.size(1))

            # Move CPU context to GPU once, not inside every chunk.
            external_mem = external_mem.to(device=device, dtype=dtype, non_blocking=True)

            # CPU_HASH_CTX emits [B, M, CPU_CTX_DIM]. Project/gate to n_embd.
            if self.cpu_ctx_proj is not None and external_mem.size(-1) != self.n_embd:
                external_mem = self.cpu_ctx_proj(external_mem)
                gate = torch.sigmoid(self.cpu_ctx_gate[0]).to(dtype=external_mem.dtype)
                external_mem = external_mem * gate
                self._diag["cpu_ctx_gate"] = float(gate.detach().float().item())
            else:
                self._diag["cpu_ctx_gate"] = 1.0

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for t0 in range(0, T, self.cfg.seq_chunk_size):
                t1 = min(t0 + self.cfg.seq_chunk_size, T)
                # Slicing [B, T, C] by time produces a non-contiguous view
                # because the batch stride remains the full original T. Make
                # the chunk contiguous once so the first norm/linear does not
                # trigger hidden layout copies.
                x_chunk = x_ff_input[:, t0:t1].contiguous()
                y_chunk = yb[:, t0:t1].contiguous() if yb is not None else None
                if working_mem is not None and (working_mem.device != device or working_mem.dtype != dtype):
                    working_mem = working_mem.to(device=device, dtype=dtype, non_blocking=True)
                wm_gpu = working_mem

                engram_feat = x_chunk.detach().mean(dim=1)

                engram_mem = None
                if self.use_engram and engram_state is not None:
                    q_key_detached = F.normalize(self.engram_key_proj(engram_feat), p=2, dim=-1)
                    retrieved_tuple = self.engram_bank.retrieve(engram_state, q_key_detached)
                    if retrieved_tuple is not None:
                        r_keys, r_vals = retrieved_tuple
                        r_vals_projected = self.engram_val_proj(r_vals)
                        q_feat_grad = x_chunk.mean(dim=1)
                        q_key_grad = F.normalize(self.engram_key_proj(q_feat_grad), p=2, dim=-1)
                        sim = torch.bmm(q_key_grad.unsqueeze(1).to(dtype=dtype), r_keys.transpose(1, 2).to(dtype=dtype))
                        attn_weights = F.softmax(sim * self.engram_attn_scale, dim=-1)
                        attended_val = torch.bmm(attn_weights, r_vals_projected.to(dtype=dtype))
                        up = self.engram_val_up(attended_val.squeeze(1))
                        engram_mem = up.unsqueeze(1).expand(-1, min(4, r_vals.size(1)), -1) * self._floored_engram_gate(dtype)

                mem_parts = []
                if external_mem is not None: mem_parts.append(external_mem)
                if wm_gpu is not None: mem_parts.append(wm_gpu)
                if engram_mem is not None: mem_parts.append(engram_mem)
                mem = torch.cat(mem_parts, dim=1) if mem_parts else None

                ema_inputs: List[torch.Tensor] = []
                ema_goodness_tensors: List[torch.Tensor] = []
                train_with_skip = os.environ.get("TRAIN_WITH_FF_SKIP", "0") == "1"
                skip_indices = self._ff_skip_indices() if ((not is_training and yb is None) or train_with_skip) else set()
                for li, blk in enumerate(self.ff_blocks):
                    if self._should_skip_ff_layer(li, skip_indices):
                        x_chunk = self._run_ff_block_with_skip(blk, x_chunk, mem, t0, skip=True)
                        continue
                    if is_training and self.cfg.use_ff_ema_bp and self.cfg.ff_ema_detach_layers:
                        x_chunk = x_chunk.detach()
                    if is_training and self.cfg.use_ff_ema_bp:
                        # Store a detached activation for the optional CE strike.
                        # _ff_layer_ce_strike detaches internally too; doing it here
                        # avoids retaining the main graph for every FF layer.
                        ema_inputs.append(x_chunk.detach())
                    x_out = blk(x_chunk, mem=mem, pos_offset=t0)

                    if is_training and self.cfg.use_ff_ema_bp:
                        ema_goodness_tensors.append(x_out.detach().pow(2).mean().sqrt())
                    x_chunk = x_out.detach() if (is_training and self.cfg.use_ff_ema_bp and self.cfg.ff_ema_detach_layers) else x_out

                if is_training and self.cfg.use_ff_ema_bp and ema_goodness_tensors:
                    trips, g_vals, z_vals = self._ff_ema_check_many(torch.stack(ema_goodness_tensors))
                    if collect_ema_diag:
                        z_values.extend(float(v) for v in z_vals.detach().float().cpu().tolist())
                        goodness_values.extend(float(v) for v in g_vals.detach().float().cpu().tolist())
                    if y_chunk is not None and self._current_step >= self.cfg.ff_ema_warmup_steps:
                        for li, trip in enumerate(trips.detach().cpu().tolist()):
                            if trip_count >= int(self.cfg.ff_ema_max_trips_per_step):
                                break
                            if trip:
                                trip_count += 1
                                strike_ce_values.append(self._ff_layer_ce_strike(
                                    blk=self.ff_blocks[li],
                                    x_in=ema_inputs[li],
                                    y_chunk=y_chunk,
                                    mem=mem,
                                    pos_offset=t0,
                                    loss_scale=1.0 / max(1, self.cfg.grad_accum_steps),
                                ))

                x_ff_raw = x_chunk
                if collect_norm_diag:
                    with torch.no_grad():
                        self._diag["ff_internal_norm"] = float(x_ff_raw.detach().norm(dim=-1).mean().item())

                if self.post_ff_norm is not None:
                    x_chunk = self.post_ff_norm(x_chunk)
                if collect_norm_diag:
                    with torch.no_grad():
                        _ff_act_norms.append(float(x_chunk.detach().norm(dim=-1).mean().item()))

                if self.ff_draft_head is not None:
                    draft_logits_chunk, _draft_hidden_chunk = self.ff_draft_head(x_chunk)
                    ff_draft_hiddens.append(x_chunk)

                    if yb is not None:
                        draft_ce_accum = draft_ce_accum + self._draft_ce(draft_logits_chunk, y_chunk)
                        n_chunks += 1
                else:
                    # No draft head: pass post-FF features directly into BP stack.
                    ff_draft_hiddens.append(x_chunk)
                    if yb is not None:
                        n_chunks += 1

                if self.mem_compressor is not None:
                    working_mem = self.mem_compressor(wm_gpu, x_chunk)
                    if is_training and (n_chunks + 1) % retain_graph_steps == 0:
                        working_mem = working_mem.detach()

                if self.use_engram and _update and engram_state is not None:
                    with torch.no_grad():
                        keys = F.normalize(self.engram_key_proj(engram_feat), p=2, dim=-1).unsqueeze(1)
                        values = F.normalize(x_ff_raw.detach().mean(dim=1), p=2, dim=-1).unsqueeze(1)
                        if self.ff_draft_head is not None:
                            scores = self._draft_entropy_scores(draft_logits_chunk)
                        else:
                            scores = torch.ones((x_chunk.size(0), 1), device=x_chunk.device, dtype=torch.float32)
                        self.engram_bank.write(engram_state, keys, values, scores)

        if _update:
            self._working_mem = working_mem.detach() if working_mem is not None else None
            self._engram_state = engram_state

        if _ff_act_norms:
            self._diag["ff_act_norm_mean"] = float(np.mean(_ff_act_norms))
            self._diag["ff_act_norm_std"] = float(np.std(_ff_act_norms))
        self._diag["ff_ema_trips"] = float(trip_count)
        self._diag["ff_ema_max_z"] = max(z_values) if z_values else float("nan")
        self._diag["ff_ema_mean_z"] = sum(z_values) / max(1, len(z_values)) if z_values else float("nan")
        self._diag["ff_ema_goodness"] = sum(goodness_values) / max(1, len(goodness_values)) if goodness_values else float("nan")
        self._diag["ff_ema_ce"] = sum(strike_ce_values) / max(1, len(strike_ce_values)) if strike_ce_values else float("nan")
        if self.use_engram:
            self._diag["engram_write_rate"] = self.engram_bank._last_write_rate
            self._diag["engram_valid_frac"] = self.engram_bank._last_valid_frac
            self._diag["engram_retrieval_sim"] = self.engram_bank._last_retrieval_sim

        x_draft = torch.cat(ff_draft_hiddens, dim=1)
        if self.ff_draft_head is not None and yb is None and self._use_draft_logit_blend():
            self._last_draft_logits = torch.cat(
                [self.ff_draft_head(h)[0] for h in ff_draft_hiddens],
                dim=1,
            )
        # In SFT mode the FF trunk is frozen. Detach here so backprop stops at
        # the FF/BP boundary: avoids wasted gradient computation through frozen
        # FF parameters and prevents a double-gradient on tied embeddings
        # (tok_emb.weight would otherwise receive gradients from both the output
        # projection path and the embedding input path through the FF trunk).
        if getattr(self.cfg, 'sft_detach_ff', False):
            x_draft = x_draft.detach()
        x_full = x_draft

        for blk in self.bp_blocks:
            if (
                self.training
                and torch.is_grad_enabled()
                and getattr(self.cfg, "bp_checkpoint", True)
            ):
                x_full = checkpoint(
                    lambda _x, _blk=blk: _blk(_x, mem=None, pos_offset=0),
                    x_full,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                x_full = blk(x_full, mem=None, pos_offset=0)

        if yb is not None:
            h_flat = self.final_proj(self.final_ln(x_full)).reshape(-1, self.n_embd)
            y_flat = yb.reshape(-1).long()
            final_ce = self._final_ce(h_flat, y_flat)
            final_ce_val = float(final_ce.detach())
            dw = self._draft_weight(current_lr=current_lr)
            if self.ff_draft_head is not None:
                avg_draft = draft_ce_accum / max(1, n_chunks)
                self._last_draft_ce = float(avg_draft.detach())
                self._diag["draft_ce_gap"] = self._last_draft_ce - final_ce_val
                total_loss = self.cfg.final_ce_weight * final_ce + dw * avg_draft
            else:
                avg_draft = final_ce.detach() * float("nan")
                self._last_draft_ce = float("nan")
                self._diag["draft_ce_gap"] = float("nan")
                total_loss = self.cfg.final_ce_weight * final_ce

            self._last_final_ce = final_ce_val
            return total_loss
        return x_full, None

    @torch.inference_mode()
    def forward_features_eval(self, xb: torch.Tensor, *,
                              external_mem: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single clean forward pass for eval/logprob scoring.

        Mirrors the training forward_features dataflow (FF blocks, optional draft
        head, memory compressor, engram, CPU context, BP blocks) but without
        chunking, EMA, diag writes, or autograd.
        """
        B, T = xb.shape
        dtype = self.tok_emb.scale.dtype if hasattr(self.tok_emb, "scale") else self.tok_emb.weight.dtype
        device = xb.device

        x_emb = self.tok_emb(xb)
        x = self.pre_ff_norm(x_emb) if self.pre_ff_norm is not None else x_emb
        self._last_draft_logits = None

        # --- init memory / engram state (no-op if already sized correctly) ---
        self._init_state(B, device, dtype)

        # --- external / CPU context ---
        if external_mem is not None:
            external_mem = external_mem.to(device=device, dtype=dtype)
            if self.cpu_ctx_proj is not None and external_mem.size(-1) != self.n_embd:
                external_mem = self.cpu_ctx_proj(external_mem)
                gate = torch.sigmoid(self.cpu_ctx_gate[0]).to(dtype=dtype)
                external_mem = external_mem * gate

        # --- engram retrieval ---
        engram_mem = None
        if self.use_engram and self._engram_state is not None:
            engram_feat = x.detach().mean(dim=1)
            q_key = F.normalize(self.engram_key_proj(engram_feat), p=2, dim=-1)
            retrieved = self.engram_bank.retrieve(self._engram_state, q_key)
            if retrieved is not None:
                r_keys, r_vals = retrieved
                r_vals_proj = self.engram_val_proj(r_vals)
                sim = torch.bmm(q_key.unsqueeze(1).to(dtype=dtype),
                                r_keys.transpose(1, 2).to(dtype=dtype))
                attn_w = F.softmax(sim * self.engram_attn_scale, dim=-1)
                attn_val = torch.bmm(attn_w, r_vals_proj.to(dtype=dtype))
                up = self.engram_val_up(attn_val.squeeze(1))
                engram_mem = up.unsqueeze(1).expand(-1, min(4, r_vals.size(1)), -1)
                engram_mem = engram_mem * self._floored_engram_gate(dtype)

        # --- assemble memory for FF attention ---
        mem_parts = []
        if external_mem is not None:
            mem_parts.append(external_mem)
        if self._working_mem is not None:
            wm = self._working_mem.to(device=device, dtype=dtype) if (
                self._working_mem.device != device or self._working_mem.dtype != dtype
            ) else self._working_mem
            mem_parts.append(wm)
        if engram_mem is not None:
            mem_parts.append(engram_mem)
        mem = torch.cat(mem_parts, dim=1) if mem_parts else None

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            skip_indices = self._ff_skip_indices()
            for li, blk in enumerate(self.ff_blocks):
                if self._should_skip_ff_layer(li, skip_indices):
                    x = self._run_ff_block_with_skip(blk, x, mem, 0, skip=True)
                    continue
                x = self._run_ff_block_with_skip(blk, x, mem, 0, skip=False)

            if self.post_ff_norm is not None:
                x = self.post_ff_norm(x)

            # --- draft head ---
            if self._use_draft_logit_blend():
                self._last_draft_logits, _draft_hidden = self.ff_draft_head(x)
            x_full = x

            for blk in self.bp_blocks:
                x_full = blk(x_full, mem=None, pos_offset=0)

        return x_full

    def _get_logits(self, x_full: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits_base(x_full)
        return self._blend_draft_logits(logits)

    def _get_logits_base(self, x_full: torch.Tensor) -> torch.Tensor:
        h = self.final_proj(self.final_ln(x_full))
        if hasattr(self.out_proj, "packed") and hasattr(self.out_proj, "scale"):
            return self.out_proj(h) * self.head_scale
        E = self.out_proj.weight
        scale = self.head_scale
        flat = h.reshape(-1, self.n_embd)
        if E.device != flat.device:
            flat_cpu = flat.detach().to(device=E.device, dtype=torch.float32)
            E_cpu = E.float()
            parts = [(flat_cpu[t0:min(t0 + self.cfg.eval_logit_chunk, flat_cpu.size(0))] @ E_cpu.T) * scale
                     for t0 in range(0, flat_cpu.size(0), self.cfg.eval_logit_chunk)]
            logits = torch.cat(parts, dim=0).view(x_full.size(0), x_full.size(1), self.vocab_size)
            return logits.to(device=x_full.device)
        parts = [(flat[t0:min(t0 + self.cfg.eval_logit_chunk, flat.size(0))] @ E.T) * scale
                 for t0 in range(0, flat.size(0), self.cfg.eval_logit_chunk)]
        return torch.cat(parts, dim=0).view(x_full.size(0), x_full.size(1), self.vocab_size)

    def _get_last_logits(self, x_full: torch.Tensor) -> torch.Tensor:
        """Generation fast path: compute vocab logits for only the final position."""
        logits = self._get_last_logits_base(x_full)
        return self._blend_draft_logits(logits)

    def _get_last_logits_base(self, x_full: torch.Tensor) -> torch.Tensor:
        h = self.final_proj(self.final_ln(x_full[:, -1:, :])).reshape(-1, self.n_embd)
        if hasattr(self.out_proj, "packed") and hasattr(self.out_proj, "scale"):
            return self.out_proj(h) * self.head_scale
        if self.out_proj.weight.device != h.device:
            logits = (h.detach().to(device=self.out_proj.weight.device, dtype=torch.float32) @ self.out_proj.weight.float().T) * self.head_scale
            return logits.to(device=h.device)
        return (h @ self.out_proj.weight.T) * self.head_scale


    @torch.inference_mode()
    def prefill_kv(self, idx: torch.Tensor, *, blend_logits: bool = True):
        """
        Build per-layer FF/BP KV caches from a prompt and return next-token logits.

        This fast path intentionally assumes standard attention only. Disable
        external memory/engram/CPU-context for clean, shareable inference.
        """
        was_training = self.training
        self.eval()
        self._working_mem = None
        self._engram_state = None
        self._last_draft_logits = None

        B, T = idx.shape
        x_emb = self.tok_emb(idx)
        x = self.pre_ff_norm(x_emb) if self.pre_ff_norm is not None else x_emb
        max_cache_len = self._runtime_kv_cache_len()
        ff_cache = []
        draft_logits = None

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            skip_indices = self._ff_skip_indices()
            for li, blk in enumerate(self.ff_blocks):
                if self._should_skip_ff_layer(li, skip_indices):
                    x, c = self._run_ff_block_kv_with_skip(blk, x, None, 0, max_cache_len, skip=True)
                    ff_cache.append(c)
                    continue
                x, c = self._run_ff_block_kv_with_skip(blk, x, None, 0, max_cache_len, skip=False)
                ff_cache.append(c)

            if self.post_ff_norm is not None:
                x = self.post_ff_norm(x)

            # Draft head is optional. Stable runtime normally keeps it disabled.
            if self._use_draft_runtime():
                draft_logits, _draft_hidden = self.ff_draft_head(x)
            x_full = x

            bp_cache = []
            for blk in self.bp_blocks:
                x_full, c = blk.forward_kv(x_full, cache=None, pos_offset=0, max_cache_len=max_cache_len)
                bp_cache.append(c)

        logits = self._get_last_logits(x_full) if blend_logits else self._get_last_logits_base(x_full)
        if blend_logits and draft_logits is not None:
            logits = self._blend_draft_logits(logits, draft_logits[:, -1, :])
        cache = {"ff": ff_cache, "bp": bp_cache, "pos": int(T), "max_cache_len": max_cache_len}
        if draft_logits is not None:
            cache["draft_logits"] = draft_logits[:, -1, :].detach()
        self.train(was_training)
        return logits, cache

    @torch.inference_mode()
    def decode_one_kv(self, next_token: torch.Tensor, cache: Dict, *, blend_logits: bool = True):
        """Decode exactly one token using existing per-layer K/V caches."""
        was_training = self.training
        self.eval()
        self._working_mem = None
        self._engram_state = None
        self._last_draft_logits = None

        pos = int(cache.get("pos", 0))
        max_cache_len = int(cache.get("max_cache_len", self._runtime_kv_cache_len()))
        x_emb = self.tok_emb(next_token)
        x = self.pre_ff_norm(x_emb) if self.pre_ff_norm is not None else x_emb
        new_ff = []
        draft_logits = None

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            skip_indices = self._ff_skip_indices()
            for li, (blk, old_c) in enumerate(zip(self.ff_blocks, cache["ff"])):
                if self._should_skip_ff_layer(li, skip_indices):
                    x, c = self._run_ff_block_kv_with_skip(blk, x, old_c, pos, max_cache_len, skip=True)
                    new_ff.append(c)
                    continue
                if old_c is None:
                    new_ff.append(old_c)
                    continue
                x, c = self._run_ff_block_kv_with_skip(blk, x, old_c, pos, max_cache_len, skip=False)
                new_ff.append(c)

            if self.post_ff_norm is not None:
                x = self.post_ff_norm(x)

            if self._use_draft_runtime():
                draft_logits, _draft_hidden = self.ff_draft_head(x)
            x_full = x

            new_bp = []
            for blk, old_c in zip(self.bp_blocks, cache["bp"]):
                x_full, c = blk.forward_kv(x_full, cache=old_c, pos_offset=pos, max_cache_len=max_cache_len)
                new_bp.append(c)

        cache["ff"] = new_ff
        cache["bp"] = new_bp
        cache["pos"] = pos + 1
        logits = self._get_last_logits(x_full) if blend_logits else self._get_last_logits_base(x_full)
        if blend_logits and draft_logits is not None:
            logits = self._blend_draft_logits(logits, draft_logits[:, -1, :])
        if draft_logits is not None:
            cache["draft_logits"] = draft_logits[:, -1, :].detach()
        self.train(was_training)
        return logits, cache

    @torch.inference_mode()
    def decode_many_kv(self, tokens: torch.Tensor, cache: Dict, *, blend_logits: bool = False):
        """Verify a sequence of tokens with one cached base-model pass."""
        was_training = self.training
        self.eval()
        self._working_mem = None
        self._engram_state = None
        self._last_draft_logits = None

        pos = int(cache.get("pos", 0))
        max_cache_len = int(cache.get("max_cache_len", self._runtime_kv_cache_len()))
        x = self.tok_emb(tokens)
        x = self.pre_ff_norm(x) if self.pre_ff_norm is not None else x
        new_ff = []
        draft_logits = None

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            skip_indices = self._ff_skip_indices()
            for li, (blk, old_c) in enumerate(zip(self.ff_blocks, cache["ff"])):
                if self._should_skip_ff_layer(li, skip_indices):
                    x, c = self._run_ff_block_kv_with_skip(blk, x, old_c, pos, max_cache_len, skip=True)
                    new_ff.append(c)
                    continue
                if old_c is None:
                    new_ff.append(old_c)
                    continue
                x, c = self._run_ff_block_kv_with_skip(blk, x, old_c, pos, max_cache_len, skip=False)
                new_ff.append(c)

            if self.post_ff_norm is not None:
                x = self.post_ff_norm(x)

            if self._use_draft_runtime():
                draft_logits, _draft_hidden = self.ff_draft_head(x)
            x_full = x

            new_bp = []
            for blk, old_c in zip(self.bp_blocks, cache["bp"]):
                x_full, c = blk.forward_kv(x_full, cache=old_c, pos_offset=pos, max_cache_len=max_cache_len)
                new_bp.append(c)

        new_cache = dict(cache)
        new_cache["ff"] = new_ff
        new_cache["bp"] = new_bp
        new_cache["pos"] = pos + int(tokens.size(1))
        logits = self._get_logits(x_full) if blend_logits else self._get_logits_base(x_full)
        if blend_logits and draft_logits is not None:
            logits = self._blend_draft_logits(logits, draft_logits)
        if draft_logits is not None:
            new_cache["draft_logits"] = draft_logits[:, -1, :].detach()
        self.train(was_training)
        return logits, new_cache

    @torch.inference_mode()
    def decode_one_draft_kv(self, next_token: torch.Tensor, cache: Dict):
        """Advance only the FF trunk and return ff_draft_head logits."""
        if self.ff_draft_head is None:
            raise RuntimeError("SPEC_DRAFT requires USE_DRAFT_HEAD=1 and ff_draft_head weights")

        was_training = self.training
        self.eval()
        pos = int(cache.get("pos", 0))
        max_cache_len = int(cache.get("max_cache_len", self._runtime_kv_cache_len()))
        x = self.tok_emb(next_token)
        x = self.pre_ff_norm(x) if self.pre_ff_norm is not None else x
        new_ff = []

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            skip_indices = self._ff_skip_indices()
            for li, (blk, old_c) in enumerate(zip(self.ff_blocks, cache["ff"])):
                if self._should_skip_ff_layer(li, skip_indices):
                    x, c = self._run_ff_block_kv_with_skip(blk, x, old_c, pos, max_cache_len, skip=True)
                    new_ff.append(c)
                    continue
                if old_c is None:
                    new_ff.append(old_c)
                    continue
                x, c = self._run_ff_block_kv_with_skip(blk, x, old_c, pos, max_cache_len, skip=False)
                new_ff.append(c)
            if self.post_ff_norm is not None:
                x = self.post_ff_norm(x)
            draft_logits, _draft_hidden = self.ff_draft_head(x)

        new_cache = dict(cache)
        new_cache["ff"] = new_ff
        new_cache["pos"] = pos + 1
        new_cache["draft_logits"] = draft_logits[:, -1, :].detach()
        self.train(was_training)
        return draft_logits[:, -1, :], new_cache

    @torch.no_grad()
    def eval_metrics(self, xb: torch.Tensor, yb: torch.Tensor, topk: Tuple[int, ...]) -> Dict[str, float]:
        saved_wm, saved_es = self._working_mem, self._engram_state
        self._working_mem = None; self._engram_state = None; self.eval()
        topk = tuple(sorted(set(int(k) for k in topk))); maxk = max(topk)
        B, T = yb.shape; total = B * T
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
            x_full, _ = self.forward_features(xb)
            logits = self._get_logits(x_full)
        flat_l = logits.reshape(-1, logits.size(-1)); flat_y = yb.reshape(-1)
        pred = flat_l.topk(maxk, dim=-1).indices
        metrics = {f"acc@{k}": 100.0 * int((pred[:, :k] == flat_y.unsqueeze(-1)).any(-1).sum()) / max(1, total) for k in topk}
        h_flat = self.final_proj(self.final_ln(x_full)).reshape(-1, self.n_embd)
        E_norm = F.normalize(self.out_proj.weight.to(dtype=h_flat.dtype), p=2, dim=-1)
        z_norm = F.normalize(h_flat, p=2, dim=-1)
        s_pos = (z_norm * E_norm[flat_y]).sum(-1)
        neg_id = torch.randint(0, self.vocab_size, (total, 32), device=yb.device)
        margin_sum = ok_sum = count = 0.0
        for t0 in range(0, total, self.cfg.head_token_chunk):
            t1 = min(t0 + self.cfg.head_token_chunk, total)
            s_neg = (E_norm[neg_id[t0:t1]] * z_norm[t0:t1].unsqueeze(1)).sum(-1)
            m = s_pos[t0:t1].unsqueeze(-1) - s_neg
            w = t1 - t0
            margin_sum += float(m.mean().item()) * w
            ok_sum += float((m > self.cfg.head_margin).float().mean().item()) * w
            count += w
        metrics["head_margin_mean"] = margin_sum / max(1, count)
        metrics["head_margin_ok_frac"] = 100.0 * ok_sum / max(1, count)
        self._working_mem = saved_wm; self._engram_state = saved_es; self.train()
        return metrics

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temp: float = 0.7) -> torch.Tensor:
        was_training = self.training
        self.eval(); self._working_mem = None; self._engram_state = None
        try:
            # Prefill once: build KV caches for the entire prompt.
            prompt = idx[:, -self.block_size:]
            logits, cache = self.prefill_kv(prompt)
            next_tok = torch.multinomial(F.softmax(logits / max(1e-5, temp), -1), num_samples=1)
            idx = torch.cat([prompt, next_tok], dim=1)

            # Decode one token at a time using incremental KV cache.
            for _ in range(max_new_tokens - 1):
                logits, cache = self.decode_one_kv(next_tok, cache)
                next_tok = torch.multinomial(F.softmax(logits / max(1e-5, temp), -1), num_samples=1)
                idx = torch.cat([idx, next_tok], dim=1)
        finally:
            self._working_mem = None; self._engram_state = None; self.train(was_training)
        return idx
