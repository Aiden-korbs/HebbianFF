from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_rmsnorm

class FFDraftHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int, use_liger: bool):
        super().__init__()
        self.n_embd = n_embd
        self.ln = make_rmsnorm(n_embd, use_liger)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out = nn.Linear(n_embd, vocab_size, bias=False)
        self.blend = nn.Parameter(torch.tensor([float(os.environ.get("DRAFT_BLEND_INIT", "-4.0"))]))
        nn.init.normal_(self.out.weight, std=0.02)
    def forward(self, x: torch.Tensor):
        h = self.proj(self.ln(x))
        return self.out(h) * (1.0 / math.sqrt(self.n_embd)), h

class ChunkMemoryCompressor(nn.Module):
    def __init__(self, n_embd: int, memory_tokens: int, gate_value: float, use_liger: bool):
        super().__init__()
        self.memory_tokens = memory_tokens
        self.gate_value = gate_value
        self.norm = make_rmsnorm(n_embd, use_liger)
        self.query = nn.Parameter(torch.randn(memory_tokens, n_embd) * 0.02)
        self.kv_proj = nn.Linear(n_embd, 2 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
    def init_memory(self, B: int, device, dtype) -> torch.Tensor:
        return torch.zeros(B, self.memory_tokens, self.query.size(-1), device=device, dtype=dtype)
    def forward(self, old_mem: Optional[torch.Tensor], x_chunk: torch.Tensor) -> torch.Tensor:
        x = self.norm(x_chunk)
        q = self.query.to(dtype=x.dtype).unsqueeze(0).expand(x.size(0), -1, -1)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)
        new_mem = self.out_proj(attn @ v)
        if old_mem is None: return new_mem
        return self.gate_value * old_mem.to(dtype=x.dtype, device=x.device) + (1.0 - self.gate_value) * new_mem

class EngramMemoryBank(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bank_size = cfg.engram_bank_size
        self.key_dim = cfg.engram_key_dim
        self.val_dim = cfg.n_embd
        self.topk = cfg.engram_topk
        self.decay = cfg.engram_decay
        self.min_score = cfg.engram_min_write_score
        self.s_blend = cfg.engram_strength_blend
        self.age_penalty = cfg.engram_age_penalty
        self._last_write_rate = float("nan")
        self._last_retrieval_sim = float("nan")
        self._last_valid_frac = float("nan")

    def init_state(self, B: int, device, dtype) -> Dict[str, torch.Tensor]:
        def gpu_buf(*shape, bool_=False):
            return torch.zeros(*shape, device=device, dtype=torch.bool if bool_ else dtype)
        return {
            "keys": gpu_buf(B, self.bank_size, self.key_dim),
            "values": gpu_buf(B, self.bank_size, self.val_dim),
            "strength": gpu_buf(B, self.bank_size),
            "age": gpu_buf(B, self.bank_size),
            "valid": gpu_buf(B, self.bank_size, bool_=True),
        }

    @torch.no_grad()
    @torch.compiler.disable
    def retrieve(self, state: Dict, query: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if state is None or not state["valid"].any(): return None
        q = F.normalize(query.detach().float(), p=2, dim=-1).to(state["keys"].dtype)
        k = F.normalize(state["keys"].float(), p=2, dim=-1).to(state["keys"].dtype)
        sim = (k * q.unsqueeze(1)).sum(-1)
        valid_sim = sim[state["valid"]]
        self._last_retrieval_sim = float(valid_sim.mean().item()) if valid_sim.numel() > 0 else float("nan")
        score = sim + state["strength"] - self.age_penalty * state["age"]
        score = score.masked_fill(~state["valid"], float("-inf"))
        ksel = min(self.topk, score.size(1))
        _, idx = torch.topk(score, k=ksel, dim=-1)
        vals = torch.gather(state["values"], 1, idx.unsqueeze(-1).expand(-1, -1, state["values"].size(-1)))
        keys = torch.gather(state["keys"], 1, idx.unsqueeze(-1).expand(-1, -1, state["keys"].size(-1)))
        return keys, vals

    @torch.no_grad()
    @torch.compiler.disable
    def write(self, state: Dict, keys: torch.Tensor, values: torch.Tensor, scores: torch.Tensor):
        B, W, _ = keys.shape
        state["strength"].mul_(self.decay)
        state["age"].add_(1.0)
        keys_d, values_d, scores_d = keys.detach(), values.detach(), scores.detach()
        slot_score = state["strength"] - self.age_penalty * state["age"]
        slot_score = slot_score.masked_fill(state["valid"], float("inf"))
        _, slots = slot_score.topk(W, dim=1, largest=False)
        mask = scores_d >= self.min_score
        self._last_write_rate = float(mask.float().mean().item())
        self._last_valid_frac = float(state["valid"].float().mean().item())
        exp = slots.unsqueeze(-1)
        def masked_scatter(target, idx, src):
            target.scatter_(1, idx.expand_as(src), (src * mask.unsqueeze(-1).to(dtype=src.dtype)).to(target.dtype))
        masked_scatter(state["keys"], exp, keys_d)
        masked_scatter(state["values"], exp, values_d)
        new_str = torch.tanh(scores_d) * mask.to(dtype=scores_d.dtype)
        old_str = torch.gather(state["strength"], 1, slots)
        blended = old_str * self.s_blend + new_str * (1.0 - self.s_blend)
        state["strength"].scatter_(1, slots, (blended * mask.to(dtype=blended.dtype)).to(state["strength"].dtype))
        old_age = torch.gather(state["age"], 1, slots)
        state["age"].scatter_(1, slots, old_age * (~mask).to(dtype=old_age.dtype))
        state["valid"].scatter_(1, slots, mask | torch.gather(state["valid"], 1, slots))
