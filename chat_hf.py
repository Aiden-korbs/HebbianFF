#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from HebbianFF import env as _repo_env  # noqa: F401
except Exception:
    pass

from HebbianFF.config import CFG
from HebbianFF.model import FF_LLM
from HebbianFF.packed import CpuOffloadedEmbedding, CpuOffloadedLinear, is_packed_entry, replace_packed_linears
from HebbianFF.ternary_runtime import PackedTernaryLoRALinear, resolved_runtime_config


def _get_child(module: torch.nn.Module, name: str) -> torch.nn.Module:
    cur = module
    for part in name.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _set_child(module: torch.nn.Module, name: str, child: torch.nn.Module) -> None:
    parts = name.split(".")
    cur = module
    for part in parts[:-1]:
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = child
    else:
        setattr(cur, last, child)


def apply_ternary_adapter(model: torch.nn.Module, adapter_path: str, dtype: torch.dtype, device: str) -> None:
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    modules = adapter.get("modules", {})
    cfg = resolved_runtime_config()
    dense_modules = set(cfg["selective_dense_modules"]) if cfg["selective_dense_cache"] else set()
    extra_mib = 0.0
    for name in dense_modules:
        state = modules.get(name)
        if state is not None:
            extra_mib += int(state["in_features"]) * int(state["out_features"]) * torch.tensor([], dtype=dtype).element_size() / 1024**2
    print(
        "[TERNARY] "
        f"adapter={adapter_path} preset={cfg['preset']} runtime={cfg['runtime']} "
        f"prefill={cfg['prefill_runtime']} decode={cfg['decode_runtime']} "
        f"auto_prefill={cfg['auto_prefill']} dense_cache={'yes' if cfg['selective_dense_cache'] else 'no'} "
        f"dense_modules={len(dense_modules)} extra_dense_cache_mib={extra_mib:.1f} "
        f"adapter_modules={len(modules)} profile={cfg['selective_dense_profile'] or 'none'}",
        flush=True,
    )
    for name, state in modules.items():
        old = _get_child(model, name)
        if not isinstance(old, torch.nn.Linear):
            raise TypeError(f"{name} is {type(old).__name__}, expected nn.Linear")
        old_device = old.weight.device
        target_device = old_device.type if old_device.type != "meta" else device
        _set_child(
            model,
            name,
            PackedTernaryLoRALinear(state, device=target_device, dtype=dtype, module_name=name).eval(),
        )


def _expand_cpu_offload_linear_name(name: str, cfg: CFG) -> list[str]:
    """Expand convenience aliases used by CPU_OFFLOAD_LINEARS."""
    name = name.strip()
    low = name.lower()
    if not name:
        return []
    if low == "final_proj":
        return ["final_proj"]
    if low == "last_ff_mlp":
        i = int(cfg.ff_n_layer) - 1
        return [f"ff_blocks.{i}.mlp.gate", f"ff_blocks.{i}.mlp.up", f"ff_blocks.{i}.mlp.down"]
    if low == "first_ff_mlp":
        return ["ff_blocks.0.mlp.gate", "ff_blocks.0.mlp.up", "ff_blocks.0.mlp.down"]

    m = re.fullmatch(r"last(\d+)_ff_mlp", low)
    if m:
        n = max(1, int(m.group(1)))
        start = max(0, int(cfg.ff_n_layer) - n)
        out: list[str] = []
        for i in range(start, int(cfg.ff_n_layer)):
            out.extend([f"ff_blocks.{i}.mlp.gate", f"ff_blocks.{i}.mlp.up", f"ff_blocks.{i}.mlp.down"])
        return out

    return [name]


def apply_cpu_linear_offloads(model: torch.nn.Module, state: dict, cfg: CFG) -> set[str]:
    """
    Keep selected dense Linear weights on CPU even when parent model.to(cuda) runs.

    Examples:
      CPU_OFFLOAD_LINEARS=last_ff_mlp
      CPU_OFFLOAD_LINEARS=last_ff_mlp,final_proj
      CPU_OFFLOAD_LINEARS=ff_blocks.27.mlp.gate,ff_blocks.27.mlp.up,ff_blocks.27.mlp.down
    """
    raw = os.environ.get("CPU_OFFLOAD_LINEARS", "").strip()
    if not raw:
        return set()

    names: list[str] = []
    for part in re.split(r"[,\s]+", raw):
        names.extend(_expand_cpu_offload_linear_name(part, cfg))

    names = list(dict.fromkeys([n for n in names if n]))
    consumed: set[str] = set()
    saved_mib = 0.0

    for name in names:
        weight_key = name + ".weight"
        bias_key = name + ".bias"
        weight = state.get(weight_key)
        bias = state.get(bias_key)

        if not torch.is_tensor(weight):
            raise RuntimeError(f"CPU_OFFLOAD_LINEARS requested {name}, but {weight_key} is missing or not dense")

        old = _get_child(model, name)
        if not isinstance(old, torch.nn.Linear):
            raise RuntimeError(f"CPU_OFFLOAD_LINEARS requested {name}, but module is {type(old).__name__}, not nn.Linear")

        _set_child(model, name, CpuOffloadedLinear(weight, bias if torch.is_tensor(bias) else None))
        consumed.add(weight_key)
        if torch.is_tensor(bias):
            consumed.add(bias_key)

        saved_mib += weight.numel() * weight.element_size() / 1024**2
        if torch.is_tensor(bias):
            saved_mib += bias.numel() * bias.element_size() / 1024**2

    print(
        f"[HFCHAT] CPU offload enabled for {len(names)} dense linears; "
        f"saved≈{saved_mib:.1f} MiB VRAM: {', '.join(names)}",
        flush=True,
    )
    return consumed


class no_init_weights:
    """Skip expensive random initialisation when immediately loading checkpoint weights."""

    def __enter__(self):
        self._old_init = {}
        import torch.nn.init as init

        for name in [
            "uniform_",
            "normal_",
            "kaiming_uniform_",
            "kaiming_normal_",
            "xavier_uniform_",
            "xavier_normal_",
            "trunc_normal_",
            "constant_",
            "zeros_",
            "ones_",
        ]:
            if hasattr(init, name):
                self._old_init[name] = getattr(init, name)
                setattr(init, name, lambda tensor, *args, **kwargs: tensor)

        self._old_tensor_uniform = torch.Tensor.uniform_
        self._old_tensor_normal = torch.Tensor.normal_
        torch.Tensor.uniform_ = lambda t, *args, **kwargs: t
        torch.Tensor.normal_ = lambda t, *args, **kwargs: t
        return self

    def __exit__(self, exc_type, exc, tb):
        import torch.nn.init as init

        for name, old in self._old_init.items():
            setattr(init, name, old)
        torch.Tensor.uniform_ = self._old_tensor_uniform
        torch.Tensor.normal_ = self._old_tensor_normal
        return False


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_from_name(name: str, device: str) -> torch.dtype:
    name = (name or "auto").lower()
    if name == "auto":
        if device == "cuda":
            return torch.bfloat16
        return torch.float32
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {name}")


def cfg_from_ckpt(base_cfg: CFG, ckpt: dict, block_size_override: int | None) -> CFG:
    saved = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(saved, dict):
        for k, v in saved.items():
            if hasattr(base_cfg, k):
                try:
                    setattr(base_cfg, k, v)
                except Exception:
                    pass

    # Runtime-safe defaults.
    base_cfg.batch_size = 1
    base_cfg.grad_accum_steps = 1
    base_cfg.use_ff_ema_bp = False
    base_cfg.ff_ema_max_trips_per_step = 0
    base_cfg.ff_ema_bp_weight = 0.0
    base_cfg.use_engram = os.environ.get("INFER_USE_ENGRAM", "0") == "1"
    base_cfg.memory_tokens = int(os.environ.get("INFER_MEMORY_TOKENS", "0"))
    base_cfg.use_draft_head = os.environ.get("USE_DRAFT_HEAD", "0") == "1"

    # Allow runtime conversion of normal checkpoints into BitNet modules.
    if os.environ.get("USE_BITNET", "0") == "1":
        base_cfg.use_bitnet = True

    base_cfg.kv_cache_int8 = os.environ.get("KV_CACHE_INT8", "0") == "1"
    base_cfg.kv_cache_max_len = int(os.environ.get("KV_CACHE_MAX_LEN", "0"))
    base_cfg.kv_cache_sink_tokens = int(os.environ.get("KV_CACHE_SINK_TOKENS", "0"))

    if block_size_override is not None:
        base_cfg.block_size = int(block_size_override)
        base_cfg.seq_chunk_size = min(int(base_cfg.seq_chunk_size), int(base_cfg.block_size))
        base_cfg.local_window = min(int(base_cfg.local_window), int(base_cfg.block_size))

    return base_cfg



def _assert_no_meta_tensors(model: torch.nn.Module, limit: int = 40) -> None:
    meta = []
    for name, param in model.named_parameters(recurse=True):
        if getattr(param, "is_meta", False):
            meta.append("param:" + name)
    for name, buf in model.named_buffers(recurse=True):
        if getattr(buf, "is_meta", False):
            meta.append("buffer:" + name)

    if meta:
        shown = "\n".join("  " + x for x in meta[:limit])
        raise RuntimeError(
            f"Model still has {len(meta)} meta tensors after packed load. "
            f"These were never materialized from the checkpoint:\n{shown}"
        )


def _set_buffer_by_name(model: torch.nn.Module, dotted_name: str, value: torch.Tensor) -> None:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        parent.register_buffer(leaf, value)


def _materialize_rope_meta_buffers(model: torch.nn.Module, cfg: CFG) -> int:
    """
    Packed/meta loading skips deterministic RoPE buffers because they are usually
    not saved in checkpoints. Rebuild them on CPU, then normal model.to(cuda)
    moves them with the rest of the packed runtime model.
    """
    fixed = 0

    # Qwen/DeepSeek-Qwen usually uses a large rope theta, but fall back safely.
    rope_base = float(
        getattr(cfg, "rope_theta", None)
        or getattr(cfg, "rope_base", None)
        or getattr(cfg, "rotary_base", None)
        or getattr(cfg, "rotary_emb_base", None)
        or 10000.0
    )

    for name, buf in list(model.named_buffers(recurse=True)):
        if not getattr(buf, "is_meta", False):
            continue
        if not name.endswith("rope_inv_freq"):
            continue

        # Existing meta buffer shape tells us how many inverse frequencies it needs.
        n = int(buf.numel())
        if n <= 0:
            head_dim = int(cfg.n_embd) // int(cfg.n_head)
            n = head_dim // 2

        dim = n * 2
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        _set_buffer_by_name(model, name, inv_freq)
        fixed += 1

    if fixed:
        print(f"[HFCHAT] Materialized {fixed} meta RoPE buffers", flush=True)

    return fixed


def load_model(args):
    device = args.device or auto_device()
    dtype = dtype_from_name(args.dtype, device)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if getattr(getattr(tok, "backend_tokenizer", None), "decoder", None) is None:
        try:
            from tokenizers import decoders

            tok.backend_tokenizer.decoder = decoders.ByteLevel()
            print("[HFCHAT] Added missing ByteLevel tokenizer decoder", flush=True)
        except Exception:
            pass
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    print(f"[HFCHAT] Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(CFG(), ckpt, args.block_size)

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    cfg.eos_token_id = int(eos_id) if eos_id is not None else -1
    cfg.pad_token_id = int(pad_id) if pad_id is not None else 0

    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state dict")

    is_packed_ckpt = ckpt.get("format") == "ff_llm_packed_1bit_transfer_v1"
    wants_cpu_offload = (
        os.environ.get("CPU_OFFLOAD_OUT_PROJ", "0") == "1"
        or os.environ.get("CPU_OFFLOAD_TOK_EMB", "0") == "1"
    )

    if "tok_emb.weight" not in state:
        raise KeyError("tok_emb.weight missing from checkpoint")

    tok_emb_weight = state["tok_emb.weight"]
    if torch.is_tensor(tok_emb_weight):
        vocab_size = int(tok_emb_weight.shape[0])
    elif is_packed_entry(tok_emb_weight):
        vocab_size = int(tok_emb_weight["shape"][0])
    else:
        raise TypeError("tok_emb.weight must be a dense tensor or packed entry")
    tok_len = len(tok)

    print(
        f"[HFCHAT] Config: vocab={vocab_size} tokenizer_len={tok_len} "
        f"d={cfg.n_embd} heads={cfg.n_head}/{cfg.n_kv_head} "
        f"FF={cfg.ff_n_layer} BP={cfg.bp_n_layer} block={cfg.block_size} "
        f"kv_max={cfg.kv_cache_max_len or cfg.block_size} "
        f"kv_sink={cfg.kv_cache_sink_tokens} kv_int8={int(cfg.kv_cache_int8)}",
        flush=True,
    )

    consumed_biases = set()
    packed_module_names = set()

    if is_packed_ckpt or wants_cpu_offload:
        packed_module_names = {
            key[: -len(".weight")]
            for key, value in state.items()
            if key.endswith(".weight") and is_packed_entry(value)
        }

        if is_packed_ckpt and not packed_module_names:
            raise RuntimeError(
                "Checkpoint says format=ff_llm_packed_1bit_transfer_v1, "
                "but no packed .weight entries were found."
            )

        print(
            f"[HFCHAT] Meta load path: constructing FF_LLM on meta, "
            f"then replacing {len(packed_module_names)} packed linears before CUDA materialization.",
            flush=True,
        )

        # Important: this prevents full dense Linear weights from ever being allocated.
        with torch.device("meta"):
            if os.environ.get("NO_INIT_LOAD", "1") == "1":
                with no_init_weights():
                    model = FF_LLM(vocab_size, cfg)
            else:
                model = FF_LLM(vocab_size, cfg)

        # Replace packed .weight dicts with TernaryLinear/PackedLinear modules
        # while the dense model is still only a meta skeleton.
        consumed_biases = replace_packed_linears(model, state)
        if os.environ.get("CPU_OFFLOAD_OUT_PROJ", "0") == "1":
            out_w = state.get("out_proj.weight")
            out_b = state.get("out_proj.bias")
            if not torch.is_tensor(out_w):
                raise RuntimeError("CPU_OFFLOAD_OUT_PROJ=1 requires dense out_proj.weight")
            model.out_proj = CpuOffloadedLinear(out_w, out_b if torch.is_tensor(out_b) else None)
            consumed_biases.add("out_proj.weight")
            if torch.is_tensor(out_b):
                consumed_biases.add("out_proj.bias")
            print("[HFCHAT] CPU offload enabled for dense out_proj.weight", flush=True)
        if os.environ.get("CPU_OFFLOAD_TOK_EMB", "0") == "1":
            emb_w = state.get("tok_emb.weight")
            if not torch.is_tensor(emb_w):
                raise RuntimeError("CPU_OFFLOAD_TOK_EMB=1 requires dense tok_emb.weight")
            model.tok_emb = CpuOffloadedEmbedding(emb_w, padding_idx=getattr(model.tok_emb, "padding_idx", None))
            consumed_biases.add("tok_emb.weight")
            print("[HFCHAT] CPU offload enabled for dense tok_emb.weight", flush=True)

        consumed_biases.update(apply_cpu_linear_offloads(model, state, cfg))

        dense_state = {
            key: value
            for key, value in state.items()
            if torch.is_tensor(value) and key not in consumed_biases
        }

        print(f"[HFCHAT] Packed checkpoint: replaced {len(packed_module_names)} linear weights", flush=True)

        # assign=True is required for meta models; it materializes real tensors
        # from checkpoint tensors instead of trying to copy into meta tensors.
        missing, unexpected = model.load_state_dict(dense_state, strict=False, assign=True)

        expected_packed_missing = {
            f"{name}.{suffix}"
            for name in packed_module_names
            for suffix in ("packed", "scale", "bias")
        }
        expected_consumed_missing = set(consumed_biases)
        missing = [
            key for key in missing
            if key not in expected_packed_missing and key not in expected_consumed_missing
        ]

        _materialize_rope_meta_buffers(model, cfg)
        _assert_no_meta_tensors(model)

        # Move only the already-packed runtime model to CUDA.
        model = model.to(device=device, dtype=dtype)

    else:
        if os.environ.get("NO_INIT_LOAD", "1") == "1":
            with no_init_weights():
                model = FF_LLM(vocab_size, cfg).to(device=device, dtype=dtype)
        else:
            model = FF_LLM(vocab_size, cfg).to(device=device, dtype=dtype)

        missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[HFCHAT WARN] Missing keys: {len(missing)}", flush=True)
        if getattr(args, "verbose_keys", False):
            print("\n".join(f"  missing: {k}" for k in missing[:80]), flush=True)

    if unexpected:
        print(f"[HFCHAT WARN] Unexpected keys: {len(unexpected)}", flush=True)
        if getattr(args, "verbose_keys", False):
            print("\n".join(f"  unexpected: {k}" for k in unexpected[:80]), flush=True)

    if cfg.use_draft_head and os.environ.get("DRAFT_BLEND_BP", "1") != "0":
        draft_missing = [k for k in missing if k.startswith("ff_draft_head.")]
        if draft_missing:
            print(
                "[HFCHAT WARN] DRAFT_BLEND_BP=1 but draft-head checkpoint weights are missing; "
                "only use nonzero DRAFT_BLEND_ALPHA after validating draft logits.",
                flush=True,
            )
        raw_blend = getattr(getattr(model, "ff_draft_head", None), "blend", None)
        if raw_blend is not None:
            blend = float(torch.sigmoid(raw_blend.detach().float())[0])
            alpha = float(os.environ.get("DRAFT_BLEND_ALPHA", "0.0"))
            print(f"[HFCHAT] Draft logit blend alpha={alpha:.4f} legacy_gate={blend:.4f}", flush=True)

    if os.environ.get("USE_FF_SKIP", "0") == "1" or os.environ.get("USE_FF_DRAFT_SKIP", "0") == "1":
        skip_layers = os.environ.get("FF_SKIP_LAYERS", "").strip() or "(none)"
        skip_mode = os.environ.get("FF_SKIP_MODE", "block").strip() or "block"
        print(f"[HFCHAT] FF skip enabled mode={skip_mode} layers={skip_layers}", flush=True)

    if args.ternary_adapter:
        apply_ternary_adapter(model, args.ternary_adapter, dtype, device)

    model.eval()
    setattr(model, "_debug_tokenizer", tok)
    setattr(getattr(model, "_orig_mod", model), "_debug_tokenizer", tok)

    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(
        f"[HFCHAT] Ready on {device} dtype={dtype} "
        f"params={n_params/1e6:.1f}M buffers={n_buffers/1e6:.1f}M",
        flush=True,
    )
    return model, getattr(model, "_orig_mod", model), tok, cfg, device, dtype


def special_token_id(tok, token: str) -> int | None:
    try:
        tid = tok.convert_tokens_to_ids(token)
    except Exception:
        return None
    if tid is None:
        return None
    if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
        return tid
    return None


def stop_token_ids(tok, args) -> set[int]:
    ids: set[int] = set()
    for maybe in [tok.eos_token_id]:
        if maybe is not None and int(maybe) >= 0:
            ids.add(int(maybe))

    for token in args.extra_stop_tokens:
        tid = special_token_id(tok, token)
        if tid is not None:
            ids.add(tid)

    # Qwen / ChatML-ish tokens. Harmless if absent.
    for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        tid = special_token_id(tok, token)
        if tid is not None:
            ids.add(tid)

    return ids


def forbidden_token_ids(tok, args) -> set[int]:
    if not args.suppress_im_start:
        return set()
    ids = set()
    for token in ["<|im_start|>"]:
        tid = special_token_id(tok, token)
        if tid is not None:
            ids.add(tid)
    return ids


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    logits = logits.clone()
    top_k = int(top_k or 0)
    top_p = float(top_p)
    if top_k > 0 and 0.0 < top_p < 1.0:
        k = min(top_k, logits.size(-1))
        top_vals, top_idx = torch.topk(logits, k, dim=-1)
        sorted_logits, sorted_order = torch.sort(top_vals, descending=True, dim=-1)
        sorted_idx = top_idx.gather(dim=-1, index=sorted_order)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        out = torch.full_like(logits, -float("inf"))
        out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        return out

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, -float("inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return logits


def repetition_penalty(logits, recent_ids: Sequence[int], penalty: float):
    if penalty <= 1.0 or not recent_ids:
        return logits
    ids = torch.tensor(list(set(int(x) for x in recent_ids)), device=logits.device, dtype=torch.long)
    vals = logits[:, ids]
    logits[:, ids] = torch.where(vals < 0, vals * penalty, vals / penalty)
    return logits


def apply_forbidden_token_mask(logits, forbidden_ids: Iterable[int]):
    ids = [int(i) for i in forbidden_ids if 0 <= int(i) < logits.size(-1)]
    if ids:
        logits[:, ids] = -float("inf")
    return logits


def autocast_context(device: str, dtype: torch.dtype):
    enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.amp.autocast("cuda", dtype=dtype, enabled=enabled)


def clone_cache_obj(obj):
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: clone_cache_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_cache_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_cache_obj(v) for v in obj)
    return obj


def decode_logits_for_choice(logits, idx, args, forbidden_ids: set[int]):
    cur = logits.float()
    if float(args.repeat_penalty) != 1.0:
        cur = repetition_penalty(cur, idx[0, -int(args.repeat_window):].tolist(), args.repeat_penalty)
    cur = apply_forbidden_token_mask(cur, forbidden_ids)
    return cur


def choose_token_from_logits(logits, args):
    if args.temp <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    cur = logits / max(1e-6, float(args.temp))
    if int(args.top_k) > 0 or float(args.top_p) < 1.0:
        cur = top_k_top_p_filter(cur, args.top_k, args.top_p)
    probs = F.softmax(cur, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def token_in_base_filter(logits, token_id: int, args) -> bool:
    if int(args.top_k) <= 0 and float(args.top_p) >= 1.0:
        return False
    filt = top_k_top_p_filter(logits, args.top_k, args.top_p)
    if token_id < 0 or token_id >= filt.size(-1):
        return False
    return bool(torch.isfinite(filt[0, token_id]).item())


def cpu_context_enabled() -> bool:
    return (
        os.environ.get("CPU_HASH_CTX", "0") == "1"
        or os.environ.get("CPU_CTX_MODE", "none").strip().lower() == "hash"
        or os.environ.get("CPU_CTX", "0") == "1"
    )


def build_external_cpu_context(raw_model, idx: torch.Tensor, cfg, device: str, dtype: torch.dtype) -> torch.Tensor | None:
    """Build compact prompt memory for the model external_mem path.

    The tensor is intentionally assembled outside the KV cache path. It lets a
    short recent window stay on GPU while older or full prompt information is
    injected as a small prefix-memory bank.
    """
    if not cpu_context_enabled():
        return None
    max_mem = int(os.environ.get("CPU_CTX_MAX_MEM", getattr(cfg, "cpu_context_max_mem_tokens", 128)))
    if max_mem <= 0:
        return None
    recent = int(os.environ.get("CPU_CTX_RECENT_TOKENS", str(getattr(cfg, "block_size", 256))))
    prefix_only = os.environ.get("CPU_CTX_PREFIX_ONLY", "1") != "0"
    source = idx[:, :-recent] if prefix_only and idx.size(1) > recent else idx
    if source.numel() == 0:
        return None

    # Average token embeddings into a small set of memory slots. This keeps the
    # memory in model embedding space, so it does not depend on a trained CPU
    # projection to be useful at runtime.
    with torch.no_grad():
        emb = raw_model.tok_emb(source.to(device=device))
        slots = min(max_mem, int(source.size(1)))
        edges = torch.linspace(0, source.size(1), steps=slots + 1, device=source.device).round().long().cpu().tolist()
        mem_parts = []
        for i in range(slots):
            lo = int(edges[i])
            hi = max(lo + 1, int(edges[i + 1]))
            mem_parts.append(emb[:, lo:hi, :].mean(dim=1, keepdim=True))
        mem = torch.cat(mem_parts, dim=1).to(device=device, dtype=dtype)
    scale = float(os.environ.get("CPU_CTX_SCALE", "1.0"))
    if scale != 1.0:
        mem = mem * scale
    return mem


def format_spec_stats(stats: dict) -> str:
    generated = int(stats.get("generated_tokens", 0))
    elapsed = float(stats.get("elapsed_sec", 0.0))
    accepted = int(stats.get("accepted_tokens", 0))
    rejected = int(stats.get("rejected_tokens", 0))
    denom = max(1, accepted + rejected)
    tok_s = generated / max(1e-9, elapsed)
    return (
        f"accepted={accepted} rejected={rejected} "
        f"acceptance={accepted / denom:.2%} "
        f"base_calls_saved={int(stats.get('base_forward_calls_saved', 0))} "
        f"tok_s={tok_s:.2f}"
    )


def ensure_runtime_arg_defaults(args) -> None:
    defaults = {
        "extra_stop_tokens": [],
        "hide_reasoning": False,
        "no_chat_template": False,
        "repeat_window": 256,
        "show_prompt": False,
        "stream": False,
        "suppress_im_start": True,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


@torch.inference_mode()
def generate_ids_speculative(raw_model, idx, cfg, device, dtype, args, stop_ids: set[int], forbidden_ids: set[int], stream_callback=None):
    if raw_model.ff_draft_head is None:
        raise RuntimeError("SPEC_DRAFT=1 requires USE_DRAFT_HEAD=1 and ff_draft_head weights")
    if not all(hasattr(raw_model, name) for name in ("prefill_kv", "decode_many_kv", "decode_one_draft_kv")):
        raise RuntimeError("SPEC_DRAFT=1 requires KV speculative helpers on the model")

    raw_model.eval()
    raw_model._working_mem = None
    raw_model._engram_state = None

    max_new = int(args.max_new)
    draft_k = max(1, int(os.environ.get("SPEC_DRAFT_K", "4")))
    t0 = time.time()
    stats = {
        "mode": "spec_draft",
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "generated_tokens": 0,
        "base_verify_calls": 0,
        "base_forward_calls_saved": 0,
    }

    try:
        idx_c = idx[:, -int(cfg.block_size):]
        with autocast_context(device, dtype):
            logits, kv_cache = raw_model.prefill_kv(idx_c, blend_logits=False)
        idx = idx_c

        while stats["generated_tokens"] < max_new:
            remaining = max_new - int(stats["generated_tokens"])
            k = min(draft_k, remaining)
            draft_cache = clone_cache_obj(kv_cache)
            draft_logits = draft_cache.get("draft_logits")
            if draft_logits is None:
                cur = decode_logits_for_choice(logits, idx, args, forbidden_ids)
                next_tok = choose_token_from_logits(cur, args)
                next_id = int(next_tok.item())
                if not args.no_stop_eos and next_id in stop_ids:
                    break
                idx = torch.cat([idx, next_tok], dim=1)
                stats["generated_tokens"] += 1
                if stream_callback is not None:
                    stream_callback(idx[0].tolist())
                with autocast_context(device, dtype):
                    logits, kv_cache = raw_model.decode_one_kv(next_tok, kv_cache, blend_logits=False)
                stats["base_verify_calls"] += 1
                continue

            proposed = []
            for _ in range(k):
                draft_choice = choose_token_from_logits(draft_logits.float(), args)
                proposed.append(draft_choice)
                with autocast_context(device, dtype):
                    draft_logits, draft_cache = raw_model.decode_one_draft_kv(draft_choice, draft_cache)

            proposed_t = torch.cat(proposed, dim=1)
            with autocast_context(device, dtype):
                verify_logits, verify_cache = raw_model.decode_many_kv(proposed_t, kv_cache, blend_logits=False)
            stats["base_verify_calls"] += 1

            accepted = []
            rejected = False
            base_reject_tok = None

            for j, draft_tok in enumerate(proposed):
                prefix_for_j = idx if j == 0 else torch.cat([idx] + accepted, dim=1)
                base_logits_j = logits if j == 0 else verify_logits[:, j - 1, :]
                base_cur = decode_logits_for_choice(base_logits_j, prefix_for_j, args, forbidden_ids)
                base_tok = choose_token_from_logits(base_cur, args)
                draft_id = int(draft_tok.item())
                base_id = int(base_tok.item())
                if draft_id == base_id or token_in_base_filter(base_cur, draft_id, args):
                    accepted.append(draft_tok)
                    stats["accepted_tokens"] += 1
                else:
                    rejected = True
                    base_reject_tok = base_tok
                    stats["rejected_tokens"] += 1
                    break

            if not rejected:
                for tok_t in accepted:
                    tok_id = int(tok_t.item())
                    if not args.no_stop_eos and tok_id in stop_ids:
                        stats["elapsed_sec"] = time.time() - t0
                        raw_model._last_gen_stats = stats
                        return idx
                    idx = torch.cat([idx, tok_t], dim=1)
                    stats["generated_tokens"] += 1
                    if stream_callback is not None:
                        stream_callback(idx[0].tolist())
                kv_cache = verify_cache
                logits = verify_logits[:, -1, :]
                stats["base_forward_calls_saved"] += max(0, len(accepted) - 1)
                continue

            commit = accepted + [base_reject_tok]
            for tok_t in commit:
                tok_id = int(tok_t.item())
                if not args.no_stop_eos and tok_id in stop_ids:
                    stats["elapsed_sec"] = time.time() - t0
                    raw_model._last_gen_stats = stats
                    return idx
                idx = torch.cat([idx, tok_t], dim=1)
                stats["generated_tokens"] += 1
                if stream_callback is not None:
                    stream_callback(idx[0].tolist())

            commit_t = torch.cat(commit, dim=1)
            with autocast_context(device, dtype):
                commit_logits, kv_cache = raw_model.decode_many_kv(commit_t, kv_cache, blend_logits=False)
            stats["base_verify_calls"] += 1
            logits = commit_logits[:, -1, :]
            stats["base_forward_calls_saved"] += max(0, len(commit) - 2)
    finally:
        raw_model._working_mem = None
        raw_model._engram_state = None

    stats["elapsed_sec"] = time.time() - t0
    raw_model._last_gen_stats = stats
    return idx


@torch.inference_mode()
def generate_ids(model, raw_model, idx, cfg, device, dtype, args, stop_ids: set[int], forbidden_ids: set[int], stream_callback=None):
    raw_model.eval()
    raw_model._working_mem = None
    raw_model._engram_state = None
    t0 = time.time()
    start_len = int(idx.size(1))

    if os.environ.get("SPEC_DRAFT", "0") == "1":
        out = generate_ids_speculative(raw_model, idx, cfg, device, dtype, args, stop_ids, forbidden_ids, stream_callback)
        stats = getattr(raw_model, "_last_gen_stats", {})
        print("[SPEC]", format_spec_stats(stats), flush=True)
        return out

    use_kv = (
        os.environ.get("USE_KV_CACHE", "1") == "1"
        and hasattr(raw_model, "prefill_kv")
        and hasattr(raw_model, "decode_one_kv")
    )
    external_mem = build_external_cpu_context(raw_model, idx, cfg, device, dtype)
    if external_mem is not None:
        use_kv = False
        if os.environ.get("CPU_CTX_VERBOSE", "0") == "1":
            print(f"[CPU_CTX] enabled external_mem={tuple(external_mem.shape)} use_kv=0", flush=True)
    kv_cache = None
    logits = None

    try:
        if use_kv:
            try:
                idx_c = idx[:, -int(cfg.block_size):]
                with autocast_context(device, dtype):
                    logits, kv_cache = raw_model.prefill_kv(idx_c)
                if os.environ.get("KV_VERBOSE", "0") == "1":
                    print(f"[KV] enabled prompt_tokens={idx_c.size(1)}", flush=True)
            except Exception as e:
                print(f"[KV WARN] prefill failed; falling back: {type(e).__name__}: {e}", flush=True)
                use_kv = False
                logits = None
                kv_cache = None

        for _ in range(int(args.max_new)):
            if not use_kv:
                idx_c = idx[:, -int(cfg.block_size):]
                with autocast_context(device, dtype):
                    x_full, _ = raw_model.forward_features(idx_c, update_state=False, external_mem=external_mem)
                    logits = raw_model._get_last_logits(x_full) if hasattr(raw_model, "_get_last_logits") else raw_model._get_logits(x_full)[:, -1, :]

            cur = logits.float()
            if float(args.repeat_penalty) != 1.0:
                cur = repetition_penalty(cur, idx[0, -int(args.repeat_window):].tolist(), args.repeat_penalty)
            cur = apply_forbidden_token_mask(cur, forbidden_ids)

            if args.temp <= 0:
                next_tok = torch.argmax(cur, dim=-1, keepdim=True)
            else:
                cur = cur / max(1e-6, float(args.temp))
                if int(args.top_k) > 0 or float(args.top_p) < 1.0:
                    cur = top_k_top_p_filter(cur, args.top_k, args.top_p)
                probs = F.softmax(cur, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            next_id = int(next_tok.item())
            if not args.no_stop_eos and next_id in stop_ids:
                break

            idx = torch.cat([idx, next_tok], dim=1)

            if stream_callback is not None:
                try:
                    stream_callback(idx[0].tolist())
                except BrokenPipeError:
                    stream_callback = None

            if use_kv:
                with autocast_context(device, dtype):
                    logits, kv_cache = raw_model.decode_one_kv(next_tok, kv_cache)
    finally:
        raw_model._working_mem = None
        raw_model._engram_state = None

    generated = max(0, int(idx.size(1)) - start_len)
    raw_model._last_gen_stats = {
        "mode": "base",
        "generated_tokens": generated,
        "elapsed_sec": time.time() - t0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "base_forward_calls_saved": 0,
    }
    return idx


def strip_reasoning(text: str) -> str:
    # Handles both complete and partially generated <think> blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def trim_stops(text: str, stops: Sequence[str]) -> str:
    cut = len(text)
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i >= 0:
            cut = min(cut, i)
    return text[:cut].strip()


def normalize_decoded_text(text: str) -> str:
    """
    Some locally trained ByteLevel BPE tokenizers in this repo were saved without
    a decoder section, so HF decode leaves whitespace sentinels in the text.
    Keep this display-only: encoding still uses the tokenizer unchanged.
    """
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def decode_generated(tok, token_ids: Sequence[int], args) -> str:
    text = tok.decode(list(token_ids), skip_special_tokens=False)
    text = normalize_decoded_text(text)
    text = trim_stops(
        text,
        [
            "<|im_end|>",
            "<|endoftext|>",
            "</s>",
            "<|im_start|>user",
            "<|im_start|>system",
            "<|im_start|>assistant",
            "<|im_start|>",
        ],
    )
    if args.hide_reasoning:
        text = strip_reasoning(text)
    return text.strip()


def fallback_chat_template(system: str, messages: List[dict]) -> str:
    parts = []
    if system.strip():
        parts.append(f"<|im_start|>system\n{system.strip()}<|im_end|>\n")
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_prompt(tok, system: str, user: str, history: List[Tuple[str, str]], args) -> str:
    messages = []
    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user})

    if not args.no_chat_template and getattr(tok, "chat_template", None):
        templated = []
        if system.strip():
            templated.append({"role": "system", "content": system.strip()})
        templated.extend(messages)
        return tok.apply_chat_template(templated, tokenize=False, add_generation_prompt=True)

    return fallback_chat_template(system, messages)


def tokenise_prompt(tok, prompt: str) -> List[int]:
    return tok(prompt, add_special_tokens=False).input_ids


def stream_text_delta(tok, generated_ids: Sequence[int], args, state: dict) -> None:
    """Decode the full generated suffix and print only the new text delta.

    Tokenizers often use byte-pair pieces, so decoding single tokens can print
    broken fragments. Decoding the whole suffix each step and printing the delta
    gives much cleaner streaming output.
    """
    text = decode_generated(tok, generated_ids, args)
    previous = state.get("printed_text", "")

    if text.startswith(previous):
        delta = text[len(previous):]
    else:
        # Rare case: stop trimming / reasoning stripping changed earlier text.
        # Start a fresh line rather than corrupting the display.
        delta = "\n" + text

    if delta:
        print(delta, end="", flush=True)
        state["printed_text"] = text


def run_prompt(args, model, raw_model, tok, cfg, device, dtype, prompt: str, history=None, stream_prefix: str | None = None):
    ensure_runtime_arg_defaults(args)
    history = history or []
    full_prompt = prompt if args.raw else build_prompt(tok, args.system, prompt, history, args)

    ids = tokenise_prompt(tok, full_prompt)
    original_len = len(ids)
    if original_len > cfg.block_size:
        keep = int(cfg.block_size)
        print(f"[HFCHAT WARN] Prompt is {original_len} tokens; using last {keep}.", flush=True)
        ids = ids[-keep:]

    if args.show_prompt:
        print("\n[HFCHAT PROMPT]")
        print(full_prompt)
        print("[/HFCHAT PROMPT]\n", flush=True)

    if not ids:
        raise ValueError("Prompt tokenized to zero tokens.")

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    stop_ids = stop_token_ids(tok, args)
    forbidden_ids = forbidden_token_ids(tok, args)

    if os.environ.get("SPEC_COMPARE", "0") == "1":
        saved_env = {k: os.environ.get(k) for k in ("SPEC_DRAFT", "DRAFT_BLEND_BP")}
        variants = [
            ("base decode", {"SPEC_DRAFT": "0", "DRAFT_BLEND_BP": "0"}),
            ("draft blend decode", {"SPEC_DRAFT": "0", "DRAFT_BLEND_BP": "1"}),
            ("speculative draft decode", {"SPEC_DRAFT": "1", "DRAFT_BLEND_BP": "0"}),
        ]
        final_text = ""
        for label, env in variants:
            os.environ.update(env)
            raw_model._last_gen_stats = {}
            out = generate_ids(model, raw_model, idx.clone(), cfg, device, dtype, args, stop_ids, forbidden_ids)[0].tolist()
            text = decode_generated(tok, out[len(ids):], args)
            stats = getattr(raw_model, "_last_gen_stats", {})
            print(f"\n[COMPARE] {label}: {format_spec_stats(stats)}", flush=True)
            print(f"[COMPARE] output: {text}", flush=True)
            final_text = text
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return final_text

    stream_enabled = bool(args.stream)
    stream_state = {"printed_text": ""}
    stream_callback = None

    if stream_enabled:
        if stream_prefix is not None:
            print(stream_prefix, end="", flush=True)

        def _callback(out_ids: Sequence[int]) -> None:
            stream_text_delta(tok, out_ids[len(ids):], args, stream_state)

        stream_callback = _callback

    out = generate_ids(
        model, raw_model, idx, cfg, device, dtype, args, stop_ids, forbidden_ids,
        stream_callback=stream_callback,
    )[0].tolist()
    generated_ids = out[len(ids):]
    final_text = decode_generated(tok, generated_ids, args)

    if stream_enabled:
        # Ensure the final trimmed/cleaned answer is represented even if the
        # last token was a stop token or a reasoning strip changed the output.
        if final_text != stream_state.get("printed_text", ""):
            stream_text_delta(tok, generated_ids, args, stream_state)
        print(flush=True)

    if os.environ.get("SPEC_DRAFT", "0") == "1":
        print(f"[SPEC] output: {final_text}", flush=True)
    if getattr(args, "show_stats", False) or os.environ.get("SHOW_GEN_STATS", "0") == "1":
        stats = getattr(raw_model, "_last_gen_stats", {}) or {}
        elapsed = float(stats.get("elapsed_sec", 0.0) or 0.0)
        generated = int(stats.get("generated_tokens", len(generated_ids)) or 0)
        tok_s = generated / max(1e-9, elapsed)
        print(f"[HFCHAT STATS] generated={generated} elapsed={elapsed:.3f}s tok/s={tok_s:.2f}", flush=True)

    return final_text


def read_paste_block(end_marker: str = "/end") -> str:
    print(f"[HFCHAT] Paste your multi-line prompt. Finish with a line containing {end_marker}")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == end_marker:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def print_help():
    print(
        """
Commands:
  /exit, /quit       Exit
  /reset             Clear history
  /raw               Toggle raw prompt mode
  /history           Toggle conversation history
  /stream            Toggle token streaming
  /paste             Multi-line prompt mode; finish with /end
  /settings          Show current decoding/chat settings
  /help              Show this help

Tip:
  For code blocks or long prompts, use /paste. Plain terminal input() treats pasted
  newlines as separate turns, which can corrupt chat history.
""".strip()
    )


def repl(args, model, raw_model, tok, cfg, device, dtype):
    history: List[Tuple[str, str]] = []
    print("[HFCHAT] Type /help for commands. Use /paste for multi-line prompts.")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user:
            continue

        cmd = user.lower()
        if cmd in {"/exit", "/quit", "exit", "quit"}:
            return
        if cmd == "/help":
            print_help()
            continue
        if cmd == "/reset":
            history.clear()
            print("[HFCHAT] History cleared.")
            continue
        if cmd == "/raw":
            args.raw = not args.raw
            print(f"[HFCHAT] raw={args.raw}")
            continue
        if cmd in {"/history", "/toggle-history"}:
            args.history = not args.history
            print(f"[HFCHAT] history={args.history}")
            continue
        if cmd in {"/stream", "/toggle-stream"}:
            args.stream = not args.stream
            print(f"[HFCHAT] stream={args.stream}")
            continue
        if cmd == "/settings":
            print(
                f"[HFCHAT] raw={args.raw} history={args.history} stream={args.stream} temp={args.temp} "
                f"top_k={args.top_k} top_p={args.top_p} repeat_penalty={args.repeat_penalty} "
                f"max_new={args.max_new} block={cfg.block_size} "
                f"KV_CACHE_INT8={os.environ.get('KV_CACHE_INT8', '0')}",
                flush=True,
            )
            continue
        if cmd == "/paste":
            user = read_paste_block()
            if not user:
                continue

        reply = run_prompt(
            args, model, raw_model, tok, cfg, device, dtype, user,
            history if args.history else [],
            stream_prefix="bot> " if args.stream else None,
        )
        if not args.stream:
            print(f"bot> {reply}\n")
        else:
            print()

        if args.history and not args.raw:
            history.append((user, reply))
            history[:] = history[-args.max_turns:]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ternary-adapter", default=None, help="Optional repaired ternary+LoRA adapter .pt file.")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--system", default=os.environ.get("SYSTEM_PROMPT", "You are a helpful coding assistant. Be concise and accurate."))
    ap.add_argument("--device", default=os.environ.get("DEVICE", None))
    ap.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    ap.add_argument("--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", "256")))
    ap.add_argument("--kv-cache-max-len", type=int, default=int(os.environ.get("KV_CACHE_MAX_LEN", "0")))
    ap.add_argument("--kv-cache-sink-tokens", type=int, default=int(os.environ.get("KV_CACHE_SINK_TOKENS", "0")))
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW", "220")))

    # Robust debug defaults: greedy generation. Set TEMP=0.4 or pass --temp for creative chat.
    ap.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "0.0")))
    ap.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "0")))
    ap.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    ap.add_argument("--repeat-penalty", type=float, default=float(os.environ.get("REPEAT_PENALTY", "1.05")))
    ap.add_argument("--repeat-window", type=int, default=int(os.environ.get("REPEAT_WINDOW", "256")))

    ap.add_argument("--no-stop-eos", action="store_true")
    ap.add_argument("--extra-stop-tokens", nargs="*", default=[])
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--suppress-im-start", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hide-reasoning", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--show-stats", action="store_true")
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=os.environ.get("STREAM", "1") != "0")
    ap.add_argument("--verbose-keys", action="store_true")

    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("MAX_TURNS", "6")))
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["KV_CACHE_MAX_LEN"] = str(args.kv_cache_max_len)
    os.environ["KV_CACHE_SINK_TOKENS"] = str(args.kv_cache_sink_tokens)
    model, raw_model, tok, cfg, device, dtype = load_model(args)
    if args.prompt is not None:
        reply = run_prompt(
            args, model, raw_model, tok, cfg, device, dtype, args.prompt, [],
            stream_prefix="" if args.stream else None,
        )
        if not args.stream:
            print(reply)
    else:
        repl(args, model, raw_model, tok, cfg, device, dtype)


if __name__ == "__main__":
    main()
