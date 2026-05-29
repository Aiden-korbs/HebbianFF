#!/usr/bin/env python3
"""
Import an open-weight HF model into FF_LLM checkpoint format.

Supported families
------------------
  Qwen2 / Qwen2.5          (1.5B, 3B, 7B, 14B, 32B)
  DeepSeek-R1-Distill-Qwen (any size — same arch as Qwen2.5)
  Llama 3 / 3.1 / 3.2      (1B, 3B, 8B, 70B*)
  Mistral / Mistral-Nemo    (7B)

  * 70B requires enough CPU RAM to load the HF weights (~140 GB).
    For large models use --ff-layers to limit how many layers are imported.

Quick start
-----------
  python tools/import_hf.py Qwen/Qwen2.5-1.5B-Instruct
  python tools/import_hf.py deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --ff-layers 14
  python tools/import_hf.py meta-llama/Meta-Llama-3.1-8B-Instruct
  python tools/import_hf.py /path/to/local/model --out models/my_import.pt
  python tools/import_hf.py Qwen/Qwen2.5-7B-Instruct --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# ── Helpers ──────────────────────────────────────────────────────────────────

def hr(char: str = "─", width: int = 64) -> str:
    return char * width

def fmt_M(n: int) -> str:
    return f"{n / 1e6:.1f}M"

def fmt_B(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else fmt_M(n)


# ── Architecture profiles ─────────────────────────────────────────────────────

# Maps HF model_type → the attribute paths needed to extract weights.
# Probed at runtime with _probe_layer() in case a model variant differs.

_PROFILES = {
    "qwen2": dict(
        attn_qkv_bias   = True,
        attn_norm_attr  = "input_layernorm",
        mlp_norm_attr   = "post_attention_layernorm",
        q_attr          = "self_attn.q_proj",
        k_attr          = "self_attn.k_proj",
        v_attr          = "self_attn.v_proj",
        o_attr          = "self_attn.o_proj",
        gate_attr       = "mlp.gate_proj",
        up_attr         = "mlp.up_proj",
        down_attr       = "mlp.down_proj",
    ),
    "llama": dict(
        attn_qkv_bias   = False,
        attn_norm_attr  = "input_layernorm",
        mlp_norm_attr   = "post_attention_layernorm",
        q_attr          = "self_attn.q_proj",
        k_attr          = "self_attn.k_proj",
        v_attr          = "self_attn.v_proj",
        o_attr          = "self_attn.o_proj",
        gate_attr       = "mlp.gate_proj",
        up_attr         = "mlp.up_proj",
        down_attr       = "mlp.down_proj",
    ),
    "mistral": dict(
        attn_qkv_bias   = False,
        attn_norm_attr  = "input_layernorm",
        mlp_norm_attr   = "post_attention_layernorm",
        q_attr          = "self_attn.q_proj",
        k_attr          = "self_attn.k_proj",
        v_attr          = "self_attn.v_proj",
        o_attr          = "self_attn.o_proj",
        gate_attr       = "mlp.gate_proj",
        up_attr         = "mlp.up_proj",
        down_attr       = "mlp.down_proj",
    ),
}

# Aliases — e.g. DeepSeek-R1-Distill-Qwen is model_type="qwen2"
_ALIASES = {}   # populated automatically by _PROFILES keys

def _get_profile(model_type: str) -> Optional[dict]:
    t = model_type.lower()
    if t in _PROFILES:
        return _PROFILES[t]
    for key in _PROFILES:
        if key in t:
            return _PROFILES[key]
    return None


def _probe_layer(layer) -> dict:
    """
    Inspect the actual HF layer object to find the right attribute names.
    Falls back to profile defaults if everything matches expectation.
    """
    profile = {}

    # MLP norm — some Qwen variants name it differently
    for cand in ("post_attention_layernorm", "post_feedforward_layernorm",
                 "ffn_norm", "post_mlp_layernorm"):
        if hasattr(layer, cand):
            profile["mlp_norm_attr"] = cand
            break

    # Attn norm
    for cand in ("input_layernorm", "attention_norm", "pre_attn_layernorm"):
        if hasattr(layer, cand):
            profile["attn_norm_attr"] = cand
            break

    # Q projection — check for bias
    q_mod = None
    for path in ("self_attn.q_proj", "attention.q_proj", "attn.q_proj"):
        obj = layer
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            q_mod = obj
            profile["q_attr"] = path
            break
        except AttributeError:
            continue

    profile["attn_qkv_bias"] = (q_mod is not None and
                                  getattr(q_mod, "bias", None) is not None)
    return profile


# ── Weight mapping ────────────────────────────────────────────────────────────

def _rget(obj, dotpath: str):
    """Resolve a dotted attribute path on obj."""
    for part in dotpath.split("."):
        obj = getattr(obj, part)
    return obj


def map_layer(hf_layer, dst_block, profile: dict, layer_idx: int, copy_dtype: torch.dtype = torch.float32) -> list[str]:
    """
    Copy one HF transformer layer → one RevBlock.
    Returns a list of warning strings (empty = clean).
    """
    warnings = []
    sd = dst_block.state_dict()

    def copy(dst_key: str, src_attr: str, bias_attr: Optional[str] = None):
        try:
            w = _rget(hf_layer, src_attr)
            tensor = w.weight if hasattr(w, "weight") else w
            sd[dst_key] = tensor.clone().to(copy_dtype)
            if bias_attr and profile.get("attn_qkv_bias"):
                b = _rget(hf_layer, bias_attr)
                bias = b.bias if hasattr(b, "bias") else b
                if bias is not None:
                    sd[bias_attr.replace(".", "_") + "_bias"] = bias.clone().to(copy_dtype)
                    # actual key name in RevBlock state dict:
                    bkey = dst_key.replace(".weight", ".bias")
                    if bkey in sd:
                        sd[bkey] = bias.clone().to(copy_dtype)
        except AttributeError as e:
            warnings.append(f"  layer {layer_idx}: {dst_key} — {e}")

    # Attention
    copy("attn.ln.weight",     profile["attn_norm_attr"] + ".weight")
    copy("attn.q_proj.weight", profile["q_attr"] + ".weight")
    copy("attn.k_proj.weight", profile["q_attr"].replace("q_proj", "k_proj") + ".weight")
    copy("attn.v_proj.weight", profile["q_attr"].replace("q_proj", "v_proj") + ".weight")
    copy("attn.c_proj.weight", profile["o_attr"] + ".weight")

    # QKV biases (Qwen family)
    if profile.get("attn_qkv_bias"):
        for src_pfx, dst_pfx in [
            (profile["q_attr"], "attn.q_proj"),
            (profile["q_attr"].replace("q_proj", "k_proj"), "attn.k_proj"),
            (profile["q_attr"].replace("q_proj", "v_proj"), "attn.v_proj"),
        ]:
            try:
                mod = _rget(hf_layer, src_pfx)
                if hasattr(mod, "bias") and mod.bias is not None:
                    bkey = dst_pfx + ".bias"
                    if bkey in sd:
                        sd[bkey] = mod.bias.clone().to(copy_dtype)
            except AttributeError:
                pass

    # MLP
    copy("mlp.ln.weight",   profile["mlp_norm_attr"] + ".weight")
    copy("mlp.gate.weight", profile["gate_attr"] + ".weight")
    copy("mlp.up.weight",   profile["up_attr"] + ".weight")
    copy("mlp.down.weight", profile["down_attr"] + ".weight")

    dst_block.load_state_dict(sd, strict=False)
    return warnings


# ── CFG builder ───────────────────────────────────────────────────────────────

def cfg_from_hf(hf_cfg, ff_layers: int, bp_layers: int, block_size: int):
    import math
    import os
    from HebbianFF.config import CFG

    inter   = int(hf_cfg.intermediate_size)
    n_embd  = int(hf_cfg.hidden_size)
    n_head  = int(hf_cfg.num_attention_heads)
    n_kv    = int(hf_cfg.num_key_value_heads)

    # HF/Qwen stores the REAL intermediate size:
    #   gate/up: [inter, n_embd]
    #
    # But this repo's RevSwiGLUMLP computes:
    #   h = round64(int(2/3 * mlp_ratio * n_embd))
    #
    # So the cfg.mlp_ratio needed by the local constructor is NOT:
    #   inter / n_embd
    #
    # It is:
    #   inter * 3 / 2 / n_embd
    hf_ratio = inter / n_embd
    ctor_ratio = (inter * 3.0) / (2.0 * n_embd)

    # Nudge upward if float rounding would undershoot after the repo's
    # int(2/3 * ratio * dim) calculation.
    def repo_hidden(ratio: float) -> int:
        return ((int((2.0 / 3.0) * ratio * n_embd) + 63) // 64) * 64

    while repo_hidden(ctor_ratio) < inter:
        ctor_ratio = math.nextafter(ctor_ratio, math.inf)

    got = repo_hidden(ctor_ratio)
    assert got == inter, (
        f"Can't match intermediate_size={inter}; repo_hidden={got}; "
        f"n_embd={n_embd}, hf_ratio={hf_ratio}, ctor_ratio={ctor_ratio}"
    )

    # These env vars are read by CFG() during construction.
    os.environ["MLP_RATIO"]    = str(ctor_ratio)
    os.environ["BP_MLP_RATIO"] = str(ctor_ratio)
    os.environ["N_EMBD"]       = str(n_embd)
    os.environ["N_HEAD"]       = str(n_head)
    os.environ["N_KV_HEAD"]    = str(n_kv)
    os.environ["FF_LAYERS"]    = str(ff_layers)
    os.environ["BP_LAYERS"]    = str(bp_layers)
    os.environ["BLOCK_SIZE"]   = str(block_size)

    cfg = CFG()

    cfg.n_embd       = n_embd
    cfg.n_head       = n_head
    cfg.n_kv_head    = n_kv
    cfg.ff_n_layer   = ff_layers
    cfg.bp_n_layer   = bp_layers
    rope_parameters = getattr(hf_cfg, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        cfg.rope_theta = float(rope_parameters["rope_theta"])
    else:
        cfg.rope_theta = float(getattr(hf_cfg, "rope_theta", 10_000.0))
    cfg.block_size   = block_size

    # IMPORTANT:
    # This repo expects the constructor ratio, not the HF effective ratio.
    cfg.mlp_ratio    = ctor_ratio
    cfg.bp_mlp_ratio = ctor_ratio

    # Keep exact HF values around for import/debug code.
    cfg.hf_mlp_ratio = hf_ratio
    cfg.intermediate_size = inter

    cfg.seq_chunk_size = min(cfg.seq_chunk_size, block_size)
    # Imported HF decoder layers use full causal attention. The local FF path
    # normally defaults to a shorter local window for experimental training,
    # but that is not architecture-compatible with native HF checkpoints.
    cfg.local_window   = block_size

    cfg.head_scale     = 1.0
    cfg.use_qk_norm    = False
    cfg.use_pre_ff_norm = False
    cfg.use_post_ff_norm = False
    cfg.use_draft_head = False
    cfg.use_engram     = False
    cfg.memory_tokens  = 0
    cfg.use_cpu_hash_context = False
    cfg.tie_token_embeddings = bool(getattr(hf_cfg, "tie_word_embeddings", False))

    computed_h = repo_hidden(cfg.mlp_ratio)
    print(
        f"  MLP check: n_embd={n_embd}, "
        f"hf_ratio={hf_ratio:.12f}, "
        f"ctor_ratio={cfg.mlp_ratio:.12f}, "
        f"h={computed_h} expected={inter}"
    )

    assert computed_h == inter, f"MLP size mismatch: {computed_h} != {inter}"

    return cfg


def _identity(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return an exact identity matrix for imported HF parity."""
    return torch.eye(n, dtype=dtype)


# ── Verification ──────────────────────────────────────────────────────────────

def verify_checkpoint(path: str, expected_vocab: int, ff_n_layer: int) -> bool:
    print(f"\n{hr()}")
    print("Verifying checkpoint …")
    ok = True

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    required = ["tok_emb.weight", "out_proj.weight", "final_ln.weight", "final_proj.weight"]
    for key in required:
        if key not in state:
            print(f"  [FAIL] missing key: {key}")
            ok = False

    # Vocab size
    if "tok_emb.weight" in state:
        vs = state["tok_emb.weight"].shape[0]
        if vs != expected_vocab:
            print(f"  [WARN] vocab mismatch: checkpoint={vs}, expected={expected_vocab}")
        else:
            print(f"  [ok]  vocab_size = {vs}")

    # FF blocks — should have meaningful norms (pretrained weights)
    ff_norms = []
    for i in range(min(ff_n_layer, 4)):
        key = f"ff_blocks.{i}.attn.q_proj.weight"
        if key in state:
            ff_norms.append(state[key].norm().item())
    if ff_norms:
        mean_norm = sum(ff_norms) / len(ff_norms)
        if mean_norm < 0.1:
            print(f"  [WARN] FF block norms look too small ({mean_norm:.3f}) — weights may not have transferred")
        else:
            print(f"  [ok]  FF block weight norm (mean of first {len(ff_norms)}): {mean_norm:.3f}")

    # BP blocks — should be ~0 (zero-init)
    bp_key = "bp_blocks.0.attn.c_proj.weight"
    if bp_key in state:
        bp_norm = state[bp_key].norm().item()
        if bp_norm > 0.1:
            print(f"  [WARN] BP c_proj norm={bp_norm:.4f} — expected ~0 (zero-init)")
        else:
            print(f"  [ok]  BP blocks zero-init confirmed (norm={bp_norm:.4f})")

    # Final proj — should be identity for imported HF parity.
    fp_key = "final_proj.weight"
    if fp_key in state:
        diff = (state[fp_key] - torch.eye(state[fp_key].shape[0])).norm().item()
        print(f"  [ok]  final_proj identity deviation: {diff:.4f}")

    n_params = sum(t.numel() for t in state.values() if isinstance(t, torch.Tensor))
    print(f"  [ok]  total params in checkpoint: {fmt_B(n_params)}")

    status = "PASSED" if ok else "FAILED"
    print(f"\n  Verification {status}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model",
        help="HF repo id (e.g. Qwen/Qwen2.5-1.5B-Instruct) or local path")
    p.add_argument("--out", default=None,
        help="Output checkpoint path (default: models/<model_name>.pt)")
    p.add_argument("--ff-layers", type=int, default=None,
        help="Number of HF layers to map into ff_blocks. "
             "Default: all available layers. Use a smaller number to save VRAM.")
    p.add_argument("--bp-layers", type=int, default=0,
        help="Number of BP refinement layers (default: 0 — not needed for FF-only training)")
    p.add_argument("--block-size", type=int, default=2048,
        help="Context length to bake into the config (default: 2048)")
    p.add_argument("--dry-run", action="store_true",
        help="Print what would happen without loading or writing anything")
    p.add_argument("--no-verify", action="store_true",
        help="Skip post-import verification")
    p.add_argument("--dtype", default="float32",
        choices=["float32", "bfloat16"],
        help="Dtype to save weights in (default: float32 for training stability)")
    return p.parse_args()



def make_layer_indices(n_src: int, n_dst: int):
    """
    If importing fewer layers than the HF source has, select layers spread across
    the whole depth instead of just taking the first n_dst layers.

    Example:
      n_src=28, n_dst=8 -> [0, 4, 8, 12, 15, 19, 23, 27]
    """
    if n_dst <= 0:
        return []
    if n_dst >= n_src:
        return list(range(n_dst))
    if n_dst == 1:
        return [n_src - 1]

    idxs = [round(i * (n_src - 1) / (n_dst - 1)) for i in range(n_dst)]

    # Make strictly non-decreasing unique-ish indices, just in case round() duplicates.
    fixed = []
    last = -1
    for x in idxs:
        x = max(last + 1, min(int(x), n_src - 1))
        fixed.append(x)
        last = x

    # If the forward pass correction pushed the tail too far, force the final layer.
    fixed[-1] = n_src - 1
    return fixed


def main():
    args = parse_args()

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        sys.exit("transformers is not installed. Run: pip install transformers")

    # ── Resolve output path ───────────────────────────────────────────────────
    model_slug = Path(args.model).name.replace("/", "_")
    out_path = Path(args.out) if args.out else Path("models") / f"{model_slug}.pt"

    # ── Load HF config (fast — no weights) ───────────────────────────────────
    print(hr())
    print(f"  Source : {args.model}")
    print(f"  Output : {out_path}")

    hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_type = getattr(hf_cfg, "model_type", "unknown").lower()
    profile = _get_profile(model_type)

    if profile is None:
        print(f"\n[WARN] Unknown model_type '{model_type}'. Attempting best-effort import.")
        print("       If it fails, open an issue with your model's config.json.\n")
        profile = _PROFILES["llama"]   # most common structure

    n_hf_layers  = hf_cfg.num_hidden_layers
    ff_layers    = args.ff_layers if args.ff_layers is not None else n_hf_layers
    bp_layers    = args.bp_layers

    if ff_layers > n_hf_layers:
        print(f"[WARN] --ff-layers {ff_layers} > model's {n_hf_layers} layers. Capping.")
        ff_layers = n_hf_layers

    cfg = cfg_from_hf(hf_cfg, ff_layers, bp_layers, args.block_size)
    cfg.attn_qkv_bias = profile["attn_qkv_bias"]

    save_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(hr())
    print(f"  Model type   : {model_type}")
    print(f"  Hidden dim   : {cfg.n_embd}")
    print(f"  Heads / KV   : {cfg.n_head} / {cfg.n_kv_head}")
    print(f"  MLP ratio    : {cfg.mlp_ratio:.3f}")
    print(f"  QKV bias     : {cfg.attn_qkv_bias}")
    print(f"  RoPE theta   : {cfg.rope_theta:.0f}")
    print(f"  Source layers: {n_hf_layers}  →  FF blocks: {ff_layers}  BP blocks: {bp_layers}")
    print(f"  Block size   : {cfg.block_size}")
    print(f"  Save dtype   : {args.dtype}")
    print(hr())

    if args.dry_run:
        print("  --dry-run: nothing written.")
        return

    # ── Load HF model (CPU, no init waste) ───────────────────────────────────
    print("Loading HF weights (CPU) …")
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=save_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  HF model loaded in {time.time() - t0:.1f}s")

    vocab_size = hf.model.embed_tokens.weight.shape[0]
    print(f"  vocab_size = {vocab_size}")

    # Probe the actual layer structure in case it diverges from profile
    if hf.model.layers:
        probed = _probe_layer(hf.model.layers[0])
        profile = {**profile, **probed}   # probed values win

    # ── Build FF_LLM ──────────────────────────────────────────────────────────
    print("\nBuilding FF_LLM …")
    from HebbianFF.config import CFG
    from HebbianFF.model import FF_LLM

    # no_init_weights is defined in scripts/inference/chat_hf.py; replicate it here
    import torch.nn.init as _init
    _noop = lambda t, *a, **kw: t
    _saved = {n: getattr(_init, n) for n in dir(_init) if callable(getattr(_init, n)) and n.endswith("_")}
    for n in _saved:
        setattr(_init, n, _noop)
    _old_uniform = torch.Tensor.uniform_
    _old_normal  = torch.Tensor.normal_
    torch.Tensor.uniform_ = _noop
    torch.Tensor.normal_  = _noop

    model = FF_LLM(vocab_size, cfg)
    if save_dtype != torch.float32:
        model = model.to(dtype=save_dtype)

    for n, fn in _saved.items():
        setattr(_init, n, fn)
    torch.Tensor.uniform_ = _old_uniform
    torch.Tensor.normal_  = _old_normal

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  FF_LLM params: {fmt_B(total_params)}")

    # ── Copy global weights ───────────────────────────────────────────────────
    print("\nCopying global weights …")
    model.tok_emb.weight.data    = hf.model.embed_tokens.weight.clone().to(save_dtype)
    model.out_proj.weight.data   = hf.lm_head.weight.clone().to(save_dtype)
    model.final_ln.weight.data   = hf.model.norm.weight.clone().to(save_dtype)

    # final_proj has no HF equivalent. Use exact identity for parity.
    model.final_proj.weight.data = _identity(cfg.n_embd, dtype=save_dtype)

    # ── Map FF blocks layer by layer, freeing HF layers as we go ─────────────
    layer_indices = list(range(ff_layers + bp_layers))
    if len(layer_indices) != ff_layers + bp_layers:
        raise RuntimeError(
            f"bad layer_indices length: {len(layer_indices)} "
            f"expected={ff_layers + bp_layers}"
        )
    print(f"  Layer map    : {layer_indices}")

    print(f"\nMapping {ff_layers} FF blocks …")
    all_warnings = []
    for i in range(ff_layers):
        hf_layer = hf.model.layers[i]
        src_i = layer_indices[i]
        hf_layer = hf.model.layers[src_i]
        warns = map_layer(hf_layer, model.ff_blocks[i], profile, src_i, copy_dtype=save_dtype)
        all_warnings.extend(warns)
        # Free immediately to keep peak RAM low
        hf.model.layers[i] = None
        torch.cuda.empty_cache()
        if (i + 1) % 4 == 0 or i == ff_layers - 1:
            print(f"  [{i+1:>3}/{ff_layers}] done")

    if all_warnings:
        print("\n  Mapping warnings:")
        for w in all_warnings:
            print(w)

    # BP blocks stay zero-initialized (set by model.__init__ via nn.init.zeros_)
    # No action needed.

    # ── Free remaining HF layers we didn't import ─────────────────────────────
    del hf
    torch.cuda.empty_cache()

    # ── Cast and save ─────────────────────────────────────────────────────────
    print(f"\nCasting to {args.dtype} and saving …")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.to(save_dtype) for k, v in model.state_dict().items()}

    ckpt = {
        "model": state_dict,
        "cfg":   cfg.__dict__,
        "step":  0,
        "source": str(args.model),
    }
    torch.save(ckpt, str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved → {out_path}  ({size_mb:.0f} MB)")

    # ── Verify ────────────────────────────────────────────────────────────────
    if not args.no_verify:
        verify_checkpoint(str(out_path), vocab_size, ff_layers)

    print(f"\n{hr()}")
    print("  Done.")
    print(f"\n  Next steps:")
    print(f"    Sanity check  :  ./scripts/check_model.sh {out_path}")
    print(f"    Quick chat    :  USE_KV_CACHE=1 python scripts/inference/chat_hf.py \\")
    print(f"                       --checkpoint {out_path} \\")
    print(f"                       --tokenizer <your_tokenizer_path>")
    print(f"    Enable draft  :  USE_DRAFT_HEAD=1 (add to your training env)")
    print(hr())


if __name__ == "__main__":
    main()
