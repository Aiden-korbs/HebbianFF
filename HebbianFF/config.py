from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CFG:
    seed:        int = 1337
    tok_path:    str = "ff_llm_spm.model"
    # Checkpoint paths are env-overridable so profiling/fresh runs can avoid
    # accidentally loading an incompatible old checkpoint.
    resume_file: str = os.environ.get("RESUME_FILE", "ff_fullres_v10_4_ema_cpu_ctx.pt")
    best_file:   str = os.environ.get("BEST_FILE", "ff_fullres_v10_4_ema_cpu_ctx_best.pt")

    # ── Throughput ───────────────────────────────────────────────────────────
    batch_size:          int  = int(os.environ.get("BATCH_SIZE", "6"))
    block_size:          int  = int(os.environ.get("BLOCK_SIZE", "1024"))
    grad_accum_steps:    int  = int(os.environ.get("GRAD_ACCUM", "8"))
    max_update_steps:    int  = int(os.environ.get("MAX_STEPS", "80000"))
    log_every_updates:   int  = int(os.environ.get("LOG_EVERY", "10"))
    save_every_updates:  int  = int(os.environ.get("SAVE_EVERY", "250"))
    eval_every_updates:  int  = int(os.environ.get("EVAL_EVERY", "50"))
    prefetch_batches:    int  = int(os.environ.get("PREFETCH", "16"))
    eval_pool_size:      int  = int(os.environ.get("EVAL_POOL", "16"))
    eval_pool_seed:      int  = 999999
    log_gpu_memory:      bool = True

    # ── Architecture ─────────────────────────────────────────────────────────
    n_embd:       int   = int(os.environ.get("N_EMBD", "1024"))
    ff_n_layer:   int   = int(os.environ.get("FF_LAYERS", "10"))
    bp_n_layer:   int   = int(os.environ.get("BP_LAYERS", "8"))
    n_head:       int   = int(os.environ.get("N_HEAD", "16"))
    n_kv_head:    int   = int(os.environ.get("N_KV_HEAD", "4"))
    mlp_ratio:    float = float(os.environ.get("MLP_RATIO", "3.6"))
    bp_mlp_ratio: float = float(os.environ.get("BP_MLP_RATIO", "4.0"))
    dropout:      float = float(os.environ.get("DROPOUT", "0.05"))
    use_qk_norm:  bool  = os.environ.get("USE_QK_NORM", "1") != "0"
    # Full residual blocks replace the old half-width reversible coupling.
    # BP checkpointing recovers most of the activation-memory saving without
    # breaking pretrained-weight shape compatibility.
    bp_checkpoint: bool = os.environ.get("BP_CHECKPOINT", "1") != "0"
    emb_init_std: float = 0.02
    # Qwen2/Qwen2.5 use q/k/v projection biases. Llama/TinyLlama do not.
    attn_qkv_bias: bool = os.environ.get("ATTN_QKV_BIAS", "0") != "0"
    # Use HEAD_SCALE=1.0 for transferred HF/Llama-family checkpoints.
    head_scale: float = float(os.environ.get("HEAD_SCALE", "nan"))

    # ── Positional encoding ──────────────────────────────────────────────────
    use_rope:      bool  = True
    rope_theta:    float = float(os.environ.get("ROPE_THETA", "10000.0"))

    # ── FF chunking ──────────────────────────────────────────────────────────
    seq_chunk_size: int = int(os.environ.get("SEQ_CHUNK", "1024"))
    local_window:   int = int(os.environ.get("LOCAL_WINDOW", "512"))
    # CPU-efficient FF stack: keep every MLP block, but replace most attention
    # blocks with a cheap causal depthwise token mixer. Disabled by default so
    # GPU/current checkpoints keep the original all-attention architecture.
    cpu_efficient_ff: bool = os.environ.get("CPU_EFFICIENT_FF", "0") == "1"
    ff_attn_every: int = int(os.environ.get("ATTN_EVERY", "1"))
    ff_force_attn_last: int = int(os.environ.get("FORCE_ATTN_LAST", "0"))
    ff_mixer: str = os.environ.get("CPU_FF_MIXER", "depthwise_conv")
    local_mixer_kernel: int = int(os.environ.get("LOCAL_MIXER_KERNEL", "5"))
    use_fused_swiglu: bool = os.environ.get("FUSED_SWIGLU", "0") == "1"
    eos_token_id:   int = 3
    pad_token_id:   int = 0
    kv_cache_int8:  bool = os.environ.get("KV_CACHE_INT8", "0") == "1"
    # Inference-only rolling cache limit. 0 means "use block_size".
    # This is intentionally separate from block_size so prompts can use the
    # full context while decode retains a smaller K/V window.
    kv_cache_max_len: int = int(os.environ.get("KV_CACHE_MAX_LEN", "0"))
    # Keep this many earliest prompt/cache positions in addition to the recent
    # decode window. This preserves system/instruction tokens better than a pure
    # sliding cache at the same total KV budget.
    kv_cache_sink_tokens: int = int(os.environ.get("KV_CACHE_SINK_TOKENS", "0"))

    # ── Chunk memory ─────────────────────────────────────────────────────────
    memory_tokens: int   = int(os.environ.get("MEMORY_TOKENS", "64"))
    memory_gate:   float = float(os.environ.get("MEMORY_GATE", "0.35"))
    # Experimental fast path for prefix/chunk memory. When enabled, attention
    # uses SDPA is_causal=True with memory prepended to K/V instead of a custom
    # boolean attn_mask. This lets Flash Attention run with MEMORY_TOKENS>0.
    # Disable with FLASH_PREFIX_MEMORY=0 to restore the exact masked local-window path.
    flash_prefix_memory: bool = os.environ.get("FLASH_PREFIX_MEMORY", "1") != "0"

    # ── CPU Context SSM sidecar ──────────────────────────────────────────────
    use_cpu_context_ssm: bool = os.environ.get("CPU_CTX", "0") == "1"
    cpu_context_ckpt: str = os.environ.get("CPU_CTX_CKPT", "cpu_ctx_ssm.pt")
    cpu_context_long_len: int = int(os.environ.get("CPU_CTX_LONG_LEN", "4096"))
    cpu_context_max_mem_tokens: int = int(os.environ.get("CPU_CTX_MAX_MEM", "128"))
    # safer for LM pretraining: only build memory from tokens before the recent window
    cpu_context_prefix_only: bool = os.environ.get("CPU_CTX_PREFIX_ONLY", "1") != "0"

    # ── CPU Context v2: fast hash compressor ─────────────────────────────────
    # CPU_CTX_MODE:
    #   none = disabled
    #   ssm  = old trained CPUContextSSM checkpoint
    #   hash = fast no-training CPU hash context compressor
    cpu_context_mode: str = os.environ.get(
        "CPU_CTX_MODE",
        "hash" if os.environ.get("CPU_HASH_CTX", "0") == "1" else ("ssm" if os.environ.get("CPU_CTX", "0") == "1" else "none")
    )
    use_cpu_hash_context: bool = (
        os.environ.get("CPU_HASH_CTX", "0") == "1"
        or os.environ.get("CPU_CTX_MODE", "none").lower() == "hash"
    )
    cpu_context_dim: int = int(os.environ.get("CPU_CTX_DIM", "512"))
    cpu_context_hash_seed: int = int(os.environ.get("CPU_CTX_HASH_SEED", "1337"))
    cpu_context_gate_init: float = float(os.environ.get("CPU_CTX_GATE_INIT", "-4.0"))

    # ── FF EMA selective-BP circuit breaker ──────────────────────────────────
    use_ff_ema_bp: bool = os.environ.get("FF_EMA_BP", "1") != "0"
    # detach each FF block from final/draft CE; FF params only train on EMA strikes
    ff_ema_detach_layers: bool = os.environ.get("FF_EMA_DETACH", "1") != "0"
    ff_ema_alpha: float = float(os.environ.get("FF_EMA_ALPHA", "0.03"))
    ff_ema_std_mult: float = float(os.environ.get("FF_EMA_STD", "2.25"))
    ff_ema_warmup_steps: int = int(os.environ.get("FF_EMA_WARMUP", "100"))
    ff_ema_max_trips_per_step: int = int(os.environ.get("FF_EMA_MAX_TRIPS", "3"))
    ff_ema_min_abs_delta: float = float(os.environ.get("FF_EMA_MIN_DELTA", "1e-4"))
    ff_ema_bp_weight: float = float(os.environ.get("FF_EMA_BP_WEIGHT", "1.0"))

    # ── Engram ───────────────────────────────────────────────────────────────
    use_engram:             bool  = os.environ.get("USE_ENGRAM", "0") == "1"
    engram_bank_size:       int   = int(os.environ.get("ENGRAM_BANK", "1024"))
    engram_topk:            int   = int(os.environ.get("ENGRAM_TOPK", "8"))
    engram_key_dim:         int   = int(os.environ.get("ENGRAM_KEY_DIM", "192"))
    engram_decay:           float = 0.9999
    engram_min_write_score: float = 0.05
    engram_strength_blend:  float = 0.20
    engram_age_penalty:     float = 0.002
    engram_gate_init:       float = 0.20
    engram_gate_floor:      float = 0.05

    # ── Pre/Post-FF normalization ────────────────────────────────────────────
    use_pre_ff_norm:  bool = os.environ.get("PRE_FF_NORM", "1") != "0"
    use_post_ff_norm: bool = os.environ.get("POST_FF_NORM", "1") != "0"


    # ── Optional modules ─────────────────────────────────────────────────────
    use_draft_head: bool = os.environ.get("USE_DRAFT_HEAD", "0") == "1"
    tie_token_embeddings: bool = os.environ.get("TIE_EMB", "1") != "0"

    # ── Draft head & loss ────────────────────────────────────────────────────
    draft_weight:            float = float(os.environ.get("DRAFT_WEIGHT", "0.08"))
    draft_warmup_steps:      int   = 2000
    draft_ce_clamp_margin:   float = 0.15
    final_ce_weight:         float = 1.0

    # ── Chunked CE ───────────────────────────────────────────────────────────
    head_ce_chunk:    int   = int(os.environ.get("HEAD_CE_CHUNK", "2048"))
    eval_logit_chunk: int   = int(os.environ.get("EVAL_LOGIT_CHUNK", "512"))
    head_token_chunk: int   = int(os.environ.get("HEAD_TOKEN_CHUNK", "1024"))
    head_margin:      float = 0.03

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr:               float = float(os.environ.get("LR", "2.5e-4"))
    emb_lr_mult:      float = 0.35
    head_lr_mult:     float = 1.00
    draft_lr_mult:    float = 1.00
    weight_decay:     float = 0.05
    emb_weight_decay: float = 0.08
    emb_max_norm:     float = 185.0
    grad_clip:        float = 1.0
    eta_min:          float = 1.0e-5
    warmup_update_steps: int = 1000

    # ── Resume behavior ──────────────────────────────────────────────────────
    warm_restart:          bool  = False
    warm_restart_lr_frac:  float = 0.20
    warm_restart_steps:    int   = 5000

    # ── BitNet b1.58 (1-bit weights) ─────────────────────────────────────────
    # Enable with: USE_BITNET=1 python train.py
    # Holdouts (always full precision): tok_emb, out_proj (lm_head),
    #   ChunkMemoryCompressor, FFDraftHead, engram/cpu_ctx projections.
    use_bitnet:            bool  = os.environ.get("USE_BITNET", "0") == "1"
    # Increase EMA warmup for 1-bit: quantised outputs are noisier early
    ff_ema_warmup_steps_bitnet: int = int(os.environ.get("FF_EMA_WARMUP_BIT", "500"))

    # ── Stability ────────────────────────────────────────────────────────────
    use_liger_ce:      bool = os.environ.get("USE_LIGER_CE", "1") != "0"
    use_liger_rmsnorm: bool = os.environ.get("USE_LIGER_RMSNORM", "0") == "1"
    save_rng_state:    bool = True

    # ── Curriculum ───────────────────────────────────────────────────────────
    use_curriculum:        bool = True
    curriculum_start_len:  int  = int(os.environ.get("CURRICULUM_START", "512"))
    curriculum_ramp_steps: int  = int(os.environ.get("CURRICULUM_RAMP", "10000"))

    # ── Data cleaning/tokenizer building ─────────────────────────────────────
    min_text_chars: int = int(os.environ.get("MIN_TEXT_CHARS", "200"))
    max_text_chars: int = int(os.environ.get("MAX_TEXT_CHARS", "12000"))

    # ── Eval / inference ─────────────────────────────────────────────────────
    eval_topk:         Tuple[int, ...] = (1, 5, 64)
    sample_temp:       float = 0.7
    sample_len:        int   = 120
    prompt_for_sample: str   = (
        "The quick brown fox jumps over the lazy dog. "
        "A language model learns by predicting the next token in a sequence. "
    )

    # ── Diagnostic logging ───────────────────────────────────────────────────
    log_recon_error:    bool = os.environ.get("LOG_BLOCK_DELTA", "1") != "0"
    recon_check_tokens: int  = 64

    # ── Gradient retention across chunks ─────────────────────────────────────
    retain_graph_steps: int = 2
