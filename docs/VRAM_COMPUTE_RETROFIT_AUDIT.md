# VRAM/Compute Retrofit Audit

Date: 2026-05-21

Scope: read-only research audit of the current FF/BP/SSM/Engram-style repo, with one report file added. No source implementation changes were made.

## Executive Summary

The repository is no longer a pure novel architecture sandbox. It already contains a practical bridge from Hugging Face transformer checkpoints into the local `FF_LLM` format, and the current block implementation is intentionally close to standard decoder blocks. That makes some behavior-preserving retrofit experiments realistic.

The highest-probability path is not replacing attention with SSM or engram memory. That is too disruptive for pretrained transformer behavior without substantial distillation or full retraining. The most realistic path is:

1. Import an existing HF model with all original transformer layers intact.
2. Add only no-op or near-no-op sidecars.
3. Measure exact behavior drift against the native HF model.
4. Adapt small frozen-weight sidecars using KL/logit distillation.
5. Then introduce conservative inference-time savings one at a time: bounded KV cache, quantized KV cache, CPU/offloaded KV for older tokens, optional draft/speculative sidecar, and possibly selective execution only where the confidence signal is strong.

Blunt assessment: if the target is preserving intelligence, attention replacement with SSM/context memory is a high-risk research project. KV-cache compression/offload and draft/speculative decoding are much more likely to reduce VRAM/latency while preserving quality.

## Repository Map

### Core model and blocks

- `HebbianFF/model.py`
  - Defines `FF_LLM`.
  - Builds token embeddings, FF blocks, BP blocks, optional draft head, optional chunk memory compressor, optional engram memory, optional CPU hash context projection, final norm/projection/head.
  - Contains training forward path `forward_features`.
  - Contains KV-cache inference path `prefill_kv` and `decode_one_kv`.
  - Contains eval helpers `eval_metrics` and `generate`.

- `HebbianFF/blocks.py`
  - Defines RoPE helpers.
  - Defines `RevGQACausalAttention`, despite the name this is standard full-width GQA causal attention.
  - Supports local-window attention when used as FF block attention, full block-size attention for BP blocks.
  - Implements KV-cache attention in `forward_kv`.
  - Defines `ResidualBlock`; `RevBlock` is now just a backwards-compatible alias.

- `HebbianFF/memory.py`
  - Defines `FFDraftHead`.
  - Defines `ChunkMemoryCompressor`.
  - Defines `EngramMemoryBank`.

- `HebbianFF/bitnet.py`
  - Defines `BitLinear`, a ternary/1.58-bit linear layer with activation quantization.
  - Intended as a training/inference architecture option, not currently a calibrated post-training transformer quantizer.

- `HebbianFF/utils.py`
  - RMSNorm fallback, seeding, AMP helpers, scheduler helpers.

- `HebbianFF/config.py`
  - Central config dataclass.
  - Includes architecture size, FF/BP layer counts, GQA counts, RoPE, local window, chunk memory, CPU context/CPU hash context, FF EMA BP, engram, draft head, BitNet, eval, and training knobs.

### Attention, SSM, memory, engram

- Attention is in `HebbianFF/blocks.py`.
- Chunk memory and engram memory are in `HebbianFF/memory.py`.
- CPU-side context is represented in config and model as:
  - `CPU_CTX_MODE=ssm` / `use_cpu_context_ssm`
  - `CPU_HASH_CTX=1` / `use_cpu_hash_context`
  - `cpu_ctx_proj` and `cpu_ctx_gate` in `FF_LLM`
- I did not find a concrete CPU SSM module implementation in the current checked-in files. The old SSM mode is config/runtime scaffolding unless the checkpoint path supplies an external module not present here.

### Training and adaptation

- `train_ff_draft_repair.py`
  - Fine-tunes selected subsets of an imported/custom checkpoint.
  - Supports modes `draft`, `draft_head`, `bp_draft_head`, and `all`.
  - Freezes most weights by default and adapts selected heads/BP modules.

- `train_ff_then_draft.py`
  - Two-stage training script.
  - Stage `ff`: trains FF blocks only.
  - Stage `draft`: trains draft head only from frozen FF hidden states.

### Inference and serving

- `chat_hf.py`
  - Loads local `FF_LLM` checkpoints.
  - Uses HF tokenizer.
  - Supports `USE_KV_CACHE=1` through `prefill_kv` and `decode_one_kv`.
  - Generates one token at a time with optional sampling controls.

- `web_chat/server.py`
  - FastAPI wrapper around `chat_hf.py`.
  - Sets stable runtime defaults: KV cache on, draft head off, memory/engram off.

- `web_chat/static/index.html`
  - Browser UI.

### Checkpoint import and validation

- `tools/import_hf.py`
  - Imports Qwen2/Qwen2.5, Llama, Mistral-family HF weights into `FF_LLM`.
  - Maps HF attention and MLP weights into local FF blocks.
  - Can import fewer layers with `--ff-layers`.
  - BP blocks default to zero layers.
  - Initializes `final_proj` near identity.

- `tools/check_checkpoint.py`
  - Checks required keys and prints config.

- `scripts/import_model.sh`
  - Wrapper around `tools/import_hf.py`.

- `scripts/check_model.sh`
  - Wrapper around `tools/check_checkpoint.py`.

### Tokenizer and data

- `ff_llm_spm.vocab`
  - SentencePiece vocab artifact, but current runtime expects an HF tokenizer path or model id.

- `chat_hf.py`, `compare_native_qwen_eval.py`, and training scripts use `transformers.AutoTokenizer`.

- Data directories present but not audited deeply:
  - `data_qwen_repair/`
  - `data_clean_1b/`
  - `data_tinyllama_tok_50m/`

### Evaluation

- `compare_native_qwen_eval.py`
  - Compares a custom checkpoint against native HF Qwen on multiple-choice tasks.
  - Records accuracy and peak allocated CUDA memory.
  - Useful but not sufficient for retrofit audit because it does not yet measure KL divergence, top-k agreement, KV memory, prefill tok/s, or decode tok/s.

## Retrofit Idea Classification

| Idea | Classification | Pretrained weights untouched | New modules required | No-op init possible | VRAM savings | Compute savings | Quality risk | Difficulty | Repo implementation points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Preserve imported HF layers exactly in `FF_LLM` and add no-op sidecars | Inference-only baseline / requires short adaptation for sidecars | All imported layer weights, embeddings, lm head | Optional gates/projections/sidecars | Yes | None initially | None initially | Low if all sidecars gated off | Low | `tools/import_hf.py`, `model.py`, `chat_hf.py` |
| Bounded KV cache / sliding window using current `max_cache_len` | Inference-only retrofit | All weights | None | Already available | High at long context | Moderate at long decode due smaller attention over cache | Medium: loses old context | Low | `RevGQACausalAttention.forward_kv`, `FF_LLM.prefill_kv`, `decode_one_kv`, `CFG.block_size` |
| Quantized KV cache | Inference-only retrofit | All weights | KV quant/dequant wrapper, cache dtype metadata | Near no-op if disabled; exact no-op impossible when quantized | High, roughly 2x for int8 KV or 4x for int4 KV cache | Mixed: memory bandwidth can improve, dequant overhead can hurt | Low-medium for int8, medium-high for int4 | Medium | `blocks.py` KV cache path, `chat_hf.py` metrics |
| CPU/offloaded KV cache for old tokens | Inference-only retrofit | All weights | Cache manager, pinned CPU buffers, async transfer, attention split | Yes when disabled | High GPU VRAM savings | Usually slower unless long context and careful paging | Low if exact KV is preserved; latency risk high | Medium-high | `blocks.py` forward_kv cache format, `model.py` cache dict |
| CPU hash context replacing distant KV | Requires short adaptation | Base weights frozen | CPU hash compressor, projection, learned gate | Yes if gate init near zero | Medium-high at long context if old KV dropped | Moderate if it avoids long attention | Medium-high; compressed memory is not behavior-preserving | Medium | Existing `use_cpu_hash_context`, `cpu_ctx_proj`, `forward_features`; missing KV decode integration |
| CPU SSM context replacing distant KV | Requires short adaptation / finetune-only retrofit | Base weights can stay frozen | Actual CPU SSM module, projection/gate, training loss | Yes if gate zero | Medium-high at long context | Possibly moderate | High; SSM summary is not equivalent to transformer KV | High | Config scaffolding in `config.py`; concrete module absent |
| Chunk memory compressor as attention prefix | Requires short adaptation | Base weights can stay frozen | `ChunkMemoryCompressor`; attention prefix already exists | Not exact no-op unless memory disabled or gate zero | Medium if full KV/history is reduced | Potential prefill savings if fewer tokens retained | Medium-high | Medium | `memory.py`, `model.forward_features`, `blocks.py` memory prefix attention |
| Engram recurrent memory as alternative to full attention | Requires full retraining for strong use; short adaptation only as weak sidecar | Base weights can stay frozen if gate tiny | `EngramMemoryBank`, key/value projections, gate | Yes with gate near zero | Low-medium unless KV is actually removed | Low-medium | High; retrieval can inject wrong latent state | Medium-high | `memory.py`, `model.forward_features` |
| Replace selected attention layers with SSM/context modules | Requires full retraining | Some original non-attention weights can stay, replaced attention weights unused | SSM blocks, projection bridges, distillation losses | Residual gate can be no-op, but replacement is not no-op once used | Medium-high if attention/KV removed | Medium-high at long contexts | Very high | High | New block type in `blocks.py`, config routing, import mapping |
| Draft head sidecar for speculative decoding | Requires short adaptation | Original model untouched | Small draft model/head, verification path | Yes if disabled; draft logits need training | Little VRAM saving; may add VRAM | Decode compute savings if accepted tokens are high | Low if verifier is original model | Medium | Existing `FFDraftHead`; generation lacks speculative verifier loop |
| FF draft head as replacement for final decode | Finetune-only retrofit / unlikely for preservation | Base can freeze | Draft head | No exact no-op unless disabled | None or negative unless skipping BP/layers | Potential large savings if skipping heavy path | High; unverified draft tokens degrade intelligence | Medium | `memory.FFDraftHead`, `train_ff_then_draft.py`, `chat_hf.py` |
| Selective layer execution / layer skipping | Requires short adaptation | Skipped layers untouched but unused on some tokens | Confidence router/gates, calibration loss | Yes if route always executes all layers | Medium activation/KV savings if layers skipped | Medium-high | High without verifier/correction | Medium-high | `model.decode_one_kv`, `ResidualBlock`, generation loop |
| Early exit / confidence routing | Requires short adaptation | Lower layers and head untouched; later layers skipped sometimes | Intermediate heads or shared head projections, confidence metric | Yes if threshold never exits | Medium compute savings | Medium-high when exits frequent | High for hard tokens | Medium-high | Add intermediate logits in `model.py`, routing in `chat_hf.py` |
| Sparse correction BP blocks | Requires short adaptation | Imported FF/base weights frozen | Zero-init BP/correction blocks, router | Yes, BP output projections already zero-init in constructor | VRAM negative unless replacing skipped layers | Compute savings only if correction rare and base path cheaper | Medium | Medium | Existing BP blocks in `model.py`, `train_ff_draft_repair.py` |
| FF-only or mostly-FF pass with BP correction when needed | Requires short adaptation / finetune-only | FF/imported weights can stay; BP trained | Router, BP correction blocks, draft/confidence signal | Yes if BP zero or router off | Possible if BP skipped most tokens | Possible if BP rare | Medium-high | Medium | `ff_blocks`, `bp_blocks`, `ff_draft_head`, `train_ff_draft_repair.py` |
| Adapters initialized as no-op | Requires short adaptation | All original weights frozen | LoRA/adapters in attention/MLP or memory bridges | Yes | None by itself | None by itself | Low | Low-medium | Add adapters to `blocks.py` projections or wrapper modules |
| Low-rank projection bridges for context/memory | Requires short adaptation | Base weights frozen | Down/up projections and gate | Yes with zero/up gate | Low-medium depending on use | Low-medium | Medium | Low-medium | `cpu_ctx_proj`, possible LoRA-style additions |
| BitNet conversion of imported weights | Requires full retraining or heavy quantization-aware adaptation | Float weights may be latent, but behavior changes | `BitLinear` already exists | No, quantization is not identity | High for weights if stored/served quantized | Not guaranteed in current PyTorch implementation | High for post-training conversion | Medium | `bitnet.py`, `make_linear`, `CFG.use_bitnet` |
| CPU/GPU split execution of layers | Inference-only retrofit | All weights | Device map/layer offload scheduler | Yes if all GPU | High if many layers on CPU | Usually negative latency unless GPU memory constrained | Low behavior risk, high latency risk | Medium | Model module placement, `chat_hf.load_model`, generation loop |
| Speculative decoding with separate HF small draft model | Inference-only retrofit | Target model untouched | Draft HF model, verifier loop | Yes if disabled | Negative VRAM if draft on GPU; can put draft on CPU/smaller GPU | Decode speedup possible | Low if exact verification used | Medium | `chat_hf.generate_ids`; can reuse native HF draft or `FFDraftHead` |

## What Is Realistic

### Realistic with high preservation

- Exact imported model plus KV-cache engineering.
- Bounded cache with explicit long-context quality measurement.
- Int8 KV-cache quantization.
- CPU/offloaded exact KV for old tokens if latency can be tolerated.
- Speculative decoding with exact verification by the original model.
- No-op initialized adapters or low-rank bridges trained by KL distillation.

These keep the original transformer computation as the authority. Quality risk is measurable and controllable.

### Plausible but research-heavy

- CPU hash context as a replacement for distant KV.
- Chunk memory replacing older prompt tokens.
- Sparse BP correction after a cheaper pass.
- Early exit / selective layer execution with confidence routing.

These can save memory/compute, but only when they actually remove original work. The moment original attention or layers are skipped, behavior is no longer preserved by construction. They need KL distillation and strict abort criteria.

### Low-probability for behavior-preserving retrofit

- Replacing attention layers with SSM modules.
- Engram memory as a replacement for attention.
- FF-only generation without verification.
- BitNet conversion as a drop-in post-training retrofit.

These may become useful if the goal changes to training a new efficient architecture, but they are not the first tools for preserving an existing pretrained model's intelligence.

## Highest-Probability Strategy

The strongest strategy is a staged, conservative retrofit:

1. Native HF baseline.
2. Imported `FF_LLM` baseline with all HF layers mapped and all sidecars off.
3. Add a no-op retrofit wrapper around KV cache handling and optional sidecars.
4. First save VRAM with KV-cache policy, not with layer/attention replacement:
   - bounded cache,
   - int8 KV cache,
   - optional CPU offload for older KV.
5. Add optional speculative decoding:
   - use a small draft model or trained `FFDraftHead`,
   - verify tokens with the unchanged target model,
   - count acceptance rate and end-to-end tokens/sec.
6. Only after the above works, test learned compressed memory for distant context as an approximation to old KV.

Reason: KV cache dominates long-context inference memory and can be changed without modifying pretrained weights. Speculative decoding can reduce decode compute while preserving exact output distribution if verification is implemented correctly. Attention replacement cannot make that guarantee.

## Minimal Experiment: Qwen2.5-Coder-0.5B or 1.5B

Preferred first model:

- `Qwen/Qwen2.5-Coder-0.5B-Instruct` if available locally or downloadable.
- Otherwise `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

### Compared systems

#### A. Baseline pretrained model

Use `transformers.AutoModelForCausalLM` directly.

Purpose:

- Establish canonical logits, memory, speed, and generation behavior.

#### B. Baseline with proposed retrofit in no-op or near-no-op mode

Use either:

- native HF model with an external cache wrapper disabled, or
- imported `FF_LLM` checkpoint with all layers imported, `USE_KV_CACHE=1`, no memory/engram/draft, and all gates disabled.

Required condition:

- Logits should match A closely. If imported `FF_LLM` differs materially from native HF before any efficiency feature is enabled, stop and debug import/parity first.

Suggested no-op settings:

- `USE_KV_CACHE=1`
- `USE_DRAFT_HEAD=0`
- `DRAFT_BLEND_BP=0`
- `INFER_MEMORY_TOKENS=0`
- `INFER_USE_ENGRAM=0`
- `CPU_HASH_CTX=0`
- `CPU_CTX=0`

#### C. Retrofitted model after short adaptation

Freeze original pretrained weights. Train only sidecars:

First adaptation target:

- int8 KV cache has no trainable module, so test directly.
- for learned modules, train low-rank/context bridge or draft sidecar with KL distillation from A.

Recommended C variants:

1. `C1`: int8 KV cache, no trainable adaptation.
2. `C2`: bounded/sliding KV plus CPU hash/chunk context sidecar, original weights frozen, train only projection/gate/adapters.
3. `C3`: speculative draft sidecar, target model unchanged, train draft for acceptance rate.

Do not start with SSM replacement.

### Adaptation objective

Use a small mixed corpus:

- code snippets,
- short reasoning samples,
- plain chat/instruction samples,
- long-context synthetic retrieval samples.

Loss:

- KL divergence from baseline logits at temperature 1 or 2.
- Optional CE on true next token with low weight.
- Optional hidden-state MSE only if representations are directly comparable.

Freeze:

- token embeddings,
- all original attention projections,
- all original MLPs,
- final norm,
- lm head.

Train:

- cache quantization calibration parameters if any,
- context projection/gate,
- low-rank adapters,
- draft sidecar,
- router thresholds if used.

### Pass/fail gates

Move to bigger models only if:

- B vs A KL is near zero.
- B vs A top-1 agreement is very high.
- C keeps perplexity close to A.
- C does not collapse on simple code/reasoning samples.
- VRAM or tokens/sec improvement is real, not just noise.

## Exact Metrics

### Memory

- Peak VRAM allocated:
  - `torch.cuda.reset_peak_memory_stats()`
  - `torch.cuda.max_memory_allocated()`

- Peak VRAM reserved:
  - `torch.cuda.max_memory_reserved()`

- Static model memory:
  - sum parameter bytes by dtype/device.

- KV cache memory:
  - exact bytes from all cached K/V tensors:
    - `sum(t.numel() * t.element_size() for layer in cache for t in layer)`
  - report separately for FF and BP caches in current `FF_LLM`.
  - for native HF, inspect `past_key_values`.

### Speed

- Prompt prefill tokens/sec:
  - fixed prompt lengths: 128, 512, 1024, 2048, and max tested context.
  - synchronize CUDA before/after timing.

- Decode tokens/sec:
  - generate 128 or 256 tokens after prefill.
  - measure steady-state decode separately from prefill.

- End-to-end tokens/sec:
  - include prompt processing and decode.

- Speculative decoding:
  - acceptance rate,
  - target forward calls per generated token,
  - draft tokens proposed per step,
  - exact-match output mode if deterministic.

### Distribution preservation

Compute on fixed tokenized batches:

- Perplexity / cross entropy.
- KL divergence:
  - `KL(baseline || retrofit)` and optionally symmetric KL.
  - Use fp32 logits for metric computation.

- Top-1 next-token agreement:
  - `argmax(logits_A) == argmax(logits_B)`.

- Top-5 agreement:
  - baseline top-1 appears in retrofit top-5,
  - retrofit top-1 appears in baseline top-5,
  - top-5 set Jaccard.

- Logit MSE and cosine similarity:
  - useful diagnostic, not a primary quality metric.

### Quality sanity

Use deterministic generation first:

- temperature 0,
- same prompts,
- compare outputs.

Sample prompts:

- Python function implementation.
- Bug fix request.
- SQL query generation.
- Simple math word problem.
- Multi-step reasoning puzzle.
- Long-context recall prompt where the answer is only in early context.
- Refusal/safety-neutral instruction following prompt.

### Mini-evals

Use small, fast evals:

- HumanEval-style few samples or local code tasks if available.
- MBPP subset if available.
- ARC-Challenge subset.
- BoolQ subset.
- PIQA subset.
- A small handcrafted code/reasoning set committed outside model code or generated as JSONL.

The existing `compare_native_qwen_eval.py` can be extended for this, but it currently measures accuracy and peak memory only. It needs added logit-parity and speed/KV metrics for this research question.

## Suggested Implementation Locations After Approval

No implementation was done in this audit. If approved, the lowest-risk code additions would be:

1. `tools/measure_retrofit.py`
   - New benchmark script for A/B/C.
   - Loads native HF and imported/custom model.
   - Reports VRAM, KV bytes, prefill tok/s, decode tok/s, perplexity, KL, top-k agreement, and sample generations.

2. `HebbianFF/kv_cache.py`
   - New cache utilities:
     - byte accounting,
     - optional int8 quant/dequant,
     - optional CPU offload manager.

3. `HebbianFF/blocks.py`
   - Minimal changes only inside `forward_kv` cache handling.
   - Preserve existing unquantized path as exact baseline.

4. `chat_hf.py`
   - Add benchmark-friendly generation hooks.
   - Later add speculative verification loop.

5. `train_retrofit_distill.py`
   - New short adaptation script.
   - Freezes original weights and trains only sidecars against baseline logits.

## Specific Notes on Current Architecture

- `ResidualBlock` is standard residual attention plus SwiGLU MLP. This is good for weight transfer.
- `RevBlock` naming is stale. It is no longer reversible.
- `tools/import_hf.py` currently maps HF layers into FF blocks and defaults BP layers to 0. That is the right starting point for behavior preservation.
- BP blocks are zero-initialized in constructor for `attn.c_proj.weight` and `mlp.down.weight`, which makes them plausible correction modules after adaptation.
- `final_proj` is near-identity in import, not exact identity. For strict parity experiments, exact identity may be preferable.
- Runtime defaults already disable memory/engram/draft for stable inference. That matches the audit conclusion.
- Current engram state lives on GPU, so it is not currently a CPU VRAM-saving memory path.
- Current CPU hash context is projected onto GPU once and then prepended as memory. It can save VRAM only if it replaces retained tokens/KV, not if it is added on top.
- The current KV path truncates cache to `block_size`. This already bounds KV memory, but it is a sliding-context approximation, not a behavior-preserving full-context method.

## Recommended First Experiment Protocol

1. Import Qwen2.5-Coder small model:
   - all layers,
   - no BP layers,
   - no draft,
   - no memory,
   - bf16 checkpoint for inference if parity is acceptable.

2. Run native HF vs imported `FF_LLM` parity:
   - same tokenizer,
   - same prompts,
   - same dtype,
   - no sampling,
   - compare logits on fixed batches.

3. Add a benchmark script before any architectural changes:
   - this prevents confusing architectural regressions with measurement noise.

4. Test inference-only cache policies:
   - full/bounded bf16 KV baseline,
   - int8 KV,
   - bounded KV at smaller window sizes,
   - optional exact CPU-offloaded old KV.

5. Only then train sidecars:
   - freeze original weights,
   - train small bridge/draft/router modules,
   - use KL to native baseline as the main objective.

## Expected Outcomes

Likely wins:

- KV-cache memory reduction from int8/offload with low quality impact.
- Decode speed improvements from speculative decoding if acceptance rate is high.
- Long-context VRAM reduction from bounded cache, with measurable quality loss on long-range recall.

Uncertain:

- CPU hash/chunk memory can recover some long-context behavior after old KV is dropped, but it will not be equivalent to full attention.
- Sparse BP correction may help repair cheap paths, but routing quality is the hard part.

Unlikely:

- Drop-in SSM replacement of pretrained attention while preserving intelligence after only short adaptation.
- Engram memory replacing full attention without retraining.
- BitNet conversion of pretrained weights as a high-quality no-retraining retrofit.

## Bottom Line

For the stated research goal, the repo should treat the existing pretrained model as the teacher and the source of truth. Start with exact or near-exact behavior, then remove memory/compute in reversible, measurable increments. The first serious target should be KV-cache memory and speculative decoding, not SSM replacement.

