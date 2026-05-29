# Architecture Implementation Plan

This plan orders the existing architecture ideas by practical implementation complexity, with the main goal of lowering inference VRAM while preserving intelligence and keeping throughput high.

The safest principle is: keep the imported transformer path as the authority, then add memory and throughput optimizations around it. Anything that removes attention, skips layers, or replaces logits needs stronger parity tests and usually adaptation.

## Current Repo Layout

The codebase has been organized around the `HebbianFF` package and grouped entry-point scripts:

- `HebbianFF/`: core model, blocks, config, BitNet layers, memory helpers, packed runtimes, and KV cache paths.
- `scripts/inference/`: local chat and legacy Flask web-chat entry points.
- `scripts/training/`: training scripts, launchers, resume helpers, and smoke tests.
- `scripts/dataset_builders/`: dataset preparation scripts.
- `scripts/benchmarks/`: CPU/GPU benchmark launchers.
- `tools/`: import, evaluation, compression, ternary repair, measurement, and diagnostic tools.
- `docs/`: research notes and runtime documentation.

Run CLI examples from the repository root so local imports resolve consistently.

## Training Plan

The training track is separate from the inference retrofit track. Inference work should preserve imported checkpoint behavior by default; training work is where the FF/BP, Hebbian/EMA, draft-head, memory, and BitNet-style ideas can be changed and evaluated directly.

Primary training entry points:

- `scripts/training/train_ff_only_ternary_ema.py`: main FF-only ternary/BitNet EMA trainer.
- `scripts/training/run_3070_tinystories.sh`: small TinyStories launcher for quick architecture checks.
- `scripts/training/run_500m_fineweb_edu.sh`: larger FineWeb-Edu training configuration.
- `scripts/training/run_500m_cpu_efficient.sh`: CPU-oriented wrapper around the FineWeb-Edu config.
- `scripts/training/train_ff_draft_repair.py`: draft/correction repair training path.
- `scripts/training/train_ff_then_draft.py`: two-stage FF plus draft-head training path.
- `scripts/training/resume_eval_tiny.sh` and `scripts/training/train_tiny_roundtrip_test*.sh`: resume and smoke-test helpers.

Dataset builders:

- `scripts/dataset_builders/build_safe_pretrain_data.py`: small safe pretraining bins, currently suited to TinyStories-style checks.
- `scripts/dataset_builders/build_large_pretrain_data.py`: larger streaming pretraining bin builder.
- `scripts/dataset_builders/build_fineweb_edu_500m_data.sh`: FineWeb-Edu wrapper around the large builder.

Recommended training order:

1. Keep a tiny smoke loop passing before changing architecture internals.
2. Train a small FF-only ternary EMA baseline on TinyStories.
3. Scale the same code path to FineWeb-Edu only after loss curves, resume, eval, and checkpoint saves are stable.
4. Add draft-head or BP correction training with the base path frozen or carefully staged.
5. Only then train memory sidecars, CPU hash context, or engram retrieval modules.
6. Compare trained checkpoints with the same inference harness used for imported models.

Current stable training defaults:

```bash
USE_BITNET=1
BP_LAYERS=0
USE_DRAFT_HEAD=0
MEMORY_TOKENS=0
USE_ENGRAM=0
CPU_HASH_CTX=0
FF_EMA_BP=1
```

Tiny smoke workflow:

```bash
python scripts/dataset_builders/build_safe_pretrain_data.py \
  --dataset roneneldan/TinyStories \
  --out-dir data/ternary_tinystories \
  --vocab-size 16000 \
  --tokenizer-records 100000 \
  --train-records 300000 \
  --val-records 10000

./scripts/training/run_3070_tinystories.sh --preset tiny
```

FineWeb-Edu workflow:

```bash
TARGET_TRAIN_TOKENS=1000000000 \
TARGET_VAL_TOKENS=10000000 \
./scripts/dataset_builders/build_fineweb_edu_500m_data.sh

./scripts/training/run_500m_fineweb_edu.sh
```

Training acceptance criteria:

- Training loss and validation loss decrease on the tiny workflow before scaling.
- Resume-from-checkpoint reproduces sensible loss continuity.
- `metrics.jsonl` contains enough information to compare runs by loss, tokens/sec, VRAM, and EMA diagnostics.
- Generated samples from checkpoints are checked before declaring an architecture change useful.
- For draft/BP/memory features, compare against the FF-only baseline at the same data, steps, and parameter budget.

Near-term training priorities:

1. Keep the FF-only ternary EMA path as the reference training baseline.
2. Improve run comparability: fixed seeds, standard tiny config, standard FineWeb-Edu config, and consistent metrics.
3. Add focused eval hooks to training outputs instead of judging by loss alone.
4. Train draft head separately and measure whether it is accurate enough for speculative decoding.
5. Train BP correction blocks only after the base FF path is stable.
6. Train memory/CPU-context sidecars with KL distillation or frozen-base adaptation, not random serving-time enablement.

## Progress Checklist

- [x] 1. Bounded KV Cache: implemented and smoke-tested.
- [x] 2. Sink + Recent KV Cache: implemented and parity-tested against sequential decode.
- [x] 3. KV Cache Measurement Harness: implemented for full, bounded, sink, and int8 policies.
- [x] 4. Int8 KV Cache: implemented and verified; not recommended as default because throughput/quality tradeoff is mixed.
- [x] 5. Flash-Compatible Prefix Memory Path: implemented and verified against exact masked fallback.
- [x] 6. Chunk Memory Compressor: evaluated and rejected as a stable no-training VRAM optimization.
- [ ] 7. CPU Hash Context: current stopping point; existing scaffolding needs distillation/adaptation before serving use.
- [ ] 8. Engram Memory Bank: existing module; research-only until evaluated.
- [ ] 9. Draft Head Logit Blend: existing path; disabled until trained and calibrated.
- [ ] 10. Speculative Decoding With Draft Head: helpers exist; blocked on trained draft head.
- [ ] 11. FF MLP Skip / Layer Skip: helper paths exist; not safe without verifier/evals.
- [ ] 12. BP Correction Blocks: supported by architecture; requires training.
- [ ] 13. BitNet / Weight Quantized Architecture: existing path; training project, not retrofit-complete.
- [ ] 14. Attention Replacement With SSM or Memory: research track, not started.

Current position: item 7, `CPU Hash Context`.

## 1. Bounded KV Cache

Complexity: Low

Status: Complete.

Verification:

- `KV_CACHE_MAX_LEN` is wired through config and chat runtime.
- Short retrofit run showed bounded cache reducing decode KV from about 1.2 MiB to about 0.8 MiB on a 128-token test.
- Native-vs-custom parity passed in the same run.

Use `KV_CACHE_MAX_LEN` to cap the decode-time K/V cache. This directly reduces VRAM and attention work during long generation without changing model weights.

Recommended default:

```bash
KV_CACHE_MAX_LEN=0
KV_CACHE_SINK_TOKENS=0
KV_CACHE_INT8=0
```

Set `KV_CACHE_MAX_LEN` below `BLOCK_SIZE` when VRAM is constrained. Quality risk increases as the cache becomes shorter because old prompt tokens become invisible during decode.

## 2. Sink + Recent KV Cache

Complexity: Low to Medium

Status: Complete.

Verification:

- Cache stores absolute positions so sink + recent eviction keeps causal masking correct.
- Full prefill parity for prompts inside the cache limit: `max_abs=0.0`, top-1 identical.
- `decode_many_kv` vs sequential `decode_one_kv` under sink cache: top-1 identical, fp16 `max_abs=0.015625`.

Use `KV_CACHE_SINK_TOKENS` together with `KV_CACHE_MAX_LEN`. The cache keeps a fixed prefix plus the most recent tokens. This is usually a better chat tradeoff than pure sliding because the system prompt and initial user instruction stay visible.

Example:

```bash
KV_CACHE_MAX_LEN=512
KV_CACHE_SINK_TOKENS=64
```

Validation required:

- Full-cache vs sink-cache generation samples.
- Long-context recall tests where the important fact is inside and outside the sink region.
- Throughput measurement, because non-contiguous cache patterns may add small overhead.

## 3. KV Cache Measurement Harness

Complexity: Low

Status: Complete.

Verification:

- `tools/measure_retrofit.py` supports `full`, `bounded`, `sink`, and `int8` cache policies.
- The harness reports parity, speed, KV memory, samples, and long-context recall fields.

Keep extending this before adding riskier architecture changes. Every new policy should report:

- Peak allocated and reserved CUDA memory.
- Parameter bytes.
- KV cache bytes by layer.
- Prefill tokens/sec.
- Decode tokens/sec.
- KL and top-k agreement against the native/full-cache baseline.
- Long-context recall behavior.

This is the guardrail that prevents memory savings from silently damaging intelligence.

## 4. Int8 KV Cache

Complexity: Medium

Status: Complete, but conditional.

Verification:

- Command:

```bash
python tools/measure_retrofit.py \
  --native-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --ff-checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --block-size 128 \
  --max-eval-tokens 48 \
  --speed-prompt-tokens 96 \
  --decode-tokens 8 \
  --sample-tokens 8 \
  --cache-policies full,int8 \
  --long-recall-lengths 128 \
  --long-recall-cache-lens 64 \
  --long-recall-new-tokens 8 \
  --parity-max-abs-threshold 1.0 \
  --no-fail-exit
```

- Native-vs-custom baseline parity passed: `KL=0.000745837`, top-1 `100.00%`.
- Full policy: `KV=1.2 MiB`, decode `116.32 tok/s`, policy KL `0.000678571`, top-1 `98.73%`.
- Int8 policy: `KV=0.6 MiB`, decode `87.47 tok/s`, policy KL `0.00691534`, top-1 `94.94%`.
- Long-context recall control passed for int8: generated `MANGO7429`.

Decision:

- Keep `KV_CACHE_INT8=0` as the stable default.
- Use `KV_CACHE_INT8=1` only when VRAM is the bottleneck and lower decode throughput is acceptable.

This stores cached K/V tensors as int8 plus scale tensors. It can reduce KV memory substantially, especially at long context, but dequantization can reduce throughput.

Recommended order:

1. Validate fp/bf16 full cache.
2. Validate bounded or sink cache.
3. Add `KV_CACHE_INT8=1` only if KV memory is still the bottleneck.

Acceptance criteria:

- Top-1 agreement remains high against the non-int8 policy.
- Long-context recall does not regress materially.
- Decode tok/s is not worse enough to erase the benefit.

## 5. Flash-Compatible Prefix Memory Path

Complexity: Medium

Status: Complete for the attention path.

The current attention path has a flash-prefix mode for chunk memory. This is useful only if memory tokens are enabled. For stable inference, keep `INFER_MEMORY_TOKENS=0` until the baseline cache policies are fully measured.

Reasoning: prefix/chunk memory is an approximation. It can help compress long context, but it is not behavior-preserving unless trained or calibrated.

Verification:

- Synthetic prefix-memory attention test compared `FLASH_PREFIX_MEMORY=1` to the exact masked fallback with the same weights and inputs.
- Result: `max_abs=8.940696716308594e-08`, `mean_abs=8.681354302098043e-09`, `allclose_1e-5=True`.

Boundary:

- This verifies the flash-compatible attention implementation.
- It does not make chunk memory a stable serving feature; that is item 6.

## 6. Chunk Memory Compressor

Complexity: Medium

Status: Complete as an evaluation; not enabled for stable inference.

`ChunkMemoryCompressor` can summarize previous hidden states into memory tokens. This may reduce dependence on full KV history, but it changes the information path and needs adaptation.

Verification:

- Tested imported `models/Qwen2.5-Coder-0.5B-Instruct.pt` with `INFER_MEMORY_TOKENS=8` against the same checkpoint with memory disabled.
- The checkpoint has no `mem_compressor.*` weights, so enabling the module introduced randomly initialized parameters.
- Result:
  - `missing_compressor_weights=True`
  - `max_abs_logit_diff=27.9375`
  - `mean_abs_logit_diff=2.6869630813598633`
  - `top1_agreement=0.25`
  - peak allocation increased from `962.45 MiB` to `967.07 MiB`
  - parameter memory increased from `943.82 MiB` to `948.43 MiB`

Decision:

- Do not enable chunk memory for stable imported-checkpoint inference.
- It does not currently lower VRAM in the tested path; it adds memory-token attention and extra parameters.
- It materially changes intelligence without training/adaptation.
- Keep `INFER_MEMORY_TOKENS=0` as the stable default.

Implementation order:

1. Run with memory disabled and establish parity.
2. Enable memory with a very small number of tokens.
3. Train or distill the compressor while base weights are frozen.
4. Compare against full-cache and sink-cache baselines.

Acceptance criteria should be stricter than sample quality alone: KL, top-k agreement, recall, and downstream task accuracy.

## 7. CPU Hash Context

Complexity: Medium to High

Status: Existing scaffolding and projection path.

The CPU hash context can provide a compact external memory representation. It is attractive for VRAM because the long-context state can live outside the GPU, but it is not equivalent to transformer KV.

Use only after sink/bounded/int8 KV have been measured. Treat it as a learned sidecar, not a drop-in cache replacement.

Recommended path:

1. Gate near zero by default.
2. Freeze the base model.
3. Train the projection/gate with KL distillation from the full-cache model.
4. Only then test reducing KV length further.

## 8. Engram Memory Bank

Complexity: High

Status: Existing module.

Engram memory is a retrieval sidecar. It can inject useful latent state, but retrieval mistakes can harm generation. It is higher risk than bounded KV or CPU hash context because it actively selects and blends stored vectors.

Recommended use:

- Research mode only.
- Keep gate low.
- Require strong recall and hallucination tests.
- Do not make it default for serving until it beats sink-cache on quality at the same VRAM budget.

## 9. Draft Head Logit Blend

Complexity: Medium to High

Status: Existing, disabled by stable defaults.

The draft head can provide cheap logits from the FF trunk. Directly blending draft logits into final logits risks quality loss unless the draft head is trained and calibrated.

Recommended use:

- Keep `USE_DRAFT_HEAD=0` and `DRAFT_BLEND_BP=0` for stable inference.
- Train draft head with the base model frozen.
- Validate draft logits separately before enabling blend.

This is not primarily a VRAM optimization. It is more useful as a stepping stone to speculative decoding.

## 10. Speculative Decoding With Draft Head

Complexity: High

Status: Partial helpers exist.

Speculative decoding can improve throughput while preserving intelligence if every proposed token is verified by the full model. It does not usually reduce VRAM, and it may increase memory if the draft path needs extra state.

Recommended order:

1. Train draft head.
2. Verify `decode_many_kv` parity against sequential `decode_one_kv`.
3. Add acceptance-rate logging.
4. Enable only if accepted tokens per verify call are high enough to improve end-to-end tok/s.

This is a throughput optimization, not the first VRAM optimization.

## 11. FF MLP Skip / Layer Skip

Complexity: High

Status: Existing environment switches and helper paths.

Skipping MLPs or full FF blocks can reduce compute, but it changes the model computation directly. Without a verifier or strong router, this is high risk for intelligence.

Recommended use:

- Do not enable by default.
- Use only with task-specific evals and strict fallback.
- Prefer speculative verification over unverified skipping.

## 12. BP Correction Blocks

Complexity: High

Status: Existing BP stack support.

BP blocks can act as correction layers after a cheaper FF path. This is promising for architecture research, but it requires training. It also adds parameters and activation/KV cost unless it allows other layers to be skipped.

Implementation order:

1. Keep BP disabled or zero-init for imported model parity.
2. Train BP as a frozen-base correction module.
3. Test whether BP lets you shorten cache, skip layers, or improve small-model quality.

## 13. BitNet / Weight Quantized Architecture

Complexity: Very High

Status: Existing `BitLinear` path.

BitNet is not a safe post-training retrofit for an imported transformer checkpoint. It changes weight behavior and needs quantization-aware training or serious adaptation.

Use only if the goal shifts from serving a compatible checkpoint to training a new efficient architecture.

## 14. Attention Replacement With SSM or Memory

Complexity: Very High

Status: Not recommended as a near-term retrofit.

Replacing transformer attention with SSM/context memory is the most disruptive option. It can reduce KV memory in theory, but it will not preserve pretrained intelligence without major distillation or retraining.

This should come last, after all cache engineering, quantized KV, and verified draft decoding have been exhausted.

## Recommended Implementation Order

1. Keep stable full-cache baseline working.
2. Use bounded KV cache.
3. Use sink + recent KV cache.
4. Extend measurement and parity tests for every cache policy.
5. Test int8 KV cache.
6. Tune `BLOCK_SIZE`, `KV_CACHE_MAX_LEN`, and `KV_CACHE_SINK_TOKENS` for target GPUs.
7. Train and evaluate draft head without blending.
8. Add speculative decoding only if draft acceptance is high.
9. Try chunk memory or CPU hash context as gated sidecars.
10. Try engram memory only in research mode.
11. Try selective layer/MLP skipping with verifier or strict evals.
12. Train BP correction blocks if skipping or compression needs repair.
13. Explore BitNet only as a training project.
14. Explore attention replacement only as a full research track.

## Current Best Path

For the current repo, the best near-term production path is:

```bash
USE_KV_CACHE=1
INFER_MEMORY_TOKENS=0
INFER_USE_ENGRAM=0
USE_DRAFT_HEAD=0
DRAFT_BLEND_BP=0
KV_CACHE_INT8=0
KV_CACHE_MAX_LEN=<gpu_budget>
KV_CACHE_SINK_TOKENS=<system_prompt_budget>
```

Then run:

```bash
python tools/measure_retrofit.py \
  --cache-policies full,bounded,sink,int8 \
  --bounded-cache-lens 128,256,512,1024 \
  --kv-cache-sink-tokens 64
```

For manual chat checks from an imported checkpoint:

```bash
USE_KV_CACHE=1 python scripts/inference/chat_hf.py \
  --checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --tokenizer Qwen/Qwen2.5-Coder-0.5B-Instruct
```

Choose the smallest cache policy that keeps parity, recall, and throughput within acceptable limits.
