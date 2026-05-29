# Run Retrofit Evaluation

This benchmark compares:

- A: native Hugging Face `AutoModelForCausalLM`
- B: imported `FF_LLM` checkpoint in no-op mode

It does not train, import automatically, enable memory sidecars, enable routing, or modify model behavior.

## 1. Import The FF_LLM Checkpoint First

The benchmark expects the imported checkpoint to already exist.

For the first pass:

```bash
python tools/import_hf.py Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --out models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --block-size 1024 \
  --bp-layers 0
```

If you skip this step, the benchmark exits clearly and prints the import command it expected you to run.

## 2. Run The Benchmark

```bash
python tools/measure_retrofit.py \
  --native-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --ff-checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --block-size 1024 \
  --dtype bf16
```

Default output directory:

```text
runs/retrofit_eval/
```

The script writes:

- timestamped JSON result: `runs/retrofit_eval/retrofit_eval_YYYYMMDD_HHMMSS.json`
- latest result copy: `runs/retrofit_eval/latest.json`

## 3. What It Measures

Memory:

- peak CUDA allocated
- peak CUDA reserved
- static parameter bytes
- KV cache bytes after prefill
- KV cache bytes after decode

Speed:

- prompt prefill tokens/sec
- decode tokens/sec

Parity and quality:

- cross entropy
- perplexity
- KL divergence from native logits to FF_LLM logits
- top-1 agreement
- native top-1 inside FF top-5
- FF top-1 inside native top-5
- top-5 Jaccard overlap
- deterministic greedy sample outputs

## 4. No-Op Runtime Settings

The harness forces or defaults these settings:

```text
USE_KV_CACHE=1
USE_DRAFT_HEAD=0
DRAFT_BLEND_BP=0
INFER_MEMORY_TOKENS=0
INFER_USE_ENGRAM=0
CPU_CTX=0
CPU_HASH_CTX=0
USE_BITNET=0
NO_INIT_LOAD=1
```

Do not use this script to evaluate SSM, engram, BitNet, BP routing, speculative decoding, CPU hash context, or training. Those are intentionally out of scope for this pass.

## 5. Parity First

The first pass condition is parity. If the imported `FF_LLM` checkpoint differs from the native HF model before any retrofit is enabled, treat that as a blocker.

Default parity thresholds:

```text
KL(native || FF_LLM) <= 1e-3
top-1 agreement >= 0.99
max absolute logit diff <= 1e-2
no first top-1 mismatch
```

On parity failure, the script:

- saves JSON results,
- prints `Parity: FAIL`,
- reports the first mismatching eval text/token position,
- prints native top-5 and FF_LLM top-5 token candidates,
- exits with status code `1`.

To keep a CI or notebook run alive while still recording failure:

```bash
python tools/measure_retrofit.py --no-fail-exit
```

## 6. Useful Variants

Use a shorter run:

```bash
python tools/measure_retrofit.py \
  --decode-tokens 16 \
  --sample-tokens 24 \
  --speed-prompt-tokens 128
```

Benchmark FF_LLM cache policies:

```bash
python tools/measure_retrofit.py \
  --native-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --ff-checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --block-size 2048 \
  --dtype fp32 \
  --cache-policies full,bounded,int8 \
  --bounded-cache-lens 64,128,256,512
```

Notes:

- `full` is the baseline full KV-cache policy.
- `bounded` trims the current FF_LLM KV cache to each requested `max_cache_len` after prefill and after each decode step.
- `int8` stores FF_LLM KV-cache tensors as int8 with per-token fp16 scale tensors and dequantizes before SDPA. It has a clean off switch: omit `int8` from `--cache-policies` or set `KV_CACHE_INT8=0`.
- Because there is no fused int8 attention kernel here, int8 can save KV memory while slowing decode due to quantize/dequantize overhead.
- The combined JSON contains all policies, and the harness also writes one JSON file per policy next to the combined result.
- The JSON also includes `long_context_recall`, which places `The secret code is MANGO7429.` near the beginning of a prompt and later asks `Reply with only the secret code. No explanation.` during KV-cache decode. This is the cache-quality check to use for bounded cache policies.
- For each recall length, bounded-cache conclusions should be used only when both `native_full` and `ff_full` controls pass.
- CE/PPL/KL from full or teacher-forced forward passes do not necessarily measure decode-time cache truncation. A bounded decode cache can shrink KV bytes without affecting short full-forward parity metrics if the evaluated tokens never require evicted context.

Long-context recall controls:

```bash
python tools/measure_retrofit.py \
  --native-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --ff-checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --block-size 2048 \
  --dtype fp32 \
  --long-recall-lengths 512,1024,2048 \
  --long-recall-new-tokens 24 \
  --long-recall-cache-lens 64,128,256,512,1024,2048
```

Use fp32 parity debugging:

```bash
python tools/measure_retrofit.py \
  --dtype fp32 \
  --block-size 512
```

Use the 1.5B model after the 0.5B path is clean:

```bash
python tools/import_hf.py Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --out models/Qwen2.5-Coder-1.5B-Instruct.pt \
  --block-size 1024 \
  --bp-layers 0

python tools/measure_retrofit.py \
  --native-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --ff-checkpoint models/Qwen2.5-Coder-1.5B-Instruct.pt \
  --block-size 1024 \
  --dtype bf16
```

## 7. Interpreting Results

If parity fails:

- do not proceed to retrofit experiments,
- inspect `comparison.first_mismatch` in the JSON,
- check tokenizer identity,
- check import layer mapping,
- check `final_proj` behavior,
- rerun in `fp32` to separate dtype noise from mapping mismatch.

If parity passes:

- use this JSON as the baseline for future retrofit work,
- only then add one efficiency mechanism at a time,
- compare each future run against this no-op baseline and the native HF baseline.
