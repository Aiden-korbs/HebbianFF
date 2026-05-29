# Ternary Runtime Presets

This repo has a ternary+LoRA inference runtime for repaired MLP gate/up modules. The adapter keeps most repaired weights compact, then chooses a runtime path per adapted linear layer.

## Presets

Use `TERNARY_PRESET` unless you are actively debugging kernels.

| Preset | Purpose | Mapping |
| --- | --- | --- |
| `low_vram` | Lowest persistent VRAM | `TERNARY_RUNTIME=hybrid`, `TERNARY_PREFILL_RUNTIME=temp_dense`, `TERNARY_DECODE_RUNTIME=triton_gemv`, no selective dense cache |
| `balanced` | Recommended default | `TERNARY_RUNTIME=auto`, `TERNARY_AUTO_PREFILL=temp_dense`, merged LoRA, selective dense cache top 8 |
| `speed` | Faster decode, more VRAM | `TERNARY_RUNTIME=auto`, `TERNARY_AUTO_PREFILL=temp_dense`, merged LoRA, selective dense cache top 16 |
| `manual` | Use explicit env vars | Leaves `TERNARY_RUNTIME`, prefill/decode, and cache envs as provided |

`balanced` and `speed` need a profile JSON to choose the top-k dense cached modules:

```bash
TERNARY_SELECTIVE_DENSE_PROFILE=ternary_repair_runs/ternary_runtime_profile.json
```

If the profile is absent, `tools/bench_ternary_generation.py --preset all` creates one.

## Runtime Modes

`dense_debug` reconstructs dense ternary weights at load and uses `F.linear`. With `TERNARY_DENSE_MERGE_LORA=1`, it folds `A @ B` into the dense weight and proves the quality/speed ceiling. It sacrifices VRAM.

`packed_fallback` keeps compact packed weights and reconstructs the current layer weight during forward. It is a correctness fallback, but decode is slow.

`triton_gemv` keeps compact bitplanes and computes decode directly in Triton without materializing a full dense weight. It is the decode path for all recommended presets.

`hybrid` uses the configured prefill path for sequence length greater than 1 and the configured decode path for sequence length 1.

`auto` applies preset policy. For the recommended presets, prefill uses `temp_dense` and decode uses `triton_gemv`, with optional selective dense cache.

## Current Benchmark

Measured with `tinyllama_ternary_lora_gate_up_rank16.pt`, prompt lengths 64/256/512, 128 decode tokens.

| Runtime | Load VRAM | Decode tok/s | Total tok/s |
| --- | ---: | ---: | ---: |
| native bf16 | 2106 MiB | 113.0 / 113.3 / 111.4 | 168.0 / 333.5 / 539.5 |
| low_vram | 1696 MiB | 86.7 / 86.2 / 85.5 | 126.1 / 249.0 / 406.7 |
| balanced top8 | 1872 MiB | 92.0 / 92.5 / 91.3 | 134.4 / 267.5 / 436.3 |
| speed top16 | 2048 MiB | 99.8 / 99.8 / 97.2 | 146.8 / 291.2 / 466.6 |

`speed` may approach or exceed native peak VRAM on longer prompts because it keeps more dense merged modules resident.

## Example Chat Commands

Low VRAM:

```bash
TERNARY_PRESET=low_vram \
python scripts/inference/chat_hf.py \
  --checkpoint tinyllama_ffbp_fullres_llama.pt \
  --ternary-adapter ternary_repair_runs/tinyllama_ternary_lora_gate_up_rank16.pt \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Balanced:

```bash
TERNARY_PRESET=balanced \
TERNARY_SELECTIVE_DENSE_PROFILE=ternary_repair_runs/ternary_runtime_profile.json \
python scripts/inference/chat_hf.py \
  --checkpoint tinyllama_ffbp_fullres_llama.pt \
  --ternary-adapter ternary_repair_runs/tinyllama_ternary_lora_gate_up_rank16.pt \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Speed:

```bash
TERNARY_PRESET=speed \
TERNARY_SELECTIVE_DENSE_PROFILE=ternary_repair_runs/ternary_runtime_profile.json \
python scripts/inference/chat_hf.py \
  --checkpoint tinyllama_ffbp_fullres_llama.pt \
  --ternary-adapter ternary_repair_runs/tinyllama_ternary_lora_gate_up_rank16.pt \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Generation Benchmark

Benchmark all presets and generate a profile if needed:

```bash
python tools/bench_ternary_generation.py \
  --checkpoint tinyllama_ffbp_fullres_llama.pt \
  --adapter ternary_repair_runs/tinyllama_ternary_lora_gate_up_rank16.pt \
  --block-size 512 \
  --prompt-lens 64,256,512 \
  --new-tokens 128 \
  --preset all \
  --profile-json ternary_repair_runs/ternary_runtime_profile.json
```

Benchmark only the recommended preset:

```bash
python tools/bench_ternary_generation.py \
  --checkpoint tinyllama_ffbp_fullres_llama.pt \
  --adapter ternary_repair_runs/tinyllama_ternary_lora_gate_up_rank16.pt \
  --preset balanced \
  --profile-json ternary_repair_runs/ternary_runtime_profile.json
```
