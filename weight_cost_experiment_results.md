# Weight Cost Experiment Results

Date: 2026-05-22

Goal: test experimental ways to reduce model weight VRAM and decide whether they pass parity well enough to use for serving.

Target hardware observed during tests:

- GPU: NVIDIA GeForce RTX 3070, 8GB class, 7.65 GiB usable
- Installed quantization support:
  - `bitsandbytes`: available
  - `accelerate`: available
  - `auto_gptq`: unavailable
  - `awq`: unavailable
  - `hqq`: unavailable
  - `quanto`: unavailable

## Summary

| Method | Status | Fits / Runs | Parity | Decision |
| --- | --- | --- | --- | --- |
| Native HF NF4 4-bit | Failed locally | No | Not measured | Failed on CUDA init spike after loading most weights |
| Native HF 8-bit offload | Failed locally | No | Not measured | Failed due local bitsandbytes/Transformers API mismatch |
| Custom partial 7B import, 14/28 layers | Tested | Does not fit current GPU cleanly; CPU test works | Failed | Not acceptable without repair training |
| Custom BitNet runtime conversion | Tested on 0.5B | Runs, but does not reduce parameter memory | Failed | Not a drop-in retrofit |
| Low-rank approximation | Probe on 0.5B first block | Runs as probe only | Failed | Needs careful SVD init plus distillation/training |
| Structured pruning | Probe on 0.5B first block | Runs as probe only | Failed | Needs architecture compaction plus repair training |
| Weight sharing | Probe on 0.5B | Runs as probe only | Failed | Not acceptable without full retraining |
| Packed ternary transfer runtime | Implemented | Runs from packed artifact | Failed | Useful as compact initialization artifact, not parity-preserving serving |
| CPU/GPU offload | Not implemented for custom runtime | N/A | N/A | Possible future work; throughput risk high |
| GPTQ/AWQ/HQQ | Not tested | Packages unavailable | N/A | Install/package work required before testing |

Current practical recommendation:

- Use `DeepSeek-R1-Distill-Qwen-1.5B` custom checkpoint for GPU web chat.
- Use native HF 7B quantization only after resolving the local 4-bit/8-bit loader failures.
- For custom `FF_LLM` 7B on this GPU, the next serious implementation path is a real quantized linear loader, not layer dropping, BitNet runtime conversion, chunk memory, or pruning.

## 1. Native HF NF4 4-Bit

Command shape:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Result:

- The loader got most weights onto GPU.
- It failed during final missing-weight initialization with a CUDA OOM.
- Error tried to allocate an additional `2.03 GiB` with about `1.40 GiB` free.
- GPU process had about `5.38 GiB` in use at failure.

Status: failed locally.

Interpretation:

- NF4 is still the best theoretical route for fitting 7B on this GPU, but the current native HF load path hits a memory spike on this environment.
- This was native HF, not the custom `FF_LLM` checkpoint.

## 2. Native HF 8-Bit With CPU Offload

Command shape:

```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
```

Result:

- Failed during load with:

```text
TypeError: Int8Params.__new__() got an unexpected keyword argument '_is_hf_initialized'
```

- Peak GPU allocation before failure: about `3745.58 MiB`.

Status: failed locally.

Interpretation:

- This looks like a local bitsandbytes/Transformers compatibility problem, not a model architecture issue.
- Even if fixed, 8-bit is likely tight on an 8GB GPU unless major modules are offloaded to CPU.

## 3. Custom Partial Import: DeepSeek 7B 14 Layers

Checkpoint created:

```text
models/DeepSeek-R1-Distill-Qwen-7B-ff14-bs8192-bf16.pt
```

Import result:

- Source layers: `28`
- Imported FF blocks: `14`
- Params: `4.37B`
- File size: about `8.7GB`
- Structural verification: passed

Parity test against native 7B on CPU prompt `The answer to 2 + 2 is`:

```text
max_abs = 31.84375
mean_abs = 5.3261027336120605
KL = 8.510856628417969
top1_agreement = 0.0
native last top5 = [374, 284, 24768, 353, 488]
ff14 last top5 = [85987, 78871, 60359, 37730, 71974]
```

Status: failed parity.

Interpretation:

- Layer dropping reduces weights, but it destroys behavior without repair training.
- It also still does not comfortably fit the current GPU as bf16 because 4.37B bf16 params exceed the usable VRAM budget.

## 4. Custom BitNet Runtime Conversion

Probe checkpoint:

```text
models/Qwen2.5-Coder-0.5B-Instruct.pt
```

Result:

```text
base_param_mib = 943.823974609375
bitnet_param_mib = 944.087646484375
base_peak_mib = 958.9609375
bitnet_peak_mib = 1643.34423828125
max_abs = 23.65625
mean_abs = 3.348907709121704
KL = 11.950925827026367
top1_agreement = 0.0
```

Status: failed parity and did not reduce runtime parameter memory.

Interpretation:

- The existing BitNet path is not a post-training quantizer.
- It does not make imported checkpoints smaller in memory by itself.
- It needs quantization-aware training or a proper conversion/calibration workflow.

## 5. Low-Rank Approximation Probe

Probe:

- Model: `Qwen2.5-Coder-0.5B-Instruct.pt`
- Applied rank-50% SVD approximation to first FF block linear weights only.

Result:

```text
max_abs = 16.8125
mean_abs = 2.1502068042755127
KL = 3.008004665374756
top1_agreement = 0.25
```

Status: failed parity as a no-training change.

Interpretation:

- Even a local first-block rank cut perturbs logits materially.
- Full-model low-rank factorization would need a proper factorized module implementation plus distillation.

## 6. Structured Pruning Probe

Probe:

- Model: `Qwen2.5-Coder-0.5B-Instruct.pt`
- Zeroed 25% of lowest-norm first-block MLP intermediate channels.

Result:

```text
max_abs = 12.25
mean_abs = 1.3985178470611572
KL = 2.090327262878418
top1_agreement = 0.5
```

Status: failed parity as a no-training change.

Interpretation:

- Pruning needs calibration, actual architecture compaction, and repair training.
- Zeroing weights alone does not reduce dense tensor memory.

## 7. Weight Sharing Probe

Probe:

- Model: `Qwen2.5-Coder-0.5B-Instruct.pt`
- Copied FF layer 0 weights into all FF blocks.

Result:

```text
max_abs = 26.375
mean_abs = 4.0379791259765625
KL = 14.468766212463379
top1_agreement = 0.0
```

Status: failed parity.

Interpretation:

- Weight sharing is not a retrofit.
- It is only plausible as a from-scratch or heavily distilled training experiment.

## 8. Packed Ternary Transfer Runtime

Implemented files:

- `ffbp_ema_cpu_ssm/packed.py`
- `tools/transfer_to_1bit.py`
- `tools/eval_packed_transfer.py`

Artifacts:

```text
models/DeepSeek-R1-Distill-Qwen-7B-ternary-g256-transfer-v2.pt
models/Qwen2.5-Coder-0.5B-Instruct-ternary-g256-transfer.pt
```

7B transfer result:

```text
source tensor bytes = 14.209 GiB
packed tensor bytes = 3.622 GiB
compression ratio = 3.92x
quantized tensors = 196
holdout tensors = 147
```

The packed runtime stores transferred ternary weights compactly and dequantizes one layer at a time during forward. This avoids keeping all dense transformed weights resident at once, but it is not an optimized bit-serial matrix multiply kernel.

7B parity result on CPU prompt `The answer to 2 + 2 is`:

```text
max_abs = 27.8125
mean_abs = 6.090705871582031
KL = 8.903946876525879
top1_agreement = 0.0
dense last top5 = [374, 284, 24768, 353, 488]
packed last top5 = [11, 220, 323, 304, 320]
```

0.5B parity result:

```text
max_abs = 22.515625
mean_abs = 2.7900397777557373
KL = 8.419567108154297
top1_agreement = 0.0
```

Status: implemented, runs, fails parity.

Interpretation:

- The mathematical transfer works as a compact initialization artifact.
- Direct ternary projection is too lossy to preserve an imported checkpoint by itself.
- The next step would be distillation/repair training with the packed ternary weights as the starting point.

## 9. CPU/GPU Offload

Status: not implemented in the custom `FF_LLM` runtime.

Interpretation:

- This could fit 7B by keeping only part of the model on GPU.
- It does not reduce total weight cost.
- Throughput is likely poor unless the offload executor is carefully designed.

## 10. GPTQ / AWQ / HQQ

Status: not tested because packages are not installed.

Interpretation:

- These are viable next experiments if package installation is acceptable.
- They target post-training weight quantization more directly than the current custom architecture.

## Conclusion

No tested no-training weight-cost method passed parity strongly enough for the custom architecture.

Best next engineering path:

1. Implement or integrate a real quantized linear path for `FF_LLM`.
2. Start with 4-bit/NF4 or HQQ-style storage for large linear weights.
3. Keep norms in bf16.
4. Consider keeping embeddings and `out_proj` bf16 at first, then quantize them only if needed.
5. Re-run parity against native/custom bf16:
   - KL
   - top-k agreement
   - recall
   - generation samples
   - decode tokens/sec
6. If parity fails, add distillation or small adapters/BP correction.
