# Ternary + residual correction

This patch adds an experimental correction channel to the selective ternary MLP path:

```text
W_dense ≈ W_ternary + A @ B
```

The ternary matrix stays packed at rest.  The `A @ B` residual is a low-rank bf16/fp16/fp32 correction initialized analytically from the dense quantization error using randomized SVD.

## Why

Pure ternary was mechanically working but produced bad generations because the reconstructed MLP weights were about 40–50% relative L2 away from the dense source.  The residual path stores a small amount of extra information to recover the largest missing directions.

## Runtime formula

For a packed linear with original weight shape `[out, in]`:

```text
y = x @ W_ternary.T + (x @ B.T) @ A.T
```

`A` has shape `[out, rank]` and `B` has shape `[rank, in]`.

For `down.forward_cols(x_sparse, idx)`, the residual uses only the selected input channels:

```text
tmp = x_sparse @ B[:, idx].T
y_corr = tmp @ A.T
```

## Packing examples

Start with `up,down` plus rank 128 residual:

```bash
python tools/pack_selective_mlp.py \
  --checkpoint ./models/DeepSeek-R1-Distill-Qwen-7B-ff28-bf16-block512.pt \
  --out ./models/DeepSeek-R1-Distill-Qwen-7B-ff28-updown-tri-r128.pt \
  --parts up,down \
  --group-size 64 \
  --threshold 0.0 \
  --residual-rank 128 \
  --residual-device cuda \
  --residual-dtype bfloat16 \
  --layers all
```

Safer but larger:

```bash
python tools/pack_selective_mlp.py \
  --checkpoint ./models/DeepSeek-R1-Distill-Qwen-7B-ff28-bf16-block512.pt \
  --out ./models/DeepSeek-R1-Distill-Qwen-7B-ff28-updown-tri-r256.pt \
  --parts up,down \
  --group-size 64 \
  --threshold 0.0 \
  --residual-rank 256 \
  --residual-device cuda \
  --residual-dtype bfloat16 \
  --layers all
```

Run full hidden first, so sparse selection is not mixed into the diagnosis:

```bash
CPU_OFFLOAD_TOK_EMB=1 \
CPU_OFFLOAD_OUT_PROJ=1 \
SELECTIVE_MLP_TOPK=18944 \
SELECTIVE_MLP_SCORE=last \
SELECTIVE_MLP_DEBUG=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chat_hf.py \
  --checkpoint ./models/DeepSeek-R1-Distill-Qwen-7B-ff28-updown-tri-r128.pt \
  --tokenizer ./models/hf/DeepSeek-R1-Distill-Qwen-7B \
  --dtype bf16 \
  --block-size 512 \
  --max-new 80
```

Only lower `SELECTIVE_MLP_TOPK` after full hidden produces sane text.

## Useful diagnostics

Add `--residual-check` during packing to print relative L2 error before and after the low-rank residual.  This is slower because it reconstructs the corrected error in chunks.

```bash
--residual-check
```

Expected pattern:

```text
residual up:   rank=128 ... rel_l2 0.44->0.xx
residual down: rank=128 ... rel_l2 0.45->0.xx
```

If the after-error is still high, increase rank or switch to 4-bit/NF4 instead of ternary.
