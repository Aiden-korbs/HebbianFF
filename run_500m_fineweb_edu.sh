#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-data/fineweb_edu_10bt_bpe32k}"
OUT_DIR="${OUT_DIR:-runs/ff_ternary_ema_fineweb_500m}"

[[ -f "$DATA_DIR/train.bin" && -f "$DATA_DIR/val.bin" ]] || {
  echo "Missing $DATA_DIR/train.bin or val.bin. Build data first:"
  echo "  ./build_fineweb_edu_500m_data.sh"
  exit 1
}

export ROUNDTRIP_CHECK="${ROUNDTRIP_CHECK:-1}"
export BITNET_CACHE_TRAIN="${BITNET_CACHE_TRAIN:-0}"
export COLLECT_TRAIN_NORM_DIAG="${COLLECT_TRAIN_NORM_DIAG:-0}"
export COLLECT_TRAIN_EMA_DIAG="${COLLECT_TRAIN_EMA_DIAG:-0}"

# ~496M parameters at vocab=32768.
export N_EMBD="${N_EMBD:-1664}"
export FF_LAYERS="${FF_LAYERS:-20}"
export BP_LAYERS="${BP_LAYERS:-0}"
export N_HEAD="${N_HEAD:-16}"
export N_KV_HEAD="${N_KV_HEAD:-4}"
export MLP_RATIO="${MLP_RATIO:-2.66}"

export USE_BITNET="${USE_BITNET:-1}"
export USE_BITNET_QUANT_ACT="${USE_BITNET_QUANT_ACT:-0}"

export FF_EMA_BP="${FF_EMA_BP:-1}"
export FF_EMA_DETACH="${FF_EMA_DETACH:-0}"
export FF_EMA_WARMUP_BIT="${FF_EMA_WARMUP_BIT:-2000}"
export FF_EMA_STD="${FF_EMA_STD:-2.0}"
export FF_EMA_MAX_TRIPS="${FF_EMA_MAX_TRIPS:-1}"
export FF_EMA_BP_WEIGHT="${FF_EMA_BP_WEIGHT:-0.03}"

export USE_DRAFT_HEAD="${USE_DRAFT_HEAD:-0}"
export MEMORY_TOKENS="${MEMORY_TOKENS:-0}"
export USE_ENGRAM="${USE_ENGRAM:-0}"
export CPU_HASH_CTX="${CPU_HASH_CTX:-0}"

"${PYTHON_BIN:-python}" train_ff_only_ternary_ema.py \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --vocab-size "${VOCAB_SIZE:-32768}" \
  --steps "${STEPS:-150000}" \
  --eval-every "${EVAL_EVERY:-1000}" \
  --eval-iters "${EVAL_ITERS:-50}" \
  --eval-first "${EVAL_FIRST:-1}" \
  --save-every "${SAVE_EVERY:-5000}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-64}" \
  --block-size "${BLOCK_SIZE:-1024}" \
  --lr "${LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.05}" \
  --grad-clip "${GRAD_CLIP:-1.0}" \
  --device "${TRAIN_DEVICE:-auto}" \
  --dtype "${TRAIN_DTYPE:-auto}" \
  --threads "${TORCH_THREADS:-0}" \
  --interop-threads "${TORCH_INTEROP_THREADS:-0}" \
  --prefetch-batches "${CPU_PREFETCH_BATCHES:-0}" \
  --pin-batches "${PIN_BATCHES:-1}" \
  --async-h2d "${ASYNC_H2D:-1}" \
  --save-final "${SAVE_FINAL:-1}" \
  --bench-result "${BENCH_RESULT:-0}" \
  --no-progress "${NO_PROGRESS:-0}" \
  --flush-denormal "${FLUSH_DENORMAL:-1}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --adamw-foreach "${ADAMW_FOREACH:-auto}" \
  --use-ipex "${USE_IPEX:-0}" \
  --compile "${TORCH_COMPILE:-0}" \
  --compile-mode "${TORCH_COMPILE_MODE:-default}" \
  --progress-every "${PROGRESS_EVERY:-5}" \
  ${RESUME_FROM:+--resume-from "$RESUME_FROM"}
