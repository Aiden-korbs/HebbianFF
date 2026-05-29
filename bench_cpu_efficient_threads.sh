#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/fineweb_edu_smoke}"
THREAD_LIST="${THREAD_LIST:-8 10 12 16}"

for threads in ${THREAD_LIST}; do
  echo
  echo "===== TORCH_THREADS=${threads} ====="
  CMD=(env
  PYTHON_BIN="${PYTHON_BIN}"
  DATA_DIR="${DATA_DIR}" \
  OUT_DIR="runs/tmp_cpu_thread_bench_${threads}" \
  TRAIN_DEVICE=cpu \
  TRAIN_DTYPE=float32 \
  CPU_EFFICIENT_FF=1 \
  ATTN_EVERY="${ATTN_EVERY:-4}" \
  FORCE_ATTN_LAST="${FORCE_ATTN_LAST:-2}" \
  CPU_FF_MIXER="${CPU_FF_MIXER:-depthwise_conv}" \
  LOCAL_MIXER_KERNEL="${LOCAL_MIXER_KERNEL:-5}" \
  FUSED_SWIGLU="${FUSED_SWIGLU:-0}" \
  TORCH_THREADS="${threads}" \
  TORCH_INTEROP_THREADS=1 \
  OMP_NUM_THREADS="${threads}" \
  MKL_NUM_THREADS="${threads}" \
  ROUNDTRIP_CHECK=0 \
  BITNET_CUSTOM_STE=1 \
  BITNET_BYPASS_TRAIN="${BITNET_BYPASS_TRAIN:-0}" \
  BITNET_CACHE_TRAIN=1 \
  BITNET_CACHE_TRAIN_MIB=256 \
  CPU_PREFETCH_BATCHES=2 \
  N_EMBD="${N_EMBD:-256}" \
  FF_LAYERS="${FF_LAYERS:-4}" \
  N_HEAD="${N_HEAD:-8}" \
  N_KV_HEAD="${N_KV_HEAD:-2}" \
  MLP_RATIO="${MLP_RATIO:-2.0}" \
  STEPS="${STEPS:-3}" \
  BATCH_SIZE="${BATCH_SIZE:-1}" \
  GRAD_ACCUM="${GRAD_ACCUM:-2}" \
  BLOCK_SIZE="${BLOCK_SIZE:-128}" \
  EVAL_FIRST=0 \
  EVAL_EVERY=999999 \
  EVAL_ITERS=1 \
  SAVE_EVERY=999999 \
  SAVE_FINAL=0 \
  BENCH_RESULT=1 \
  NO_PROGRESS=1 \
  FLUSH_DENORMAL="${FLUSH_DENORMAL:-1}" \
  GRAD_CLIP="${GRAD_CLIP:-0}" \
  OPTIMIZER="${OPTIMIZER:-adamw}" \
  ADAMW_FOREACH="${ADAMW_FOREACH:-auto}" \
  USE_IPEX="${USE_IPEX:-0}" \
  TORCH_COMPILE="${TORCH_COMPILE:-0}" \
  TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-default}" \
  PROGRESS_EVERY=0 \
  ./run_500m_fineweb_edu.sh --no-bnb)
  if [[ -n "${CPUSET:-}" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "${CPUSET}" "${CMD[@]}"
  else
    "${CMD[@]}"
  fi
done
