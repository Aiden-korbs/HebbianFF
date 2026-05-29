#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/fineweb_edu_smoke}"
STEPS="${STEPS:-8}"
BENCH_WARMUP_STEPS="${BENCH_WARMUP_STEPS:-1}"
WARMUP_NOTE="BENCH_RESULT excludes the first ${BENCH_WARMUP_STEPS} warmup step(s)"

common_env=(
  PYTHON_BIN="${PYTHON_BIN}"
  DATA_DIR="${DATA_DIR}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  TRAIN_DEVICE=cuda
  TRAIN_DTYPE=auto
  CPU_EFFICIENT_FF=1
  ATTN_EVERY="${ATTN_EVERY:-4}"
  FORCE_ATTN_LAST="${FORCE_ATTN_LAST:-2}"
  CPU_FF_MIXER="${CPU_FF_MIXER:-depthwise_conv}"
  FUSED_SWIGLU="${FUSED_SWIGLU:-0}"
  TORCH_THREADS="${TORCH_THREADS:-8}"
  TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
  ROUNDTRIP_CHECK=0
  EVAL_FIRST=0
  EVAL_EVERY=999999
  EVAL_ITERS=1
  SAVE_EVERY=999999
  SAVE_FINAL=0
  BENCH_RESULT=1
  BENCH_WARMUP_STEPS="${BENCH_WARMUP_STEPS}"
  NO_PROGRESS=1
  BITNET_CUSTOM_STE=1
  BITNET_CACHE_TRAIN=0
  STEPS="${STEPS}"
  BATCH_SIZE="${BATCH_SIZE:-1}"
  GRAD_ACCUM="${GRAD_ACCUM:-8}"
  BLOCK_SIZE="${BLOCK_SIZE:-512}"
  PROGRESS_EVERY=0
)

run_case() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  echo "${WARMUP_NOTE}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
  fi
  env "${common_env[@]}" OUT_DIR="runs/tmp_hybrid_bench_${name}" "$@" ./run_500m_fineweb_edu.sh --no-bnb
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
  fi
}

run_case baseline CPU_PREFETCH_BATCHES=0 PIN_BATCHES=0 ASYNC_H2D=0
run_case pinned_prefetch CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}" PIN_BATCHES=1 ASYNC_H2D=1
run_case low_sync CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}" PIN_BATCHES=1 ASYNC_H2D=1 PROGRESS_EVERY=0
run_case fused_swiglu CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}" PIN_BATCHES=1 ASYNC_H2D=1 FUSED_SWIGLU=1
run_case bitnet_cache768 CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}" PIN_BATCHES=1 ASYNC_H2D=1 FUSED_SWIGLU=1 BITNET_CACHE_TRAIN=1 BITNET_CACHE_TRAIN_MIB=768

if [[ "${RUN_THREAD_SWEEP:-1}" == "1" ]]; then
  for threads in 4 6 8 12; do
    run_case "threads_${threads}" \
      CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}" PIN_BATCHES=1 ASYNC_H2D=1 \
      TORCH_THREADS="${threads}" OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}"
  done
fi

if [[ "${RUN_BLOCK1024:-0}" == "1" ]]; then
  run_case block1024 CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}" PIN_BATCHES=1 ASYNC_H2D=1 BLOCK_SIZE=1024
fi
