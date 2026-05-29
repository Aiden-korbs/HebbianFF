#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
export TRAIN_DTYPE="${TRAIN_DTYPE:-float32}"

export CPU_EFFICIENT_FF="${CPU_EFFICIENT_FF:-1}"
export ATTN_EVERY="${ATTN_EVERY:-4}"
export FORCE_ATTN_LAST="${FORCE_ATTN_LAST:-2}"
export CPU_FF_MIXER="${CPU_FF_MIXER:-depthwise_conv}"
export LOCAL_MIXER_KERNEL="${LOCAL_MIXER_KERNEL:-5}"
export FUSED_SWIGLU="${FUSED_SWIGLU:-0}"

export TORCH_THREADS="${TORCH_THREADS:-8}"
export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export CPU_PREFETCH_BATCHES="${CPU_PREFETCH_BATCHES:-4}"

export ROUNDTRIP_CHECK="${ROUNDTRIP_CHECK:-0}"
export BITNET_CUSTOM_STE="${BITNET_CUSTOM_STE:-1}"
export BITNET_BYPASS_TRAIN="${BITNET_BYPASS_TRAIN:-0}"
export BITNET_CACHE_TRAIN="${BITNET_CACHE_TRAIN:-1}"
export BITNET_CACHE_TRAIN_MIB="${BITNET_CACHE_TRAIN_MIB:-2048}"

export OUT_DIR="${OUT_DIR:-runs/ff_ternary_ema_fineweb_500m_cpu_efficient_ctx512}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-64}"
export BLOCK_SIZE="${BLOCK_SIZE:-512}"
export EVAL_EVERY="${EVAL_EVERY:-5000}"
export EVAL_ITERS="${EVAL_ITERS:-5}"
export EVAL_FIRST="${EVAL_FIRST:-0}"
export SAVE_EVERY="${SAVE_EVERY:-10000}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
export GRAD_CLIP="${GRAD_CLIP:-0}"
export OPTIMIZER="${OPTIMIZER:-adamw}"
export ADAMW_FOREACH="${ADAMW_FOREACH:-0}"
export USE_IPEX="${USE_IPEX:-0}"
export TORCH_COMPILE="${TORCH_COMPILE:-0}"
export CPUSET="${CPUSET:-0,2,4,6,8,10,12,14}"

if [[ -n "${CPUSET}" ]] && command -v taskset >/dev/null 2>&1; then
  exec taskset -c "${CPUSET}" ./run_500m_fineweb_edu.sh "$@"
fi

exec ./run_500m_fineweb_edu.sh "$@"
