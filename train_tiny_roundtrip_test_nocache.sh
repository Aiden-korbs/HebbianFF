#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export ROUNDTRIP_CHECK=1
export BITNET_CACHE_TRAIN=0

export N_EMBD=384
export FF_LAYERS=8
export BP_LAYERS=0
export N_HEAD=6
export N_KV_HEAD=2
export MLP_RATIO=2.66

export USE_BITNET=1
export USE_BITNET_QUANT_ACT=0

export FF_EMA_BP=1
export FF_EMA_DETACH=0
export FF_EMA_WARMUP_BIT=800
export FF_EMA_STD=3.0
export FF_EMA_MAX_TRIPS=1
export FF_EMA_BP_WEIGHT=0.05

export USE_DRAFT_HEAD=0
export MEMORY_TOKENS=0
export USE_ENGRAM=0
export CPU_HASH_CTX=0

python train_ff_only_ternary_ema.py \
  --data-dir data/ternary_tinystories \
  --out-dir runs/ff_ternary_ema_tiny_roundtrip_clearcache \
  --vocab-size 16000 \
  --steps 1500 \
  --eval-every 250 \
  --eval-iters 20 \
  --save-every 1000 \
  --batch-size 6 \
  --grad-accum 4 \
  --block-size 512 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --grad-clip 1.0
