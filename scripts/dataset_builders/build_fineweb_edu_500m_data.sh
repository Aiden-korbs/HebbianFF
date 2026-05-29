#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# FineWeb-Edu sample-10BT is the intended pretraining corpus for the 500M run.
# uint16 bins use about 2 bytes/token, so the default 10B-token train split
# writes roughly 20GB plus a small validation bin.
export HF_HOME="${HF_HOME:-/home/corbs/datasets/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/corbs/datasets/hf_datasets_cache}"

OUT_DIR="${OUT_DIR:-data/fineweb_edu_10bt_bpe32k}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
TOKENIZER_DOCS="${TOKENIZER_DOCS:-1000000}"
TARGET_TRAIN_TOKENS="${TARGET_TRAIN_TOKENS:-10000000000}"
TARGET_VAL_TOKENS="${TARGET_VAL_TOKENS:-50000000}"

python scripts/dataset_builders/build_large_pretrain_data.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --out-dir "$OUT_DIR" \
  --vocab-size "$VOCAB_SIZE" \
  --tokenizer-docs "$TOKENIZER_DOCS" \
  --target-train-tokens "$TARGET_TRAIN_TOKENS" \
  --target-val-tokens "$TARGET_VAL_TOKENS" \
  --val-fraction 0.002 \
  --min-chars 400 \
  --max-chars 80000 \
  --min-alpha-frac 0.55 \
  --max-digit-frac 0.20 \
  --max-symbol-frac 0.35 \
  --max-urls 12
