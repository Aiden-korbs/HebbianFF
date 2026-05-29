#!/usr/bin/env bash
set -Eeuo pipefail

# FF-only ternary EMA launcher for a 3070-class 8GB GPU.
# Usage:
#   ./scripts/training/run_3070_tinystories.sh --preset tiny
#   ./scripts/training/run_3070_tinystories.sh --preset small --steps 10000
#   N_EMBD=768 FF_LAYERS=10 BATCH_SIZE=2 GRAD_ACCUM=16 ./scripts/training/run_3070_tinystories.sh

# ----------------------------- UI helpers -----------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; MAG=$'\033[35m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAG=""; CYAN=""; RESET=""
fi

say() { printf "%b\n" "$*"; }
hr()  { printf "%b\n" "${DIM}──────────────────────────────────────────────────────────────────────────────${RESET}"; }
keyval() { printf "  ${CYAN}%-24s${RESET} %s\n" "$1" "$2"; }
fatal() { say "${RED}${BOLD}ERROR:${RESET} $*" >&2; exit 1; }

usage() {
  cat <<EOF
${BOLD}FF-only ternary EMA launcher${RESET}

Options:
  --preset tiny|small|medium   Hardware-safe model presets. Default: tiny
  --data-dir DIR               Dataset dir containing train.bin and val.bin
  --out-dir DIR                Output run directory
  --name NAME                  Run name used if --out-dir is not supplied
  --steps N                    Training steps. Default: 5000
  --eval-every N               Eval interval. Default: 250
  --save-every N               Save interval. Default: 1000
  --eval-iters N               Eval batches. Default: 20
  --lr LR                      Learning rate. Default: 3e-4
  --weight-decay WD            Weight decay. Default: 0.05
  --grad-clip N                Grad clip. Default: 1.0
  --no-bnb                     Force torch AdamW instead of bitsandbytes AdamW8bit
  --dry-run                    Print config and command, do not train
  -h, --help                   Show this help

Environment overrides still work, for example:
  N_EMBD=512 FF_LAYERS=10 BATCH_SIZE=6 GRAD_ACCUM=8 ./scripts/training/run_3070_tinystories.sh --preset small

Useful env toggles:
  USE_BITNET_QUANT_ACT=1       Try quantised activations after the stable run works
  FF_EMA_STD=4.0               Softer EMA trips
  FF_EMA_BP_WEIGHT=0.02        Softer EMA loss weight
  GPU_LOG=0                    Disable GPU csv logging
EOF
}

# ----------------------------- CLI defaults -----------------------------
PRESET="${PRESET:-tiny}"
DATA_DIR="${DATA_DIR:-data/ternary_tinystories}"
RUN_NAME="${RUN_NAME:-}"
OUT_DIR="${OUT_DIR:-}"
STEPS="${MAX_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_ITERS="${EVAL_ITERS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
VOCAB_SIZE="${VOCAB_SIZE:-16000}"
NO_BNB="${NO_BNB:-0}"
DRY_RUN=0
GPU_LOG="${GPU_LOG:-1}"
GPU_LOG_INTERVAL="${GPU_LOG_INTERVAL:-5}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET="${2:?missing preset}"; shift 2 ;;
    --data-dir) DATA_DIR="${2:?missing data dir}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?missing out dir}"; shift 2 ;;
    --name) RUN_NAME="${2:?missing run name}"; shift 2 ;;
    --steps) STEPS="${2:?missing steps}"; shift 2 ;;
    --eval-every) EVAL_EVERY="${2:?missing eval interval}"; shift 2 ;;
    --save-every) SAVE_EVERY="${2:?missing save interval}"; shift 2 ;;
    --eval-iters) EVAL_ITERS="${2:?missing eval iters}"; shift 2 ;;
    --lr) LR="${2:?missing lr}"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="${2:?missing weight decay}"; shift 2 ;;
    --grad-clip) GRAD_CLIP="${2:?missing grad clip}"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="${2:?missing vocab size}"; shift 2 ;;
    --no-bnb) NO_BNB=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fatal "Unknown option: $1. Use --help." ;;
  esac
done

# ----------------------------- Presets -----------------------------
case "$PRESET" in
  tiny)
    : "${N_EMBD:=384}"; : "${FF_LAYERS:=8}"; : "${N_HEAD:=6}"; : "${N_KV_HEAD:=2}"; : "${MLP_RATIO:=2.66}"; : "${BLOCK_SIZE:=256}"; : "${BATCH_SIZE:=12}"; : "${GRAD_ACCUM:=4}" ;;
  small)
    : "${N_EMBD:=512}"; : "${FF_LAYERS:=10}"; : "${N_HEAD:=8}"; : "${N_KV_HEAD:=2}"; : "${MLP_RATIO:=2.66}"; : "${BLOCK_SIZE:=384}"; : "${BATCH_SIZE:=6}"; : "${GRAD_ACCUM:=8}" ;;
  medium)
    : "${N_EMBD:=768}"; : "${FF_LAYERS:=10}"; : "${N_HEAD:=12}"; : "${N_KV_HEAD:=3}"; : "${MLP_RATIO:=2.5}"; : "${BLOCK_SIZE:=512}"; : "${BATCH_SIZE:=2}"; : "${GRAD_ACCUM:=16}" ;;
  *) fatal "Bad preset '$PRESET'. Use tiny, small, or medium." ;;
esac

# ----------------------------- Required files -----------------------------
[[ -f scripts/training/train_ff_only_ternary_ema.py ]] || fatal "scripts/training/train_ff_only_ternary_ema.py not found. Run from the repo root."
[[ -f "$DATA_DIR/train.bin" && -f "$DATA_DIR/val.bin" ]] || fatal "Missing $DATA_DIR/train.bin or val.bin. Build data first, e.g.:
python scripts/dataset_builders/build_safe_pretrain_data.py \\
  --dataset roneneldan/TinyStories \\
  --out-dir $DATA_DIR \\
  --vocab-size $VOCAB_SIZE \\
  --tokenizer-records 100000 \\
  --train-records 300000 \\
  --val-records 10000"

# ----------------------------- Env config -----------------------------
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_BITNET="${USE_BITNET:-1}"
export USE_BITNET_QUANT_ACT="${USE_BITNET_QUANT_ACT:-0}"
export BP_LAYERS="${BP_LAYERS:-0}"
export USE_DRAFT_HEAD="${USE_DRAFT_HEAD:-0}"
export FF_EMA_BP="${FF_EMA_BP:-1}"
export FF_EMA_DETACH="${FF_EMA_DETACH:-0}"
export FF_EMA_WARMUP_BIT="${FF_EMA_WARMUP_BIT:-800}"
export FF_EMA_STD="${FF_EMA_STD:-3.0}"
export FF_EMA_MAX_TRIPS="${FF_EMA_MAX_TRIPS:-1}"
export FF_EMA_BP_WEIGHT="${FF_EMA_BP_WEIGHT:-0.05}"
export TIE_EMB="${TIE_EMB:-1}"
export MEMORY_TOKENS="${MEMORY_TOKENS:-0}"
export INFER_MEMORY_TOKENS="${INFER_MEMORY_TOKENS:-0}"
export USE_ENGRAM="${USE_ENGRAM:-0}"
export CPU_HASH_CTX="${CPU_HASH_CTX:-0}"
export CPU_CTX="${CPU_CTX:-0}"
export USE_LIGER_CE="${USE_LIGER_CE:-0}"

export N_EMBD FF_LAYERS N_HEAD N_KV_HEAD MLP_RATIO BLOCK_SIZE BATCH_SIZE GRAD_ACCUM LR WEIGHT_DECAY

TOKENS_PER_STEP=$(( BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE ))
TOTAL_TOKENS=$(( TOKENS_PER_STEP * STEPS ))

if [[ -z "$RUN_NAME" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_NAME="ff_ternary_ema_${PRESET}_d${N_EMBD}_l${FF_LAYERS}_ctx${BLOCK_SIZE}_${TS}"
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="runs/${RUN_NAME}"
fi
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/train.log"
GPU_CSV="$OUT_DIR/gpu.csv"
ENV_FILE="$OUT_DIR/launcher_env.sh"

cat > "$ENV_FILE" <<EOF
# Recreate launcher environment
export PRESET="$PRESET"
export DATA_DIR="$DATA_DIR"
export OUT_DIR="$OUT_DIR"
export MAX_STEPS="$STEPS"
export EVAL_EVERY="$EVAL_EVERY"
export SAVE_EVERY="$SAVE_EVERY"
export EVAL_ITERS="$EVAL_ITERS"
export VOCAB_SIZE="$VOCAB_SIZE"
export N_EMBD="$N_EMBD"
export FF_LAYERS="$FF_LAYERS"
export N_HEAD="$N_HEAD"
export N_KV_HEAD="$N_KV_HEAD"
export MLP_RATIO="$MLP_RATIO"
export BLOCK_SIZE="$BLOCK_SIZE"
export BATCH_SIZE="$BATCH_SIZE"
export GRAD_ACCUM="$GRAD_ACCUM"
export LR="$LR"
export WEIGHT_DECAY="$WEIGHT_DECAY"
export GRAD_CLIP="$GRAD_CLIP"
export USE_BITNET="$USE_BITNET"
export USE_BITNET_QUANT_ACT="$USE_BITNET_QUANT_ACT"
export FF_EMA_BP="$FF_EMA_BP"
export FF_EMA_DETACH="$FF_EMA_DETACH"
export FF_EMA_WARMUP_BIT="$FF_EMA_WARMUP_BIT"
export FF_EMA_STD="$FF_EMA_STD"
export FF_EMA_MAX_TRIPS="$FF_EMA_MAX_TRIPS"
export FF_EMA_BP_WEIGHT="$FF_EMA_BP_WEIGHT"
EOF

# ----------------------------- Pretty summary -----------------------------
clear 2>/dev/null || true
say "${BOLD}${MAG}FF-only ternary EMA training${RESET}"
hr
keyval "preset" "$PRESET"
keyval "data" "$DATA_DIR"
keyval "out" "$OUT_DIR"
keyval "log" "$LOG_FILE"
keyval "model" "d=$N_EMBD layers=$FF_LAYERS heads=$N_HEAD/$N_KV_HEAD mlp=$MLP_RATIO"
keyval "batch" "micro=$BATCH_SIZE accum=$GRAD_ACCUM block=$BLOCK_SIZE"
keyval "tokens/step" "$TOKENS_PER_STEP"
keyval "target tokens" "$TOTAL_TOKENS"
keyval "steps" "$STEPS eval_every=$EVAL_EVERY save_every=$SAVE_EVERY eval_iters=$EVAL_ITERS"
keyval "lr/wd/clip" "$LR / $WEIGHT_DECAY / $GRAD_CLIP"
keyval "ternary" "USE_BITNET=$USE_BITNET quant_act=$USE_BITNET_QUANT_ACT"
keyval "EMA" "on=$FF_EMA_BP detach=$FF_EMA_DETACH warmup=$FF_EMA_WARMUP_BIT std=$FF_EMA_STD max_trips=$FF_EMA_MAX_TRIPS weight=$FF_EMA_BP_WEIGHT"
keyval "disabled" "BP=$BP_LAYERS draft=$USE_DRAFT_HEAD memory=$MEMORY_TOKENS engram=$USE_ENGRAM cpu_ctx=$CPU_CTX"

if command -v nvidia-smi >/dev/null 2>&1; then
  hr
  say "${BOLD}${BLUE}GPU snapshot${RESET}"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits \
    | awk -F, '{printf "  gpu                       %s | %s/%s MiB | util %s%% | temp %sC | power %sW\n", $1,$2,$3,$4,$5,$6}'
fi

CMD=(python scripts/training/train_ff_only_ternary_ema.py
  --data-dir "$DATA_DIR"
  --out-dir "$OUT_DIR"
  --vocab-size "$VOCAB_SIZE"
  --steps "$STEPS"
  --eval-every "$EVAL_EVERY"
  --eval-iters "$EVAL_ITERS"
  --save-every "$SAVE_EVERY"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --block-size "$BLOCK_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP")

if [[ "$NO_BNB" == "1" ]]; then
  CMD+=(--no-bnb)
fi

hr
say "${BOLD}${GREEN}Command${RESET}"
printf '  %q' "${CMD[@]}"; printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  say "${YELLOW}Dry run only. Nothing started.${RESET}"
  exit 0
fi

# ----------------------------- GPU logger -----------------------------
GPU_LOG_PID=""
cleanup() {
  local code=$?
  if [[ -n "${GPU_LOG_PID:-}" ]]; then
    kill "$GPU_LOG_PID" 2>/dev/null || true
  fi
  if [[ $code -eq 0 ]]; then
    say "\n${GREEN}${BOLD}Done.${RESET} Output: $OUT_DIR"
  else
    say "\n${RED}${BOLD}Training exited with code $code.${RESET} Check: $LOG_FILE" >&2
  fi
  exit $code
}
trap cleanup EXIT INT TERM

if [[ "$GPU_LOG" == "1" && -x "$(command -v nvidia-smi || true)" ]]; then
  echo "timestamp,memory_used_mib,memory_total_mib,util_gpu_pct,power_w,temp_c" > "$GPU_CSV"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits \
        | awk -F, '{gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $2); gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $4); gsub(/^ +| +$/, "", $5); gsub(/^ +| +$/, "", $6); print $1","$2","$3","$4","$5","$6}' >> "$GPU_CSV"
      sleep "$GPU_LOG_INTERVAL"
    done
  ) &
  GPU_LOG_PID=$!
  keyval "gpu log" "$GPU_CSV every ${GPU_LOG_INTERVAL}s"
fi

hr
say "${BOLD}${GREEN}Starting training${RESET} ${DIM}(Ctrl+C saves whatever the trainer has already written)${RESET}"
say "${DIM}Live log: tail -f $LOG_FILE${RESET}"
hr

# Capture both stdout/stderr. pipefail preserves Python failure exit code.
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
