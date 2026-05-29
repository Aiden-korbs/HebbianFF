#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ "${1:-}" != "" ]; then export CHAT_CHECKPOINT="$1"; fi
export CHAT_CHECKPOINT="${CHAT_CHECKPOINT:-models/model.pt}"
export CHAT_TOKENIZER="${CHAT_TOKENIZER:-weights/qwen25-1.5b-instruct}"
export USE_KV_CACHE="${USE_KV_CACHE:-1}"
export USE_FF_DRAFT_SKIP="${USE_FF_DRAFT_SKIP:-0}"
export USE_DRAFT_HEAD="${USE_DRAFT_HEAD:-0}"
export DRAFT_BLEND_BP="${DRAFT_BLEND_BP:-0}"
export INFER_MEMORY_TOKENS="${INFER_MEMORY_TOKENS:-0}"
export INFER_USE_ENGRAM="${INFER_USE_ENGRAM:-0}"
export BLOCK_SIZE="${BLOCK_SIZE:-256}"
export KV_CACHE_MAX_LEN="${KV_CACHE_MAX_LEN:-0}"
export KV_CACHE_SINK_TOKENS="${KV_CACHE_SINK_TOKENS:-0}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-7860}"
python tools/check_checkpoint.py "$CHAT_CHECKPOINT"
exec uvicorn web_chat.server:app --host "$HOST" --port "$PORT"
