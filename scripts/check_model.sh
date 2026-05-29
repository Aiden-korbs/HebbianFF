#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/check_checkpoint.py "${1:-${CHAT_CHECKPOINT:-models/model.pt}}"
