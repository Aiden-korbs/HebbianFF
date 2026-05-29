#!/usr/bin/env bash
# import_model.sh — one-line wrapper around tools/import_hf.py
#
# Usage
# -----
#   ./scripts/import_model.sh <hf_repo_or_local_path> [extra python args]
#
# Examples
# --------
#   # Qwen 2.5 1.5B  (fastest, lowest VRAM)
#   ./scripts/import_model.sh Qwen/Qwen2.5-1.5B-Instruct
#
#   # DeepSeek R1 distill (reasoning, same arch as Qwen2.5 — just works)
#   ./scripts/import_model.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#
#   # Qwen 7B — import 14 of 28 layers as FF path to keep VRAM reasonable
#   ./scripts/import_model.sh Qwen/Qwen2.5-7B-Instruct --ff-layers 14
#
#   # DeepSeek R1 7B distill — same flag, same idea
#   ./scripts/import_model.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --ff-layers 14
#
#   # Llama 3.1 8B
#   ./scripts/import_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct
#
#   # Local path, custom output, longer context
#   ./scripts/import_model.sh /data/my_model --out models/custom.pt --block-size 4096
#
#   # See what would happen without writing anything
#   ./scripts/import_model.sh Qwen/Qwen2.5-7B-Instruct --dry-run
#
# Output
# ------
#   models/<model_name>.pt   (or whatever --out you set)
#
# After import
# ------------
#   Check the checkpoint:
#     ./scripts/check_model.sh models/<model_name>.pt
#
#   Run a quick chat to confirm it works:
#     USE_KV_CACHE=1 python chat_hf.py \
#       --checkpoint models/<model_name>.pt \
#       --tokenizer Qwen/Qwen2.5-1.5B-Instruct
#
# Notes
# -----
#   - BP layers default to 0. This is intentional: the pretrained FF weights
#     alone match or exceed native quality at this model scale, so BP layers
#     add VRAM cost for no gain. Pass --bp-layers N to re-enable them.
#
#   - Weights are saved as float32 by default for training stability.
#     Pass --dtype bfloat16 to halve the file size (fine for inference-only).
#
#   - For models >7B, make sure you have enough CPU RAM to load the HF weights
#     before the script starts freeing them layer by layer (~2× model size peak).
#
# Requirements
# ------------
#   pip install transformers accelerate
#   (PyTorch already required by the repo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <hf_repo_or_local_path> [--ff-layers N] [--out path] [--dry-run] [...]"
    echo "Run with --help for full options:"
    echo "  python tools/import_hf.py --help"
    exit 1
fi

cd "$REPO_ROOT"

# Check transformers is available
python - <<'EOF'
try:
    import transformers
except ImportError:
    print("\n[ERROR] transformers is not installed.")
    print("  Run:  pip install transformers accelerate\n")
    raise SystemExit(1)
EOF

exec python tools/import_hf.py "$@"
