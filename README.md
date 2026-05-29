# HebbianFF

`HebbianFF` is an experimental PyTorch research repo for FF/BP-style causal language models, ternary/BitNet-style training, Hugging Face checkpoint import, and low-VRAM inference experiments. The active code centers on `FF_LLM`, a decoder-only model with configurable feed-forward and backprop/correction blocks, grouped-query attention, RoPE, optional draft/memory sidecars, checkpoint import tools, and packed ternary runtime utilities.

This is research code. It is meant for experimentation and inspection, not as a stable library or a production serving stack.

## Key Ideas

- **FF/BP model layout:** `HebbianFF.model.FF_LLM` builds a stack of FF blocks plus optional BP/correction blocks. The current block implementation is a full-width residual decoder block; `RevBlock` remains as a compatibility alias.
- **Ternary / BitNet training path:** `HebbianFF.bitnet.BitLinear` and `train_ff_only_ternary_ema.py` support FF-only ternary training with EMA-triggered auxiliary losses.
- **Hugging Face retrofit path:** `tools/import_hf.py` imports selected HF decoder checkpoints into the local `FF_LLM` checkpoint format for parity and retrofit experiments.
- **Inference and serving:** `chat_hf.py` provides local checkpoint chat/inference. `web_chat.py` and `web_chat/server.py` provide web serving paths.
- **Packed ternary runtime:** `HebbianFF.ternary_runtime`, `tools/transfer_to_1bit.py`, and the ternary repair/eval scripts explore compact packed weights and repaired ternary adapters.
- **Low-VRAM experiments:** bounded KV cache, sink tokens, CPU offload helpers, packed linears, and benchmarking scripts are included, but many paths are explicitly experimental.

## Folder Structure

```text
HebbianFF/      Core model, blocks, config, BitNet layers, packed runtimes, memory helpers
scripts/               Small wrappers for importing/checking models and launching the local server
tools/                 Import, evaluation, compression, ternary repair, benchmark, and diagnostic tools
docs/                  Research notes and runtime/evaluation documentation
web_chat/              FastAPI web-chat server and static UI
*.py                   Training, inference, data preparation, evaluation, and serving entry points
*.sh                   Launchers for TinyStories, FineWeb-Edu, CPU/GPU benchmarking, and smoke tests
```

## Requirements

Install PyTorch separately for your CUDA/CPU environment, then install the Python package requirements:

```bash
pip install -r requirements.txt
```

The repo commonly uses:

- Python 3.10+
- PyTorch
- `transformers`, `sentencepiece`
- `datasets`, `tokenizers`, `numpy`, `tqdm`
- `fastapi`, `uvicorn`, `pydantic` for the web server

Optional acceleration packages such as `bitsandbytes`, Triton, Intel IPEX, or Liger kernels may improve or unlock some paths, but the code contains fallbacks for several of them.

## Setup

```bash
git clone <your-fork-url> HebbianFF
cd HebbianFF
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For CUDA builds, install the PyTorch wheel that matches your driver before installing the remaining requirements.

## Important Workflows

### Build a small TinyStories-style dataset

```bash
python build_safe_pretrain_data.py \
  --dataset roneneldan/TinyStories \
  --out-dir data/ternary_tinystories \
  --vocab-size 16000 \
  --tokenizer-records 100000 \
  --train-records 300000 \
  --val-records 10000
```

### Train a small FF-only ternary EMA model

```bash
./run_3070_tinystories.sh --preset tiny
```

The launcher expects `data/ternary_tinystories/train.bin` and `val.bin` and writes checkpoints/metrics under `runs/`.

### Build a FineWeb-Edu pretraining dataset

```bash
TARGET_TRAIN_TOKENS=1000000000 \
TARGET_VAL_TOKENS=10000000 \
./build_fineweb_edu_500m_data.sh
```

The default script target is much larger, so set token limits deliberately before running it.

### Launch the 500M-style FineWeb-Edu training config

```bash
./run_500m_fineweb_edu.sh
```

For CPU-oriented experiments:

```bash
./run_500m_cpu_efficient.sh
```

### Import a Hugging Face checkpoint

```bash
./scripts/import_model.sh Qwen/Qwen2.5-1.5B-Instruct \
  --out models/qwen25-1.5b-ff.pt \
  --block-size 1024 \
  --bp-layers 0
```

Then check it:

```bash
./scripts/check_model.sh models/qwen25-1.5b-ff.pt
```

### Run local chat from an imported checkpoint

```bash
USE_KV_CACHE=1 python chat_hf.py \
  --checkpoint models/qwen25-1.5b-ff.pt \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct
```

Or start the FastAPI web server:

```bash
cp config.example.env .env
./scripts/run_local.sh models/qwen25-1.5b-ff.pt
```

### Compare an imported checkpoint with native HF

```bash
python tools/measure_retrofit.py \
  --native-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --ff-checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \
  --block-size 1024 \
  --dtype bf16
```

See `docs/RUN_RETROFIT_EVAL.md` for details.

## Not Included

This clean repo intentionally excludes generated and machine-local state:

- model weights and checkpoints (`*.pt`, `*.safetensors`, `models/`, `runs/`)
- datasets and token bins (`data/`, `data_*/`, `*.bin`, `*.jsonl`)
- logs, metrics outputs, benchmark run folders, and `wandb/`
- virtual environments, Hugging Face caches, Python caches, and backup snapshots

You need to rebuild datasets or import/download models locally before running training or inference commands that depend on those artifacts.

## Current Status

The codebase is an active experiment bench. The TinyStories/FineWeb-Edu ternary training path, HF import path, chat runtime, retrofit measurement tools, and ternary runtime experiments are present. Several architecture features are research scaffolding or require trained checkpoints/adapters before they are useful, especially draft heads, engram memory, CPU hash context, BP correction blocks, and attention-replacement ideas.

Expect to read the scripts, run small smoke tests first, and verify parity before trusting any retrofit or compression result.
