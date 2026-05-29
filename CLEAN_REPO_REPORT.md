# Clean Repo Report

## Paths

- Original repo path: `/home/corbs/datasets/ffbp_codex`
- New repo path: `/home/corbs/datasets/ternary-revforge`
- Chosen repo name: `ternary-revforge`

The name fits because the active repo combines ternary/BitNet training and packed ternary runtime work with an FF/BP residual-block architecture that still carries reversible-block naming compatibility.

## Copied Files and Folders

Copied source and project files:

- Core package: `ffbp_ema_cpu_ssm/`
- Tooling: `tools/`
- Web server package: `web_chat/`
- Documentation: `docs/`, `architecture_plan.md`, `weight_cost_experiment_results.md`
- Main scripts: `chat_hf.py`, `web_chat.py`, `train_ff_only_ternary_ema.py`, `train_ff_draft_repair.py`, `train_ff_then_draft.py`, `compare_native_qwen_eval.py`, `seed_features.py`
- Data builders: `build_safe_pretrain_data.py`, `build_large_pretrain_data.py`, `build_fineweb_edu_500m_data.sh`
- Launchers and smoke scripts: `run_3070_tinystories.sh`, `run_500m_fineweb_edu.sh`, `run_500m_cpu_efficient.sh`, `bench_cpu_efficient_threads.sh`, `bench_hybrid_gpu_cpu.sh`, `resume_eval_tiny.sh`, `train_tiny_roundtrip_test.sh`, `train_tiny_roundtrip_test_nocache.sh`
- Setup/config: `requirements.txt`, `config.example.env`, `.gitignore`
- Small UI asset: `training_viewer.html`, `web_chat/static/index.html`

The copied root shell launchers were adjusted in the clean repo to `cd` relative to their own location instead of the original absolute source path.

## Intentionally Skipped

Skipped generated or local-only content:

- `.git/`, `.venv/`, `.hf_cache/`, `__pycache__/`
- `logs/`, `runs/`, `wandb/`, run metrics, benchmark outputs, and generated CSV/log files
- `models/` and all model checkpoints/weights
- `data/`, `data_clean_1b/`, `data_tinyllama_tok_50m/`, `data_qwen_repair/`, tokenizer/data bins, and JSONL datasets
- `old/` snapshots and all `*.bak*` backup files
- large binary artifacts such as `*.pt`, `*.bin`, `*.npy`, `*.npz`, `*.pkl`, `*.safetensors`, `*.gguf`

## Important Files Skipped Due to Size or Artifact Status

- `models/DeepSeek-R1-Distill-Qwen-7B-ff28-bf16-block512.pt` was skipped because it is a 15GB model checkpoint.
- `data/fineweb_edu_10bt_bpe32k/train.bin` and `val.bin` were skipped because they are generated dataset binaries.
- Tokenizer/model files under `data_clean_1b/` and `data_tinyllama_tok_50m/` were skipped because they are generated dataset artifacts rather than source. Rebuild or copy them locally if a specific old run needs exact reproduction.
- `ff_llm_spm.vocab` and `data_manifest.json` were skipped as root-level generated data/tokenizer leftovers, not active source requirements.

## Follow-Up Before Pushing

- Run a small dataset build and short training smoke test in the clean repo if you want runtime validation beyond syntax checks.
- Decide whether to add a license before publishing.
- Consider adding pinned dependency versions once you settle on the exact CUDA/PyTorch environment.
- Keep model weights, datasets, checkpoints, and logs out of GitHub; publish large artifacts separately if needed.
