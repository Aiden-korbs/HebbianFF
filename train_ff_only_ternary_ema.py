#!/usr/bin/env python3
"""
Small FF-only ternary+EMA trainer for the existing FFBP-clean repo.

Drop this script into the repo root. It uses ffbp_ema_cpu_ssm.FF_LLM with:
  USE_BITNET=1, BP_LAYERS=0, USE_DRAFT_HEAD=0, FF_EMA_BP=1
and trains on uint16 train.bin / val.bin files.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange

from ffbp_ema_cpu_ssm.config import CFG
from ffbp_ema_cpu_ssm.model import FF_LLM
from ffbp_ema_cpu_ssm.bitnet import (
    bitnet_train_cache_peak_mib,
    bitnet_train_cache_stats,
    clear_bitnet_weight_caches,
    count_bitnet_params,
)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    name = name.strip().lower()
    aliases = {
        "fp32": "float32",
        "f32": "float32",
        "fp16": "float16",
        "f16": "float16",
        "bf16": "bfloat16",
    }
    name = aliases.get(name, name)
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name!r}; use auto, float32, float16, or bfloat16")


def resolve_device(name: str) -> torch.device:
    name = name.strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return torch.device("cuda")
    raise ValueError(f"Unsupported device {name!r}; use auto, cpu, or cuda")


def set_default_env() -> None:
    defaults = {
        "USE_BITNET": "1",
        # Can speed up grad-accum runs, but costs extra VRAM. Keep off by
        # default for the 500M/8GB run; opt in with BITNET_CACHE_TRAIN=1.
        "BITNET_CACHE_TRAIN": "0",
        "USE_BITNET_QUANT_ACT": "0",  # start stable; turn on later once loss moves
        "BP_LAYERS": "0",
        "USE_DRAFT_HEAD": "0",
        "FF_EMA_BP": "1",
        "FF_EMA_DETACH": "0",         # normal CE trains trunk; EMA is auxiliary
        "FF_EMA_WARMUP_BIT": "800",
        "FF_EMA_STD": "3.0",
        "FF_EMA_MAX_TRIPS": "1",
        "FF_EMA_BP_WEIGHT": "0.05",
        "INFER_MEMORY_TOKENS": "0",
        "MEMORY_TOKENS": "0",
        "USE_ENGRAM": "0",
        "CPU_HASH_CTX": "0",
        "CPU_CTX": "0",
        "TIE_EMB": "1",
        "USE_LIGER_CE": "0",          # safer for custom tokenizer/CPU fallback
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def load_bin(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.uint16, mode="r")


_BATCH_OFFSETS: dict[int, np.ndarray] = {}


def _pin_if_requested(x: torch.Tensor, pin_memory: bool) -> torch.Tensor:
    if pin_memory and torch.cuda.is_available():
        return x.pin_memory()
    return x


def make_cpu_batch(data: np.memmap, batch_size: int, block_size: int, pin_memory: bool = False):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise RuntimeError(f"Data is too short: {len(data)} tokens for block_size={block_size}")
    if batch_size == 1:
        start = int(np.random.randint(0, max_start))
        window = np.asarray(data[start : start + block_size + 1], dtype=np.int64)
        xb = torch.from_numpy(np.ascontiguousarray(window[:-1])).view(1, block_size)
        yb = torch.from_numpy(np.ascontiguousarray(window[1:])).view(1, block_size)
        xb = _pin_if_requested(xb, pin_memory)
        yb = _pin_if_requested(yb, pin_memory)
        return xb, yb
    offsets = _BATCH_OFFSETS.get(block_size)
    if offsets is None:
        offsets = np.arange(block_size + 1, dtype=np.int64)
        _BATCH_OFFSETS[block_size] = offsets
    starts = np.random.randint(0, max_start, size=batch_size, dtype=np.int64)

    # One vectorized memmap gather replaces two Python loops over random slices.
    # The extra token lets x/y share the same sampled windows.
    batch = np.asarray(data[starts[:, None] + offsets[None, :]], dtype=np.int64)
    xb = torch.from_numpy(np.ascontiguousarray(batch[:, :-1]))
    yb = torch.from_numpy(np.ascontiguousarray(batch[:, 1:]))
    xb = _pin_if_requested(xb, pin_memory)
    yb = _pin_if_requested(yb, pin_memory)
    return xb, yb


def make_batch(data: np.memmap, batch_size: int, block_size: int, device: torch.device, pin_memory: bool = False):
    xb, yb = make_cpu_batch(data, batch_size, block_size, pin_memory=(pin_memory and device.type == "cuda"))
    if device.type == "cuda":
        return xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
    return xb.to(device), yb.to(device)


class BatchPrefetcher:
    def __init__(self, data: np.memmap, batch_size: int, block_size: int, depth: int, pin_memory: bool = False):
        self.data = data
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.pin_memory = bool(pin_memory)
        self.q: queue.Queue = queue.Queue(maxsize=max(1, int(depth)))
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="batch-prefetch", daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        while not self.stop.is_set():
            try:
                item = make_cpu_batch(self.data, self.batch_size, self.block_size, pin_memory=self.pin_memory)
            except BaseException as e:
                item = e
            while not self.stop.is_set():
                try:
                    self.q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if isinstance(item, BaseException):
                return

    def next(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1.0)


class CudaBatchPipeline:
    def __init__(self, cpu_prefetcher: BatchPrefetcher, device: torch.device, enabled: bool = True):
        self.cpu_prefetcher = cpu_prefetcher
        self.device = device
        self.enabled = bool(enabled and device.type == "cuda")
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None
        self.next_batch = None
        if self.enabled:
            self._preload()

    def _copy_to_device(self, batch):
        xb, yb = batch
        if not self.enabled:
            if self.device.type == "cuda":
                return xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            return xb.to(self.device), yb.to(self.device)
        with torch.cuda.stream(self.stream):
            return xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)

    def _preload(self) -> None:
        self.next_batch = self._copy_to_device(self.cpu_prefetcher.next())

    def next(self):
        if not self.enabled:
            return self._copy_to_device(self.cpu_prefetcher.next())
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        for tensor in batch:
            tensor.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return batch


@torch.no_grad()
def evaluate(model: FF_LLM, val_data: np.memmap, args, device: torch.device, dtype: torch.dtype, step: int) -> tuple[float, float]:
    # forward_features writes model._diag. During eval/no_grad the EMA path is disabled,
    # which used to overwrite useful train-step EMA diagnostics with NaNs.
    saved_diag = dict(getattr(model, "_diag", {}))
    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for _ in range(args.eval_iters):
            xb, yb = make_batch(val_data, args.batch_size, args.block_size, device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                loss = model.forward_features(xb, yb, update_state=False, current_lr=args.lr)
            losses.append(float(loss.detach().float().item()))
    if was_training:
        model.train()
    model._diag = saved_diag
    mean = sum(losses) / max(1, len(losses))
    ppl = math.exp(min(20.0, mean))
    return mean, ppl


def _safe_float(v, default=float("nan")) -> float:
    try:
        if torch.is_tensor(v):
            v = v.detach().float().item()
        return float(v)
    except Exception:
        return float(default)


def _copy_numeric_diag(model: nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in dict(getattr(model, "_diag", {})).items():
        if isinstance(v, (int, float)) or torch.is_tensor(v):
            out[k] = _safe_float(v)
    return out


def _gpu_mem_gib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.max_memory_allocated() / 1024**3


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def _fmt(v, digits=3, missing="nan") -> str:
    try:
        v = float(v)
    except Exception:
        return missing
    if not math.isfinite(v):
        return missing
    return f"{v:.{digits}f}"


def _ema_line(diag: dict[str, float]) -> str:
    trips = _safe_float(diag.get("ff_ema_trips", 0.0), 0.0)
    zmax = diag.get("ff_ema_max_z", float("nan"))
    zmean = diag.get("ff_ema_mean_z", float("nan"))
    goodness = diag.get("ff_ema_goodness", float("nan"))
    ema_ce = diag.get("ff_ema_ce", float("nan"))
    return (
        f"trips={trips:.0f} z_max={_fmt(zmax,2)} z_mean={_fmt(zmean,2)} "
        f"good={_fmt(goodness,2)} ema_ce={_fmt(ema_ce,3)}"
    )

def _parse_optional_bool(value: str):
    value = str(value).strip().lower()
    if value in {"auto", "none", ""}:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected auto/0/1 bool value, got {value!r}")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, no_bnb: bool, optimizer_name: str, adamw_foreach):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not name.endswith("tok_emb.weight"):
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    optimizer_name = optimizer_name.strip().lower()
    if optimizer_name == "sgd":
        print("[OPT] torch SGD")
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer {optimizer_name!r}; use adamw or sgd")

    if not no_bnb:
        try:
            import bitsandbytes as bnb
            print("[OPT] bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        except Exception as e:
            print(f"[OPT WARN] bitsandbytes unavailable: {e}")
    print(f"[OPT] torch AdamW foreach={adamw_foreach}")
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=torch.cuda.is_available(),
        foreach=adamw_foreach,
    )


def save_ckpt(path: Path, model: FF_LLM, cfg: CFG, step: int, val_loss: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "cfg": asdict(cfg) if is_dataclass(cfg) else vars(cfg),
        "step": int(step),
        "val_loss": float(val_loss),
        "args": vars(args),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)



def roundtrip_checkpoint_check(ckpt_path, make_model_fn, val_data, args, device, dtype, ref_val, step, eval_rng_state=None):
    """
    Reload the just-saved checkpoint into a fresh model and verify val_loss still matches.
    This catches BitNet/ternary state_dict bugs immediately.
    """
    if os.environ.get("ROUNDTRIP_CHECK", "0") != "1":
        return

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))

        rt_model = make_model_fn().to(device)
        missing, unexpected = rt_model.load_state_dict(state, strict=False)

        rt_model.eval()
        current_rng_state = np.random.get_state()
        try:
            if eval_rng_state is not None:
                np.random.set_state(eval_rng_state)
            with torch.no_grad():
                rt_val, rt_ppl = evaluate(rt_model, val_data, args, device, dtype, step)
        finally:
            np.random.set_state(current_rng_state)

        delta = float(rt_val) - float(ref_val)
        print(
            f"[ROUNDTRIP] file={ckpt_path} ref_val={float(ref_val):.4f} "
            f"reload_val={float(rt_val):.4f} delta={delta:+.4f} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

        if abs(delta) > 0.25:
            print(
                "[ROUNDTRIP WARN] Reloaded checkpoint does not reproduce live validation loss. "
                "Checkpoint is probably incomplete for this BitNet/ternary model.",
                flush=True,
            )

        del rt_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ROUNDTRIP ERROR] {type(e).__name__}: {e}", flush=True)

def main():
    set_default_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/ternary_tinystories"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/ff_only_ternary_ema_tiny"))
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=int(os.environ.get("MAX_STEPS", "5000")))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "12")))
    ap.add_argument("--grad-accum", type=int, default=int(os.environ.get("GRAD_ACCUM", "4")))
    ap.add_argument("--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", "256")))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", "3e-4")))
    ap.add_argument("--weight-decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.05")))
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--eval-first", type=int, default=int(os.environ.get("EVAL_FIRST", "1")), help="Run validation/save-best at step 1")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint to resume model weights from, e.g. runs/.../best.pt")
    ap.add_argument("--ema-diag-every", type=int, default=int(os.environ.get("EMA_DIAG_EVERY", "250")), help="Print train-time EMA diagnostics every N steps")
    ap.add_argument("--progress-every", type=int, default=int(os.environ.get("PROGRESS_EVERY", "5")), help="Update tqdm/GPU-memory diagnostics every N optimizer steps")
    ap.add_argument("--device", default=os.environ.get("TRAIN_DEVICE", "auto"), help="Training device: auto, cpu, or cuda")
    ap.add_argument("--dtype", default=os.environ.get("TRAIN_DTYPE", "auto"), help="Training dtype: auto, float32/fp32, float16/fp16, or bfloat16/bf16")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TORCH_THREADS", "0")), help="CPU intra-op threads; 0 leaves PyTorch default")
    ap.add_argument("--interop-threads", type=int, default=int(os.environ.get("TORCH_INTEROP_THREADS", "0")), help="CPU inter-op threads; 0 leaves PyTorch default")
    ap.add_argument("--prefetch-batches", type=int, default=int(os.environ.get("CPU_PREFETCH_BATCHES", "0")), help="Background train batch prefetch depth; useful for CPU runs")
    ap.add_argument("--pin-batches", type=int, default=int(os.environ.get("PIN_BATCHES", "1")), help="Use pinned host memory for CUDA train batches")
    ap.add_argument("--async-h2d", type=int, default=int(os.environ.get("ASYNC_H2D", "1")), help="Use a CUDA copy stream for prefetched host batches")
    ap.add_argument("--save-final", type=int, default=int(os.environ.get("SAVE_FINAL", "1")), help="Write final last.pt at training end")
    ap.add_argument("--bench-result", type=int, default=int(os.environ.get("BENCH_RESULT", "0")), help="Print a parseable BENCH_RESULT line at training end")
    ap.add_argument("--bench-warmup-steps", type=int, default=int(os.environ.get("BENCH_WARMUP_STEPS", "1")), help="Exclude this many initial steps from BENCH_RESULT timing")
    ap.add_argument("--no-progress", type=int, default=int(os.environ.get("NO_PROGRESS", "0")), help="Disable tqdm progress UI")
    ap.add_argument("--flush-denormal", type=int, default=int(os.environ.get("FLUSH_DENORMAL", "1")), help="Flush CPU denormal floats to zero")
    ap.add_argument("--optimizer", default=os.environ.get("OPTIMIZER", "adamw"), help="Optimizer: adamw or sgd")
    ap.add_argument("--adamw-foreach", default=os.environ.get("ADAMW_FOREACH", "auto"), help="AdamW foreach implementation: auto, 0, or 1")
    ap.add_argument("--use-ipex", type=int, default=int(os.environ.get("USE_IPEX", "0")), help="Use Intel Extension for PyTorch optimize() if installed")
    ap.add_argument("--compile", type=int, default=int(os.environ.get("TORCH_COMPILE", "0")), help="Wrap model with torch.compile; experimental for CPU training")
    ap.add_argument("--compile-mode", default=os.environ.get("TORCH_COMPILE_MODE", "default"), help="torch.compile mode")
    ap.add_argument("--no-bnb", action="store_true")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.interop_threads > 0:
        torch.set_num_interop_threads(args.interop_threads)
    if args.flush_denormal:
        torch.set_flush_denormal(True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Mirror CLI values into env-backed CFG before constructing CFG.
    os.environ["BATCH_SIZE"] = str(args.batch_size)
    os.environ["GRAD_ACCUM"] = str(args.grad_accum)
    os.environ["BLOCK_SIZE"] = str(args.block_size)

    cfg = CFG()
    cfg.batch_size = args.batch_size
    cfg.grad_accum_steps = args.grad_accum
    cfg.block_size = args.block_size
    cfg.seq_chunk_size = min(cfg.seq_chunk_size, args.block_size)
    cfg.bp_n_layer = 0
    cfg.use_bitnet = True
    cfg.use_draft_head = False
    cfg.memory_tokens = 0
    cfg.use_engram = False
    cfg.use_cpu_hash_context = False
    cfg.cpu_context_mode = "none"
    cfg.tie_token_embeddings = True

    train_data = load_bin(args.data_dir / "train.bin")
    val_data = load_bin(args.data_dir / "val.bin")

    model = FF_LLM(args.vocab_size, cfg).to(device=device, dtype=dtype)
    bit_params, total_params = count_bitnet_params(model)
    print(f"[MODEL] params={total_params/1e6:.2f}M bitnet_linear_params={bit_params/1e6:.2f}M")
    print(f"[MODEL] FF_LAYERS={cfg.ff_n_layer} BP_LAYERS={cfg.bp_n_layer} d={cfg.n_embd} heads={cfg.n_head}/{cfg.n_kv_head} block={cfg.block_size}")
    if getattr(cfg, "cpu_efficient_ff", False):
        attn_layers = sorted(getattr(model, "ff_attention_layers", set()))
        print(
            f"[CPU_FF] enabled=1 attn_every={cfg.ff_attn_every} "
            f"force_attn_last={cfg.ff_force_attn_last} mixer={cfg.ff_mixer} "
            f"kernel={cfg.local_mixer_kernel} fused_swiglu={int(cfg.use_fused_swiglu)} "
            f"attn_layers={attn_layers}"
        )
    else:
        print(f"[CPU_FF] enabled=0 all FF blocks use attention fused_swiglu={int(cfg.use_fused_swiglu)}")
    print(
        f"[DEVICE] device={device.type} dtype={dtype} threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} pin_batches={args.pin_batches} "
        f"async_h2d={args.async_h2d} prefetch={args.prefetch_batches}"
    )
    print(
        f"[BITNET] cache_train={os.environ.get('BITNET_CACHE_TRAIN', '0')} "
        f"cache_train_mib={os.environ.get('BITNET_CACHE_TRAIN_MIB', '0')} "
        f"custom_ste={os.environ.get('BITNET_CUSTOM_STE', '0')} "
        f"bypass_train={os.environ.get('BITNET_BYPASS_TRAIN', '0')} "
        f"quant_act={os.environ.get('USE_BITNET_QUANT_ACT', '0')}"
    )
    print(f"[EMA] enabled={cfg.use_ff_ema_bp} detach={cfg.ff_ema_detach_layers} warmup={cfg.ff_ema_warmup_steps} std={cfg.ff_ema_std_mult} max_trips={cfg.ff_ema_max_trips_per_step} weight={cfg.ff_ema_bp_weight}")

    start_step = 0
    best = float("inf")
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        start_step = int(ckpt.get("step", 0))
        best = float(ckpt.get("val_loss", float("inf")))
        print(f"[RESUME] loaded={args.resume_from} step={start_step} best_val={best:.4f} missing={len(missing)} unexpected={len(unexpected)}")

        if os.environ.get("EVAL_RESUME_FIRST", "0") == "1":
            model.eval()
            with torch.no_grad():
                resume_val, resume_ppl = evaluate(model, val_data, args, device, dtype, start_step)
            print(
                f"[RESUME EVAL] val_loss={float(resume_val):.4f} "
                f"val_ppl={resume_ppl:.2f} ckpt_best={best:.4f}",
                flush=True,
            )
            model.train()
        if start_step >= args.steps:
            print(f"[RESUME WARN] checkpoint step {start_step} >= --steps {args.steps}; increase --steps to continue training")

    opt = build_optimizer(
        model,
        args.lr,
        args.weight_decay,
        args.no_bnb or device.type == "cpu",
        args.optimizer,
        _parse_optional_bool(args.adamw_foreach),
    )
    if args.use_ipex:
        if device.type != "cpu":
            print("[IPEX WARN] USE_IPEX requested but device is not CPU; skipping")
        else:
            try:
                import intel_extension_for_pytorch as ipex
                model, opt = ipex.optimize(model, optimizer=opt, dtype=dtype)
                print("[IPEX] ipex.optimize applied")
            except Exception as e:
                print(f"[IPEX WARN] unavailable or failed: {type(e).__name__}: {e}")
    if args.compile:
        if not hasattr(torch, "compile"):
            print("[COMPILE WARN] torch.compile is not available; skipping")
        else:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[COMPILE] torch.compile applied mode={args.compile_mode}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_config.json").write_text(json.dumps({"args": vars(args), "cfg": asdict(cfg)}, indent=2, default=str))

    t0 = time.time()
    model.train()
    tokens_per_step = int(args.batch_size) * int(args.block_size) * int(args.grad_accum)
    metrics_path = args.out_dir / "metrics.jsonl"
    pbar = trange(start_step + 1, args.steps + 1, dynamic_ncols=True, disable=bool(args.no_progress))
    bench_start_step = start_step
    bench_t0 = t0
    prefetcher = None
    batch_pipeline = None
    if args.prefetch_batches > 0:
        prefetcher = BatchPrefetcher(
            train_data,
            args.batch_size,
            args.block_size,
            args.prefetch_batches,
            pin_memory=bool(args.pin_batches and device.type == "cuda"),
        )
        batch_pipeline = CudaBatchPipeline(prefetcher, device, enabled=bool(args.async_h2d))
    try:
        for step in pbar:
            model._current_step = step
            opt.zero_grad(set_to_none=True)
            accum_loss_t = None
            for _ in range(args.grad_accum):
                if batch_pipeline is not None:
                    xb, yb = batch_pipeline.next()
                else:
                    xb, yb = make_batch(
                        train_data,
                        args.batch_size,
                        args.block_size,
                        device,
                        pin_memory=bool(args.pin_batches and device.type == "cuda"),
                    )
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
                    loss = model.forward_features(xb, yb, update_state=True, current_lr=args.lr)
                    loss = loss / args.grad_accum
                loss.backward()
                loss_detached = loss.detach()
                accum_loss_t = loss_detached if accum_loss_t is None else accum_loss_t + loss_detached

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            clear_bitnet_weight_caches(model)

            elapsed = max(1e-9, time.time() - t0)
            done_steps = max(1, step - start_step)
            tok_s = (done_steps * tokens_per_step) / elapsed
            should_report = (
                step == start_step + 1
                or step == args.steps
                or (args.progress_every > 0 and step % args.progress_every == 0)
                or (args.eval_every > 0 and step % args.eval_every == 0)
                or (args.ema_diag_every > 0 and step % args.ema_diag_every == 0)
                or (args.save_every > 0 and step % args.save_every == 0)
            )
            if should_report:
                accum_loss = float(accum_loss_t.float().item()) if accum_loss_t is not None else float("nan")
                train_diag = _copy_numeric_diag(model)
                alloc_gib, peak_gib = _gpu_mem_gib()
                cache_mib, cache_budget_mib = bitnet_train_cache_stats()
                zmax = train_diag.get("ff_ema_max_z", float("nan"))
                trips = train_diag.get("ff_ema_trips", 0.0)
                best_short = _fmt(best, 3) if math.isfinite(best) else "----"
                pbar.set_description(f"loss={accum_loss:.3f} best={best_short}")
                pbar.set_postfix_str(
                    f"tok/s={tok_s:,.0f} vram={_fmt(alloc_gib,2)}/{_fmt(peak_gib,2)}GiB "
                    f"cache={cache_mib:.0f}/{cache_budget_mib:.0f}MiB "
                    f"ema_trips={_safe_float(trips,0):.0f} z={_fmt(zmax,2)}"
                )

            if args.ema_diag_every > 0 and (step % args.ema_diag_every == 0):
                print(f"\n[EMA TRAIN] step={step} {_ema_line(train_diag)}")

            if step % args.eval_every == 0 or (args.eval_first and step == 1):
                if not should_report:
                    accum_loss = float(accum_loss_t.float().item()) if accum_loss_t is not None else float("nan")
                    train_diag = _copy_numeric_diag(model)
                    alloc_gib, peak_gib = _gpu_mem_gib()
                eval_rng_state = np.random.get_state()
                val_loss, val_ppl = evaluate(model, val_data, args, device, dtype, step)
                clear_bitnet_weight_caches(model)
                elapsed = time.time() - t0
                improved = val_loss < best
                if improved:
                    best = val_loss
                    best_path = args.out_dir / "best.pt"
                    save_ckpt(best_path, model, cfg, step, val_loss, args)
                    roundtrip_checkpoint_check(
                        best_path,
                        lambda: FF_LLM(args.vocab_size, cfg),
                        val_data,
                        args,
                        device,
                        dtype,
                        val_loss,
                        step,
                        eval_rng_state,
                    )
                metrics = {
                    "step": int(step),
                    "train_loss": float(accum_loss),
                    "val_loss": float(val_loss),
                    "best_val_loss": float(best),
                    "val_ppl": float(val_ppl),
                    "elapsed_min": float(elapsed / 60.0),
                    "tok_s": float(tok_s),
                    "gpu_alloc_gib": float(alloc_gib),
                    "gpu_peak_gib": float(peak_gib),
                    **{f"diag_{k}": float(v) for k, v in train_diag.items() if isinstance(v, (int, float))},
                }
                _append_jsonl(metrics_path, metrics)
                star = " *best*" if improved else ""
                print(
                    f"\n[EVAL] step={step} train_loss={accum_loss:.4f} "
                    f"val_loss={val_loss:.4f} best={best:.4f} val_ppl={val_ppl:.2f} "
                    f"tok/s={tok_s:,.0f} vram={_fmt(alloc_gib,2)}/{_fmt(peak_gib,2)}GiB "
                    f"elapsed_min={elapsed/60:.1f}{star}"
                )
                print(f"[EMA TRAIN] {_ema_line(train_diag)}")
                print(f"[DIAG TRAIN] {json.dumps(train_diag, sort_keys=True)[:1200]}")

            if step % args.save_every == 0:
                save_ckpt(args.out_dir / f"step_{step}.pt", model, cfg, step, best, args)
                save_ckpt(args.out_dir / "last.pt", model, cfg, step, best, args)

            if args.bench_result and args.bench_warmup_steps > 0 and step == start_step + args.bench_warmup_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                bench_start_step = step
                bench_t0 = time.time()
    finally:
        if prefetcher is not None:
            prefetcher.close()

    if args.bench_result and device.type == "cuda":
        torch.cuda.synchronize(device)
    final_elapsed = max(1e-9, time.time() - t0)
    bench_elapsed = max(1e-9, time.time() - bench_t0)
    final_steps = max(0, args.steps - start_step)
    bench_steps = max(0, args.steps - bench_start_step)
    final_tok_s = (final_steps * tokens_per_step) / final_elapsed if final_steps > 0 else 0.0
    bench_tok_s = (bench_steps * tokens_per_step) / bench_elapsed if bench_steps > 0 else 0.0
    if args.bench_result:
        alloc_gib, peak_gib = _gpu_mem_gib()
        print(
            f"BENCH_RESULT steps={bench_steps} tokens={bench_steps * tokens_per_step} "
            f"elapsed_sec={bench_elapsed:.3f} tok_s={bench_tok_s:.3f} "
            f"total_steps={final_steps} total_elapsed_sec={final_elapsed:.3f} total_tok_s={final_tok_s:.3f} "
            f"gpu_alloc_gib={alloc_gib:.3f} gpu_peak_gib={peak_gib:.3f} "
            f"bitnet_cache_peak_mib={bitnet_train_cache_peak_mib():.1f}"
        )
    if args.save_final:
        save_ckpt(args.out_dir / "last.pt", model, cfg, args.steps, best, args)


if __name__ == "__main__":
    main()
