#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = PROJECT_ROOT / "scripts" / "inference"
for path in (PROJECT_ROOT, INFERENCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
import torch.nn.functional as F
from tqdm import tqdm

import chat_hf


def load_jsonl_tokens(path: str, tok, max_records: int = 0):
    ids_all = []
    n = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if "messages" in obj:
                text = tok.apply_chat_template(
                    obj["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            elif "text" in obj:
                text = obj["text"]
            elif "prompt" in obj and "response" in obj:
                text = tok.apply_chat_template(
                    [
                        {"role": "user", "content": obj["prompt"]},
                        {"role": "assistant", "content": obj["response"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                continue

            ids = tok(text, add_special_tokens=False).input_ids
            if tok.eos_token_id is not None:
                ids.append(int(tok.eos_token_id))

            if len(ids) >= 4:
                ids_all.extend(ids)
                n += 1

            if max_records and n >= max_records:
                break

    if not ids_all:
        raise RuntimeError(f"No usable records found in {path}")

    return torch.tensor(ids_all, dtype=torch.long), n


def make_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    if data.numel() < block_size + 2:
        reps = math.ceil((block_size + 2) / max(1, data.numel()))
        data = data.repeat(reps)

    max_start = data.numel() - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))

    xs, ys = [], []
    for s in starts:
        s = int(s)
        chunk = data[s:s + block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    return (
        torch.stack(xs).to(device, non_blocking=True),
        torch.stack(ys).to(device, non_blocking=True),
    )


def build_optimizer(params, lr: float, wd: float, no_bnb: bool):
    params = [p for p in params if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters selected")

    if not no_bnb:
        try:
            import bitsandbytes as bnb
            print("[OPT] bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=wd)
        except Exception as e:
            print(f"[OPT WARN] bitsandbytes unavailable: {e}")

    print("[OPT] torch AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def ff_block_index(name: str):
    m = re.match(r"ff_blocks\.(\d+)\.", name)
    if not m:
        return None
    return int(m.group(1))


def set_trainable_ff(raw_model, ff_last_n: int = 0, include_norms: bool = False):
    for p in raw_model.parameters():
        p.requires_grad_(False)

    n_ff = len(raw_model.ff_blocks)
    min_idx = 0 if ff_last_n <= 0 else max(0, n_ff - ff_last_n)

    trainable = 0
    total = 0

    for name, p in raw_model.named_parameters():
        total += p.numel()
        train = False

        idx = ff_block_index(name)
        if idx is not None and idx >= min_idx:
            train = True

        if include_norms and (
            name.startswith("pre_ff_norm.")
            or name.startswith("post_ff_norm.")
        ):
            train = True

        if train:
            p.requires_grad_(True)
            trainable += p.numel()

    print(
        f"[TRAINABLE] FF only: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
        f"(ff_last_n={ff_last_n or 'all'}, include_norms={include_norms})"
    )

    for name, p in raw_model.named_parameters():
        if p.requires_grad:
            print("  train:", name, tuple(p.shape))

    return [p for p in raw_model.parameters() if p.requires_grad]


def set_trainable_draft(raw_model, train_out: bool = True):
    for p in raw_model.parameters():
        p.requires_grad_(False)

    if raw_model.ff_draft_head is None:
        raise RuntimeError("No ff_draft_head. Use a DRAFT_SEEDED checkpoint with USE_DRAFT_HEAD=1.")

    trainable = 0
    total = 0

    for name, p in raw_model.named_parameters():
        total += p.numel()

        if not name.startswith("ff_draft_head."):
            continue

        if not train_out and name.startswith("ff_draft_head.out."):
            continue

        p.requires_grad_(True)
        trainable += p.numel()

    print(f"[TRAINABLE] draft only: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    for name, p in raw_model.named_parameters():
        if p.requires_grad:
            print("  train:", name, tuple(p.shape))

    return [p for p in raw_model.parameters() if p.requires_grad]


def save_ckpt(path: Path, raw_model, cfg, step: int, loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(cfg):
        cfg_obj = asdict(cfg)
    else:
        cfg_obj = dict(vars(cfg))

    payload = {
        "model": raw_model.state_dict(),
        "cfg": cfg_obj,
        "step": int(step),
        "loss": float(loss),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def full_loss_with_draft_temporarily_disabled(raw_model, xb, yb, current_lr: float):
    """
    Stage 1 trains FF layers only using the normal final loss.
    But we do not want ff_draft_head to run, because it creates huge vocab logits.
    Keep the module in the model for saving, but disable it for this forward only.
    """
    saved_draft = raw_model.ff_draft_head
    raw_model.ff_draft_head = None
    try:
        loss = raw_model.forward_features(
            xb,
            yb,
            update_state=False,
            current_lr=current_lr,
        )
    finally:
        raw_model.ff_draft_head = saved_draft
    return loss


@torch.no_grad()
def frozen_ff_hidden(raw_model, xb):
    """
    Stage 2 trains draft head only.
    FF trunk is frozen and runs under no_grad.
    No BP blocks, no final logits.
    """
    raw_model._working_mem = None
    raw_model._engram_state = None

    x_emb = raw_model.tok_emb(xb)
    x = raw_model.pre_ff_norm(x_emb) if raw_model.pre_ff_norm is not None else x_emb

    for blk in raw_model.ff_blocks:
        x = blk(x, mem=None, pos_offset=0)

    if raw_model.post_ff_norm is not None:
        x = raw_model.post_ff_norm(x)

    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ff", "draft"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default="data_qwen_repair/repair_seed_plus_reasoning.jsonl")
    ap.add_argument("--outdir", default="repair_runs/ff_then_draft")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--no-bnb", action="store_true")

    # FF-stage controls.
    ap.add_argument("--ff-last-n", type=int, default=0, help="0 means train all FF blocks. Use 2/4/6 if OOM.")
    ap.add_argument("--include-ff-norms", action="store_true")

    # Draft-stage controls.
    ap.add_argument("--freeze-draft-out", action="store_true", help="Freeze giant draft vocab out weight; tune ln/proj/blend only.")

    args = ap.parse_args()

    os.environ.setdefault("USE_DRAFT_HEAD", "1")
    os.environ.setdefault("DRAFT_BLEND_BP", "0")
    os.environ.setdefault("USE_ENGRAM", "0")
    os.environ.setdefault("MEMORY_TOKENS", "0")
    os.environ.setdefault("CPU_CTX", "0")
    os.environ.setdefault("CPU_HASH_CTX", "0")

    run_name = args.run_name or f"{args.stage}_" + time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    load_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
    )

    model, raw_model, tok, cfg, device, dtype = chat_hf.load_model(load_args)
    raw_model.train()

    cfg.block_size = int(args.block_size)
    cfg.seq_chunk_size = min(int(getattr(cfg, "seq_chunk_size", args.block_size)), int(args.block_size))
    cfg.local_window = min(int(getattr(cfg, "local_window", args.block_size)), int(args.block_size))
    cfg.use_engram = False
    cfg.memory_tokens = 0
    cfg.use_cpu_hash_context = False

    if args.stage == "ff":
        params = set_trainable_ff(raw_model, ff_last_n=args.ff_last_n, include_norms=args.include_ff_norms)
    else:
        params = set_trainable_draft(raw_model, train_out=not args.freeze_draft_out)

    opt = build_optimizer(params, args.lr, args.weight_decay, args.no_bnb)

    data, n_records = load_jsonl_tokens(args.data, tok, args.max_records)
    print(f"[DATA] records={n_records} tokens={data.numel():,} path={args.data}")

    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    best = float("inf")
    ema = None
    t0 = time.time()

    opt.zero_grad(set_to_none=True)

    for step in tqdm(range(1, args.steps + 1), desc=args.stage):
        accum = 0.0

        for _ in range(args.grad_accum):
            xb, yb = make_batch(data, args.batch_size, args.block_size, device)

            raw_model._working_mem = None
            raw_model._engram_state = None

            if args.stage == "ff":
                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
                    loss = full_loss_with_draft_temporarily_disabled(raw_model, xb, yb, args.lr)

            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
                        ff_hidden = frozen_ff_hidden(raw_model, xb)

                with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
                    draft_logits, _draft_hidden = raw_model.ff_draft_head(ff_hidden)
                    loss = F.cross_entropy(
                        draft_logits.reshape(-1, raw_model.vocab_size).float(),
                        yb.reshape(-1).long(),
                    )

            (loss / args.grad_accum).backward()
            accum += float(loss.detach())

            raw_model._working_mem = None
            raw_model._engram_state = None

        if args.clip and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip)

        opt.step()
        opt.zero_grad(set_to_none=True)

        step_loss = accum / max(1, args.grad_accum)
        ema = step_loss if ema is None else 0.97 * ema + 0.03 * step_loss

        if step_loss < best:
            best = step_loss
            save_ckpt(outdir / "best.pt", raw_model, cfg, step, step_loss)

        if step % args.save_every == 0:
            save_ckpt(outdir / f"step_{step}.pt", raw_model, cfg, step, step_loss)

        if step % args.log_every == 0 or step == 1:
            blend_obj = getattr(raw_model, "blend_gate", float("nan"))
            blend = float(blend_obj() if callable(blend_obj) else blend_obj)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            elapsed = (time.time() - t0) / 60.0
            print(
                f"[LOG] stage={args.stage} step={step} loss={step_loss:.4f} "
                f"ema={ema:.4f} best={best:.4f} blend={blend:.4f} "
                f"vram={mem:.2f}GiB elapsed={elapsed:.1f}m",
                flush=True,
            )

        gc.collect()

    save_ckpt(outdir / "final.pt", raw_model, cfg, args.steps, ema if ema is not None else best)
    print(f"[DONE] outdir={outdir}")
    print(f"[BEST] {outdir / 'best.pt'}")


if __name__ == "__main__":
    main()
