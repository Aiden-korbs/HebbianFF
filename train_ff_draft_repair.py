#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

import chat_hf


def load_jsonl_messages(path: str, tok, max_records: int = 0):
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
                messages = [
                    {"role": "user", "content": obj["prompt"]},
                    {"role": "assistant", "content": obj["response"]},
                ]
                text = tok.apply_chat_template(
                    messages,
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

    return torch.tensor(ids_all, dtype=torch.long), n


def make_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    if data.numel() < block_size + 2:
        reps = math.ceil((block_size + 2) / max(1, data.numel()))
        data = data.repeat(reps)

    max_start = data.numel() - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))

    xs = []
    ys = []
    for s in starts:
        s = int(s)
        chunk = data[s:s + block_size + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    x = torch.stack(xs).to(device, non_blocking=True)
    y = torch.stack(ys).to(device, non_blocking=True)
    return x, y


def set_trainable(model, mode: str):
    raw = getattr(model, "_orig_mod", model)

    for p in raw.parameters():
        p.requires_grad_(False)

    def enable_if(name: str):
        if mode == "draft":
            return name.startswith("ff_draft_head.")

        if mode == "draft_head":
            return (
                name.startswith("ff_draft_head.")
                or name.startswith("final_ln.")
                or name.startswith("final_proj.")
                or name.startswith("out_proj.")
            )

        if mode == "bp_draft_head":
            return (
                name.startswith("ff_draft_head.")
                or name.startswith("bp_blocks.")
                or name.startswith("final_ln.")
                or name.startswith("final_proj.")
                or name.startswith("out_proj.")
            )

        if mode == "all":
            return True

        raise ValueError(f"unknown train mode: {mode}")

    total = 0
    trainable = 0

    for name, p in raw.named_parameters():
        total += p.numel()
        if enable_if(name):
            p.requires_grad_(True)
            trainable += p.numel()

    print(f"[TRAINABLE] mode={mode} trainable={trainable/1e6:.1f}M / total={total/1e6:.1f}M")
    return raw


def build_optimizer(model, lr: float, weight_decay: float, use_bnb: bool):
    params = [p for p in model.parameters() if p.requires_grad]

    if not params:
        raise RuntimeError("No trainable params")

    if use_bnb:
        try:
            import bitsandbytes as bnb
            print("[OPT] bitsandbytes AdamW8bit")
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception as e:
            print(f"[OPT WARN] bitsandbytes unavailable: {e}")

    print("[OPT] torch AdamW")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=torch.cuda.is_available())


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", default="data_qwen_repair/repair_seed_plus_reasoning.jsonl")
    ap.add_argument("--outdir", default="repair_runs/draft_repair")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--train-mode", default="bp_draft_head", choices=["draft", "draft_head", "bp_draft_head", "all"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--draft-weight", type=float, default=0.12)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--no-bnb", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("USE_DRAFT_HEAD", "1")
    os.environ.setdefault("DRAFT_BLEND_BP", "1")
    os.environ.setdefault("USE_ENGRAM", "0")
    os.environ.setdefault("MEMORY_TOKENS", "0")
    os.environ.setdefault("CPU_CTX", "0")
    os.environ.setdefault("CPU_HASH_CTX", "0")

    run_name = args.run_name or time.strftime("ff_draft_repair_%Y%m%d_%H%M%S")
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

    if raw_model.ff_draft_head is None:
        raise RuntimeError(
            "Model has no ff_draft_head. Use a DRAFT_SEEDED checkpoint and run with USE_DRAFT_HEAD=1."
        )

    cfg.batch_size = args.batch_size
    cfg.grad_accum_steps = args.grad_accum
    cfg.block_size = args.block_size
    cfg.seq_chunk_size = min(int(cfg.seq_chunk_size), int(args.block_size))
    cfg.local_window = min(int(cfg.local_window), int(args.block_size))
    cfg.draft_weight = float(args.draft_weight)
    cfg.use_engram = False
    cfg.memory_tokens = 0
    cfg.use_cpu_hash_context = False

    raw_model = set_trainable(raw_model, args.train_mode)
    raw_model.train()

    data, n_records = load_jsonl_messages(args.data, tok, max_records=args.max_records)
    print(f"[DATA] records={n_records} tokens={data.numel():,} path={args.data}")

    opt = build_optimizer(raw_model, args.lr, args.weight_decay, use_bnb=not args.no_bnb)

    scaler = None
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    best_loss = float("inf")
    loss_ema = None
    micro = 0
    t0 = time.time()

    opt.zero_grad(set_to_none=True)

    pbar = tqdm(range(1, args.steps + 1), desc="repair")
    for step in pbar:
        raw_model.train()

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            xb, yb = make_batch(data, args.batch_size, args.block_size, device)

            raw_model._working_mem = None
            raw_model._engram_state = None

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_autocast):
                loss = raw_model.forward_features(
                    xb,
                    yb,
                    update_state=False,
                    current_lr=args.lr,
                )

            loss_to_back = loss / args.grad_accum
            loss_to_back.backward()
            accum_loss += float(loss.detach())

            micro += 1

        if args.clip and args.clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                args.clip,
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        step_loss = accum_loss / max(1, args.grad_accum)
        loss_ema = step_loss if loss_ema is None else 0.97 * loss_ema + 0.03 * step_loss

        if step_loss < best_loss:
            best_loss = step_loss
            save_ckpt(outdir / "best.pt", raw_model, cfg, step, step_loss)

        if step % args.save_every == 0:
            save_ckpt(outdir / f"step_{step}.pt", raw_model, cfg, step, step_loss)

        if step % args.log_every == 0 or step == 1:
            diag = raw_model.diagnostics() if hasattr(raw_model, "diagnostics") else {}
            blend_obj = getattr(raw_model, "blend_gate", None)
            if blend_obj is None:
                blend = float("nan")
            elif callable(blend_obj):
                blend = float(blend_obj())
            else:
                blend = float(blend_obj)
            mem_gib = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
            elapsed = time.time() - t0

            msg = (
                f"step={step} loss={step_loss:.4f} ema={loss_ema:.4f} "
                f"best={best_loss:.4f} blend={blend:.4f} "
                f"draft_gap={diag.get('draft_ce_gap', float('nan')):.4f} "
                f"vram={mem_gib:.2f}GiB elapsed={elapsed/60:.1f}m"
            )
            print("[LOG]", msg, flush=True)
            pbar.set_postfix(loss=f"{step_loss:.3f}", ema=f"{loss_ema:.3f}", blend=f"{blend:.3f}")

        raw_model._working_mem = None
        raw_model._engram_state = None
        gc.collect()

    save_ckpt(outdir / "final.pt", raw_model, cfg, args.steps, loss_ema if loss_ema is not None else best_loss)
    print(f"[DONE] outdir={outdir}")
    print(f"[BEST] {outdir / 'best.pt'}")


if __name__ == "__main__":
    main()
