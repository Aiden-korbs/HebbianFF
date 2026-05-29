#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import chat_hf


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def device_name(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return device


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"bad dtype: {name}")


def chat_prompt(tok, question: str, choices: List[str], system: str) -> Tuple[str, List[str]]:
    lines = [question.strip(), ""]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {str(c).strip()}")
    lines.append("")
    lines.append("Answer with only the correct letter.")

    user = "\n".join(lines)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    candidates = [f" {LETTERS[i]}" for i in range(len(choices))]
    return prompt, candidates


def load_samples(tasks: List[str], n_per_task: int, seed: int, tok, system: str) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    def add_sample(task: str, sid: str, question: str, choices: List[str], label: int):
        if label is None or label < 0 or label >= len(choices):
            return
        prompt, candidates = chat_prompt(tok, question, choices, system)
        samples.append({
            "task": task,
            "id": sid,
            "question": question,
            "choices": choices,
            "label": int(label),
            "prompt": prompt,
            "candidates": candidates,
        })

    for task in tasks:
        before = len(samples)

        try:
            if task == "boolq":
                ds = load_dataset("boolq", split="validation")
                rows = list(ds)
                rng.shuffle(rows)
                for i, r in enumerate(rows[:n_per_task]):
                    q = f"Passage:\n{r['passage']}\n\nQuestion: {r['question']}?"
                    choices = ["No", "Yes"]
                    label = 1 if bool(r["answer"]) else 0
                    add_sample(task, str(i), q, choices, label)

            elif task == "piqa":
                ds = load_dataset("piqa", split="validation")
                rows = list(ds)
                rng.shuffle(rows)
                for i, r in enumerate(rows[:n_per_task]):
                    q = f"Goal: {r['goal']}\n\nWhich solution is more sensible?"
                    choices = [r["sol1"], r["sol2"]]
                    add_sample(task, str(i), q, choices, int(r["label"]))

            elif task == "arc_challenge":
                ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
                rows = list(ds)
                rng.shuffle(rows)
                for i, r in enumerate(rows[:n_per_task]):
                    labels = list(r["choices"]["label"])
                    choices = list(r["choices"]["text"])
                    ans = str(r["answerKey"]).strip()
                    if ans in labels:
                        label = labels.index(ans)
                    elif ans.isdigit():
                        label = int(ans) - 1
                    else:
                        continue
                    add_sample(task, str(i), r["question"], choices, label)

            elif task == "hellaswag":
                ds = load_dataset("hellaswag", split="validation")
                rows = list(ds)
                rng.shuffle(rows)
                for i, r in enumerate(rows[:n_per_task]):
                    ctx = (r.get("ctx") or (r.get("ctx_a", "") + " " + r.get("ctx_b", ""))).strip()
                    q = f"Choose the most likely continuation.\n\nContext: {ctx}"
                    choices = list(r["endings"])
                    label = int(r["label"])
                    add_sample(task, str(i), q, choices, label)

            elif task == "winogrande":
                ds = load_dataset("winogrande", "winogrande_xl", split="validation")
                rows = list(ds)
                rng.shuffle(rows)
                for i, r in enumerate(rows[:n_per_task]):
                    sent = r["sentence"].replace("_", "[blank]")
                    q = f"Choose the option that best fills the blank.\n\nSentence: {sent}"
                    choices = [r["option1"], r["option2"]]
                    label = int(r["answer"]) - 1
                    add_sample(task, str(i), q, choices, label)

            elif task == "mmlu":
                # If this dataset name changes on your machine, the script will skip it cleanly.
                ds = load_dataset("cais/mmlu", "all", split="test")
                rows = list(ds)
                rng.shuffle(rows)
                for i, r in enumerate(rows[:n_per_task]):
                    q = str(r["question"])
                    choices = list(r["choices"])
                    label = int(r["answer"])
                    subject = r.get("subject", "unknown")
                    add_sample(task, f"{subject}:{i}", q, choices, label)

            else:
                print(f"[WARN] unknown task skipped: {task}")

        except Exception as e:
            print(f"[WARN] failed to load task={task}: {type(e).__name__}: {e}")

        added = len(samples) - before
        print(f"[DATA] {task}: {added} samples")

    return samples


def trim_to_block(tok, prompt: str, cand: str, block_size: int):
    pids = tok(prompt, add_special_tokens=False).input_ids
    cids = tok(cand, add_special_tokens=False).input_ids

    if len(cids) < 1:
        raise ValueError("candidate tokenized to empty")

    max_prompt = max(1, block_size - len(cids))
    if len(pids) > max_prompt:
        pids = pids[-max_prompt:]

    ids = pids + cids
    return pids, cids, ids


@torch.inference_mode()
def score_custom_one(raw_model, tok, sample, device, dtype, block_size: int) -> Dict[str, Any]:
    scores = []
    toks = []

    raw_model.eval()
    raw_model._working_mem = None
    raw_model._engram_state = None

    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    for cand in sample["candidates"]:
        pids, cids, ids = trim_to_block(tok, sample["prompt"], cand, block_size)

        x = torch.tensor([ids], device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            if hasattr(raw_model, "forward_features_eval"):
                x_full = raw_model.forward_features_eval(x)
            else:
                x_full, _ = raw_model.forward_features(x, update_state=False)
            logits = raw_model._get_logits(x_full).float()[0]

        # Candidate token j is predicted at position prompt_len + j - 1.
        start = len(pids) - 1
        cand_logprob = 0.0
        for j, tid in enumerate(cids):
            pos = start + j
            lp = F.log_softmax(logits[pos], dim=-1)[tid]
            cand_logprob += float(lp.item())

        scores.append(cand_logprob)
        toks.append(len(cids))

        raw_model._working_mem = None
        raw_model._engram_state = None

    pred = max(range(len(scores)), key=lambda i: scores[i])
    scores_norm = [s / max(1, t) for s, t in zip(scores, toks)]
    pred_norm = max(range(len(scores_norm)), key=lambda i: scores_norm[i])

    return {
        "scores": scores,
        "scores_norm": scores_norm,
        "pred": pred,
        "pred_norm": pred_norm,
        "correct": pred == sample["label"],
        "correct_norm": pred_norm == sample["label"],
    }


@torch.inference_mode()
def score_native_one(model, tok, sample, device, dtype, block_size: int) -> Dict[str, Any]:
    scores = []
    toks = []

    model.eval()
    autocast_enabled = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    for cand in sample["candidates"]:
        pids, cids, ids = trim_to_block(tok, sample["prompt"], cand, block_size)
        x = torch.tensor([ids], device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
            logits = model(x).logits.float()[0]

        start = len(pids) - 1
        cand_logprob = 0.0
        for j, tid in enumerate(cids):
            pos = start + j
            lp = F.log_softmax(logits[pos], dim=-1)[tid]
            cand_logprob += float(lp.item())

        scores.append(cand_logprob)
        toks.append(len(cids))

    pred = max(range(len(scores)), key=lambda i: scores[i])
    scores_norm = [s / max(1, t) for s, t in zip(scores, toks)]
    pred_norm = max(range(len(scores_norm)), key=lambda i: scores_norm[i])

    return {
        "scores": scores,
        "scores_norm": scores_norm,
        "pred": pred,
        "pred_norm": pred_norm,
        "correct": pred == sample["label"],
        "correct_norm": pred_norm == sample["label"],
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by_task: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    total = 0
    good = 0
    good_norm = 0

    for task, rs in sorted(by_task.items()):
        n = len(rs)
        c = sum(1 for r in rs if r["correct"])
        cn = sum(1 for r in rs if r["correct_norm"])
        out[task] = {
            "n": n,
            "acc": c / max(1, n),
            "acc_norm": cn / max(1, n),
        }
        total += n
        good += c
        good_norm += cn

    out["overall"] = {
        "n": total,
        "acc": good / max(1, total),
        "acc_norm": good_norm / max(1, total),
    }
    return out


def run_custom(args, samples, tok, outdir: Path):
    print("\n" + "=" * 100)
    feature_tags = ",".join(sorted({f.strip() for f in args.features.split(",") if f.strip()})) if args.features else "none"
    print(f"[LOAD] custom FF/BP model  features={feature_tags}")

    chat_args = SimpleNamespace(
        checkpoint=args.custom_checkpoint,
        tokenizer=args.tokenizer,
        prompt=None,
        system=args.system,
        device=args.device,
        dtype=args.dtype,
        block_size=args.block_size,
        max_new=1,
        temp=0.0,
        top_k=0,
        top_p=1.0,
        repeat_penalty=1.0,
        no_stop_eos=True,
        raw=True,
        history=False,
        max_turns=0,
    )

    model, raw_model, _tok2, cfg, device, dtype = chat_hf.load_model(chat_args)
    assert _tok2.get_vocab() == tok.get_vocab(), "tokenizer mismatch"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    t0 = time.time()

    for s in tqdm(samples, desc="custom"):
        r = score_custom_one(raw_model, tok, s, device, dtype, args.block_size)
        rows.append({
            "model": "custom",
            "task": s["task"],
            "id": s["id"],
            "label": s["label"],
            **r,
        })

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

    summary = summarize(rows)
    summary["_meta"] = {
        "elapsed_sec": elapsed,
        "samples_per_sec": len(samples) / max(1e-9, elapsed),
        "peak_allocated_gib": peak,
        "checkpoint": args.custom_checkpoint,
    }

    (outdir / "custom_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (outdir / "custom_summary.json").write_text(json.dumps(summary, indent=2))

    del model, raw_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def run_native(args, samples, tok, outdir: Path):
    print("\n" + "=" * 100)
    print("[LOAD] native HF Qwen")

    device = args.device
    dtype = dtype_from_name(args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.native_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    t0 = time.time()

    for s in tqdm(samples, desc="native"):
        r = score_native_one(model, tok, s, device, dtype, args.block_size)
        rows.append({
            "model": "native",
            "task": s["task"],
            "id": s["id"],
            "label": s["label"],
            **r,
        })

    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

    summary = summarize(rows)
    summary["_meta"] = {
        "elapsed_sec": elapsed,
        "samples_per_sec": len(samples) / max(1e-9, elapsed),
        "peak_allocated_gib": peak,
        "native_model": args.native_model,
    }

    (outdir / "native_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (outdir / "native_summary.json").write_text(json.dumps(summary, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def print_comparison(custom_summary, native_summary):
    tasks = sorted(k for k in custom_summary.keys() if not k.startswith("_") and k in native_summary)

    print("\n" + "=" * 100)
    print("COMPARISON")
    print("=" * 100)
    print(f"{'task':<18} {'n':>6} {'custom':>10} {'native':>10} {'delta':>10} {'custom_norm':>13} {'native_norm':>13} {'delta_norm':>12}")
    print("-" * 100)

    for task in tasks:
        c = custom_summary[task]
        n = native_summary[task]
        print(
            f"{task:<18} {c['n']:>6} "
            f"{100*c['acc']:>9.2f}% {100*n['acc']:>9.2f}% {100*(c['acc']-n['acc']):>+9.2f}% "
            f"{100*c['acc_norm']:>12.2f}% {100*n['acc_norm']:>12.2f}% {100*(c['acc_norm']-n['acc_norm']):>+11.2f}%"
        )

    print("-" * 100)
    c = custom_summary["overall"]
    n = native_summary["overall"]
    print(
        f"{'overall':<18} {c['n']:>6} "
        f"{100*c['acc']:>9.2f}% {100*n['acc']:>9.2f}% {100*(c['acc']-n['acc']):>+9.2f}% "
        f"{100*c['acc_norm']:>12.2f}% {100*n['acc_norm']:>12.2f}% {100*(c['acc_norm']-n['acc_norm']):>+11.2f}%"
    )

    print("\nRuntime:")
    print(f"  custom: {custom_summary['_meta']['elapsed_sec']:.1f}s, "
          f"{custom_summary['_meta']['samples_per_sec']:.2f} samples/s, "
          f"peak={custom_summary['_meta']['peak_allocated_gib']:.2f} GiB")
    print(f"  native: {native_summary['_meta']['elapsed_sec']:.1f}s, "
          f"{native_summary['_meta']['samples_per_sec']:.2f} samples/s, "
          f"peak={native_summary['_meta']['peak_allocated_gib']:.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom-checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--native-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--skip-native", action="store_true", help="Only evaluate --custom-checkpoint; do not load the native HF model.")
    ap.add_argument("--tasks", default="boolq,piqa,arc_challenge,hellaswag,winogrande,mmlu")
    ap.add_argument("--n-per-task", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="eval_runs/qwen15_compare")
    ap.add_argument("--system", default="You are a helpful assistant. Choose the correct answer.")
    ap.add_argument("--features", default="",
                    help="Comma-separated features: draft,memory,engram,cpuctx.  Applies env vars before load.")
    args = ap.parse_args()

    # Apply feature toggles via env so chat_hf.load_model picks them up.
    features = {f.strip() for f in args.features.split(",") if f.strip()}
    os.environ.setdefault("USE_DRAFT_HEAD", "1" if "draft" in features else "0")
    os.environ.setdefault("DRAFT_BLEND_BP", "1" if "draft" in features else "0")
    os.environ.setdefault("INFER_MEMORY_TOKENS", "64" if "memory" in features else "0")
    os.environ.setdefault("INFER_USE_ENGRAM", "1" if "engram" in features else "0")
    os.environ.setdefault("CPU_HASH_CTX", "1" if "cpuctx" in features else "0")

    feature_tags = ",".join(sorted(features)) if features else "none"

    outdir = Path(args.outdir) / time.strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] device:", args.device, device_name(args.device))
    print("[INFO] dtype:", args.dtype)
    print("[INFO] features:", feature_tags)
    print("[INFO] outdir:", outdir)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    samples = load_samples(tasks, args.n_per_task, args.seed, tok, args.system)

    if not samples:
        raise SystemExit("No samples loaded. Check dataset downloads / internet / task names.")

    (outdir / "samples.jsonl").write_text(
        "\n".join(json.dumps({
            "task": s["task"],
            "id": s["id"],
            "label": s["label"],
            "question": s["question"],
            "choices": s["choices"],
        }) for s in samples) + "\n"
    )

    custom_summary = run_custom(args, samples, tok, outdir)
    native_summary = None if args.skip_native else run_native(args, samples, tok, outdir)

    combined = {
        "custom": custom_summary,
        "native": native_summary,
    }
    (outdir / "comparison_summary.json").write_text(json.dumps(combined, indent=2))

    if native_summary is None:
        print("\n" + "=" * 100)
        print("CUSTOM-ONLY SUMMARY")
        print("=" * 100)
        for task in sorted(k for k in custom_summary.keys() if not k.startswith("_")):
            item = custom_summary[task]
            print(f"{task:<18} n={item['n']:>6} acc={100*item['acc']:>6.2f}% acc_norm={100*item['acc_norm']:>6.2f}%")
        meta = custom_summary["_meta"]
        print(
            f"\nRuntime: {meta['elapsed_sec']:.1f}s, "
            f"{meta['samples_per_sec']:.2f} samples/s, "
            f"peak={meta['peak_allocated_gib']:.2f} GiB"
        )
    else:
        print_comparison(custom_summary, native_summary)
    print("\n[FILES]")
    print(" ", outdir / "comparison_summary.json")
    print(" ", outdir / "custom_results.jsonl")
    if native_summary is not None:
        print(" ", outdir / "native_results.jsonl")


if __name__ == "__main__":
    main()
