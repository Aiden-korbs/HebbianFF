#!/usr/bin/env python3
"""
Build large clean uint16 pretraining bins for FF/BP ternary runs.

The default source is FineWeb-Edu sample-10BT.  The script streams from
Hugging Face datasets, applies conservative text-quality filters, trains a
ByteLevel BPE tokenizer, and writes train.bin / val.bin in the format expected
by train_ff_only_ternary_ema.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MANY_NEWLINES_RE = re.compile(r"\n{3,}")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


@dataclass
class CleanStats:
    seen: int = 0
    kept: int = 0
    duplicate: int = 0
    too_short: int = 0
    too_long: int = 0
    low_alpha: int = 0
    high_digit: int = 0
    high_symbol: int = 0
    too_many_urls: int = 0
    bad_repetition: int = 0
    no_text: int = 0


def stable_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=16).hexdigest()


def extract_text(row: object) -> str | None:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        return None

    for key in ("text", "content", "document", "raw_content", "story"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value

    if isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
        return f"User: {row['prompt']}\nAssistant: {row['response']}"

    messages = row.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content}")
        if parts:
            return "\n".join(parts)
    return None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return MANY_NEWLINES_RE.sub("\n\n", "\n".join(lines)).strip()


def repetition_bad(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 40:
        return False
    window = words[: min(len(words), 400)]
    unique_frac = len(set(window)) / max(1, len(window))
    if unique_frac < 0.28:
        return True
    for n in (3, 4, 5):
        grams = [" ".join(window[i : i + n]) for i in range(max(0, len(window) - n + 1))]
        if grams and (len(grams) - len(set(grams))) / len(grams) > 0.35:
            return True
    return False


def quality_filter(text: str, args: argparse.Namespace, stats: CleanStats) -> str | None:
    text = normalize_text(text)
    n_chars = len(text)
    if n_chars < args.min_chars:
        stats.too_short += 1
        return None
    if args.max_chars and n_chars > args.max_chars:
        stats.too_long += 1
        return None

    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        stats.no_text += 1
        return None
    alpha_frac = sum(ch.isalpha() for ch in visible) / len(visible)
    digit_frac = sum(ch.isdigit() for ch in visible) / len(visible)
    symbol_frac = sum((not ch.isalnum()) for ch in visible) / len(visible)

    if alpha_frac < args.min_alpha_frac:
        stats.low_alpha += 1
        return None
    if digit_frac > args.max_digit_frac:
        stats.high_digit += 1
        return None
    if symbol_frac > args.max_symbol_frac:
        stats.high_symbol += 1
        return None
    if len(URL_RE.findall(text)) > args.max_urls:
        stats.too_many_urls += 1
        return None
    if repetition_bad(text):
        stats.bad_repetition += 1
        return None
    return text


def load_stream(args: argparse.Namespace):
    from datasets import load_dataset

    kwargs = {"split": args.split, "streaming": True}
    if args.config:
        return load_dataset(args.dataset, args.config, **kwargs)
    return load_dataset(args.dataset, **kwargs)


def iter_clean_texts(args: argparse.Namespace, stats: CleanStats, *, limit_docs: int = 0) -> Iterator[str]:
    seen_hashes: set[str] = set()
    yielded = 0
    for row in load_stream(args):
        stats.seen += 1
        raw = extract_text(row)
        if not raw:
            stats.no_text += 1
            continue
        text = quality_filter(raw, args, stats)
        if text is None:
            continue
        digest = stable_hash(text)
        if digest in seen_hashes:
            stats.duplicate += 1
            continue
        seen_hashes.add(digest)
        stats.kept += 1
        yielded += 1
        yield text
        if limit_docs and yielded >= limit_docs:
            return


def train_tokenizer(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC, Sequence
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = Sequence([NFKC()])
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.tokenizer_min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    stats = CleanStats()
    texts = iter_clean_texts(args, stats, limit_docs=args.tokenizer_docs)
    tok.train_from_iterator(texts, trainer=trainer, length=args.tokenizer_docs or None)

    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    if bos is None or eos is None:
        raise RuntimeError("Tokenizer did not create <bos>/<eos> ids")
    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos), ("<eos>", eos)],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.out_dir / "tokenizer.json"))
    (args.out_dir / "tokenizer_clean_stats.json").write_text(json.dumps(asdict(stats), indent=2))


def append_ids(handle, ids: list[int], buffer: list[int], flush_tokens: int) -> int:
    buffer.extend(ids)
    written = 0
    if len(buffer) >= flush_tokens:
        arr = np.asarray(buffer, dtype=np.uint16)
        arr.tofile(handle)
        written = int(arr.size)
        buffer.clear()
    return written


def flush_ids(handle, buffer: list[int]) -> int:
    if not buffer:
        return 0
    arr = np.asarray(buffer, dtype=np.uint16)
    arr.tofile(handle)
    n = int(arr.size)
    buffer.clear()
    return n


def encode_bins(args: argparse.Namespace) -> tuple[int, int, CleanStats]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.out_dir / "tokenizer.json"))
    if tok.get_vocab_size() > 65535:
        raise RuntimeError("Tokenizer vocab exceeds uint16 storage; use --vocab-size <= 65535")

    stats = CleanStats()
    train_tokens = 0
    val_tokens = 0
    train_buffer: list[int] = []
    val_buffer: list[int] = []
    train_path = args.out_dir / "train.bin"
    val_path = args.out_dir / "val.bin"
    tmp_train = args.out_dir / "train.bin.tmp"
    tmp_val = args.out_dir / "val.bin.tmp"

    target_train = int(args.target_train_tokens)
    target_val = int(args.target_val_tokens)
    if target_train <= 0 or target_val <= 0:
        raise RuntimeError("--target-train-tokens and --target-val-tokens must be positive")

    pbar_total = target_train + target_val
    with tmp_train.open("wb") as train_f, tmp_val.open("wb") as val_f:
        pbar = tqdm(total=pbar_total, unit="tok", desc="encode")
        for text in iter_clean_texts(args, stats):
            ids = tok.encode(text).ids
            if not ids:
                continue
            if max(ids) >= 65535:
                raise RuntimeError("Token id exceeds uint16 storage")

            digest = stable_hash(text)
            as_val = (int(digest[:8], 16) / 0xFFFFFFFF) < args.val_fraction
            if val_tokens < target_val and (as_val or train_tokens >= target_train):
                before = val_tokens
                val_tokens += append_ids(val_f, ids, val_buffer, args.flush_tokens)
                pbar.update(val_tokens - before)
            elif train_tokens < target_train:
                before = train_tokens
                train_tokens += append_ids(train_f, ids, train_buffer, args.flush_tokens)
                pbar.update(train_tokens - before)

            if train_tokens >= target_train and val_tokens >= target_val:
                break

        before = train_tokens
        train_tokens += flush_ids(train_f, train_buffer)
        pbar.update(train_tokens - before)
        before = val_tokens
        val_tokens += flush_ids(val_f, val_buffer)
        pbar.update(val_tokens - before)
        pbar.close()

    if train_tokens < target_train:
        raise RuntimeError(f"Only wrote {train_tokens:,} train tokens; target was {target_train:,}")
    if val_tokens < target_val:
        raise RuntimeError(f"Only wrote {val_tokens:,} val tokens; target was {target_val:,}")

    tmp_train.replace(train_path)
    tmp_val.replace(val_path)
    return train_tokens, val_tokens, stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-dir", type=Path, default=Path("data/fineweb_edu_10bt_bpe32k"))
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--tokenizer-docs", type=int, default=500_000)
    ap.add_argument("--tokenizer-min-frequency", type=int, default=2)
    ap.add_argument("--target-train-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--target-val-tokens", type=int, default=10_000_000)
    ap.add_argument("--val-fraction", type=float, default=0.002)
    ap.add_argument("--flush-tokens", type=int, default=2_000_000)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=80_000)
    ap.add_argument("--min-alpha-frac", type=float, default=0.55)
    ap.add_argument("--max-digit-frac", type=float, default=0.20)
    ap.add_argument("--max-symbol-frac", type=float, default=0.35)
    ap.add_argument("--max-urls", type=int, default=12)
    ap.add_argument("--skip-tokenizer", action="store_true")
    ap.add_argument("--dry-run-docs", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.vocab_size > 65535:
        raise SystemExit("Use --vocab-size <= 65535 because this trainer reads uint16 bins")
    if not (0.0 < args.val_fraction < 1.0):
        raise SystemExit("--val-fraction must be between 0 and 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run_docs:
        stats = CleanStats()
        token_est = 0
        char_count = 0
        for text in tqdm(iter_clean_texts(args, stats, limit_docs=args.dry_run_docs), total=args.dry_run_docs, desc="dry-run"):
            char_count += len(text)
            token_est += max(1, math.ceil(len(text) / 4))
        print(json.dumps({"clean_stats": asdict(stats), "chars": char_count, "rough_tokens": token_est}, indent=2))
        return

    if not args.skip_tokenizer:
        print(f"[DATA] training tokenizer: {args.out_dir / 'tokenizer.json'}")
        train_tokenizer(args)
    elif not (args.out_dir / "tokenizer.json").exists():
        raise SystemExit("--skip-tokenizer requires an existing tokenizer.json in --out-dir")

    print(f"[DATA] encoding bins: {args.out_dir}")
    train_tokens, val_tokens, encode_stats = encode_bins(args)
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "vocab_size_requested": args.vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "dtype": "uint16",
        "tokenizer": "tokenizer.json",
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_alpha_frac": args.min_alpha_frac,
            "max_digit_frac": args.max_digit_frac,
            "max_symbol_frac": args.max_symbol_frac,
            "max_urls": args.max_urls,
            "dedup": "exact blake2b-128 per pass",
        },
        "clean_stats": asdict(encode_stats),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
    else:
        # On this workstation, Hugging Face streaming can abort in a native
        # shutdown path after all Python work has completed.  Flush explicitly
        # and bypass interpreter teardown so successful long builds report a
        # clean exit code.
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
