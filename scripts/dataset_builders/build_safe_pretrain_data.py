#!/usr/bin/env python3
"""
Build small uint16 pretraining bins for FF-only ternary experiments.

Default dataset is TinyStories because it is synthetic and lightweight.
This script trains a small BPE tokenizer and writes train.bin / val.bin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm


def iter_texts(dataset_name: str, split: str, config: str | None, limit: int = 0) -> Iterator[str]:
    from datasets import load_dataset

    kwargs = dict(split=split, streaming=True)
    if config:
        ds = load_dataset(dataset_name, config, **kwargs)
    else:
        ds = load_dataset(dataset_name, **kwargs)

    n = 0
    for row in ds:
        text = None
        if isinstance(row, dict):
            if isinstance(row.get("text"), str):
                text = row["text"]
            elif isinstance(row.get("content"), str):
                text = row["content"]
            elif isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str):
                text = f"User: {row['prompt']}\nAssistant: {row['response']}"
            elif isinstance(row.get("messages"), list):
                parts = []
                for m in row["messages"]:
                    role = m.get("role", "user") if isinstance(m, dict) else "user"
                    content = m.get("content", "") if isinstance(m, dict) else str(m)
                    parts.append(f"{role}: {content}")
                text = "\n".join(parts)
        if not text:
            continue
        text = " ".join(text.replace("\x00", " ").split())
        if len(text) < 40:
            continue
        yield text
        n += 1
        if limit and n >= limit:
            return


def train_tokenizer(args) -> None:
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
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        show_progress=True,
    )

    texts = iter_texts(args.dataset, args.train_split, args.config, args.tokenizer_records)
    tok.train_from_iterator(texts, trainer=trainer, length=args.tokenizer_records or None)

    bos = tok.token_to_id("<bos>")
    eos = tok.token_to_id("<eos>")
    tok.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[("<bos>", bos), ("<eos>", eos)],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.out_dir / "tokenizer.json"))


def encode_split(args, split: str, output_name: str, limit: int) -> int:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.out_dir / "tokenizer.json"))
    ids: list[int] = []
    for text in tqdm(iter_texts(args.dataset, split, args.config, limit), desc=f"encode {split}"):
        enc = tok.encode(text)
        ids.extend(enc.ids)

    if not ids:
        raise RuntimeError(f"No tokens produced for split={split}")
    max_id = max(ids)
    if max_id >= 65535:
        raise RuntimeError(f"vocab id {max_id} exceeds uint16 safe range; lower --vocab-size")

    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(args.out_dir / output_name)
    return int(arr.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--out-dir", type=Path, default=Path("data/ternary_tinystories"))
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--tokenizer-records", type=int, default=100_000)
    ap.add_argument("--train-records", type=int, default=300_000)
    ap.add_argument("--val-records", type=int, default=10_000)
    args = ap.parse_args()

    if args.vocab_size > 65535:
        raise SystemExit("Use --vocab-size <= 65535 because the trainer writes uint16 bins")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DATA] training tokenizer -> {args.out_dir / 'tokenizer.json'}")
    train_tokenizer(args)
    print("[DATA] encoding train/val bins")
    n_train = encode_split(args, args.train_split, "train.bin", args.train_records)
    n_val = encode_split(args, args.val_split, "val.bin", args.val_records)

    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "vocab_size_requested": args.vocab_size,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "dtype": "uint16",
        "tokenizer": "tokenizer.json",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
