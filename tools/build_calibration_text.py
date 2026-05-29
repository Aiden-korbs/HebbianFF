#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYSTEM = "You are a helpful assistant. Be concise, accurate, and show reasoning when useful."


def chat(user: str, assistant: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def seed_records() -> list[str]:
    rows = [
        chat("Explain why the sky looks blue in simple terms.", "Sunlight contains many colors. Air molecules scatter blue light more than red light, so more blue light reaches our eyes from across the sky."),
        chat("Write a Python function that returns the factorial of n.", "```python\ndef factorial(n: int) -> int:\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n```"),
        chat("A shop has 12 apples and sells 5. Then it receives 9 more. How many apples are there?", "There are 12 - 5 = 7 apples after the sale. After receiving 9 more, there are 7 + 9 = 16 apples."),
        chat("Summarize: The committee postponed the vote because two members were absent and the budget figures had not been verified.", "The vote was delayed due to missing members and unverified budget numbers."),
        chat("Classify the sentiment: 'The app is fast, but the login process is frustrating.'", "Mixed. The speed is positive, while the login experience is negative."),
        chat("Convert this to JSON: name Alice, role engineer, active true.", '{"name":"Alice","role":"engineer","active":true}'),
        chat("What is the difference between precision and recall?", "Precision measures how many selected items are correct. Recall measures how many correct items were selected."),
        chat("Give three safety checks before deploying a model change.", "Run automated tests, compare quality metrics on held-out examples, and monitor runtime errors and latency after release."),
    ]
    return rows


def generated_records(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    topics = [
        "photosynthesis", "database indexes", "gradient descent", "HTTP caching",
        "battery storage", "unit testing", "regular expressions", "memory allocation",
        "attention mechanisms", "file permissions", "binary search", "load balancing",
    ]
    tasks = [
        "Explain {topic} to a beginner.",
        "List two common failure modes for {topic}.",
        "Give a compact checklist for debugging {topic}.",
        "Write a short example involving {topic}.",
        "Compare {topic} with a simpler alternative.",
    ]
    records: list[str] = []
    for i in range(n):
        topic = rng.choice(topics)
        prompt = rng.choice(tasks).format(topic=topic)
        if "failure modes" in prompt:
            answer = f"Two common issues with {topic} are incorrect assumptions about inputs and poor handling of edge cases."
        elif "checklist" in prompt:
            answer = f"For {topic}: reproduce the issue, inspect inputs and outputs, isolate the smallest failing case, then verify the fix with a test."
        elif "example" in prompt:
            answer = f"Example: when using {topic}, start with a small controlled case, observe the result, then scale only after the behavior is clear."
        elif "Compare" in prompt:
            answer = f"{topic.capitalize()} is more specialized and powerful, while the simpler alternative is easier to inspect and less likely to hide mistakes."
        else:
            answer = f"{topic.capitalize()} is a concept that helps solve a specific class of problems by organizing information and reducing repeated work."
        records.append(chat(prompt, answer))
    return records


def load_extra_jsonl(path: str, limit: int) -> list[str]:
    if not path:
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit > 0 and len(out) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("prompt") or row.get("content")
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_calibration/ternary_calibration.jsonl")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--extra-jsonl", default="data_qwen_repair/repair_seed_plus_reasoning.jsonl")
    ap.add_argument("--extra-limit", type=int, default=256)
    ap.add_argument("--max-chars", type=int, default=4096)
    args = ap.parse_args()

    records = seed_records()
    records += generated_records(max(0, args.n - len(records)), args.seed)
    records += load_extra_jsonl(args.extra_jsonl, args.extra_limit)
    records = [r for r in records if len(r) <= args.max_chars]
    random.Random(args.seed + 99).shuffle(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for text in records:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    chars = sum(len(r) for r in records)
    print(f"[calibration] wrote {len(records)} records to {out}")
    print(f"[calibration] chars={chars} token_est={chars // 4}")


if __name__ == "__main__":
    main()
