#!/usr/bin/env python3
"""
scripts/inference/web_chat.py  —  Browser front-end for FF_LLM / chat_hf models.

Defaults follow the "Current Best Path" in architecture_plan.md:
  USE_KV_CACHE=1, KV_CACHE_INT8=0, INFER_MEMORY_TOKENS=0,
  INFER_USE_ENGRAM=0, USE_DRAFT_HEAD=0, DRAFT_BLEND_BP=0

Usage:
  pip install flask
  python scripts/inference/web_chat.py \\
      --checkpoint models/Qwen2.5-Coder-0.5B-Instruct.pt \\
      --tokenizer  Qwen/Qwen2.5-Coder-0.5B-Instruct

Share over Tailscale:
  Point friends at  http://<your-tailscale-ip>:7860
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

# ── Stable inference env defaults (architecture_plan.md §"Current Best Path") ─
os.environ.setdefault("USE_KV_CACHE",        "1")
os.environ.setdefault("INFER_MEMORY_TOKENS", "0")
os.environ.setdefault("INFER_USE_ENGRAM",    "0")
os.environ.setdefault("USE_DRAFT_HEAD",      "0")
os.environ.setdefault("DRAFT_BLEND_BP",      "0")
os.environ.setdefault("KV_CACHE_INT8",       "0")
# KV_CACHE_MAX_LEN / KV_CACHE_SINK_TOKENS are set after arg parsing (see main())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from flask import Flask, Response, jsonify, request, session
except ImportError:
    sys.exit("[web_chat] Flask is required:  pip install flask")

import torch

from chat_hf import (
    autocast_context,       # noqa: F401 (used indirectly via generate_ids)
    build_prompt,
    decode_generated,
    ensure_runtime_arg_defaults,
    forbidden_token_ids,
    generate_ids,
    load_model,
    stop_token_ids,
    tokenise_prompt,
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Per-session chat history  {session_id: [(user, assistant), ...]}
_histories: Dict[str, List[Tuple[str, str]]] = {}
_histories_lock = threading.Lock()

# Serialize generation to the single GPU
_gen_lock = threading.Lock()

# Globals filled by main()
_model = _raw_model = _tok = _cfg = _device = _dtype = None
_server_args = None   # argparse Namespace used as default generation config


def _split_thinking_sections(raw: str) -> tuple[str, str, bool]:
    """Split Qwen-style reasoning text into (thinking, answer, open_think).

    Handles both normal <think>...</think> blocks and orphan </think> output.
    Some chat templates suppress the opening tag but the model can still emit the
    closing tag, leaving the reasoning text at the start of the decoded output.
    In that case, treat everything before the first orphan close tag as thinking.
    """
    text = raw or ""
    lower = text.lower()
    open_tag = "<think>"
    close_tag = "</think>"

    first_open = lower.find(open_tag)
    first_close = lower.find(close_tag)
    if first_close >= 0 and (first_open < 0 or first_close < first_open):
        thinking = text[:first_close].strip()
        answer = text[first_close + len(close_tag):].strip()
        return thinking, answer, False

    pos = 0
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    open_think = False

    while pos < len(text):
        start = lower.find(open_tag, pos)
        if start < 0:
            answer_parts.append(text[pos:])
            break

        answer_parts.append(text[pos:start])
        inner_start = start + len(open_tag)
        end = lower.find(close_tag, inner_start)
        if end < 0:
            thinking_parts.append(text[inner_start:])
            open_think = True
            break

        thinking_parts.append(text[inner_start:end])
        pos = end + len(close_tag)

    thinking = "\n\n".join(part.strip() for part in thinking_parts if part.strip()).strip()
    answer = "".join(answer_parts).replace(open_tag, "").replace(close_tag, "").strip()
    return thinking, answer, open_think


def _count_tokens_for_display(text: str) -> int:
    """Best-effort tokenizer count for UI metadata; falls back safely."""
    if not text:
        return 0
    try:
        if hasattr(_tok, "encode"):
            return len(_tok.encode(text, add_special_tokens=False))
    except TypeError:
        try:
            return len(_tok.encode(text))
        except Exception:
            pass
    except Exception:
        pass
    try:
        return len(tokenise_prompt(_tok, text))
    except Exception:
        return len(text.split())


# ── Core streaming generator ──────────────────────────────────────────────────

def _sse_reply(session_id: str, user_msg: str, req_overrides: dict) -> Iterator[str]:
    """Yield SSE-formatted lines. Sends delta chunks then a final [DONE] frame."""

    history = _histories.get(session_id, [])

    # Merge server defaults with per-request overrides from the UI settings panel
    la = copy.copy(_server_args)
    la.temp            = float(req_overrides.get("temp",           la.temp))
    la.top_k           = int(  req_overrides.get("top_k",          la.top_k))
    la.top_p           = float(req_overrides.get("top_p",          la.top_p))
    la.repeat_penalty  = float(req_overrides.get("repeat_penalty", la.repeat_penalty))
    la.max_new         = int(  req_overrides.get("max_new",        la.max_new))
    la.system          = str(  req_overrides.get("system",         la.system))
    la.stream_every    = int(  req_overrides.get("stream_every",   getattr(la, "stream_every", 1)))

    # Qwen-style reasoning support: preserve literal <think>...</think>
    # sections when requested so the browser can render them separately.
    show_thinking      = bool(req_overrides.get("show_thinking", not bool(getattr(la, "hide_reasoning", False))))
    la.hide_reasoning  = not show_thinking

    la.stream          = False   # we handle streaming via the queue below
    ensure_runtime_arg_defaults(la)

    if bool(getattr(la, "raw", False)):
        if history:
            prior = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
            full_prompt = f"{prior}\nUser: {user_msg}\nAssistant:"
        else:
            full_prompt = user_msg
    else:
        full_prompt = build_prompt(_tok, la.system, user_msg, history, la)
    ids = tokenise_prompt(_tok, full_prompt)
    if len(ids) > _cfg.block_size:
        ids = ids[-int(_cfg.block_size):]

    idx         = torch.tensor([ids], dtype=torch.long, device=_device)
    stop_ids    = stop_token_ids(_tok, la)
    forbidden   = forbidden_token_ids(_tok, la)
    prompt_len  = len(ids)

    delta_q:    queue.Queue[str | None] = queue.Queue()
    stream_every = max(1, int(getattr(la, "stream_every", 1)))
    state       = {"printed_text": "", "callbacks": 0}
    full_chunks: list[str] = []

    def _callback(out_ids):
        state["callbacks"] += 1
        if state["callbacks"] % stream_every != 0:
            return
        text  = decode_generated(_tok, out_ids[prompt_len:], la)
        prev  = state["printed_text"]
        delta = text[len(prev):] if text.startswith(prev) else ("\n" + text)
        if delta:
            state["printed_text"] = text
            full_chunks.append(delta)
            delta_q.put(delta)

    def _worker():
        try:
            with _gen_lock:
                out = generate_ids(
                    _model, _raw_model, idx, _cfg, _device, _dtype,
                    la, stop_ids, forbidden,
                    stream_callback=_callback,
                )
                final_text = decode_generated(_tok, out[0].tolist()[prompt_len:], la)
                prev = state["printed_text"]
                delta = final_text[len(prev):] if final_text.startswith(prev) else ("\n" + final_text)
                if delta:
                    state["printed_text"] = final_text
                    full_chunks.append(delta)
                    delta_q.put(delta)
        except Exception as exc:
            delta_q.put(f"\n\n**[generation error: {exc}]**")
        finally:
            delta_q.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    # Stream deltas
    while True:
        delta = delta_q.get()
        if delta is None:
            break
        yield f"data: {json.dumps({'delta': delta})}\n\n"

    # Persist only the clean answer to history; keep thinking visible in the UI
    # without feeding long reasoning traces back into later prompts.
    raw_reply = "".join(full_chunks).strip()
    thinking_text, answer_text, open_think = _split_thinking_sections(raw_reply)
    final_reply = answer_text or raw_reply
    thinking_tokens = _count_tokens_for_display(thinking_text)
    stats = getattr(_raw_model, "_last_gen_stats", {}) or {}
    elapsed_sec = float(stats.get("elapsed_sec", 0.0) or 0.0)
    generated_tokens = int(stats.get("generated_tokens", 0) or 0)
    tok_s = generated_tokens / max(1e-9, elapsed_sec) if generated_tokens > 0 else 0.0

    if final_reply:
        with _histories_lock:
            h = _histories.setdefault(session_id, [])
            h.append((user_msg, final_reply))
            _histories[session_id] = h[-int(_server_args.max_turns):]

    payload = {
        "done": True,
        "reply": final_reply,
        "raw_reply": raw_reply,
        "thinking": thinking_text,
        "thinking_tokens": thinking_tokens,
        "thinking_open": open_think,
        "elapsed_sec": elapsed_sec,
        "generated_tokens": generated_tokens,
        "tok_s": tok_s,
    }
    yield f"data: {json.dumps(payload)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return _HTML_PAGE


@app.route("/chat", methods=["POST"])
def chat():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    data     = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    return Response(
        _sse_reply(session["sid"], user_msg, data),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("sid")
    if sid:
        with _histories_lock:
            _histories.pop(sid, None)
    return jsonify({"ok": True})


@app.route("/info")
def info():
    return jsonify({
        "device":     str(_device),
        "dtype":      str(_dtype).replace("torch.", ""),
        "block_size": int(_cfg.block_size),
        "kv_max":     int(_cfg.kv_cache_max_len or _cfg.block_size),
        "kv_sink":    int(_cfg.kv_cache_sink_tokens),
        "kv_int8":    bool(_cfg.kv_cache_int8),
        "params_m":   round(sum(p.numel() for p in _raw_model.parameters()) / 1e6, 1),
    })


@app.route("/defaults")
def defaults():
    return jsonify({
        "system":         _server_args.system,
        "temp":           _server_args.temp,
        "top_k":          _server_args.top_k,
        "top_p":          _server_args.top_p,
        "repeat_penalty": _server_args.repeat_penalty,
        "max_new":        _server_args.max_new,
        "show_thinking":  not bool(getattr(_server_args, "hide_reasoning", False)),
    })


# ── Embedded UI ───────────────────────────────────────────────────────────────

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FF_LLM Chat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    color-scheme: dark;
    --bg: #070910;
    --bg-2: #0b1020;
    --surface: rgba(15, 20, 34, 0.82);
    --surface-strong: rgba(20, 27, 44, 0.96);
    --surface-soft: rgba(255, 255, 255, 0.045);
    --surface-hover: rgba(255, 255, 255, 0.075);
    --border: rgba(148, 163, 184, 0.18);
    --border-strong: rgba(148, 163, 184, 0.32);
    --text: #e5e7eb;
    --text-soft: #cbd5e1;
    --text-dim: #94a3b8;
    --muted: #64748b;
    --accent: #f4b63f;
    --accent-2: #8b5cf6;
    --accent-3: #38bdf8;
    --good: #34d399;
    --danger: #fb7185;
    --code-bg: rgba(2, 6, 23, 0.82);
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
    --shadow-soft: 0 16px 50px rgba(0, 0, 0, 0.24);
    --radius-xl: 26px;
    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-sm: 9px;
  }

  *, *::before, *::after { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }

  body {
    min-height: 100%;
    overflow: hidden;
    color: var(--text);
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background:
      radial-gradient(circle at 14% 12%, rgba(139, 92, 246, 0.23), transparent 30%),
      radial-gradient(circle at 86% 10%, rgba(56, 189, 248, 0.14), transparent 32%),
      radial-gradient(circle at 50% 100%, rgba(244, 182, 63, 0.12), transparent 35%),
      linear-gradient(135deg, var(--bg), var(--bg-2));
  }

  button, textarea, input { font: inherit; }
  button { -webkit-tap-highlight-color: transparent; }

  #app {
    height: 100vh;
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    padding: 18px;
    gap: 18px;
  }

  .sidebar {
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    background: linear-gradient(180deg, rgba(15, 20, 34, 0.92), rgba(15, 20, 34, 0.68));
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(18px);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
  }
  .brand-mark {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    color: #0f172a;
    font-weight: 900;
    letter-spacing: -0.06em;
    background: linear-gradient(135deg, var(--accent), #fde68a);
    box-shadow: 0 12px 34px rgba(244, 182, 63, 0.22);
  }
  .brand h1 {
    margin: 0;
    font-size: 18px;
    line-height: 1.05;
    letter-spacing: -0.04em;
  }
  .brand p {
    margin: 4px 0 0;
    color: var(--text-dim);
    font-size: 12px;
  }

  .status-card {
    display: grid;
    gap: 10px;
    padding: 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: rgba(255, 255, 255, 0.045);
  }
  .status-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    color: var(--text-soft);
    font-weight: 700;
    font-size: 13px;
  }
  .live-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--good);
    border: 1px solid rgba(52, 211, 153, 0.26);
    background: rgba(52, 211, 153, 0.08);
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
  }
  .live-dot {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--good);
    box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.12);
  }
  .metric-grid {
    display: grid;
    gap: 8px;
  }
  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    color: var(--text-dim);
    font-size: 12px;
  }
  .metric strong {
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--text);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 11px;
    font-weight: 700;
  }

  .actions {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 40px;
    padding: 9px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    color: var(--text-soft);
    background: rgba(255, 255, 255, 0.045);
    cursor: pointer;
    transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease, color 0.16s ease;
    font-size: 13px;
    font-weight: 700;
  }
  .btn:hover {
    transform: translateY(-1px);
    border-color: var(--border-strong);
    color: var(--text);
    background: var(--surface-hover);
  }
  .btn.primary {
    color: #0f172a;
    border-color: rgba(244, 182, 63, 0.28);
    background: linear-gradient(135deg, var(--accent), #fde68a);
    box-shadow: 0 12px 30px rgba(244, 182, 63, 0.18);
  }
  .btn.primary:hover { color: #0f172a; }
  .btn:disabled { opacity: 0.45; cursor: default; transform: none; }

  .hint-card {
    margin-top: auto;
    padding: 14px;
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: var(--radius-lg);
    color: var(--text-dim);
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(139, 92, 246, 0.07));
    font-size: 12px;
    line-height: 1.55;
  }
  .hint-card strong { color: var(--text-soft); }
  .kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    padding: 1px 6px;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-soft);
    background: rgba(2, 6, 23, 0.38);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 11px;
  }

  .main {
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    overflow: hidden;
    background: rgba(9, 13, 24, 0.68);
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
  }

  #header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
    background: rgba(15, 20, 34, 0.7);
  }
  .header-title { min-width: 0; }
  .header-title h2 {
    margin: 0;
    color: var(--text);
    font-size: 16px;
    letter-spacing: -0.02em;
  }
  .header-title p {
    margin: 3px 0 0;
    color: var(--text-dim);
    font-size: 12px;
  }
  #model-chips {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    flex-wrap: wrap;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 250px;
    min-height: 28px;
    padding: 5px 9px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--text-dim);
    background: rgba(255, 255, 255, 0.04);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 11px;
  }
  .chip.good {
    color: var(--good);
    border-color: rgba(52, 211, 153, 0.23);
    background: rgba(52, 211, 153, 0.07);
  }

  #messages {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 22px;
    scroll-behavior: smooth;
  }
  #messages::-webkit-scrollbar { width: 10px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.18);
    border: 3px solid transparent;
    background-clip: content-box;
    border-radius: 999px;
  }

  #welcome {
    min-height: 100%;
    display: grid;
    place-items: center;
    padding: 28px;
  }
  .welcome-card {
    width: min(560px, 100%);
    padding: 34px;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 32px;
    background:
      radial-gradient(circle at top left, rgba(244, 182, 63, 0.13), transparent 35%),
      rgba(255, 255, 255, 0.045);
  }
  .welcome-icon {
    width: 64px;
    height: 64px;
    margin: 0 auto 16px;
    display: grid;
    place-items: center;
    border-radius: 22px;
    color: #0f172a;
    font-size: 26px;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent), #fde68a);
    box-shadow: 0 20px 42px rgba(244, 182, 63, 0.18);
  }
  .welcome-card h2 {
    margin: 0;
    font-size: clamp(28px, 4vw, 44px);
    letter-spacing: -0.07em;
    line-height: 1;
  }
  .welcome-card p {
    width: min(430px, 100%);
    margin: 14px auto 0;
    color: var(--text-dim);
    font-size: 14px;
    line-height: 1.65;
  }
  .welcome-prompts {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 24px;
  }
  .prompt-card {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px;
    color: var(--text-soft);
    background: rgba(2, 6, 23, 0.24);
    font-size: 12px;
    text-align: left;
  }
  #welcome.hidden { display: none; }

  .msg-row {
    display: flex;
    gap: 12px;
    width: 100%;
    margin: 0 0 18px;
    animation: fadeIn 0.2s ease both;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: none; }
  }
  .msg-row.user { flex-direction: row-reverse; }
  .msg-row.system-note { justify-content: center; }
  .avatar {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    margin-top: 22px;
    border-radius: 13px;
    color: var(--text);
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.055);
    font-size: 13px;
    font-weight: 900;
  }
  .msg-row.user .avatar {
    color: #0f172a;
    border-color: rgba(56, 189, 248, 0.34);
    background: linear-gradient(135deg, #38bdf8, #bae6fd);
  }
  .msg-row.ai .avatar {
    color: #0f172a;
    border-color: rgba(244, 182, 63, 0.34);
    background: linear-gradient(135deg, var(--accent), #fde68a);
  }
  .msg-stack {
    max-width: min(820px, 82%);
    min-width: 0;
    display: grid;
    gap: 6px;
  }
  .msg-row.user .msg-stack { justify-items: end; }
  .msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
  }
  .copy-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 24px;
    padding: 2px 8px;
    border: 1px solid transparent;
    border-radius: 999px;
    color: var(--muted);
    background: transparent;
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
  }
  .copy-btn:hover {
    color: var(--text-soft);
    border-color: var(--border);
    background: rgba(255,255,255,0.045);
  }
  .bubble {
    min-width: 0;
    padding: 13px 15px;
    border: 1px solid var(--border);
    border-radius: 18px;
    color: var(--text-soft);
    background: rgba(255, 255, 255, 0.045);
    line-height: 1.7;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .msg-row.ai .bubble {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.035));
  }
  .msg-row.user .bubble {
    color: #e0f2fe;
    border-color: rgba(56, 189, 248, 0.22);
    background: linear-gradient(135deg, rgba(14, 116, 144, 0.22), rgba(30, 64, 175, 0.18));
  }
  .system-note .bubble {
    max-width: 520px;
    color: var(--text-dim);
    text-align: center;
    font-size: 13px;
    background: rgba(244, 182, 63, 0.07);
    border-color: rgba(244, 182, 63, 0.18);
  }

  .thinking-panel {
    width: fit-content;
    max-width: 100%;
    overflow: hidden;
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: 999px;
    background: rgba(139, 92, 246, 0.08);
    transition: border-radius .16s ease, background .16s ease, border-color .16s ease;
  }
  .thinking-panel.hidden { display: none; }
  .thinking-panel[open] {
    width: 100%;
    border-radius: 16px;
    border-color: rgba(139, 92, 246, 0.26);
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.10), rgba(56, 189, 248, 0.055));
  }
  .thinking-panel summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 7px 10px;
    color: #d8b4fe;
    cursor: pointer;
    user-select: none;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.02em;
    list-style: none;
  }
  .thinking-panel[open] summary {
    padding: 10px 12px;
  }
  .thinking-panel summary::-webkit-details-marker { display: none; }
  .thinking-title {
    display: inline-flex;
    align-items: center;
    gap: 7px;
  }
  .thinking-title::before {
    content: "✦";
    display: inline-grid;
    place-items: center;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    color: #f5d0fe;
    background: rgba(139, 92, 246, 0.20);
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.08);
    font-size: 11px;
    line-height: 1;
  }
  .thinking-title::after {
    content: "click to expand";
    color: var(--text-dim);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0;
  }
  .thinking-panel[open] .thinking-title::after {
    content: "click to collapse";
  }
  .thinking-count {
    color: var(--text-dim);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 10px;
    font-weight: 700;
  }
  .thinking-body {
    display: none;
    padding: 0 12px 12px;
    color: #c4b5fd;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 12px;
    line-height: 1.65;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    opacity: 0.92;
  }
  .thinking-panel[open] .thinking-body { display: block; }
  .stream-placeholder {
    color: var(--muted);
    font-style: italic;
  }
  .checkbox-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: 13px;
    background: rgba(2, 6, 23, 0.26);
  }
  .checkbox-row input {
    margin-top: 3px;
    width: 16px;
    height: 16px;
    accent-color: var(--accent);
  }
  .checkbox-row strong {
    display: block;
    color: var(--text-soft);
    font-size: 13px;
  }
  .checkbox-row span {
    display: block;
    margin-top: 2px;
    color: var(--text-dim);
    font-size: 12px;
    line-height: 1.45;
  }

  .bubble code.inline {
    padding: 2px 6px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 7px;
    color: #bfdbfe;
    background: rgba(2, 6, 23, 0.54);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 0.92em;
  }
  .bubble pre {
    margin: 10px 0;
    padding: 13px 14px;
    overflow: auto;
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    color: #dbeafe;
    background: var(--code-bg);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
  }
  .bubble pre code {
    padding: 0;
    border: 0;
    background: transparent;
    color: inherit;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 12.5px;
    white-space: pre;
  }
  .code-lang {
    display: block;
    margin-bottom: 8px;
    color: var(--accent);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .bubble strong { color: #fff7ed; }
  .bubble em { color: #c4b5fd; }

  .cursor {
    display: inline-block;
    width: 7px;
    height: 1.08em;
    margin-left: 2px;
    border-radius: 2px;
    vertical-align: text-bottom;
    background: var(--accent);
    animation: blink 0.8s step-end infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }

  #input-bar {
    padding: 16px;
    border-top: 1px solid var(--border);
    background: rgba(15, 20, 34, 0.72);
  }
  .composer {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: end;
    gap: 12px;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 22px;
    background: rgba(2, 6, 23, 0.34);
  }
  #user-input {
    width: 100%;
    min-height: 44px;
    max-height: 180px;
    resize: none;
    overflow-y: auto;
    padding: 11px 12px;
    border: 0;
    outline: 0;
    color: var(--text);
    background: transparent;
    line-height: 1.5;
  }
  #user-input::placeholder { color: var(--muted); }
  #send-btn {
    min-width: 96px;
    height: 44px;
    border-radius: 16px;
  }

  #settings-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 100;
  }
  #settings-overlay.open { display: block; }
  #settings-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(2, 6, 23, 0.58);
    backdrop-filter: blur(8px);
  }
  #settings-panel {
    position: absolute;
    top: 16px;
    right: 16px;
    bottom: 16px;
    width: min(420px, calc(100vw - 32px));
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 18px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 26px;
    background: var(--surface-strong);
    box-shadow: var(--shadow);
    animation: slideIn 0.2s ease both;
  }
  @keyframes slideIn {
    from { opacity: 0; transform: translateX(16px) scale(0.98); }
    to { opacity: 1; transform: none; }
  }
  .settings-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .settings-top h2 {
    margin: 0;
    font-size: 18px;
    letter-spacing: -0.04em;
  }
  .settings-top p {
    margin: 4px 0 0;
    color: var(--text-dim);
    font-size: 12px;
  }
  .icon-btn {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border: 1px solid var(--border);
    border-radius: 12px;
    color: var(--text-dim);
    background: rgba(255,255,255,0.04);
    cursor: pointer;
  }
  .icon-btn:hover { color: var(--text); border-color: var(--border-strong); }
  .setting-group { display: grid; gap: 8px; }
  .setting-group label {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    color: var(--text-dim);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .setting-group input[type=number],
  .setting-group textarea {
    width: 100%;
    padding: 10px 11px;
    border: 1px solid var(--border);
    border-radius: 13px;
    outline: 0;
    color: var(--text);
    background: rgba(2, 6, 23, 0.32);
  }
  .setting-group textarea { min-height: 105px; resize: vertical; line-height: 1.5; }
  .setting-group input:focus, .setting-group textarea:focus { border-color: rgba(244, 182, 63, 0.48); }
  .range-row {
    display: grid;
    grid-template-columns: 1fr 54px;
    align-items: center;
    gap: 10px;
  }
  .range-row input[type=range] { width: 100%; accent-color: var(--accent); }
  .range-row span {
    color: var(--accent);
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 12px;
    font-weight: 800;
    text-align: right;
  }

  .waiting-dots {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 22px;
  }
  .waiting-dots span {
    width: 7px;
    height: 7px;
    border-radius: 999px;
    background: var(--accent);
    animation: dotPulse 1.2s ease-in-out infinite;
  }
  .waiting-dots span:nth-child(2) { animation-delay: 0.16s; }
  .waiting-dots span:nth-child(3) { animation-delay: 0.32s; }
  @keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.25; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-3px); }
  }

  @media (max-width: 900px) {
    #app { grid-template-columns: 1fr; padding: 10px; gap: 10px; }
    .sidebar { display: none; }
    .main { border-radius: 22px; }
    #header { align-items: flex-start; flex-direction: column; }
    #model-chips { justify-content: flex-start; }
    #messages { padding: 16px 12px; }
    .msg-stack { max-width: min(100%, calc(100vw - 78px)); }
    .welcome-prompts { grid-template-columns: 1fr; }
    #send-btn { min-width: 76px; }
  }
</style>
</head>
<body>
<div id="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">FF</div>
      <div>
        <h1>FF_LLM</h1>
        <p>Local model chat console</p>
      </div>
    </div>

    <div class="status-card">
      <div class="status-head">
        <span>Runtime</span>
        <span class="live-pill"><span class="live-dot"></span>online</span>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>Device</span><strong id="side-device">loading…</strong></div>
        <div class="metric"><span>Parameters</span><strong id="side-params">—</strong></div>
        <div class="metric"><span>KV policy</span><strong id="side-kv">—</strong></div>
      </div>
    </div>

    <div class="actions">
      <button id="btn-reset" class="btn">↺ Reset</button>
      <button id="btn-settings" class="btn">⚙ Settings</button>
    </div>

    <div class="hint-card">
      <strong>Tip:</strong> Press <span class="kbd">Enter</span> to send and <span class="kbd">Shift</span> + <span class="kbd">Enter</span> for a new line. Keep the Flask app bound to localhost when sharing through Tailscale Serve.
    </div>
  </aside>

  <main class="main">
    <div id="header">
      <div class="header-title">
        <h2>Chat session</h2>
        <p>Streaming responses from your local checkpoint</p>
      </div>
      <div id="model-chips">
        <span class="chip" id="chip-device">loading…</span>
        <span class="chip" id="chip-params"></span>
        <span class="chip good" id="chip-kv"></span>
      </div>
    </div>

    <div id="messages">
      <div id="welcome">
        <div class="welcome-card">
          <div class="welcome-icon">✦</div>
          <h2>Ready to chat</h2>
          <p>Ask a question, test a prompt, or tune generation settings from the sidebar.</p>
          <div class="welcome-prompts">
            <div class="prompt-card">Explain this bug clearly</div>
            <div class="prompt-card">Rewrite this function</div>
            <div class="prompt-card">Generate a test plan</div>
          </div>
        </div>
      </div>
    </div>

    <div id="input-bar">
      <div class="composer">
        <textarea id="user-input" placeholder="Message FF_LLM…" rows="1"></textarea>
        <button id="send-btn" class="btn primary">Send</button>
      </div>
    </div>
  </main>
</div>

<div id="settings-overlay">
  <div id="settings-backdrop"></div>
  <div id="settings-panel">
    <div class="settings-top">
      <div>
        <h2>Generation settings</h2>
        <p>These override the server defaults for the next message, including optional thinking display.</p>
      </div>
      <button id="close-settings" class="icon-btn" aria-label="Close settings">×</button>
    </div>

    <div class="setting-group">
      <label for="s-system">System prompt</label>
      <textarea id="s-system" rows="4"></textarea>
    </div>

    <label class="checkbox-row" for="s-show-thinking">
      <input type="checkbox" id="s-show-thinking">
      <span>
        <strong>Show thinking tokens</strong>
        <span>Show a collapsed thinking chip when &lt;think&gt;...&lt;/think&gt; output is present. Click it to expand.</span>
      </span>
    </label>

    <div class="setting-group">
      <label for="s-temp">Temperature</label>
      <div class="range-row">
        <input type="range" id="s-temp" min="0" max="1.5" step="0.05">
        <span id="s-temp-label">0.00</span>
      </div>
    </div>

    <div class="setting-group">
      <label for="s-max-new">Max new tokens</label>
      <input type="number" id="s-max-new" min="16" max="4096" step="16">
    </div>

    <div class="setting-group">
      <label for="s-top-p">Top-P</label>
      <div class="range-row">
        <input type="range" id="s-top-p" min="0" max="1" step="0.05">
        <span id="s-top-p-label">1.00</span>
      </div>
    </div>

    <div class="setting-group">
      <label for="s-top-k">Top-K <span>0 = off</span></label>
      <input type="number" id="s-top-k" min="0" max="200" step="1">
    </div>

    <div class="setting-group">
      <label for="s-repeat-penalty">Repeat penalty</label>
      <input type="number" id="s-repeat-penalty" min="1.0" max="2.0" step="0.01">
    </div>
  </div>
</div>

<script>
const cfg = {};
let generating = false;

function esc(t) {
  return String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderMarkdown(raw) {
  const blocks = [];
  let text = String(raw).replace(/```([\w.+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const idx = blocks.length;
    blocks.push({ lang, code });
    return `§§CODE_BLOCK_${idx}§§`;
  });

  text = esc(text);
  text = text.replace(/`([^`\n]+)`/g, (_, c) => `<code class="inline">${c}</code>`);
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  text = text.replace(/\n/g, '<br>');

  blocks.forEach((block, idx) => {
    const label = block.lang ? `<span class="code-lang">${esc(block.lang)}</span>` : '';
    const html = `<pre>${label}<code>${esc(block.code.trimEnd())}</code></pre>`;
    text = text.replace(`§§CODE_BLOCK_${idx}§§`, html);
  });

  return text;
}

function indexOfCI(text, needle, start = 0) {
  return text.toLowerCase().indexOf(needle.toLowerCase(), start);
}

function splitThinking(raw) {
  const text = String(raw || '');
  const openTag = '<think>';
  const closeTag = '</think>';

  const firstOpen = indexOfCI(text, openTag);
  const firstClose = indexOfCI(text, closeTag);
  if (firstClose >= 0 && (firstOpen < 0 || firstClose < firstOpen)) {
    return {
      thinking: text.slice(0, firstClose).trim(),
      answer: text.slice(firstClose + closeTag.length).trimStart(),
      thinkingOpen: false,
    };
  }

  let pos = 0;
  let thinkingParts = [];
  let answerParts = [];
  let thinkingOpen = false;

  while (pos < text.length) {
    const start = indexOfCI(text, openTag, pos);
    if (start < 0) {
      answerParts.push(text.slice(pos));
      break;
    }

    answerParts.push(text.slice(pos, start));
    const innerStart = start + openTag.length;
    const end = indexOfCI(text, closeTag, innerStart);
    if (end < 0) {
      thinkingParts.push(text.slice(innerStart));
      thinkingOpen = true;
      break;
    }

    thinkingParts.push(text.slice(innerStart, end));
    pos = end + closeTag.length;
  }

  return {
    thinking: thinkingParts.map(x => x.trim()).filter(Boolean).join('\n\n').trim(),
    answer: answerParts.join('').replaceAll(openTag, '').replaceAll(closeTag, '').trimStart(),
    thinkingOpen,
  };
}

function approxTokenCount(text) {
  const t = String(text || '').trim();
  if (!t) return 0;
  return t.split(/\s+/).filter(Boolean).length;
}

function updateAIContent(row, rawText, finalPayload = null, streaming = false) {
  const parts = splitThinking(rawText);
  const thinking = finalPayload?.thinking ?? parts.thinking;
  const answer = finalPayload?.reply ?? parts.answer;
  const tokenCount = finalPayload?.thinking_tokens ?? approxTokenCount(thinking);
  const panel = row.querySelector('.thinking-panel');
  const body = row.querySelector('.thinking-body');
  const count = row.querySelector('.thinking-count');
  const bubble = row.querySelector('.bubble');

  if (thinking && panel && body && count) {
    panel.classList.remove('hidden');
    // Keep thinking collapsed by default. If the user opens it mid-stream,
    // preserve that choice instead of snapping it shut on every token.
    if (!panel.dataset.userToggled) panel.open = false;
    body.innerHTML = renderMarkdown(thinking);
    count.textContent = tokenCount ? `${tokenCount} tokens` : 'saved';
  } else if (panel) {
    panel.classList.add('hidden');
    panel.open = false;
    delete panel.dataset.userToggled;
  }

  const shownAnswer = answer || '';
  row.dataset.rawText = shownAnswer.trim() || rawText;
  if (bubble) {
    if (shownAnswer.trim()) {
      bubble.innerHTML = renderMarkdown(shownAnswer) + (streaming ? '<span class="cursor"></span>' : '');
    } else {
      bubble.innerHTML = `<span class="stream-placeholder">${thinking ? 'Thinking…' : 'Waiting for response…'}</span>${streaming ? '<span class="cursor"></span>' : ''}`;
    }
  }
}

function scrollDown() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function setGenerating(v) {
  generating = v;
  document.getElementById('send-btn').disabled = v;
  document.getElementById('user-input').disabled = v;
}

function copyText(text, btn) {
  navigator.clipboard?.writeText(text).then(() => {
    const old = btn.textContent;
    btn.textContent = 'copied';
    setTimeout(() => { btn.textContent = old; }, 1000);
  }).catch(() => {});
}

function appendRow(role, htmlContent, id, rawText = '') {
  document.getElementById('welcome').classList.add('hidden');
  const msgs = document.getElementById('messages');
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;
  if (id) row.id = id;

  if (role === 'system-note') {
    row.innerHTML = `<div class="bubble">${htmlContent}</div>`;
  } else {
    const isUser = role === 'user';
    const label = isUser ? 'You' : 'FF_LLM';
    const avatar = isUser ? 'A' : '✦';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    row.innerHTML = `
      <div class="avatar">${avatar}</div>
      <div class="msg-stack">
        <div class="msg-meta">
          <span>${label}</span>
          <span>·</span>
          <span>${time}</span>
          ${isUser ? '' : '<button class="copy-btn" type="button">copy answer</button>'}
        </div>
        ${isUser ? '' : '<details class="thinking-panel hidden"><summary><span class="thinking-title">Thinking</span><span class="thinking-count"></span></summary><div class="thinking-body"></div></details>'}
        <div class="bubble">${htmlContent}</div>
      </div>`;

    const thinkPanel = row.querySelector('.thinking-panel');
    if (thinkPanel) {
      thinkPanel.addEventListener('toggle', () => {
        thinkPanel.dataset.userToggled = '1';
      });
    }

    const copyBtn = row.querySelector('.copy-btn');
    if (copyBtn) copyBtn.addEventListener('click', () => copyText(row.dataset.rawText || rawText, copyBtn));
    row.dataset.rawText = rawText;
  }

  msgs.appendChild(row);
  scrollDown();
  return row;
}

function addWaiting() {
  return appendRow('ai', `<span class="waiting-dots"><span></span><span></span><span></span></span>`, 'waiting-row');
}

async function sendMessage() {
  if (generating) return;
  const input = document.getElementById('user-input');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  input.style.height = 'auto';
  setGenerating(true);

  appendRow('user', renderMarkdown(msg), null, msg);
  const waitRow = addWaiting();

  const payload = {
    message: msg,
    system: document.getElementById('s-system').value,
    temp: parseFloat(document.getElementById('s-temp').value),
    top_k: parseInt(document.getElementById('s-top-k').value),
    top_p: parseFloat(document.getElementById('s-top-p').value),
    repeat_penalty: parseFloat(document.getElementById('s-repeat-penalty').value),
    max_new: parseInt(document.getElementById('s-max-new').value),
    show_thinking: document.getElementById('s-show-thinking').checked,
  };

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let aiRow = null;
    let bubble = null;
    let accText = '';
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.delta !== undefined) {
          if (!aiRow) {
            waitRow.remove();
            aiRow = appendRow('ai', '', 'ai-streaming');
            bubble = aiRow.querySelector('.bubble');
          }
          accText += data.delta;
          updateAIContent(aiRow, accText, null, true);
          scrollDown();
        }

        if (data.done) {
          if (bubble) {
            const finalRaw = data.raw_reply || data.reply || accText;
            updateAIContent(aiRow, finalRaw, data, false);
            aiRow.removeAttribute('id');
          } else {
            waitRow.remove();
            appendRow('system-note', 'No response generated.');
          }
        }
      }
    }
  } catch (err) {
    waitRow.remove();
    appendRow('system-note', `Error: ${esc(String(err))}`);
  } finally {
    setGenerating(false);
    document.getElementById('user-input').focus();
  }
}

document.getElementById('btn-reset').addEventListener('click', async () => {
  await fetch('/reset', { method: 'POST' });
  const msgs = document.getElementById('messages');
  [...msgs.querySelectorAll('.msg-row')].forEach(r => r.remove());
  document.getElementById('welcome').classList.remove('hidden');
});

document.getElementById('btn-settings').addEventListener('click', () => {
  document.getElementById('settings-overlay').classList.add('open');
});
document.getElementById('settings-backdrop').addEventListener('click', () => {
  document.getElementById('settings-overlay').classList.remove('open');
});
document.getElementById('close-settings').addEventListener('click', () => {
  document.getElementById('settings-overlay').classList.remove('open');
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('settings-overlay').classList.remove('open');
});

['temp', 'top-p'].forEach(id => {
  const range = document.getElementById(`s-${id}`);
  const label = document.getElementById(`s-${id}-label`);
  range.addEventListener('input', () => { label.textContent = parseFloat(range.value).toFixed(2); });
});

const ta = document.getElementById('user-input');
ta.addEventListener('input', () => {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
});
ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
document.getElementById('send-btn').addEventListener('click', sendMessage);

(async () => {
  try {
    const [info, defs] = await Promise.all([
      fetch('/info').then(r => r.json()),
      fetch('/defaults').then(r => r.json()),
    ]);

    const deviceText = `${info.device} · ${info.dtype}`;
    const paramsText = `${info.params_m}M params`;
    const kvText = `kv=${info.kv_max} sink=${info.kv_sink}${info.kv_int8 ? ' int8' : ''}`;

    document.getElementById('chip-device').textContent = deviceText;
    document.getElementById('chip-params').textContent = paramsText;
    document.getElementById('chip-kv').textContent = kvText;
    document.getElementById('side-device').textContent = deviceText;
    document.getElementById('side-params').textContent = paramsText;
    document.getElementById('side-kv').textContent = kvText;

    document.getElementById('s-system').value = defs.system;
    document.getElementById('s-show-thinking').checked = Boolean(defs.show_thinking);
    document.getElementById('s-temp').value = defs.temp;
    document.getElementById('s-temp-label').textContent = parseFloat(defs.temp).toFixed(2);
    document.getElementById('s-top-p').value = defs.top_p;
    document.getElementById('s-top-p-label').textContent = parseFloat(defs.top_p).toFixed(2);
    document.getElementById('s-top-k').value = defs.top_k;
    document.getElementById('s-repeat-penalty').value = defs.repeat_penalty;
    document.getElementById('s-max-new').value = defs.max_new;
  } catch (e) {
    console.warn('Failed to load model info:', e);
  }
})();
</script>
</body>
</html>
"""


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Web chat UI for FF_LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model / checkpoint
    ap.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    ap.add_argument("--ternary-adapter", default=None, help="Optional repaired ternary+LoRA adapter .pt file.")
    ap.add_argument("--tokenizer",   required=True, help="HF tokenizer name or path")
    ap.add_argument("--device",      default=os.environ.get("DEVICE", None))
    ap.add_argument("--dtype",       default=os.environ.get("DTYPE",  "auto"))
    ap.add_argument("--block-size",  type=int, default=int(os.environ.get("BLOCK_SIZE", "256")))

    # KV cache (defaults from architecture_plan.md "Current Best Path")
    ap.add_argument("--kv-cache-max-len",     type=int, default=int(os.environ.get("KV_CACHE_MAX_LEN",     "0")),
                    help="0 = full block-size window (safe default)")
    ap.add_argument("--kv-cache-sink-tokens", type=int, default=int(os.environ.get("KV_CACHE_SINK_TOKENS", "64")),
                    help="Keep first N tokens (system prompt) visible even under cache eviction")

    # Generation defaults (overridable per-request from settings panel)
    ap.add_argument("--max-new",        type=int,   default=int(  os.environ.get("MAX_NEW",        "512")))
    ap.add_argument("--stream-every",   type=int,   default=int(  os.environ.get("STREAM_EVERY",   "1")),
                    help="Decode and send SSE deltas every N generated tokens.")
    ap.add_argument("--temp",           type=float, default=float(os.environ.get("TEMP",           "0.4")),
                    help="0.0 = greedy; 0.4 is a good chat default")
    ap.add_argument("--top-k",          type=int,   default=int(  os.environ.get("TOP_K",          "0")))
    ap.add_argument("--top-p",          type=float, default=float(os.environ.get("TOP_P",          "1.0")))
    ap.add_argument("--repeat-penalty", type=float, default=float(os.environ.get("REPEAT_PENALTY", "1.05")))
    ap.add_argument("--repeat-window",  type=int,   default=int(  os.environ.get("REPEAT_WINDOW",  "256")))
    ap.add_argument("--max-turns",      type=int,   default=int(  os.environ.get("MAX_TURNS",      "12")),
                    help="Max conversation turns kept in context per user")
    ap.add_argument("--system", default=os.environ.get(
        "SYSTEM_PROMPT", "You are a helpful coding assistant. Be concise and accurate."))

    # Prompt/token flags (passed through to chat_hf helpers)
    ap.add_argument("--no-chat-template",   action="store_true")
    ap.add_argument("--raw",                action="store_true", help="Use plain completion-style prompts instead of chat templates.")
    ap.add_argument("--suppress-im-start",  action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hide-reasoning",     action="store_true")
    ap.add_argument("--extra-stop-tokens",  nargs="*", default=[])
    ap.add_argument("--no-stop-eos",        action="store_true")
    ap.add_argument("--verbose-keys",       action="store_true")

    # Server
    ap.add_argument("--host",  default="0.0.0.0",
                    help="Bind address. 0.0.0.0 = reachable from Tailscale peers")
    ap.add_argument("--port",  type=int, default=9090)
    ap.add_argument("--debug", action="store_true", help="Flask debug mode (single-process, no reload)")
    return ap.parse_args()


def main():
    global _model, _raw_model, _tok, _cfg, _device, _dtype, _server_args

    _server_args = parse_args()

    # Wire KV env vars before load_model reads them
    os.environ["KV_CACHE_MAX_LEN"]     = str(_server_args.kv_cache_max_len)
    os.environ["KV_CACHE_SINK_TOKENS"] = str(_server_args.kv_cache_sink_tokens)

    # chat_hf.load_model needs these attributes present on args
    _server_args.stream = False
    _server_args.raw    = False
    _server_args.history = True

    print("[web_chat] Loading model…", flush=True)
    _model, _raw_model, _tok, _cfg, _device, _dtype = load_model(_server_args)

    print(f"\n[web_chat] ✓ Ready", flush=True)
    print(f"[web_chat]   Local:     http://localhost:{_server_args.port}", flush=True)
    print(f"[web_chat]   Tailscale: http://<your-ts-ip>:{_server_args.port}", flush=True)
    print(f"[web_chat]   KV policy: max={_cfg.kv_cache_max_len or _cfg.block_size} "
          f"sink={_cfg.kv_cache_sink_tokens} int8={int(_cfg.kv_cache_int8)}\n", flush=True)

    # use_reloader=False is important — model must not be loaded twice
    app.run(
        host=_server_args.host,
        port=_server_args.port,
        debug=_server_args.debug,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
