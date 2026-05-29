#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chat_hf import load_model, run_prompt  # noqa: E402

CHECKPOINT = os.environ.get("CHAT_CHECKPOINT", "models/model.pt")
TOKENIZER = os.environ.get("CHAT_TOKENIZER", "weights/qwen25-1.5b-instruct")
SHARE_KEY = os.environ.get("SHARE_KEY", "").strip()
TITLE = os.environ.get("CHAT_TITLE", "FF/BP Chat Runtime")

# Stable tested runtime defaults.
os.environ.setdefault("USE_KV_CACHE", "1")
os.environ.setdefault("USE_FF_DRAFT_SKIP", "0")
os.environ.setdefault("USE_DRAFT_HEAD", "0")
os.environ.setdefault("DRAFT_BLEND_BP", "0")
os.environ.setdefault("INFER_MEMORY_TOKENS", "0")
os.environ.setdefault("INFER_USE_ENGRAM", "0")
os.environ.setdefault("KV_CACHE_MAX_LEN", "0")

app = FastAPI(title=TITLE)
app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "web_chat" / "static")), name="static")

generate_lock = asyncio.Lock()
histories: Dict[str, List[Tuple[str, str]]] = {}
model = raw_model = tok = cfg = device = dtype = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    system: str = "You are a helpful assistant. Be concise. If writing code, give runnable code."
    max_new: int = 220
    temp: float = 0.4
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.15
    history: bool = True
    max_turns: int = 6

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    elapsed_sec: float


def require_key(x_share_key: Optional[str]):
    if SHARE_KEY and x_share_key != SHARE_KEY:
        raise HTTPException(status_code=401, detail="Invalid share key")

@app.on_event("startup")
def startup():
    global model, raw_model, tok, cfg, device, dtype
    args = SimpleNamespace(
        checkpoint=CHECKPOINT,
        tokenizer=TOKENIZER,
        device=os.environ.get("DEVICE", None),
        dtype=os.environ.get("DTYPE", "auto"),
        block_size=int(os.environ.get("BLOCK_SIZE", "256")),
        kv_cache_max_len=int(os.environ.get("KV_CACHE_MAX_LEN", "0")),
    )
    print("[WEBCHAT] Loading model once...", flush=True)
    model, raw_model, tok, cfg, device, dtype = load_model(args)
    print("[WEBCHAT] Ready.", flush=True)

@app.get("/")
def index():
    return FileResponse(str(REPO_ROOT / "web_chat" / "static" / "index.html"))

@app.get("/health")
def health():
    return {
        "ok": True,
        "title": TITLE,
        "checkpoint": CHECKPOINT,
        "device": str(device),
        "kv_cache": os.environ.get("USE_KV_CACHE"),
        "kv_cache_max_len": os.environ.get("KV_CACHE_MAX_LEN"),
        "kv_cache_int8": os.environ.get("KV_CACHE_INT8", "0"),
    }

@app.post("/api/reset")
def reset(session_id: str, x_share_key: Optional[str] = Header(default=None)):
    require_key(x_share_key)
    histories.pop(session_id, None)
    return {"ok": True}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_share_key: Optional[str] = Header(default=None)):
    require_key(x_share_key)
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Empty message")
    session_id = req.session_id or str(uuid.uuid4())
    history = histories.setdefault(session_id, [])
    args = SimpleNamespace(
        raw=False,
        system=req.system,
        max_new=max(1, min(int(req.max_new), 768)),
        temp=float(req.temp),
        top_k=int(req.top_k),
        top_p=float(req.top_p),
        repeat_penalty=float(req.repeat_penalty),
        no_stop_eos=False,
        history=bool(req.history),
        max_turns=max(0, min(int(req.max_turns), 20)),
    )
    start = time.time()
    async with generate_lock:
        try:
            reply = await asyncio.to_thread(run_prompt, args, model, raw_model, tok, cfg, device, dtype, msg, history if req.history else [])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}: {e}")
    if req.history:
        history.append((msg, reply))
        histories[session_id] = history[-args.max_turns:]
    return ChatResponse(session_id=session_id, reply=reply, elapsed_sec=round(time.time() - start, 3))
