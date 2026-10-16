"""Isolated Faz36 relay to the already-configured Qwen 27B long-context API.

Only the chat-completion payload crosses this relay. Benchmark files, IDs,
annotations, local model weights, and credentials are never uploaded or logged.
"""
from __future__ import annotations

import modal

BASE = "https://metekizilcik--tokenpack-qwen27b-longcontext-inference-server.modal.run"
MODEL = "qwen3.8-27b-orcarouter-q4km"
APP = modal.App("tokenpack-phase36-iirc-qwen27b-relay")


@APP.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    secrets=[modal.Secret.from_name("qwen-modal-secrets")],
    cpu=0.125,
    memory=512,
    timeout=365,
)
def infer_once(payload: dict) -> dict:
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    key = os.environ.get("LLM_API_KEY", "")
    if not key:
        return {"ok": False, "phase": "secret", "error": "missing secret"}

    def http(path: str, data: dict | None = None, timeout: int = 15):
        req = urllib.request.Request(
            BASE + path,
            method="POST" if data is not None else "GET",
            data=json.dumps(data).encode("utf8") if data is not None else None,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf8"))
        except urllib.error.HTTPError as exc:
            return exc.code, {}

    for _ in range(8):
        try:
            status, _ = http("/health", timeout=12)
        except Exception as exc:
            return {"ok": False, "phase": "health", "exception": type(exc).__name__}
        if status == 200:
            break
        if status not in (502, 503, 504):
            return {"ok": False, "phase": "health", "http_status": status}
        time.sleep(3)
    else:
        return {"ok": False, "phase": "health", "error": "not ready"}

    started = time.monotonic()
    try:
        status, result = http("/v1/chat/completions", data=payload, timeout=300)
    except Exception as exc:
        return {"ok": False, "phase": "inference", "exception": type(exc).__name__}
    if status != 200:
        return {"ok": False, "phase": "inference", "http_status": status}

    choices = result.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    usage = result.get("usage") or {}
    return {
        "ok": True,
        "answer": str(message.get("content") or "").strip()[:4000],
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
        "elapsed_s": round(time.monotonic() - started, 2),
    }
