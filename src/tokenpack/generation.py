from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from tokenpack.export import render_context
from tokenpack.models import Chunk


def answer_from_selection(
    query: str,
    selection_path: str | Path,
    provider: str = "none",
    model: str = "gpt-4o-mini",
    ollama_url: str = "http://localhost:11434",
) -> dict:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(item["chunk"]) for item in payload.get("selected", [])]
    context = render_context(chunks)
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    if provider == "none":
        answer = ""
    elif provider in {"ollama", "local"}:
        local_model = _default_ollama_model(model)
        answer = _ollama_answer(prompt, model=local_model, base_url=ollama_url)
        model = local_model
    elif provider == "cerebras":
        answer = _cerebras_answer(prompt, model=model)
    elif provider == "groq":
        answer = _groq_answer(prompt, model=model)
    else:
        raise ValueError(f"Unknown answer provider: {provider}")
    return {
        "provider": provider,
        "model": model if provider != "none" else None,
        "query": query,
        "answer": answer,
        "context_tokens": sum(chunk.token_count for chunk in chunks),
        "source_selection": str(selection_path),
    }


def save_answer(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cerebras_answer(prompt: str, model: str) -> str:
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required for provider=cerebras.")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 220,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.cerebras.ai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"]).strip()


def _groq_answer(prompt: str, model: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required for provider=groq.")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 220,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"]["content"]).strip()


def _ollama_answer(prompt: str, model: str, base_url: str = "http://localhost:11434") -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Ollama generation failed. Ensure Ollama is running and the requested model is installed "
            f"(for example: ollama pull {model})."
        ) from exc
    return str(payload.get("response") or "").strip()


def _default_ollama_model(model: str) -> str:
    if model and model != "gpt-4o-mini":
        return model
    return os.environ.get("TOKENPACK_OLLAMA_MODEL", "llama3.2:1b")
