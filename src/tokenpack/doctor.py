from __future__ import annotations

import importlib.util
import json
import os
import urllib.request


def collect_diagnostics(ollama_url: str = "http://localhost:11434") -> dict:
    return {
        "python_packages": {
            "sentence_transformers": _has_package("sentence_transformers"),
            "tiktoken": _has_package("tiktoken"),
            "pymupdf_fitz": _has_package("fitz"),
            "pypdf": _has_package("pypdf"),
            "python_docx": _has_package("docx"),
            "python_pptx": _has_package("pptx"),
            "openpyxl": _has_package("openpyxl"),
            "mcp": _has_package("mcp"),
        },
        "environment": {
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
            "tokenpack_hf_offline": os.environ.get("TOKENPACK_HF_OFFLINE") == "1",
            "tokenpack_ollama_model": os.environ.get("TOKENPACK_OLLAMA_MODEL"),
        },
        "ollama": _ollama_status(ollama_url),
    }


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _ollama_status(base_url: str) -> dict:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"available": False, "error": str(exc), "models": []}
    models = [item.get("name", "") for item in payload.get("models", []) if item.get("name")]
    return {"available": True, "models": models}
