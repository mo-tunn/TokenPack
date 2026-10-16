"""Frozen Faz36 IIRC semantic citation comparison helpers.

The treatment is the unchanged Phase28 semantic-joint-citations selector,
applied to the IIRC main passage plus every available linked article. Gold
annotations are never needed by this module.
"""
from __future__ import annotations

import hashlib
import json

from submission.experiments.phase28_semantic_core import (
    MAX_OUTPUT_TOKENS as VERIFIER_MAX_OUTPUT_TOKENS,
    SYSTEM as VERIFIER_SYSTEM,
    USER as VERIFIER_USER,
    make_task,
    parse_ids,
    payload as verifier_payload,
    select as phase28_select,
)

MODEL = "qwen3.8-27b-orcarouter-q4km"
READER_SEED = 20261014
READER_MAX_OUTPUT_TOKENS = 96
ARMS = ("atomic-sentence-rank", "semantic-joint-citations")
READER_SYSTEM = (
    "Answer the question using only the supplied context. Return the shortest "
    "answer supported by the text: a name or phrase, a number with its unit "
    "when needed, Yes, No, or NONE if the context does not contain enough "
    "information. If the answer needs multiple separate spans, separate them "
    "with semicolons. Return only the answer, with no explanation."
)


def reader_payload(task: dict) -> dict:
    user = (
        "Context:\n" + task["context"] +
        "\n\nQuestion:\n" + task["question"] +
        "\n\nAnswer only in the requested format:"
    )
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "min_p": 0,
        "seed": READER_SEED,
        "max_tokens": READER_MAX_OUTPUT_TOKENS,
        "reasoning_effort": "none",
        "stream": False,
    }


def digest_payload(payload: dict) -> str:
    packed = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(packed.encode("utf8")).hexdigest()


def answer_parts(raw: object) -> list[str]:
    """Apply the preregistered multi-span output delimiter."""
    text = str(raw or "").strip()
    parts = [part.strip() for part in text.split(";")]
    return [part for part in parts if part] or [""]


__all__ = [
    "ARMS", "MODEL", "READER_MAX_OUTPUT_TOKENS", "READER_SEED",
    "READER_SYSTEM", "VERIFIER_MAX_OUTPUT_TOKENS", "VERIFIER_SYSTEM",
    "VERIFIER_USER", "answer_parts", "digest_payload", "make_task",
    "parse_ids", "phase28_select", "reader_payload", "verifier_payload",
]
