from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tokenpack.index import ChunkIndex


@dataclass(slots=True)
class GoldRecord:
    query: str
    answer: str
    evidence_chunk_ids: list[str]
    source_path: str | None = None
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoldRecord":
        query = str(payload.get("query") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        evidence = payload.get("evidence_chunk_ids") or payload.get("evidence_chunk_id") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence_ids = [str(item).strip() for item in evidence if str(item).strip()]
        if not query:
            raise ValueError("Gold record is missing required field: query")
        if not evidence_ids:
            raise ValueError("Gold record is missing required field: evidence_chunk_ids")
        return cls(
            query=query,
            answer=answer,
            evidence_chunk_ids=evidence_ids,
            source_path=payload.get("source_path"),
            notes=str(payload.get("notes") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "between",
    "context",
    "from",
    "have",
    "into",
    "large",
    "language",
    "model",
    "models",
    "that",
    "their",
    "there",
    "these",
    "this",
    "using",
    "with",
}


def load_gold_records(path: str | Path) -> list[GoldRecord]:
    records: list[GoldRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(GoldRecord.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid gold record at line {line_number}: {exc}") from exc
    return records


def save_gold_records(records: list[GoldRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=False) for record in records]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def validate_gold_records(records: list[GoldRecord], index: ChunkIndex) -> list[str]:
    chunk_ids = {chunk.id for chunk in index.chunks}
    errors: list[str] = []
    for record_index, record in enumerate(records, start=1):
        for chunk_id in record.evidence_chunk_ids:
            if chunk_id not in chunk_ids:
                errors.append(f"record {record_index}: unknown evidence chunk id {chunk_id}")
    return errors


def propose_gold_records(index: ChunkIndex, sample_size: int = 12, keyword_count: int = 6) -> list[GoldRecord]:
    records: list[GoldRecord] = []
    for chunk_index in _spaced_indices(len(index.chunks), sample_size):
        chunk = index.chunks[chunk_index]
        keywords = _keywords(chunk.text, keyword_count)
        if not keywords:
            continue
        records.append(
            GoldRecord(
                query=" ".join(keywords),
                answer=_first_sentence(chunk.text),
                evidence_chunk_ids=[chunk.id],
                source_path=chunk.source_path,
                notes="Auto-proposed candidate; review query, answer, and evidence before using as gold.",
                metadata={"proposal": "keyword", "chunk_index": chunk_index},
            )
        )
    return records


def _keywords(text: str, limit: int) -> list[str]:
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z-]{3,}", text.lower())
        if word not in STOPWORDS
    ]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(limit)]


def _first_sentence(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return sentence[:500]


def _spaced_indices(length: int, sample_size: int) -> list[int]:
    if length <= 0:
        return []
    if length <= sample_size:
        return list(range(length))
    step = length / sample_size
    return sorted({min(length - 1, int(index * step)) for index in range(sample_size)})
