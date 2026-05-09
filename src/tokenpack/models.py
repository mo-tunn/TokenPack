from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TextBlock:
    """A document text unit with provenance needed for faithful reconstruction."""

    text: str
    source_path: str
    document_index: int
    page: int | None = None
    paragraph_index: int = 0
    char_start: int = 0
    char_end: int = 0
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBlock":
        payload = dict(data)
        if payload.get("bbox") is not None:
            payload["bbox"] = tuple(payload["bbox"])
        return cls(**payload)


@dataclass(slots=True)
class Chunk:
    """A semantically coherent candidate item for budgeted selection."""

    id: str
    text: str
    source_path: str
    document_index: int
    start_page: int | None
    end_page: int | None
    start_paragraph: int
    end_paragraph: int
    char_start: int
    char_end: int
    token_count: int
    block_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def order_key(self) -> tuple[int, int, int, int]:
        page = self.start_page if self.start_page is not None else -1
        return (self.document_index, page, self.start_paragraph, self.char_start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    value: float
    raw_similarity: float
    weight: int
    redundancy_penalty: float = 0.0
    embedding: list[float] | None = None
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "value": self.value,
            "raw_similarity": self.raw_similarity,
            "weight": self.weight,
            "redundancy_penalty": self.redundancy_penalty,
            "score_components": self.score_components,
        }


@dataclass(slots=True)
class SelectionResult:
    strategy: str
    budget: int
    used_tokens: int
    total_value: float
    selected: list[ScoredChunk]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget": self.budget,
            "used_tokens": self.used_tokens,
            "total_value": self.total_value,
            "elapsed_seconds": self.elapsed_seconds,
            "selected": [item.to_dict() for item in self.selected],
        }

