from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tokenpack.models import Chunk


@dataclass(slots=True)
class ChunkIndex:
    chunks: list[Chunk]
    embeddings: list[list[float]]
    model_name: str

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "model_name": self.model_name,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "embeddings": self.embeddings,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ChunkIndex":
        return cls(
            chunks=[Chunk.from_dict(item) for item in payload["chunks"]],
            embeddings=[list(map(float, row)) for row in payload["embeddings"]],
            model_name=payload.get("model_name", "unknown"),
        )


def save_index(index: ChunkIndex, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(index.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
        errors="replace",
    )


def load_index(path: str | Path) -> ChunkIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ChunkIndex.from_dict(payload)

