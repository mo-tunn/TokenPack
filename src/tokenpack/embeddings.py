from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """Sentence-transformers embedding backend used by TokenPack."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer  # type: ignore

        offline = (
            os.environ.get("TOKENPACK_HF_OFFLINE") == "1"
            or os.environ.get("HF_HUB_OFFLINE") == "1"
        )
        use_local_files = offline if local_files_only is None else local_files_only
        self._model = SentenceTransformer(model_name, local_files_only=use_local_files)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            [_clean_text(text) for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, row)) for row in embeddings]


def make_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool | None = None,
) -> Embedder:
    return SentenceTransformerEmbedder(model_name=model_name, local_files_only=local_files_only)


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(limit))


class EmbeddingCache:
    """Small JSON cache keyed by model and text hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, list[float]] = {}
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._records = {key: list(map(float, value)) for key, value in payload.items()}

    def get_or_embed(self, texts: list[str], embedder: Embedder) -> list[list[float]]:
        keys = [self._key(text, embedder.model_name) for text in texts]
        missing_texts: list[str] = []
        missing_keys: list[str] = []
        for key, text in zip(keys, texts, strict=True):
            if key not in self._records:
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            for key, vector in zip(missing_keys, embedder.embed(missing_texts), strict=True):
                self._records[key] = vector
            self.save()
        return [self._records[key] for key in keys]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records), encoding="utf-8")

    @staticmethod
    def _key(text: str, model_name: str) -> str:
        digest = hashlib.sha256(_clean_text(text).encode("utf-8", errors="replace")).hexdigest()
        return f"{model_name}:{digest}"


def _clean_text(text: str) -> str:
    """Normalize extractor artifacts that are not valid standalone Unicode."""

    return text.encode("utf-8", errors="replace").decode("utf-8")

