from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_LOCK_TIMEOUT_SECONDS = 10.0
CACHE_LOCK_STALE_SECONDS = 60.0


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """Sentence-transformers embedding backend used by TokenPack.

    When network access is allowed implicitly, prefer already-cached local model
    files first. This avoids slow Hugging Face network checks on machines that
    have used TokenPack before, while still allowing a first-run download when
    the model is not cached.
    """

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
        if local_files_only is None and not offline:
            try:
                self._model = SentenceTransformer(model_name, local_files_only=True)
                return
            except Exception:
                self._model = SentenceTransformer(model_name, local_files_only=False)
                return

        use_local_files = True if offline else bool(local_files_only)
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
    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: left has {len(left)} dimensions, right has {len(right)}."
        )
    return sum(left[index] * right[index] for index in range(len(left)))


class EmbeddingCache:
    """Small JSON cache keyed by model and text hash."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records = self._read_records()

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
        with self._lock():
            latest = self._read_records()
            latest.update(self._records)
            self._records = latest
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(self._records), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_records(self) -> dict[str, list[float]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Embedding cache payload must be a JSON object.")
            return {key: list(map(float, value)) for key, value in payload.items()}
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, TypeError, ValueError):
            return {}

    @contextmanager
    def _lock(self) -> Iterator[None]:
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        deadline = time.monotonic() + CACHE_LOCK_TIMEOUT_SECONDS
        descriptor: int | None = None
        while descriptor is None:
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(descriptor, str(os.getpid()).encode("ascii", errors="replace"))
                except OSError:
                    os.close(descriptor)
                    descriptor = None
                    lock_path.unlink(missing_ok=True)
                    raise
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > CACHE_LOCK_STALE_SECONDS:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for embedding cache lock: {lock_path}")
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    @staticmethod
    def _key(text: str, model_name: str) -> str:
        digest = hashlib.sha256(_clean_text(text).encode("utf-8", errors="replace")).hexdigest()
        return f"{model_name}:{digest}"


def _clean_text(text: str) -> str:
    """Normalize extractor artifacts that are not valid standalone Unicode."""

    return text.encode("utf-8", errors="replace").decode("utf-8")

