from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from tokenpack.chunking import SemanticThresholdChunker, StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, Embedder
from tokenpack.index import ChunkIndex, save_index
from tokenpack.loaders import iter_supported_files, load_blocks
from tokenpack.models import Chunk, TextBlock


MANIFEST_VERSION = 1


def ingest_path(
    source: str | Path,
    index_path: str | Path,
    embedder: Embedder,
    target_tokens: int = 650,
    min_tokens: int = 120,
    max_tokens: int = 900,
    chunker_name: str = "structure-aware",
    semantic_threshold: float = 0.35,
    source_type: str = "auto",
    cache_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> ChunkIndex:
    source_path = Path(source)
    if chunker_name not in {"structure-aware", "semantic-threshold"}:
        raise ValueError(f"Unknown chunker: {chunker_name}")
    cache = EmbeddingCache(cache_path or Path(index_path).with_suffix(".embeddings.json"))
    if source_path.is_dir():
        index = _ingest_directory_incremental(
            source_path,
            embedder=embedder,
            cache=cache,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            chunker_name=chunker_name,
            semantic_threshold=semantic_threshold,
            source_type=source_type,
            manifest_path=Path(manifest_path or Path(index_path).with_suffix(".manifest.json")),
        )
        save_index(index, index_path)
        return index

    blocks = load_blocks(source_path, source_type=source_type)
    chunks = _chunk_blocks(
        blocks,
        embedder=embedder,
        cache=cache,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        chunker_name=chunker_name,
        semantic_threshold=semantic_threshold,
    )
    embeddings = cache.get_or_embed([chunk.text for chunk in chunks], embedder)
    index = ChunkIndex(chunks=chunks, embeddings=embeddings, model_name=embedder.model_name)
    save_index(index, index_path)
    return index


def _chunk_blocks(
    blocks: list[TextBlock],
    *,
    embedder: Embedder,
    cache: EmbeddingCache,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    chunker_name: str,
    semantic_threshold: float,
) -> list[Chunk]:
    if chunker_name == "structure-aware":
        block_embeddings = cache.get_or_embed([block.text for block in blocks], embedder)
        chunker = StructureAwareChunker(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            block_embeddings=block_embeddings,
            semantic_threshold=semantic_threshold,
        )
    elif chunker_name == "semantic-threshold":
        block_embeddings = cache.get_or_embed([block.text for block in blocks], embedder)
        chunker = SemanticThresholdChunker(
            block_embeddings=block_embeddings,
            similarity_threshold=semantic_threshold,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown chunker: {chunker_name}")
    return chunker.chunk(blocks)


def _ingest_directory_incremental(
    source: Path,
    *,
    embedder: Embedder,
    cache: EmbeddingCache,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    chunker_name: str,
    semantic_threshold: float,
    source_type: str,
    manifest_path: Path,
) -> ChunkIndex:
    config = {
        "version": MANIFEST_VERSION,
        "source": str(source.resolve()),
        "model_name": embedder.model_name,
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "chunker_name": chunker_name,
        "semantic_threshold": semantic_threshold,
        "source_type": source_type,
    }
    previous = _load_manifest(manifest_path)
    previous_files = previous.get("files", {}) if previous.get("config") == config else {}
    if not isinstance(previous_files, dict):
        previous_files = {}

    all_chunks: list[Chunk] = []
    all_embeddings: list[list[float]] = []
    next_files: dict[str, dict[str, Any]] = {}
    for document_index, file_path in enumerate(iter_supported_files(source, source_type=source_type)):
        relative_path = file_path.relative_to(source).as_posix()
        digest = _hash_file(file_path)
        cached = previous_files.get(relative_path)
        if isinstance(cached, dict) and cached.get("sha256") == digest:
            chunks = [Chunk.from_dict(item) for item in cached.get("chunks", [])]
            embeddings = [list(map(float, row)) for row in cached.get("embeddings", [])]
            if len(chunks) != len(embeddings):
                chunks, embeddings = _ingest_file(
                    file_path,
                    document_index=document_index,
                    source_type=source_type,
                    embedder=embedder,
                    cache=cache,
                    target_tokens=target_tokens,
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    chunker_name=chunker_name,
                    semantic_threshold=semantic_threshold,
                )
            else:
                for chunk in chunks:
                    chunk.document_index = document_index
        else:
            chunks, embeddings = _ingest_file(
                file_path,
                document_index=document_index,
                source_type=source_type,
                embedder=embedder,
                cache=cache,
                target_tokens=target_tokens,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                chunker_name=chunker_name,
                semantic_threshold=semantic_threshold,
            )

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        next_files[relative_path] = {
            "sha256": digest,
            "chunks": [chunk.to_dict() for chunk in chunks],
            "embeddings": embeddings,
        }

    _save_manifest(manifest_path, {"config": config, "files": next_files})
    return ChunkIndex(chunks=all_chunks, embeddings=all_embeddings, model_name=embedder.model_name)


def _ingest_file(
    file_path: Path,
    *,
    document_index: int,
    source_type: str,
    embedder: Embedder,
    cache: EmbeddingCache,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    chunker_name: str,
    semantic_threshold: float,
) -> tuple[list[Chunk], list[list[float]]]:
    blocks = load_blocks(file_path, source_type=source_type)
    for block in blocks:
        block.document_index = document_index
    chunks = _chunk_blocks(
        blocks,
        embedder=embedder,
        cache=cache,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        chunker_name=chunker_name,
        semantic_threshold=semantic_threshold,
    )
    embeddings = cache.get_or_embed([chunk.text for chunk in chunks], embedder)
    return chunks, embeddings


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", errors="replace")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

