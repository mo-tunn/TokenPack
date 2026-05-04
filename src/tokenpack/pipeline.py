from __future__ import annotations

from pathlib import Path

from tokenpack.chunking import ParagraphGroupChunker, SemanticThresholdChunker
from tokenpack.embeddings import EmbeddingCache, Embedder
from tokenpack.index import ChunkIndex, save_index
from tokenpack.loaders import load_blocks


def ingest_path(
    source: str | Path,
    index_path: str | Path,
    embedder: Embedder,
    target_tokens: int = 650,
    min_tokens: int = 120,
    max_tokens: int = 900,
    chunker_name: str = "paragraph",
    semantic_threshold: float = 0.35,
    cache_path: str | Path | None = None,
) -> ChunkIndex:
    blocks = load_blocks(source)
    cache = EmbeddingCache(cache_path or Path(index_path).with_suffix(".embeddings.json"))
    if chunker_name == "paragraph":
        chunker = ParagraphGroupChunker(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
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
    chunks = chunker.chunk(blocks)
    embeddings = cache.get_or_embed([chunk.text for chunk in chunks], embedder)
    index = ChunkIndex(chunks=chunks, embeddings=embeddings, model_name=embedder.model_name)
    save_index(index, index_path)
    return index

