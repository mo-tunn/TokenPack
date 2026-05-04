from __future__ import annotations

from tokenpack.embeddings import cosine
from tokenpack.models import Chunk, ScoredChunk


def score_chunks(
    query_embedding: list[float],
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    redundancy_penalty: float = 0.0,
) -> list[ScoredChunk]:
    similarities = [cosine(query_embedding, embedding) for embedding in chunk_embeddings]
    values = _minmax(similarities)
    scored = [
        ScoredChunk(
            chunk=chunk,
            value=value,
            raw_similarity=similarity,
            weight=chunk.token_count,
            embedding=embedding,
        )
        for chunk, value, similarity, embedding in zip(chunks, values, similarities, chunk_embeddings, strict=True)
    ]
    if redundancy_penalty > 0:
        _apply_redundancy_penalty(scored, chunk_embeddings, redundancy_penalty)
    return scored


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _apply_redundancy_penalty(
    scored: list[ScoredChunk],
    embeddings: list[list[float]],
    penalty_strength: float,
) -> None:
    ranked_indices = sorted(range(len(scored)), key=lambda index: scored[index].value, reverse=True)
    stronger: list[int] = []
    for index in ranked_indices:
        max_overlap = 0.0
        for previous in stronger:
            max_overlap = max(max_overlap, max(0.0, cosine(embeddings[index], embeddings[previous])))
        penalty = penalty_strength * max_overlap
        scored[index].redundancy_penalty = penalty
        scored[index].value = max(0.0, scored[index].value * (1.0 - penalty))
        stronger.append(index)

