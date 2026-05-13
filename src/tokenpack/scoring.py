from __future__ import annotations

import math
import re
from collections import Counter

from tokenpack.embeddings import cosine
from tokenpack.models import Chunk, ScoredChunk

DEFAULT_SCORING_PROFILE = "evidence-hybrid"
SCORING_PROFILES = (DEFAULT_SCORING_PROFILE,)

EVIDENCE_HYBRID_WEIGHTS = {
    "cosine": 0.35,
    "bm25": 0.25,
    "query_coverage": 0.20,
    "structural_prior": 0.15,
    "neighbor_coherence": 0.05,
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", flags=re.UNICODE)
_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}", flags=re.UNICODE)


def score_chunks(
    query_embedding: list[float],
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    redundancy_penalty: float = 0.0,
    scoring: str = DEFAULT_SCORING_PROFILE,
    query_text: str | None = None,
    redundancy_candidate_pool: int | None = 250,
) -> list[ScoredChunk]:
    """Score chunks with the production evidence-hybrid value function.

    Historical and research-only profiles live in ``tokenpack.scoring_experimental``.
    They are intentionally not exposed through the production scoring registry.
    """

    if scoring != DEFAULT_SCORING_PROFILE:
        raise ValueError(
            f"Unsupported production scoring profile: {scoring}. "
            "Use 'evidence-hybrid' or import tokenpack.scoring_experimental for ablations."
        )

    similarities = [cosine(query_embedding, embedding) for embedding in chunk_embeddings]
    components = _evidence_hybrid_components(query_text or "", chunks, chunk_embeddings, similarities)
    values = [
        sum(EVIDENCE_HYBRID_WEIGHTS[name] * item[name] for name in EVIDENCE_HYBRID_WEIGHTS)
        for item in components
    ]
    scored = [
        ScoredChunk(
            chunk=chunk,
            value=value,
            raw_similarity=similarity,
            weight=chunk.token_count,
            embedding=embedding,
            score_components=component,
        )
        for chunk, value, similarity, embedding, component in zip(
            chunks,
            values,
            similarities,
            chunk_embeddings,
            components,
            strict=True,
        )
    ]
    if redundancy_penalty > 0:
        _apply_redundancy_penalty(
            scored,
            chunk_embeddings,
            redundancy_penalty,
            candidate_pool=redundancy_candidate_pool,
        )
    return scored


def _evidence_hybrid_components(
    query_text: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    similarities: list[float],
) -> list[dict[str, float]]:
    cosine_values = _minmax(similarities)
    bm25_values = _normalized_signal(_bm25_scores(query_text, chunks))
    coverage_values = _query_coverage(query_text, chunks)
    structural_values = _structural_prior(query_text, chunks)
    neighbor_values = _normalized_signal(_neighbor_coherence(chunks, chunk_embeddings))
    return [
        {
            "cosine": cosine_value,
            "bm25": bm25_value,
            "query_coverage": coverage_value,
            "structural_prior": structural_value,
            "neighbor_coherence": neighbor_value,
        }
        for cosine_value, bm25_value, coverage_value, structural_value, neighbor_value in zip(
            cosine_values,
            bm25_values,
            coverage_values,
            structural_values,
            neighbor_values,
            strict=True,
        )
    ]


def _bm25_scores(query_text: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = _tokenize(query_text)
    if not query_terms or not chunks:
        return [0.0 for _ in chunks]

    documents = [_tokenize(chunk.text) for chunk in chunks]
    frequencies = [Counter(document) for document in documents]
    lengths = [len(document) for document in documents]
    avg_length = sum(lengths) / max(1, len(lengths))
    document_frequency: Counter[str] = Counter()
    for frequency in frequencies:
        document_frequency.update(frequency.keys())

    scores: list[float] = []
    total_documents = len(documents)
    for frequency, length in zip(frequencies, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            term_frequency = frequency.get(term, 0)
            if term_frequency == 0:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * length / max(1.0, avg_length))
            score += idf * (term_frequency * (k1 + 1.0)) / denominator
        scores.append(score)
    return scores


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _query_coverage(query_text: str, chunks: list[Chunk]) -> list[float]:
    query_terms = set(_tokenize(query_text))
    if not query_terms:
        return [0.0 for _ in chunks]
    scores: list[float] = []
    for chunk in chunks:
        chunk_terms = set(_tokenize(chunk.text))
        scores.append(len(query_terms & chunk_terms) / max(1, len(query_terms)))
    return scores


def _structural_prior(query_text: str, chunks: list[Chunk]) -> list[float]:
    query_terms = set(_tokenize(query_text))
    scores: list[float] = []
    for chunk in chunks:
        metadata = chunk.metadata
        score = 0.0
        content_type = metadata.get("content_type")
        if content_type == "code":
            score += 0.10
            symbol_text = " ".join(
                str(metadata.get(key, ""))
                for key in ("symbol_name", "symbol_kind", "language")
            )
            symbol_terms = set(_tokenize(symbol_text))
            if query_terms and query_terms & symbol_terms:
                score += 0.35
            header = "\n".join(chunk.text.splitlines()[:6])
            header_terms = set(_tokenize(header))
            if query_terms and query_terms & header_terms:
                score += 0.20
            if _has_dependency_hint(chunk.text, query_terms):
                score += 0.15
        section_hint = str(metadata.get("section_hint", ""))
        if section_hint and query_terms & set(_tokenize(section_hint)):
            score += 0.25
        scores.append(min(1.0, score))
    return scores


def _has_dependency_hint(text: str, query_terms: set[str]) -> bool:
    if not query_terms:
        return False
    dependency_lines = [
        line
        for line in text.splitlines()[:20]
        if re.match(r"\s*(import|from|#include|package|use)\b", line)
    ]
    if not dependency_lines:
        return False
    dependency_terms = set(_tokenize("\n".join(dependency_lines)))
    return bool(query_terms & dependency_terms)


def _neighbor_coherence(chunks: list[Chunk], embeddings: list[list[float]]) -> list[float]:
    scores: list[float] = []
    for index, chunk in enumerate(chunks):
        neighbors: list[float] = []
        if index > 0 and _same_document(chunk, chunks[index - 1]):
            neighbors.append(max(0.0, cosine(embeddings[index], embeddings[index - 1])))
        if index + 1 < len(chunks) and _same_document(chunk, chunks[index + 1]):
            neighbors.append(max(0.0, cosine(embeddings[index], embeddings[index + 1])))
        scores.append(sum(neighbors) / len(neighbors) if neighbors else 0.0)
    return scores


def _same_document(left: Chunk, right: Chunk) -> bool:
    return left.document_index == right.document_index and left.source_path == right.source_path


def _normalized_signal(values: list[float]) -> list[float]:
    if not values:
        return []
    if max(values) <= 0.0:
        return [0.0 for _ in values]
    return _minmax(values)


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
    candidate_pool: int | None = 250,
) -> None:
    ranked_indices = sorted(range(len(scored)), key=lambda index: scored[index].value, reverse=True)
    if candidate_pool is not None and candidate_pool > 0:
        ranked_indices = ranked_indices[:candidate_pool]
    stronger: list[int] = []
    for index in ranked_indices:
        best_overlap = {
            "embedding_overlap": 0.0,
            "lexical_overlap": 0.0,
            "structural_overlap": 0.0,
            "redundancy_overlap": 0.0,
        }
        for previous in stronger:
            overlap = _redundancy_overlap(
                scored[index].chunk,
                scored[previous].chunk,
                embeddings[index],
                embeddings[previous],
            )
            if overlap["redundancy_overlap"] > best_overlap["redundancy_overlap"]:
                best_overlap = overlap
        penalty = penalty_strength * best_overlap["redundancy_overlap"]
        scored[index].redundancy_penalty = penalty
        scored[index].value = max(0.0, scored[index].value * (1.0 - penalty))
        scored[index].score_components.update(best_overlap)
        scored[index].score_components["novelty"] = 1.0 - best_overlap["redundancy_overlap"]
        stronger.append(index)


def _redundancy_overlap(
    left: Chunk,
    right: Chunk,
    left_embedding: list[float],
    right_embedding: list[float],
) -> dict[str, float]:
    embedding_overlap = max(0.0, cosine(left_embedding, right_embedding))
    lexical_overlap = _lexical_overlap(left.text, right.text)
    structural_overlap = _structural_overlap(left, right)
    combined = min(
        1.0,
        0.50 * embedding_overlap + 0.35 * lexical_overlap + 0.15 * structural_overlap,
    )
    return {
        "embedding_overlap": embedding_overlap,
        "lexical_overlap": lexical_overlap,
        "structural_overlap": structural_overlap,
        "redundancy_overlap": combined,
    }


def _lexical_overlap(left: str, right: str) -> float:
    left_terms = _content_terms(left)
    right_terms = _content_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))


def _content_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _CONTENT_TOKEN_RE.finditer(text)}


def _structural_overlap(left: Chunk, right: Chunk) -> float:
    if left.source_path != right.source_path:
        return 0.0
    left_meta = left.metadata
    right_meta = right.metadata
    if left_meta.get("symbol_name") and left_meta.get("symbol_name") == right_meta.get("symbol_name"):
        return 1.0
    if left_meta.get("section_hint") and left_meta.get("section_hint") == right_meta.get("section_hint"):
        return 0.75
    if left_meta.get("content_type") == "code" and right_meta.get("content_type") == "code":
        line_overlap = _line_overlap(left_meta, right_meta)
        if line_overlap > 0:
            return line_overlap
        if left_meta.get("language") == right_meta.get("language"):
            return 0.25
    return 0.0


def _line_overlap(left_meta: dict, right_meta: dict) -> float:
    left_start = left_meta.get("start_line")
    left_end = left_meta.get("end_line")
    right_start = right_meta.get("start_line")
    right_end = right_meta.get("end_line")
    if not all(isinstance(value, int) for value in (left_start, left_end, right_start, right_end)):
        return 0.0
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    span = max(left_end, right_end) - min(left_start, right_start) + 1
    return overlap / max(1, span)
