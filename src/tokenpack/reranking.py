from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from tokenpack.models import ScoredChunk


class Reranker(Protocol):
    def score(self, query: str, items: list[ScoredChunk]) -> list[float]:
        """Return one relevance score for each scored chunk."""


class CrossEncoderReranker:
    """Thin adapter around sentence-transformers cross encoders."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install tokenpack[reranking] or add sentence-transformers>=3.0.0."
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device, trust_remote_code=True, local_files_only=local_files_only)

    def score(self, query: str, items: list[ScoredChunk]) -> list[float]:
        if not items:
            return []
        pairs = [(query, item.chunk.text) for item in items]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(score) for score in scores]


def apply_reranker(
    scored: list[ScoredChunk],
    *,
    query: str,
    reranker: Reranker,
    candidate_pool: int = 80,
    weight: float = 0.35,
) -> list[ScoredChunk]:
    candidates = sorted(scored, key=lambda item: item.value, reverse=True)[: max(0, candidate_pool)]
    reranker_scores = reranker.score(query, candidates)
    return blend_reranker_scores(scored, candidates, reranker_scores, weight=weight)


def blend_reranker_scores(
    scored: list[ScoredChunk],
    reranked_items: list[ScoredChunk],
    reranker_scores: list[float],
    *,
    weight: float = 0.35,
) -> list[ScoredChunk]:
    """Blend base TokenPack scores with normalized reranker scores for candidate items."""

    if not reranked_items or not reranker_scores:
        return list(scored)
    clipped_weight = min(1.0, max(0.0, weight))
    normalized = _minmax(reranker_scores)
    by_id = {
        item.chunk.id: (float(raw_score), float(norm_score))
        for item, raw_score, norm_score in zip(reranked_items, reranker_scores, normalized)
    }
    blended: list[ScoredChunk] = []
    for item in scored:
        scores = by_id.get(item.chunk.id)
        if scores is None:
            blended.append(item)
            continue
        raw_score, norm_score = scores
        value = (1.0 - clipped_weight) * item.value + clipped_weight * norm_score
        blended.append(
            replace(
                item,
                value=max(0.0, value),
                score_components={
                    **item.score_components,
                    "base_value": item.value,
                    "reranker_score": raw_score,
                    "reranker_norm": norm_score,
                    "reranker_weight": clipped_weight,
                },
            )
        )
    return blended


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]
