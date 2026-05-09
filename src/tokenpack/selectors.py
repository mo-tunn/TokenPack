from __future__ import annotations

import math
import re
import time
from dataclasses import replace

from tokenpack.embeddings import cosine
from tokenpack.models import ScoredChunk, SelectionResult


def select_chunks(
    scored: list[ScoredChunk],
    strategy: str,
    budget: int,
    candidate_pool: int = 250,
    relevance_threshold: float = 0.0,
    mmr_lambda: float = 0.75,
    token_granularity: int = 1,
    embeddings: list[list[float]] | None = None,
    coverage_query: str | None = None,
) -> SelectionResult:
    started = time.perf_counter()
    filtered = _filtered_candidates(scored, relevance_threshold)
    if strategy == "document-prefix":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _document_prefix(filtered, budget)
    elif strategy == "full-document":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _full_document(filtered)
    elif strategy == "top-k":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _top_k(candidates)
    elif strategy == "budget-top-k":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _budget_top_k(candidates, budget)
    elif strategy == "greedy-value":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _greedy_value(candidates, budget)
    elif strategy == "greedy-density":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _greedy_density(candidates, budget)
    elif strategy == "mmr":
        candidates = _candidate_pool(filtered, candidate_pool)
        selected = _mmr(candidates, budget, embeddings=embeddings, mmr_lambda=mmr_lambda)
    elif strategy == "knapsack":
        candidates = _knapsack_candidate_pool(filtered, candidate_pool)
        selected = _knapsack(candidates, budget, token_granularity=token_granularity)
    elif strategy == "knapsack-redundancy":
        candidates = _knapsack_candidate_pool(filtered, candidate_pool)
        selected = _knapsack(
            _redundancy_adjusted_candidates(candidates, penalty_strength=1.0 - mmr_lambda),
            budget,
            token_granularity=token_granularity,
        )
    elif strategy == "knapsack-coverage":
        candidates = _knapsack_candidate_pool(filtered, candidate_pool)
        selected = _coverage_greedy(
            _redundancy_adjusted_candidates(candidates, penalty_strength=1.0 - mmr_lambda),
            budget,
            coverage_query=coverage_query,
        )
    elif strategy == "knapsack-augment":
        value_candidates = _candidate_pool(filtered, candidate_pool)
        seed = _budget_top_k(value_candidates, budget)
        used = sum(item.weight for item in seed)
        seed_ids = {item.chunk.id for item in seed}
        remaining = [
            item
            for item in _knapsack_candidate_pool(filtered, candidate_pool)
            if item.chunk.id not in seed_ids
        ]
        selected = [*seed, *_knapsack(remaining, budget - used, token_granularity=token_granularity)]
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    selected = sorted(selected, key=lambda item: item.chunk.order_key)
    elapsed = time.perf_counter() - started
    return SelectionResult(
        strategy=strategy,
        budget=budget,
        used_tokens=sum(item.weight for item in selected),
        total_value=sum(item.value for item in selected),
        selected=selected,
        elapsed_seconds=elapsed,
    )


def _filtered_candidates(scored: list[ScoredChunk], threshold: float) -> list[ScoredChunk]:
    return [item for item in scored if item.value >= threshold and item.weight > 0]


def _candidate_pool(scored: list[ScoredChunk], candidate_pool: int) -> list[ScoredChunk]:
    return sorted(scored, key=lambda item: item.value, reverse=True)[:candidate_pool]


def _knapsack_candidate_pool(scored: list[ScoredChunk], candidate_pool: int) -> list[ScoredChunk]:
    value_pool = _candidate_pool(scored, candidate_pool)
    density_pool = sorted(
        scored,
        key=lambda item: item.value / math.sqrt(max(1, item.weight)),
        reverse=True,
    )[:candidate_pool]
    by_id: dict[str, ScoredChunk] = {}
    for item in [*value_pool, *density_pool]:
        by_id.setdefault(item.chunk.id, item)
    return sorted(by_id.values(), key=lambda item: item.value, reverse=True)


def _redundancy_adjusted_candidates(
    candidates: list[ScoredChunk],
    penalty_strength: float,
) -> list[ScoredChunk]:
    ranked = sorted(candidates, key=lambda item: item.value, reverse=True)
    adjusted: list[ScoredChunk] = []
    stronger: list[ScoredChunk] = []
    strength = min(1.0, max(0.0, penalty_strength))
    for item in ranked:
        overlap = 0.0
        if item.embedding is not None and stronger:
            overlap = max(
                max(0.0, cosine(item.embedding, previous.embedding or []))
                for previous in stronger
            )
        penalty = strength * overlap
        clone = replace(
            item,
            value=max(0.0, item.value * (1.0 - penalty)),
            redundancy_penalty=penalty,
            score_components={
                **item.score_components,
                "selector_redundancy_overlap": overlap,
                "selector_redundancy_penalty": penalty,
                "selector_novelty": 1.0 - overlap,
            },
        )
        adjusted.append(clone)
        stronger.append(item)
    return sorted(adjusted, key=lambda item: item.value, reverse=True)


def _document_prefix(candidates: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
    selected: list[ScoredChunk] = []
    used = 0
    for item in sorted(candidates, key=lambda item: item.chunk.order_key):
        if used + item.weight > budget:
            break
        selected.append(item)
        used += item.weight
    return selected


def _full_document(candidates: list[ScoredChunk]) -> list[ScoredChunk]:
    return sorted(candidates, key=lambda item: item.chunk.order_key)


def _top_k(candidates: list[ScoredChunk], k: int = 5) -> list[ScoredChunk]:
    return candidates[:k]


def _budget_top_k(candidates: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
    return _greedy_by(candidates, budget, key=lambda item: item.value)


def _greedy_value(candidates: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
    return _greedy_by(candidates, budget, key=lambda item: item.value)


def _greedy_density(candidates: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
    return _greedy_by(candidates, budget, key=lambda item: item.value / max(1, item.weight))


def _greedy_by(candidates: list[ScoredChunk], budget: int, key) -> list[ScoredChunk]:
    selected: list[ScoredChunk] = []
    used = 0
    for item in sorted(candidates, key=key, reverse=True):
        if used + item.weight <= budget:
            selected.append(item)
            used += item.weight
    return selected


def _mmr(
    candidates: list[ScoredChunk],
    budget: int,
    embeddings: list[list[float]] | None,
    mmr_lambda: float,
) -> list[ScoredChunk]:
    if embeddings is None and not any(item.embedding is not None for item in candidates):
        return _budget_top_k(candidates, budget)

    selected: list[ScoredChunk] = []
    remaining = list(candidates)
    used = 0
    while remaining:
        best_item: ScoredChunk | None = None
        best_score = -math.inf
        for item in remaining:
            if used + item.weight > budget:
                continue
            novelty_penalty = 0.0
            item_embedding = item.embedding
            if item_embedding is not None and selected:
                novelty_penalty = max(
                    cosine(item_embedding, chosen.embedding or []) for chosen in selected
                )
            mmr_score = mmr_lambda * item.value - (1.0 - mmr_lambda) * novelty_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
        if best_item is None:
            break
        selected.append(best_item)
        used += best_item.weight
        remaining.remove(best_item)
    return selected


def _knapsack(
    candidates: list[ScoredChunk],
    budget: int,
    token_granularity: int = 1,
) -> list[ScoredChunk]:
    if budget <= 0 or not candidates:
        return []
    granularity = max(1, token_granularity)
    if len(candidates) * max(1, budget // granularity) > 10_000_000:
        granularity = max(granularity, math.ceil(len(candidates) * budget / 10_000_000))
    capacity = budget // granularity
    weights = [math.ceil(item.weight / granularity) for item in candidates]
    dp = [0.0] * (capacity + 1)
    keep = [bytearray(capacity + 1) for _ in candidates]

    for item_index, item in enumerate(candidates):
        weight = weights[item_index]
        if weight > capacity:
            continue
        for token_budget in range(capacity, weight - 1, -1):
            candidate_value = dp[token_budget - weight] + item.value
            if candidate_value > dp[token_budget]:
                dp[token_budget] = candidate_value
                keep[item_index][token_budget] = 1

    selected: list[ScoredChunk] = []
    token_budget = max(range(capacity + 1), key=lambda index: dp[index])
    for item_index in range(len(candidates) - 1, -1, -1):
        if keep[item_index][token_budget]:
            selected.append(candidates[item_index])
            token_budget -= weights[item_index]
    return list(reversed(selected))


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "both",
    "can",
    "does",
    "from",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "not",
    "only",
    "our",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}


def _coverage_greedy(
    candidates: list[ScoredChunk],
    budget: int,
    *,
    coverage_query: str | None,
) -> list[ScoredChunk]:
    if budget <= 0 or not candidates:
        return []
    query_terms = _content_terms(coverage_query or "")
    selected: list[ScoredChunk] = []
    selected_ids: set[str] = set()
    covered_terms: set[str] = set()
    used = 0

    while True:
        best_item: ScoredChunk | None = None
        best_score = -math.inf
        for item in candidates:
            if item.chunk.id in selected_ids or used + item.weight > budget:
                continue
            item_terms = _content_terms(item.chunk.text)
            new_terms = item_terms & query_terms - covered_terms
            query_bonus = len(new_terms) / max(1, len(query_terms)) if query_terms else 0.0
            support_bonus = max(
                float(item.score_components.get("query_coverage", 0.0)),
                float(item.score_components.get("support_likelihood", 0.0)),
            )
            novelty = float(item.score_components.get("selector_novelty", 1.0))
            adjusted_value = item.value * (0.80 + 0.20 * novelty) + 0.20 * query_bonus + 0.05 * support_bonus
            score = adjusted_value / math.sqrt(max(1, item.weight))
            if score > best_score:
                best_score = score
                best_item = replace(
                    item,
                    value=max(0.0, adjusted_value),
                    score_components={
                        **item.score_components,
                        "selector_query_coverage_bonus": query_bonus,
                        "selector_support_bonus": support_bonus,
                    },
                )
        if best_item is None:
            break
        selected.append(best_item)
        selected_ids.add(best_item.chunk.id)
        covered_terms.update(_content_terms(best_item.chunk.text) & query_terms)
        used += best_item.weight
    return selected


def _content_terms(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _TOKEN_RE.finditer(text)
        if match.group(0).lower() not in _STOPWORDS
    }

