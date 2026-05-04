from __future__ import annotations

import math
import time

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
) -> SelectionResult:
    started = time.perf_counter()
    filtered = _filtered_candidates(scored, relevance_threshold)
    candidates = _candidate_pool(filtered, candidate_pool)
    if strategy == "document-prefix":
        selected = _document_prefix(filtered, budget)
    elif strategy == "full-document":
        selected = _full_document(filtered)
    elif strategy == "top-k":
        selected = _top_k(candidates)
    elif strategy == "budget-top-k":
        selected = _budget_top_k(candidates, budget)
    elif strategy == "mmr":
        selected = _mmr(candidates, budget, embeddings=embeddings, mmr_lambda=mmr_lambda)
    elif strategy in {"knapsack", "knapsack-redundancy"}:
        selected = _knapsack(candidates, budget, token_granularity=token_granularity)
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
    selected: list[ScoredChunk] = []
    used = 0
    for item in candidates:
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

