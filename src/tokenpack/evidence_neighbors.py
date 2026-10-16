"""Opt-in, deterministic, source-adjacent evidence completion (Phase 2).

This is a cheap *structural* neighbor baseline, not semantic evidence-chain
inference. It never uses answers, gold spans, inference models or network IO.

The unchanged budget-top-k selector is the seed. A bounded part of the
available budget is reserved for immediate source-adjacent supporting chunks.
Unused budget is filled with the same existing greedy-value ranking.

Selected content remains verbatim and never exceeds the chunk-token budget.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time

from tokenpack.models import Chunk, ScoredChunk, SelectionResult
from tokenpack.selectors import select_chunks

MAX_ADJACENCY_GAP_CHARS = 32


def source_adjacent(left: Chunk, right: Chunk, *, max_gap_chars: int = MAX_ADJACENCY_GAP_CHARS) -> bool:
    """True only for physical source neighbours with no inferred cross-doc edge."""
    if (left.source_path != right.source_path or
            left.document_index != right.document_index):
        return False
    a = left.metadata.get("section_hint")
    b = right.metadata.get("section_hint")
    if a and b and a != b:
        return False
    if not (0 <= left.char_start <= left.char_end <= right.char_start <= right.char_end):
        return False
    return right.char_start - left.char_end <= max_gap_chars


def select_neighbor_augmented(
    scored: list[ScoredChunk],
    budget: int,
    *,
    candidate_pool: int = 300,
    neighbor_fraction: float = 0.15,
    max_gap_chars: int = MAX_ADJACENCY_GAP_CHARS,
) -> SelectionResult:
    """Reserve at most neighbor_fraction of the total budget for adjacent chunks.

    The seed portion uses the exact existing budget-top-k selector. Physical
    immediate neighbours may come from *outside* the top candidate pool, but
    are never selected across provenance/section boundaries. Afterwards any
    unused space returns to ordinary top-value filling.

    This is not an optimal group knapsack and makes no semantic entailment
    claim. All deterministic ties are resolved by source order/chunk ID.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")
    if not 0 <= neighbor_fraction < 1:
        raise ValueError("neighbor_fraction must be in [0, 1)")
    if max_gap_chars < 0:
        raise ValueError("max_gap_chars must be non-negative")
    start = time.perf_counter()
    available = [x for x in scored if 0 < x.weight <= budget and x.value >= 0]
    ids = [x.chunk.id for x in available]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scored chunk IDs")
    reserve = math.floor(budget * neighbor_fraction)
    seeds = select_chunks(
        available,
        strategy="budget-top-k",
        budget=budget - reserve,
        candidate_pool=candidate_pool,
    ).selected
    selected: dict[str, ScoredChunk] = {item.chunk.id: item for item in seeds}
    used = sum(item.weight for item in selected.values())

    groups: dict[tuple[int, str], list[ScoredChunk]] = defaultdict(list)
    for item in available:
        c = item.chunk
        groups[(c.document_index, c.source_path)].append(item)

    adjacency: dict[str, list[str]] = defaultdict(list)
    by_id = {item.chunk.id: item for item in available}
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (item.chunk.char_start, item.chunk.char_end, item.chunk.id))
        for left, right in zip(ordered, ordered[1:]):
            if source_adjacent(left.chunk, right.chunk, max_gap_chars=max_gap_chars):
                adjacency[left.chunk.id].append(right.chunk.id)
                adjacency[right.chunk.id].append(left.chunk.id)

    proposals: dict[str, float] = {}
    for seed in seeds:
        for neighbor_id in adjacency[seed.chunk.id]:
            if neighbor_id in selected:
                continue
            neighbor = by_id[neighbor_id]
            # Link to a high-ranked seed; prefer a useful neighbor if ties.
            utility = (max(0.0, seed.value) + max(0.0, neighbor.value) * 0.5) / math.sqrt(neighbor.weight)
            proposals[neighbor_id] = max(proposals.get(neighbor_id, -math.inf), utility)

    expanded = 0
    extra_used = 0
    for neighbor_id in sorted(
        proposals,
        key=lambda k: (-proposals[k], by_id[k].chunk.order_key, k),
    ):
        item = by_id[neighbor_id]
        if extra_used + item.weight > reserve or used + item.weight > budget:
            continue
        selected[neighbor_id] = item
        extra_used += item.weight
        used += item.weight
        expanded += 1

    remaining = [item for item in available if item.chunk.id not in selected]
    if remaining and used < budget:
        filler = select_chunks(
            remaining,
            strategy="budget-top-k",
            budget=budget - used,
            candidate_pool=candidate_pool,
        )
        for item in filler.selected:
            selected[item.chunk.id] = item
            used += item.weight

    out = sorted(selected.values(), key=lambda item: item.chunk.order_key)
    if used > budget:
        raise AssertionError("neighbor selector exceeded chunk-token budget")
    return SelectionResult(
        strategy="neighbor-augment",
        budget=budget,
        used_tokens=used,
        total_value=sum(item.value for item in out),
        selected=out,
        elapsed_seconds=time.perf_counter()-start,
    )
