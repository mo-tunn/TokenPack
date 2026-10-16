"""Opt-in source-contiguous evidence rendering (Phase 2).

Groups only verifiable source-adjacent chunks; headings describe their
provenance, NOT inferred cause/effect, entailment, or document relations.
The original selected texts remain exact and in source order.
"""
from __future__ import annotations

from collections.abc import Callable

from tokenpack.evidence_neighbors import source_adjacent
from tokenpack.models import ScoredChunk


def render_source_evidence_groups(items: list[ScoredChunk]) -> str:
    if not items:
        return ""
    ranked = sorted(
        items, key=lambda item: (item.chunk.document_index, item.chunk.source_path,
                                item.chunk.char_start, item.chunk.char_end, item.chunk.id)
    )
    groups: list[list[ScoredChunk]] = []
    for item in ranked:
        if groups and source_adjacent(groups[-1][-1].chunk, item.chunk):
            groups[-1].append(item)
        else:
            groups.append([item])

    parts: list[str] = []
    for number, group in enumerate(groups,1):
        first, last = group[0].chunk, group[-1].chunk
        parts.append(
            f"[Evidence Group {number}: source={first.source_path}, "
            f"document_index={first.document_index}, "
            f"paragraphs={first.start_paragraph}-{last.end_paragraph}, "
            "relation=source-contiguous]"
        )
        for item in group:
            chunk = item.chunk
            parts.append(
                f"[Evidence Chunk: id={chunk.id}, "
                f"paragraphs={chunk.start_paragraph}-{chunk.end_paragraph}, "
                f"tokens={chunk.token_count}]"
            )
            parts.append(chunk.text)
    return "\n\n".join(parts).strip()+"\n"


def fit_render_to_token_budget(
    items: list[ScoredChunk],
    *,
    render: Callable[[list[ScoredChunk]],str],
    token_counter: object,
    budget: int,
    drop_priority: Callable[[ScoredChunk], float] | None = None,
) -> tuple[str, list[ScoredChunk], int]:
    """Bound actual rendered context INCLUDING headers to the experiment cap.

    If necessary, remove lowest-valued chunks. This is deterministic and
    reported as an ablation artifact rather than silently truncating tokens.
    """
    if budget < 0:
        raise ValueError("budget must be >= 0")
    unique = {item.chunk.id for item in items}
    if len(unique) != len(items):
        raise ValueError("duplicate chunk IDs in rendering")
    priority = drop_priority or (lambda item: item.value)
    remaining = list(items)
    dropped = 0
    while remaining:
        context = render(remaining)
        if token_counter.count(context) <= budget:
            return context, remaining, dropped
        victim = min(
            remaining,
            key=lambda item: (priority(item), item.chunk.order_key, item.chunk.id)
        )
        remaining.remove(victim)
        dropped += 1
    return "",[],dropped
