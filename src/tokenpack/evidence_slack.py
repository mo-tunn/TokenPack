"""Opt-in, zero-displacement evidence augmentation using only spare render budget.

Phase 2b exploratory baseline.  It never reads answer labels or gold evidence.
Unlike reserved-budget neighbors, the originally rendered chunks can NEVER be
evicted.  Includes a non-neighbor slack fill control to distinguish proximity
from just using more of the same budget.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Callable

from tokenpack.evidence_neighbors import source_adjacent
from tokenpack.models import ScoredChunk

_TERMS = re.compile(r"(?u)\b[^\W\d_]\w{2,}\b")
_STOP = {
    "the", "and", "for", "are", "was", "were", "from", "with", "this",
    "that", "which", "what", "when", "where", "does", "how", "can",
    "has", "have", "been", "into", "between", "their", "about",
}


def terms(text: str) -> set[str]:
    return {x.group().casefold() for x in _TERMS.finditer(text)
            if x.group().casefold() not in _STOP}


def fill_render_slack(
    scored: list[ScoredChunk],
    base_kept: list[ScoredChunk],
    *,
    question: str,
    budget: int,
    render: Callable[[list[ScoredChunk]], str],
    counter: object,
    only_neighbors: bool,
    relative_neighbor_score: float = 0.40,
) -> tuple[str, list[ScoredChunk], dict]:
    """Use spare tokens while preserving EVERY already-rendered original chunk.

    Selection is deterministic and gold-blind. A neighbor must actually touch
    one selected chunk, share query terms, and have reasonable hybrid relevance
    compared to its adjacent seed. The other arm fills slack by global score.
    Returns the exact context with no truncated chunks or removed baseline.
    """
    if budget < 0 or not 0 <= relative_neighbor_score <= 1:
        raise ValueError("Invalid budget or threshold")
    base_ids = [x.chunk.id for x in base_kept]
    candidate_ids = [x.chunk.id for x in scored]
    if len(set(base_ids)) != len(base_ids) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate chunk IDs")
    selected = list(base_kept)
    by_id = {x.chunk.id: x for x in scored}
    if not set(base_ids) <= set(by_id):
        raise ValueError("Seed absent from scored pool")
    rendered = render(selected)
    initial_tokens = counter.count(rendered)
    if initial_tokens > budget:
        raise ValueError("Seed already exceeds rendered budget")
    source_query = terms(question)
    in_set = set(base_ids)
    by_source = defaultdict(list)
    for item in scored:
        by_source[(item.chunk.source_path, item.chunk.document_index)].append(item)
    proximity = defaultdict(set)
    for group in by_source.values():
        ordered = sorted(group,key=lambda x:(x.chunk.char_start,x.chunk.char_end,x.chunk.id))
        for left,right in zip(ordered,ordered[1:]):
            if source_adjacent(left.chunk,right.chunk):
                proximity[left.chunk.id].add(right.chunk.id)
                proximity[right.chunk.id].add(left.chunk.id)
    # Only score currently addable candidates, not reserved replacements.
    attempts = 0
    accepted = []
    while True:
        selected_query = set()
        for item in selected:
            selected_query.update(terms(item.chunk.text) & source_query)
        choices=[]
        for item in scored:
            cid = item.chunk.id
            if cid in in_set or item.weight <= 0 or item.value <= 0:
                continue
            adjacent_seeds = [by_id[n] for n in proximity[cid] if n in in_set]
            item_query = terms(item.chunk.text) & source_query
            novelty = len(item_query - selected_query) / max(1,len(source_query))
            if only_neighbors:
                if not adjacent_seeds or not item_query:
                    continue
                if item.value < relative_neighbor_score * max(x.value for x in adjacent_seeds):
                    continue
                priority = (item.value + .20*novelty) / math.sqrt(max(1,item.weight))
            else:
                priority = item.value / math.sqrt(max(1,item.weight))
            choices.append((priority,item.chunk.order_key,cid,item))
        # Descending relevance, stable source-index tie-breaks.
        choices.sort(key=lambda row:(-row[0],row[1],row[2]))
        added = None
        for _,_,cid,item in choices:
            attempts += 1
            trial = [*selected,item]
            trial_context = render(trial)
            if counter.count(trial_context) <= budget:
                selected = trial
                rendered = trial_context
                in_set.add(cid)
                accepted.append(cid)
                added = item
                break
        if added is None:
            break
    if not set(base_ids) <= in_set:
        raise AssertionError("Baseline displaced")
    used = counter.count(rendered)
    if used > budget:
        raise AssertionError("Rendered token budget exceeded")
    return rendered,selected,{
        "added_chunks":len(accepted),
        "added_chunk_ids":accepted,
        "displaced_baseline_chunks":0,
        "source_tokens_used_before":initial_tokens,
        "rendered_tokens":used,
        "candidate_checks":attempts,
        "only_neighbors":only_neighbors,
    }
