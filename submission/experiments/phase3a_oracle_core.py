"""Gold-aware diagnostic ONLY: exact small-instance / heuristic large-instance oracle.

Never import this into production retrieval, scoring, selectors or packing.
Gold is used in this diagnostic to establish an attainable performance
frontier; gold-derived outcomes must never be reported as blind predictions.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import inf
from typing import Callable

from submission.experiments.evidence_failure_audit import tokens
from tokenpack.models import ScoredChunk


def gram_counts(s: str, n: int = 5) -> Counter:
    words=tokens(s)
    return Counter(tuple(words[i:i+n]) for i in range(max(0,len(words)-n+1)))


def gold_counts(spans: list[str]) -> tuple[Counter,list[Counter]]:
    parts=[gram_counts(x) for x in spans if isinstance(x,str) and x.strip()]
    return sum(parts,Counter()),parts


def matched_count(available: Counter, gold: Counter) -> int:
    return sum(min(q,available.get(term,0)) for term,q in gold.items())


def gold_objective(selected, gold: Counter, span_grams: list[Counter],
                   *, per_chunk: dict[str,Counter]) -> tuple[int,int]:
    count=Counter()
    for item in selected:
        count.update(per_chunk[item.chunk.id])
    total=matched_count(count,gold)
    span_complete=sum(
        bool(span) and matched_count(count,span)>=.5*sum(span.values())
        for span in span_grams
    )
    return total,span_complete


def oracle_search(
    candidates: list[ScoredChunk],
    *,
    gold: Counter,
    span_grams: list[Counter],
    budget: int,
    render: Callable[[list[ScoredChunk]],str],
    counter: object,
    exact_max_relevant: int = 12,
) -> dict:
    """Search positive-overlap candidates with exact final rendered-token checks.

    For at most exact_max_relevant positive-overlap chunks, exhaustively
    enumerates all subsets, giving an *exact* optimum for the declared
    chunk-internal 5-gram multiset proxy at the actual rendered-token cap.
    Larger cases use deterministic, multiple-start greedy and one-for-one
    exchange; feasible result is NOT a certified upper bound.

    Independently finds two optima: 5gram-first and span-count-first.
    """
    if budget<=0:
        raise ValueError("Budget must be positive")
    ids=[x.chunk.id for x in candidates]
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate candidate chunk id")
    per_chunk={item.chunk.id:gram_counts(item.chunk.text) for item in candidates}
    relevant=[x for x in candidates if matched_count(per_chunk[x.chunk.id],gold)>0]
    def evaluate(selection):
        txt=render(selection)
        if counter.count(txt)>budget:
            return None
        hits,spans=gold_objective(selection,gold,span_grams,per_chunk=per_chunk)
        return (hits,spans,counter.count(txt))
    winners={}
    for objective in ("5gram","joint"):
        # All candidates irrelevant to gold are safely ignored in this
        # diagnostic: they cannot improve chunk-internal lexical coverage.
        winner=[]
        best=(0,0,0)
        checked=0
        def key(stats):
            hits,spans,used=stats
            return (hits,spans,-used) if objective=="5gram" else (spans,hits,-used)
        if len(relevant)<=exact_max_relevant:
            for n in range(len(relevant)+1):
                for subset in combinations(relevant,n):
                    checked+=1
                    stats=evaluate(subset)
                    if stats is not None and key(stats)>key(best):
                        best=stats
                        winner=list(subset)
            optimality="exact_for_chunk_internal_lexical_proxy"
        else:
            # Multiple deterministic greedy modes, followed by swaps to get
            # valid lower bounds without falsely claiming an exact ceiling.
            for exponent in (0,0.5,1.0):
                selected=[]
                remaining=list(relevant)
                while remaining:
                    old_hits,old_spans=gold_objective(
                        selected,gold,span_grams,per_chunk=per_chunk
                    )
                    choices=[]
                    for item in remaining:
                        trial=[*selected,item]
                        stats=evaluate(trial)
                        checked+=1
                        if stats is None: continue
                        hits,spans,used=stats
                        dh,ds=hits-old_hits,spans-old_spans
                        gain=(dh+ds*.001) if objective=="5gram" else (ds*100000+dh)
                        if gain<=0:continue
                        priority=gain/max(1,item.weight)**exponent
                        choices.append((-priority,item.chunk.order_key,item.chunk.id,item,stats))
                    if not choices:break
                    choices.sort(key=lambda a:(a[0],a[1],a[2]))
                    chosen=choices[0][3]
                    selected.append(chosen)
                    remaining.remove(chosen)
                stats=evaluate(selected)
                if stats is not None and key(stats)>key(best):
                    best=stats
                    winner=selected
            # Strictly-improving 1-for-1 swaps, never exceeding token cap.
            for _ in range(3):
                improved=False
                selection_ids={x.chunk.id for x in winner}
                for old in winner:
                    for candidate in relevant:
                        if candidate.chunk.id in selection_ids: continue
                        trial=[x for x in winner if x.chunk.id!=old.chunk.id]+[candidate]
                        stats=evaluate(trial)
                        checked+=1
                        if stats is not None and key(stats)>key(best):
                            best=stats;winner=trial;improved=True;break
                    if improved:break
                if not improved:break
            optimality="heuristic_feasible_lower_bound"
        hits,spans,used=best
        winners[objective]={
            "gold_5gram_hits":hits,
            "gold_5gram_recall":hits/max(1,sum(gold.values())),
            "completed_spans":spans,
            "total_spans":len(span_grams),
            "all_spans_present":len(span_grams)>0 and spans==len(span_grams),
            "rendered_tokens":used,
            "selected_chunks":len(winner),
            "selected_ids":sorted(x.chunk.id for x in winner),
            "mode":optimality,
            "candidate_subsets_checked":checked,
        }
    return {
        "eligible_relevant_chunks":len(relevant),
        "oracle_5gram":winners["5gram"],
        "oracle_joint":winners["joint"],
    }


def pool_coverage(candidates: list[ScoredChunk], gold: Counter,
                  span_grams: list[Counter]) -> dict:
    # Capacity-free representation diagnostic. NOT a feasible-budget result.
    per_chunk={x.chunk.id:gram_counts(x.chunk.text) for x in candidates}
    hits,spans=gold_objective(candidates,gold,span_grams,per_chunk=per_chunk)
    return {"gold_5gram_hits":hits,
            "gold_5gram_recall":hits/max(1,sum(gold.values())),
            "completed_spans":spans,
            "total_spans":len(span_grams),
            "all_spans_present":len(span_grams)>0 and spans==len(span_grams)}
