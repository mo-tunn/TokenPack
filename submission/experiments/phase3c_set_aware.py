"""Research-only budget-exact set-aware selection. NO GOLD INPUTS.

Predeclared Phase 3C policies:
  render-value: select relevance-first by singleton rendered cost,
                enforce ACTUAL fully rendered budget before returning.
  set-aware:    greedy marginal relevance, query-term novelty, source overlap
                and singleton rendered-token cost, with final exact cap.

Gold evidence cannot enter selection: neither gold text, nor labels, nor
annotator groups appear in the function signature or any feature.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from typing import Callable

from tokenpack.embeddings import cosine
from tokenpack.models import ScoredChunk

TOKEN=re.compile(r"(?u)\b[^\W\d_]\w{2,}\b")
STOP=frozenset({
    "the","and","for","from","with","what","which","where","when","how",
    "this","that","have","has","were","was","are","can","does","about",
    "into","their","our","its","these","those","such","also","they",
    "paper","study","research","results","model","models",
})
METHODS=("render-value","set-aware")


def terms(s: str)->frozenset[str]:
    return frozenset(m.group(0).casefold() for m in TOKEN.finditer(s)
                     if m.group(0).casefold() not in STOP)


def relevance_values(items:list[ScoredChunk])->dict[str,float]:
    values=[float(x.value) for x in items]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite relevance score")
    low=min(values,default=0.)
    high=max(values,default=0.)
    if high==low:
        return {x.chunk.id:.5 for x in items}
    return {x.chunk.id:.1+.9*(x.value-low)/(high-low) for x in items}


@dataclass
class Selection:
    selected:list[ScoredChunk]
    context:str
    real_tokens:int
    elapsed_seconds:float
    candidates_checked:int
    policy:str
    total_query_terms:int
    covered_query_terms:int


def select_render_aware(
    scored:list[ScoredChunk],
    *,
    query:str,
    budget:int,
    counter,
    render:Callable[[list[ScoredChunk]],str],
    policy:str="set-aware",
    candidate_pool:int=300,
    singleton_cost_cache:dict[str,int]|None=None,
) -> Selection:
    """Choose whole chunks under final rendered cl100k_base cap.

    Render each candidate singleton at most once (cache shared per question).
    Greedily plan by additive singleton-cost estimates. Validate final
    source-ordered render, dropping LAST selected pieces until strictly
    within budget, then use spare budget only for exact-fit whole chunks.
    No partial chunk trimming; no gold supervision at inference.
    """
    if policy not in METHODS:
        raise ValueError("Unknown set-aware research policy")
    if budget<=0 or candidate_pool<=0:
        raise ValueError("Invalid token budget or candidate count")
    t0=time.perf_counter()
    candidates=sorted(scored,key=lambda x:(-x.value,x.chunk.order_key,x.chunk.id))[:candidate_pool]
    ids=[x.chunk.id for x in candidates]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate chunk IDs")
    normalized=relevance_values(candidates)
    query_terms=terms(query)
    item_terms={x.chunk.id:terms(x.chunk.text) for x in candidates}
    # Source-ordered rendering is deterministic and is NEVER the gold target.
    def render_selected(items):
        return render(sorted(items,key=lambda x:(x.chunk.order_key,x.chunk.id)))
    base_context=render_selected([])
    base_tokens=counter.count(base_context)
    if base_tokens>budget:
        raise ValueError("Empty template exceeds budget")
    checks=0
    costs=singleton_cost_cache if singleton_cost_cache is not None else {}
    for item in candidates:
        cid=item.chunk.id
        if cid not in costs:
            costs[cid]=max(1,counter.count(render_selected([item]))-base_tokens)
            checks+=1
    selected=[]
    seen=set()
    covered=frozenset()
    estimated_used=base_tokens
    context=base_context
    used=base_tokens
    # Cache best overlap with the current selected set, updated just once
    # per newly chosen chunk. Avoid cubic repeated embedding comparisons
    # on long QASPER papers with hundreds of candidate chunks.
    max_lex={x.chunk.id:0.0 for x in candidates}
    max_sem={x.chunk.id:0.0 for x in candidates}
    def update_overlap(chosen):
        if policy!="set-aware":return
        cid=chosen.chunk.id
        ts=item_terms[cid]
        for other in candidates:
            oid=other.chunk.id
            if oid in seen:continue
            lexical=len(ts&item_terms[oid])/max(1,len(ts|item_terms[oid]))
            semantic=max(0.,cosine(chosen.embedding or [],other.embedding or []))
            max_lex[oid]=max(max_lex[oid],lexical)
            max_sem[oid]=max(max_sem[oid],semantic)

    def priorities_for(selected,seen,covered):
        priorities=[]
        for item in candidates:
            cid=item.chunk.id
            if cid in seen:continue
            if policy=="render-value":
                utility=normalized[cid]
            else:
                lexical=max_lex[cid]
                semantic=max_sem[cid]
                new_query=len((item_terms[cid]&query_terms)-covered)/max(1,len(query_terms))
                novelty=max(.15,1-.50*lexical-.15*semantic)
                utility=(normalized[cid]*(.78+.22*novelty)+.22*new_query)/max(1,costs[cid])**.30
            priorities.append((-utility,costs[cid],item.chunk.order_key,cid,item))
        priorities.sort(key=lambda x:x[:4])
        return priorities

    while len(seen)<len(candidates):
        chosen=next(
            (item for _,cost,_,_,item in priorities_for(selected,seen,covered)
             if estimated_used+cost<=budget),
            None,
        )
        if chosen is None:break
        selected.append(chosen)
        seen.add(chosen.chunk.id)
        covered=covered|(item_terms[chosen.chunk.id]&query_terms)
        estimated_used+=costs[chosen.chunk.id]
        update_overlap(chosen)
    context=render_selected(selected)
    used=counter.count(context)
    checks+=1
    # Tokenizer costs are not strictly additive at adjoining segment edges.
    while used>budget and selected:
        removed=selected.pop()
        seen.remove(removed.chunk.id)
        covered=frozenset().union(*(
            item_terms[x.chunk.id]&query_terms for x in selected))
        context=render_selected(selected)
        used=counter.count(context)
        checks+=1
    if policy=="set-aware" and len(seen)<len(candidates):
        max_lex={x.chunk.id:0.0 for x in candidates}
        max_sem={x.chunk.id:0.0 for x in candidates}
        for chosen in selected:
            update_overlap(chosen)
    # Exact-budget augmentation attempts each remaining chunk at most once,
    # in current greedy priority order.
    for _,_,_,_,item in priorities_for(selected,seen,covered):
        candidate_context=render_selected([*selected,item])
        next_tokens=counter.count(candidate_context)
        checks+=1
        if next_tokens<=budget:
            selected.append(item)
            seen.add(item.chunk.id)
            covered=covered|(item_terms[item.chunk.id]&query_terms)
            context,used=candidate_context,next_tokens
    if counter.count(context)>budget:
        raise AssertionError("Final render exceeds real token budget")
    return Selection(
        selected=sorted(selected,key=lambda x:(x.chunk.order_key,x.chunk.id)),
        context=context,real_tokens=used,
        elapsed_seconds=time.perf_counter()-t0,
        candidates_checked=checks,policy=policy,
        total_query_terms=len(query_terms),
        covered_query_terms=len(covered),
    )
