"""Gold-aware Phase 4A diagnostic; NEVER import in online inference.

One QASPER annotator evidence list is an alternative, not a union requirement.
Objective = chunk-INTERNAL 5-gram multiset recall; every chosen subset must
respect the TRUE, source-ordered, cl100k_base rendered context token cap.

If at most 9 relevant chunks exist: exact exhaustive objective optimum.
For larger cases: feasible multi-start greedy LOWER bound, not a ceiling.
The unconstrained candidate-pool evidence recall is an UPPER bound for this
chunk-internal objective regardless of token budget.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from dataclasses import dataclass

from submission.experiments.phase3a_oracle_core import (
    gram_counts,gold_counts,matched_count,
)


def as_grams(spans):
    gold,_=gold_counts(spans)
    return gold


def objective(chunks,selected_indices,gold,precomputed=None):
    counts=Counter()
    if precomputed is None:
        precomputed=[gram_counts(x.chunk.text) for x in chunks]
    for i in selected_indices:
        counts.update(precomputed[i])
    return matched_count(counts,gold)


@dataclass(frozen=True)
class OracleResult:
    gold_total:int
    pool_hits:int
    achieved_hits:int
    chosen_indices:tuple[int,...]
    actual_render_tokens:int
    relevant_chunks:int
    mode:str
    checks:int

    @property
    def certified_optimal(self)->bool:
        return self.mode in ("exact","pool_ceiling_attained")

    @property
    def recall(self)->float:
        return self.achieved_hits/max(1,self.gold_total)

    @property
    def pool_upper_recall(self)->float:
        return self.pool_hits/max(1,self.gold_total)


def solve_group(candidates,gold,budget,counter,render,
                *,max_exact:int=9,precomputed=None)->OracleResult:
    if budget<=0:raise ValueError("Positive token limit required")
    denom=sum(gold.values())
    if not denom:raise ValueError("At least one evidence 5gram required")
    grams=precomputed if precomputed is not None else [
        gram_counts(x.chunk.text) for x in candidates]
    relevant=[i for i,g in enumerate(grams) if matched_count(g,gold)>0]
    if len(set(x.chunk.id for x in candidates))!=len(candidates):
        raise ValueError("Duplicate candidate chunk IDs")
    counts=Counter()
    for i in relevant:counts.update(grams[i])
    pool=matched_count(counts,gold)
    if not relevant:
        return OracleResult(denom,0,0,(),0,0,"exact",0)
    best_hits=0
    best_chosen=()
    best_tokens=0
    checks=0
    def feasible(indices):
        nonlocal checks
        text=render([candidates[i] for i in sorted(indices,
                   key=lambda i:(candidates[i].chunk.order_key,
                                 candidates[i].chunk.id))])
        used=counter.count(text)
        checks+=1
        return used if used<=budget else None
    def adopt(indices,used):
        nonlocal best_hits,best_chosen,best_tokens
        hits=objective(candidates,indices,gold,grams)
        if (hits,-used)>(best_hits,-best_tokens):
            best_hits=hits
            best_chosen=tuple(indices)
            best_tokens=used
        return hits
    # If all positive candidates fit, full representational ceiling is proven
    # achievable. No exponential enumeration needed.
    full_used=feasible(relevant)
    if full_used is not None:
        return OracleResult(denom,pool,pool,tuple(relevant),full_used,
                            len(relevant),"pool_ceiling_attained",checks)
    # Singletons: cheap initial attainable solution, also prunes too-big ones.
    feasible_single={}
    for i in relevant:
        used=feasible([i])
        if used is not None:
            feasible_single[i]=used
            adopt([i],used)
    if len(relevant)<=max_exact:
        for size in range(2,len(relevant)):
            for indices in combinations(relevant,size):
                if any(i not in feasible_single for i in indices):
                    continue
                used=feasible(indices)
                if used is not None:
                    adopt(indices,used)
        return OracleResult(denom,pool,best_hits,best_chosen,best_tokens,
                            len(relevant),"exact",checks)
    # Feasible deterministic heuristic lower bound, NOT exact if it falls
    # short of pool ceiling. No gold leaks into score/real selector.
    for exponent in (0.,.45,.85,1.2):
        chosen=[]
        remaining=list(feasible_single)
        cum=Counter()
        while remaining:
            current=matched_count(cum,gold)
            options=[]
            for i in remaining:
                new=cum+grams[i]
                gain=matched_count(new,gold)-current
                if gain<=0:continue
                weight=max(1,candidates[i].weight)
                options.append((-(gain/(weight**exponent)),
                                weight,candidates[i].chunk.order_key,i))
            options.sort()
            added=False
            for *_,i in options:
                trial=[*chosen,i]
                used=feasible(trial)
                if used is None:continue
                chosen=trial
                cum.update(grams[i])
                remaining.remove(i)
                adopt(chosen,used)
                added=True
                break
            if not added:break
            if best_hits==pool:
                return OracleResult(denom,pool,best_hits,best_chosen,best_tokens,
                                    len(relevant),"pool_ceiling_attained",checks)
    return OracleResult(denom,pool,best_hits,best_chosen,best_tokens,
                        len(relevant),"feasible_heuristic_lower_bound",checks)
