"""Faz 4C3: budget-realistic, gold-blind manual rollout training states.

Use exactly the singleton cost estimate and source-ordered final-render
mechanism from Phase 4C's conditional selector. States are prefixes of an
actual manual-score greedy rollout under the requested token budget, NOT
unconstrained Top1/Top3 score prefixes. Gold annotations never enter here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from submission.experiments.phase4c_train_marginal import minmax
from tokenpack.export import render_context

STATE_COUNTS=(0,1,2,3,5,7,9,12)
COST_POWER=.20


@dataclass
class RolloutTrace:
    ordered:tuple[int,...]
    states:tuple[tuple[int,...],...]
    singleton_cost:dict[int,int]
    reached_budget:int
    checks:int


def manual_budget_rollout(q,counter,budget)->RolloutTrace:
    """Compute manual greedy states with no supervision and final cap audit."""
    n=len(q.scored)
    if budget<1:
        raise ValueError("Positive context token cap required")
    def render(indices):
        items=sorted(indices,key=lambda j:(q.scored[j].chunk.order_key,
                                             q.scored[j].chunk.id))
        return render_context([q.scored[j].chunk for j in items])
    base=counter.count(render([]))
    if base>budget:
        return RolloutTrace((),((),),{},base,0)
    cost={
        i:max(1,counter.count(render([i]))-base)
        for i in range(n)
    }
    normalized=minmax([item.value for item in q.scored])
    priorities=sorted(
        range(n),
        key=lambda i:(
            -normalized[i]/max(1,cost[i])**COST_POWER,
            q.scored[i].chunk.order_key,i,
        ),
    )
    chosen=[]
    estimated=base
    checkpoints={0:()}
    for i in priorities:
        if estimated+cost[i]>budget:
            continue
        chosen.append(i)
        estimated+=cost[i]
        if len(chosen) in STATE_COUNTS:
            checkpoints[len(chosen)]=tuple(chosen)
    # If singleton cost nonadditivity made any saved prefix too big, use
    # only genuine feasible states. Every earlier source-ordered subset
    # is rechecked before use in supervised training.
    final=len(chosen)
    checkpoints[final]=tuple(chosen)
    feasible=[]
    checks=0
    for _,state in sorted(checkpoints.items()):
        used=counter.count(render(state))
        checks+=1
        if used<=budget:
            feasible.append(state)
    if not feasible:
        feasible=[()]
    return RolloutTrace(
        ordered=tuple(chosen),
        states=tuple(feasible),
        singleton_cost=cost,
        reached_budget=estimated,
        checks=checks,
    )


def candidates_with_remaining_budget(q,trace,state,budget,counter=None):
    """Same singleton-additive screening as the inference greedy phase.

    For training, candidate labels are calculated ONLY for items that may
    be selected next under the remaining estimated token allowance. This
    avoids teaching the ranker on impossible actions in late states.
    """
    selected=set(state)
    base=0
    if counter is not None:
        base=counter.count(render_context([]))
    estimated=base+sum(trace.singleton_cost[i] for i in state)
    return [
        i for i in range(len(q.scored))
        if i not in selected and estimated+trace.singleton_cost[i]<=budget
    ]
