"""Phase19 gold-blind, token-matched SOFT bridge packing.

Unlike Faz18 citation-closed, allows uncited Wikipedia docs when linked to
positively cited evidence. Unlike Faz17, hard-pins source-grounded positive
citations first. Every pair must match ACTUAL rendered cl100k token counts:
otherwise returned EXACT Phase16 baseline with a documented fallback.

An entity-title co-mention or Qwen approval is not a verified gold relation.
"""
from __future__ import annotations
import itertools, math
from submission.experiments.phase19_protocol import (
    TOKEN_RATIO_TOLERANCE,TOKEN_ABSOLUTE_TOLERANCE,MAX_BRIDGE_DOCS)
from submission.experiments.phase9_crossdoc_core import (
    render_atomic,rank_rescale)
from submission.experiments.phase13_graph_core import bridge_link

def select(case,judgment,baseline,counter):
    atoms=case["atoms"]
    edges=[e for b in case["branches"] for e in b["edges"]]
    assert len(edges)<=4 and edges
    budget=case["budget"]
    target=baseline["tokens"]
    tolerance=max(TOKEN_ABSOLUTE_TOLERANCE, math.ceil(TOKEN_RATIO_TOLERANCE*target))
    low=max(0,target-tolerance)
    high=min(budget,target+tolerance)
    assert target<=budget and high>=target>=low
    baseline_set=set(baseline["selected_sentence_indexes"])
    decisions=judgment["decision_ids"]
    assert set(decisions)=={f"e{j}" for j in range(len(edges))}
    approved={}
    abstained=set()
    for j in range(len(edges)):
        pool=case["candidate_pools"][j]+case["expanded_pools"][j]
        cid=decisions[f"e{j}"]
        if cid is None:
            abstained.update(p["atom_index"] for p in pool)
        else:
            found=[p["atom_index"] for p in pool if p["candidate_id"]==cid]
            assert len(found)==1
            approved[j]=found[0]
    if not approved:
        result=dict(baseline)
        result.update({"soft_fallback_reason":"no_cross_edge_positive_citation",
           "positive_edge_anchors":0,"bridge_exception_sources":0,
           "matched_delta_tokens":0,"token_parity_tolerance":tolerance,
           "model_approval_not_gold":True})
        return result
    positive_set=set(approved.values())
    normalized=[rank_rescale(v) for v in case["edge_scores"]]
    original=rank_rescale(case["original_scores"])
    edge_to_branch={}
    index=0
    for bi,b in enumerate(case["branches"]):
        for _ in b["edges"]:edge_to_branch[index]=bi;index+=1
    assert index==len(edges)
    best=None
    anchors={}
    for mask in itertools.product((False,True),repeat=len(approved)):
        chosen={j:approved[j] for j,keep in zip(approved,mask) if keep}
        ix=sorted(set(chosen.values()))
        cost=counter.count(render_atomic(ix,atoms))
        if cost>high:continue
        complete=sum(all(j in chosen for j,bi in edge_to_branch.items() if bi==b)
                     for b in range(len(case["branches"])))
        represented=len({edge_to_branch[j] for j in chosen})
        rank=(complete,represented,len(chosen),
              sum(normalized[j][i] for j,i in chosen.items()),-cost)
        if best is None or rank>best:
            best=rank;anchors=chosen
    assert best is not None
    selected=sorted(set(anchors.values()))
    used=set(selected)
    doccount={}
    for i in selected:
        doc=atoms[i]["article"];doccount[doc]=doccount.get(doc,0)+1
    cited_docs={atoms[i]["article"] for i in used}
    abstained_only=abstained-positive_set
    checks=2**len(approved)
    def try_add(i):
        nonlocal checks
        if i in used:return False
        checks+=1
        if counter.count(render_atomic([*selected,i],atoms))>high:return False
        selected.append(i);used.add(i)
        d=atoms[i]["article"];doccount[d]=doccount.get(d,0)+1
        return True
    def strongest_link(i):
        return max((bridge_link(atoms[i],atoms[j])
                    for j in used if atoms[i]["article"]!=atoms[j]["article"]),default=0.)
    def rel(i):
        return .68*max(scores[i] for scores in normalized)+.32*original[i]
    def relevance(i):
        return (rel(i)+.26*strongest_link(i)
                -.07*min(3,doccount.get(atoms[i]["article"],0))
                -(.18 if i in abstained_only else 0))
    # Strictly budgeted cross-document evidence seeds, at most 2 docs.
    # Prefer original-source sentences absent from the baseline when their
    # question/edge relevance is competitive; no gold-based candidate choice.
    bridge_count=0
    source_added=set()
    for _ in range(MAX_BRIDGE_DOCS):
        docs={atoms[i]["article"] for i in used}
        choices=[i for i in range(len(atoms))
                 if atoms[i]["article"] not in docs
                 and atoms[i]["article"] not in source_added]
        if not choices:break
        order=sorted(choices,key=lambda i:(
            -(relevance(i)+(.08 if i not in baseline_set else 0)),
            atoms[i]["article"],atoms[i]["sent_id"],i))
        added=None
        for i in order:
            if try_add(i):added=i;break
        if added is None:break
        source_added.add(atoms[added]["article"])
        bridge_count+=1
    # First salvage high-value original baseline evidence, avoiding a
    # strict document veto. Penalize Qwen-abstained candidate sentences, but
    # never forbid them if they may be bridges to another document.
    original_order=sorted(baseline_set-used,key=lambda i:
        (-relevance(i),atoms[i]["article"],atoms[i]["sent_id"],i))
    for i in original_order:try_add(i)
    # Fill to the baseline token count, if possible, with any source sentence
    # under the same exact render cap. Pick closest-to-target tokens first
    # among reasonably relevant candidates (bounded top 50 by relevance).
    remaining=set(range(len(atoms)))-used
    while remaining and counter.count(render_atomic(selected,atoms))<low:
        order=sorted(remaining,key=lambda i:
            (-relevance(i),atoms[i]["article"],atoms[i]["sent_id"],i))[:60]
        feasible=[]
        for i in order:
            checks+=1
            tokens=counter.count(render_atomic([*selected,i],atoms))
            if tokens<=high:
                feasible.append((i,tokens))
        if not feasible:break
        # Keep semantic scoring dominant; prefer closeness when scores
        # are within .12 of the best score.
        bestval=max(relevance(i) for i,_ in feasible)
        high_rel=[(i,n) for i,n in feasible if relevance(i)>=bestval-.12]
        i,n=max(high_rel,key=lambda x:(
            -abs(target-x[1]),relevance(x[0]),-x[0]))
        assert try_add(i)
        remaining.remove(i)
    final=render_atomic(selected,atoms)
    tokens=counter.count(final)
    assert tokens<=budget
    if tokens<low or tokens>high:
        out=dict(baseline)
        out.update({"soft_fallback_reason":"cannot_match_actual_baseline_token_count",
            "positive_edge_anchors":len(anchors),
            "bridge_exception_sources":bridge_count,
            "matched_delta_tokens":0,
            "token_parity_tolerance":tolerance,
            "model_approval_not_gold":True})
        return out
    return {
        "text":final,"tokens":tokens,"budget":budget,
        "selected_sentence_indexes":sorted(used),
        "selected_source_idxs":sorted({atoms[i]["original_paragraph_idx"] for i in used}),
        "selected_sentence_count":len(used),
        "selected_paragraph_count":len({atoms[i]["article"] for i in used}),
        "positive_edge_anchors":len(anchors),
        "bridge_exception_sources":bridge_count,
        "unverified_abstained_candidates_retained":len(used&abstained_only),
        "matched_delta_tokens":tokens-target,
        "token_parity_tolerance":tolerance,
        "soft_fallback_reason":None,
        "model_approval_not_gold":True,
        "render_checks":checks
    }
