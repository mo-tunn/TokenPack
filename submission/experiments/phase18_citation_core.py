"""Faz18 hard citation-closed evidence selection, gold blind.

Model rejects/abstains: never add top8 *edge-specific* candidate sentence as
unverified filler, unless it is also approved by a different edge. Filler
is restricted to documents from which Qwen cited a positive edge sentence.
This is a deliberately strong, preregistered intervention, not NLI proof.
"""
from __future__ import annotations
import itertools
from submission.experiments.phase9_crossdoc_core import (
    render_atomic,rank_rescale)
from submission.experiments.phase13_graph_core import (
    bridge_link,EDGE_FILL_WEIGHT,ORIGINAL_FILL_WEIGHT,
    LINKED_FILL_BONUS,DOC_REPEAT_PENALTY)

def select(case,judgment,baseline,counter):
    edges=[e for b in case["branches"] for e in b["edges"]]
    assert 1<=len(edges)<=4
    atom=case["atoms"];limit=case["budget"]
    verdict=judgment["decision_ids"]
    assert set(verdict)=={f"e{i}" for i in range(len(edges))}
    approved={}
    rejected=set()
    for j,edge in enumerate(edges):
        pool=case["candidate_pools"][j]+case["expanded_pools"][j]
        value=verdict[f"e{j}"]
        if value is None:
            rejected|={c["atom_index"] for c in pool}
        else:
            matches=[p["atom_index"] for p in pool if p["candidate_id"]==value]
            assert len(matches)==1
            approved[j]=matches[0]
    if not approved:
        out=dict(baseline)
        out.update({"citation_anchors":0,"citation_positive_edges":0,
            "closed_source_articles":0,"vetoed_pool_sentences":len(rejected),
            "unverified_filler_candidates":0,"rejected_candidate_sentences_in_final":None,
            "fallback_no_positive":True,"citation_model_not_gold":True})
        return out
    approved_indices=set(approved.values())
    # Subset selection under true render cl100k cost: maximum distinct
    # anchored branches, then approved edges, then normalized evidence rank.
    branch={}
    j=0
    for bi,b in enumerate(case["branches"]):
        for e in b["edges"]:branch[j]=bi;j+=1
    assert j==len(edges)
    norm=[rank_rescale(v) for v in case["edge_scores"]]
    original=rank_rescale(case["original_scores"])
    choices=list(approved)
    best=None;seed={}
    for mask in itertools.product((0,1),repeat=len(choices)):
        anchors={j:approved[j] for j,k in zip(choices,mask) if k}
        ix=sorted(set(anchors.values()))
        cost=counter.count(render_atomic(ix,atom))
        if cost>limit:continue
        covered=len({branch[j] for j in anchors})
        complete=sum(all(j in anchors for j,x in branch.items() if x==bi)
                     for bi in range(len(case["branches"])))
        rank=(complete,covered,len(anchors),
              sum(norm[j][ix] for j,ix in anchors.items()),-cost)
        if best is None or rank>best:
            best=rank;seed=anchors
    assert best is not None
    selected=sorted(set(seed.values()));seen=set(selected)
    selected_docs={atom[i]["article"] for i in selected}
    # If any approved anchors are infeasible, do NOT include their uncited
    # documents as filler. Distinct document set is fixed before filling.
    banned=rejected-approved_indices
    n_checks=2**len(choices)
    bydoc={}
    for i in seen:
        d=atom[i]["article"];bydoc[d]=bydoc.get(d,0)+1
    def utility(i):
        link=max((bridge_link(atom[i],atom[j])
            for j in selected if atom[i]["article"]!=atom[j]["article"]),default=0.)
        return (EDGE_FILL_WEIGHT*max(v[i] for v in norm)
                +ORIGINAL_FILL_WEIGHT*original[i]
                +LINKED_FILL_BONUS*link
                -DOC_REPEAT_PENALTY*min(3,bydoc.get(atom[i]["article"],0)))
    remaining={i for i in range(len(atom))
               if i not in seen and i not in banned
               and atom[i]["article"] in selected_docs}
    while remaining:
        order=sorted(remaining,key=lambda i:
            (-utility(i),atom[i]["article"],atom[i]["sent_id"],i))
        chosen=None
        for i in order:
            n_checks+=1
            if counter.count(render_atomic([*selected,i],atom))<=limit:
                chosen=i;break
        if chosen is None:break
        selected.append(chosen);seen.add(chosen)
        d=atom[chosen]["article"];bydoc[d]=bydoc.get(d,0)+1
        remaining.remove(chosen)
    text=render_atomic(selected,atom)
    tokens=counter.count(text)
    assert tokens<=limit
    assert not set(seen)&banned
    return {
        "text":text,"tokens":tokens,"budget":limit,
        "selected_sentence_indexes":sorted(seen),
        "selected_source_idxs":sorted(
            {atom[i]["original_paragraph_idx"] for i in seen}),
        "selected_paragraph_count":len({atom[i]["article"] for i in seen}),
        "selected_sentence_count":len(seen),
        "citation_positive_edges":len(approved),
        "citation_anchors":len(seed),
        "closed_source_articles":len(selected_docs),
        "vetoed_pool_sentences":len(banned),
        "rejected_candidate_sentences_in_final":len(set(seen)&rejected),
        "fallback_no_positive":False,
        "citation_model_not_gold":True,
        "render_checks":n_checks,
    }
