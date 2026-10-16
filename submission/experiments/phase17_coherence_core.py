"""Gold-blind cross-edge consistency candidates and fixed selection.

The Qwen verdict is a MODEL JUDGMENT of relation consistency, not proof.
No gold or answer references are used.
"""
from __future__ import annotations
import itertools
from submission.experiments.phase17_protocol import TOP_CANDIDATES_PER_EDGE
from submission.experiments.phase9_crossdoc_core import render_atomic,rank_rescale
from submission.experiments.phase13_graph_core import (
    bridge_link,EDGE_FILL_WEIGHT,ORIGINAL_FILL_WEIGHT,
    LINKED_FILL_BONUS,DOC_REPEAT_PENALTY
)

def build_coherence_input(case,initial,rescue):
    edges=[e for branch in case["branches"] for e in branch["edges"]]
    assert len(edges)==len(case["candidate_pools"])==len(case["expanded_pools"])<=4
    prior=dict(initial["decision_ids"])
    for k,p in rescue["rescue_decisions"].items():
        assert k in prior and prior[k] is None
        if p is not None:prior[k]=p
    items=[]
    for j,e in enumerate(edges):
        pool=case["candidate_pools"][j]+case["expanded_pools"][j]
        chosen=[];used=set()
        priority=[]
        if prior[f"e{j}"] is not None:
            priority=[p for p in pool if p["candidate_id"]==prior[f"e{j}"]]
            assert len(priority)==1
        # Prior chosen positive and independent source alternatives. Deterministic.
        priority+=case["candidate_pools"][j][:2]+case["expanded_pools"][j][:2]
        for item in priority:
            if item["atom_index"] in used:continue
            used.add(item["atom_index"])
            chosen.append(item)
            if len(chosen)>=TOP_CANDIDATES_PER_EDGE:break
        assert chosen
        items.append({"edge_id":f"e{j}",
            "subject":e["subject"],"relation":e["relation"],"object":e["object"],
            "retrieval_query":e["query"],
            "previously_verified_candidate_id":prior[f"e{j}"],
            "candidates":[{
                "candidate_id":x["candidate_id"],"document_title":x["title"],
                "original_sentence_id":x["sentence_id"],
                "original_sentence":x["sentence"]}
                for x in chosen]})
    return {"question":case["question"],
            "hypothesized_edges_with_source_candidates":items}


def coherence_select(case,decision,counter):
    """Max branch completeness over model-approved candidates; abstentions not forced.

    Strict token cap; remaining space filled by frozen Phase16 relevance policy.
    """
    atoms=case["atoms"]
    budget=case["budget"]
    all_edges=[e for b in case["branches"] for e in b["edges"]]
    mapping={}
    for j,cid in enumerate(all_edges):
        key=f"e{j}"
        selected_id=decision["decision_ids"].get(key)
        if selected_id is None:continue
        pool=case["candidate_pools"][j]+case["expanded_pools"][j]
        matches=[p for p in pool if p["candidate_id"]==selected_id]
        assert len(matches)==1,f"Model returned candidate outside matching edge: {key}"
        mapping[j]=matches[0]["atom_index"]
    n_edges=len(all_edges)
    assert set(decision["decision_ids"])=={f"e{i}" for i in range(n_edges)}
    edge_to_branch={}
    x=0
    for bi,b in enumerate(case["branches"]):
        for _ in b["edges"]:edge_to_branch[x]=bi;x+=1
    normalized=[rank_rescale(s) for s in case["edge_scores"]]
    original=rank_rescale(case["original_scores"])
    best=None;anchored={}
    possibilities=list(mapping)
    # At most 4 edges => at most 16 candidate subsets.
    for mask in itertools.product((0,1),repeat=len(possibilities)):
        candidate={j:mapping[j] for j,b in zip(possibilities,mask) if b}
        chosen=sorted(set(candidate.values()))
        cost=counter.count(render_atomic(chosen,atoms))
        if cost>budget:continue
        complete=sum(all(j in candidate for j,bi in edge_to_branch.items() if bi==b)
                     for b in range(len(case["branches"])))
        edge_n=len(candidate)
        rel=sum(normalized[j][idx] for j,idx in candidate.items())
        rank=(complete,edge_n,rel,-cost)
        if best is None or rank>best:best=rank;anchored=candidate
    assert best is not None
    selected=sorted(set(anchored.values()));seen=set(selected)
    bydoc={}
    for i in selected:
        d=atoms[i]["article"];bydoc[d]=bydoc.get(d,0)+1
    checks=2**len(possibilities)
    def try_add(i):
        nonlocal checks
        if i in seen:return False
        checks+=1
        if counter.count(render_atomic([*selected,i],atoms))>budget:return False
        selected.append(i);seen.add(i)
        d=atoms[i]["article"];bydoc[d]=bydoc.get(d,0)+1
        return True
    def utility(i):
        link=max((bridge_link(atoms[i],atoms[j])
                for j in selected if atoms[i]["article"]!=atoms[j]["article"]),default=0)
        return (EDGE_FILL_WEIGHT*max(v[i] for v in normalized)
                +ORIGINAL_FILL_WEIGHT*original[i]+LINKED_FILL_BONUS*link
                -DOC_REPEAT_PENALTY*min(3,bydoc.get(atoms[i]["article"],0)))
    rest=set(range(len(atoms)))-seen
    while rest:
        order=sorted(rest,key=lambda i:(-utility(i),
            atoms[i]["article"],atoms[i]["sent_id"],i))
        added=None
        for i in order:
            if try_add(i):added=i;break
        if added is None:break
        rest.remove(added)
    text=render_atomic(selected,atoms)
    tokens=counter.count(text)
    assert tokens<=budget
    complete=sum(all(j in anchored for j,bi in edge_to_branch.items() if bi==b)
        for b in range(len(case["branches"])))
    return {"text":text,"tokens":tokens,"budget":budget,
        "selected_sentence_indexes":sorted(seen),
        "selected_source_idxs":sorted({atoms[i]["original_paragraph_idx"] for i in seen}),
        "selected_paragraph_count":len({atoms[i]["article"] for i in seen}),
        "selected_sentence_count":len(seen),
        "coherence_approved_edge_n":len(mapping),
        "coherence_approved_anchors_retained_n":len(anchored),
        "coherence_approved_complete_branches":complete,
        "abstained_edge_n":n_edges-len(mapping),
        "coherence_approval_is_not_gold_evidence":True,
        "render_checks":checks}
