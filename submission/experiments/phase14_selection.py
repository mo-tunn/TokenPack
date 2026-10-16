"""Faz14 candidate-verified edge coverage: frozen, gold-blind selector.

Gold-blind Qwen selects (or abstains from) candidate original sentence IDs for
question-derived edge hypotheses. Choose subset covering most graph BRANCHES
then edges subject to anchor token cap; fill using the old unmodified graph
candidate order and fixed MS MARCO utility. No gold features.
"""
import itertools,math
from submission.experiments.phase14_protocol import EDGE_ANCHOR_SHARE
from submission.experiments.phase13_graph_core import (
    graph_select,bridge_link,rank_rescale,EDGE_FILL_WEIGHT,
    ORIGINAL_FILL_WEIGHT,LINKED_FILL_BONUS,DOC_REPEAT_PENALTY
)
from submission.experiments.phase9_crossdoc_core import render_atomic

def edge_verified_select(case,verdict,counter):
    atoms=case["atoms"];base=case["original_scores"]
    scores=case["edge_scores"];branches=case["branches"]
    budget=case["budget"]
    pools=case["candidate_pools"]
    graph=graph_select(atoms,base,scores,branches,budget,counter)
    valid={}
    edgeidx=0
    edge_to_branch={}
    for b,branch in enumerate(branches):
        for e in branch["edges"]:
            edge_to_branch[edgeidx]=b;edgeidx+=1
    assert len(scores)==len(pools)==len(edge_to_branch)
    decisions=verdict["decision_ids"]
    for j,pool in enumerate(pools):
        candidate_id=decisions.get(f"e{j}")
        if candidate_id is None:continue
        m={x["candidate_id"]:x["atom_index"] for x in pool}
        if candidate_id in m:
            valid[j]=m[candidate_id]
    if not valid:
        out=dict(graph)
        out.update({
            "verified_edge_pick_count":0,"verified_edge_selected":0,
            "selected_anchor_branches":0,"verified_anchor_sentence_count":0,
            "rejected_verified_due_budget":0,
            "verification_fallback_to_graph":True,
        })
        return out
    cap=max(1,math.floor(EDGE_ANCHOR_SHARE*budget))
    original_norm=rank_rescale(base)
    edges_norm=[rank_rescale(x) for x in scores]
    best_key=None;best_items=[]
    edgelist=list(valid)
    # At most four edges. Subset maximizes distinct graph branch presence and
    # number of verified edges under the SAME actual rendered token cost cap.
    for mask in itertools.product([False,True],repeat=len(edgelist)):
        chosen_edges=[edgelist[i] for i,m in enumerate(mask) if m]
        indices=sorted({valid[i] for i in chosen_edges})
        tokens=counter.count(render_atomic(indices,atoms))
        if tokens>cap:continue
        branches_hit=len({edge_to_branch[i] for i in chosen_edges})
        relation_score=sum(edges_norm[i][valid[i]] for i in chosen_edges)
        rank=(branches_hit,len(chosen_edges),relation_score,-tokens)
        if best_key is None or rank>best_key:
            best_key=rank;best_items=chosen_edges
    if not best_items:
        out=dict(graph)
        out.update({
            "verified_edge_pick_count":len(valid),"verified_edge_selected":0,
            "selected_anchor_branches":0,"verified_anchor_sentence_count":0,
            "rejected_verified_due_budget":len(valid),
            "verification_fallback_to_graph":True,
        })
        return out
    selected=sorted({valid[e] for e in best_items})
    seen=set(selected)
    render_checks=2**len(edgelist)
    bydoc={}
    for i in selected:
        d=atoms[i]["article"];bydoc[d]=bydoc.get(d,0)+1
    def add(i):
        nonlocal render_checks
        if i in seen:return False
        render_checks+=1
        if counter.count(render_atomic([*selected,i],atoms))>budget:
            return False
        selected.append(i);seen.add(i)
        d=atoms[i]["article"];bydoc[d]=bydoc.get(d,0)+1
        return True
    def relevance(i):
        link=max((bridge_link(atoms[i],atoms[j])
            for j in selected if atoms[i]["article"]!=atoms[j]["article"]),default=0.)
        return (EDGE_FILL_WEIGHT*max(v[i] for v in edges_norm)
                +ORIGINAL_FILL_WEIGHT*original_norm[i]
                +LINKED_FILL_BONUS*link
                -DOC_REPEAT_PENALTY*min(3,bydoc.get(atoms[i]["article"],0)))
    # Explicitly carry as much of the original graph as fits after verified
    # evidence anchors. Tie-break by fixed rerank utility, no gold.
    graph_indices=set(graph["selected_sentence_indexes"])-seen
    graph_order=sorted(graph_indices,key=lambda i:
        (-relevance(i),atoms[i]["article"],atoms[i]["sent_id"],i))
    for i in graph_order:add(i)
    # Greedy rest to use budget, recompute link as selection changes.
    remaining=set(range(len(atoms)))-seen
    while remaining:
        order=sorted(remaining,key=lambda i:
            (-relevance(i),atoms[i]["article"],atoms[i]["sent_id"],i))
        added=None
        for i in order:
            if add(i):
                added=i;break
        if added is None:break
        remaining.remove(added)
    text=render_atomic(selected,atoms)
    n=counter.count(text)
    assert n<=budget
    edge_selected=sum(valid[e] in seen for e in valid)
    return {"text":text,"tokens":n,"budget":budget,
        "selected_sentence_indexes":sorted(seen),
        "selected_source_idxs":sorted(
            {atoms[i]["original_paragraph_idx"] for i in seen}),
        "selected_sentence_count":len(seen),
        "selected_paragraph_count":len(bydoc),
        "verified_edge_pick_count":len(valid),
        "verified_edge_selected":edge_selected,
        "selected_anchor_branches":len(
            {edge_to_branch[e] for e in best_items}),
        "verified_anchor_sentence_count":len(set(valid[e] for e in best_items)),
        "rejected_verified_due_budget":len(valid)-len(best_items),
        "verification_fallback_to_graph":False,
        "render_checks":render_checks}
