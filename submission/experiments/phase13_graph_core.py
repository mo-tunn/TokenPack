"""Frozen graph-pair selector for Phase13 2Wiki. No labels, answers, or gold edges.

Competes with an unstructured selector using the SAME generated edge queries
and the SAME reranker outputs and exact cl100k_base context cap.
"""
from __future__ import annotations
import math
from submission.experiments.phase9_crossdoc_core import (
    bridge_link, rank_rescale, render_atomic,
)

TOP_PER_EDGE=16
MAX_EDGES=4
PAIR_LINK_WEIGHT=.38
PAIR_RELEVANCE_WEIGHT=.12
PAIR_COST_WEIGHT=.05
ANCHOR_CAP_SHARE=.64
DOC_REPEAT_PENALTY=.10
LINKED_FILL_BONUS=.16
EDGE_FILL_WEIGHT=.54
ORIGINAL_FILL_WEIGHT=.46

def normalized_edges(question,raw):
    """Return bounded validated branches; deterministic safe fallback."""
    import json,re
    blob=str(raw or "").strip()
    if blob.startswith("```"):
        blob=re.sub(r"^\s*```(?:json)?","",blob).split("```")[0].strip()
    if not blob.startswith("{") and "{" in blob and "}" in blob:
        blob=blob[blob.find("{"):blob.rfind("}")+1]
    try:
        obj=json.loads(blob)
        assert isinstance(obj,dict)
        branches=obj["branches"]
        assert isinstance(branches,list) and 1<=len(branches)<=2
        out=[]
        for branch in branches:
            assert isinstance(branch,dict)
            edges=branch["edges"]
            assert isinstance(edges,list) and 1<=len(edges)<=2
            valid=[]
            for x in edges:
                assert isinstance(x,dict)
                row={}
                for k in ("subject","relation","object","query"):
                    v=x.get(k)
                    assert isinstance(v,str) and 1<=len(v.strip())<=180
                    row[k]=" ".join(v.split())[:180]
                row["query"]=" ".join(row["query"].split()[:18])
                valid.append(row)
            out.append({"edges":valid})
        assert 1<=sum(len(b["edges"]) for b in out)<=MAX_EDGES
        return out,"valid_structured_graph"
    except (ValueError,TypeError,KeyError,AssertionError):
        return [{"edges":[{"subject":"?","relation":"question evidence",
            "object":"?","query":str(question)[:180]}]}],"fallback_original_question"


def _pack(selected,atoms,counter):
    return counter.count(render_atomic(selected,atoms))


def graph_select(atoms,base_scores,edge_scores,branches,budget,counter):
    """Pre-budget connected edge pairs per branch, then relevance+link greedy.

    A graph *hypothesis*: branches and edges are generated from question only.
    Links between source sentences are lexical, not verified entailment.
    """
    if budget<=0 or not atoms or len(base_scores)!=len(atoms):
        raise ValueError("Invalid inputs")
    edges=[e for branch in branches for e in branch["edges"]]
    assert 1<=len(edges)<=MAX_EDGES
    assert len(edge_scores)==len(edges)
    assert all(len(v)==len(atoms) for v in edge_scores)
    q=rank_rescale(base_scores)
    e=[rank_rescale(v) for v in edge_scores]
    n=len(atoms)
    selected=[]
    seen=set()
    bydoc={}
    render_checks=0
    seed_pairs=0
    edges_hit=set()
    def fits(items,limit):
        nonlocal render_checks
        render_checks+=1
        return _pack(items,atoms,counter)<=limit
    def add(i,limit):
        if i in seen:return True
        if not fits([*selected,i],limit):return False
        selected.append(i);seen.add(i)
        d=atoms[i]["article"]
        bydoc[d]=bydoc.get(d,0)+1
        return True

    anchor_cap=max(1,int(math.floor(ANCHOR_CAP_SHARE*budget)))
    offset=0
    for branch in branches:
        edgeids=list(range(offset,offset+len(branch["edges"])))
        offset+=len(edgeids)
        if len(edgeids)==2:
            a,b=edgeids
            tops=[]
            for k in edgeids:
                tops.append(sorted(range(n),key=lambda i:
                    (-e[k][i],atoms[i]["article"],atoms[i]["sent_id"],i))[:TOP_PER_EDGE])
            proposals=[]
            for i in tops[0]:
                for j in tops[1]:
                    if i==j or atoms[i]["article"]==atoms[j]["article"]:continue
                    link=bridge_link(atoms[i],atoms[j])
                    # Do not require lexical link (entity aliases may differ);
                    # score smoothly and preserve a relevance-based fallback.
                    score=(e[a][i]+e[b][j])/2 + PAIR_LINK_WEIGHT*link + (
                        PAIR_RELEVANCE_WEIGHT*(q[i]+q[j])/2)
                    proposed=sorted(set([*selected,i,j]))
                    cost=_pack(proposed,atoms,counter)
                    render_checks+=1
                    if cost>anchor_cap:continue
                    score-=PAIR_COST_WEIGHT*(cost/budget)
                    proposals.append((score,link,-cost,-i,-j,i,j))
            if proposals:
                best=max(proposals)
                i,j=best[-2:]
                if add(i,anchor_cap) and add(j,anchor_cap):
                    seed_pairs+=1
                    edges_hit.update(edgeids)
                    continue
        # Single-edge or unaffordable pair fallback.
        for k in edgeids:
            order=sorted(range(n),key=lambda i:(-e[k][i],-q[i],
                bydoc.get(atoms[i]["article"],0),atoms[i]["article"],
                atoms[i]["sent_id"],i))
            for i in order:
                if i not in seen and add(i,anchor_cap):
                    edges_hit.add(k);break

    remaining=set(range(n))-seen
    while remaining:
        def utility(i):
            link=max((bridge_link(atoms[i],atoms[j]) for j in selected
                if atoms[i]["article"]!=atoms[j]["article"]),default=0.)
            return (EDGE_FILL_WEIGHT*max(x[i] for x in e)
                +ORIGINAL_FILL_WEIGHT*q[i]+LINKED_FILL_BONUS*link
                -DOC_REPEAT_PENALTY*min(3,bydoc.get(atoms[i]["article"],0)))
        order=sorted(remaining,key=lambda i:
            (-utility(i),atoms[i]["article"],atoms[i]["sent_id"],i))
        added=None
        for i in order:
            if add(i,budget):
                added=i;break
        if added is None:break
        remaining.remove(added)
    text=render_atomic(selected,atoms)
    tokens=counter.count(text)
    assert tokens<=budget
    return {
        "text":text,"budget":budget,"tokens":tokens,
        "selected_sentence_indexes":sorted(seen),
        "selected_source_idxs":sorted({atoms[i]["original_paragraph_idx"] for i in seen}),
        "selected_sentence_count":len(seen),
        "selected_paragraph_count":len(bydoc),
        "graph_branch_count":len(branches),
        "graph_edge_count":len(edges),
        "paired_branch_seeds":seed_pairs,
        "anchor_edges_covered_by_seeds":len(edges_hit),
        "render_checks":render_checks,
    }
