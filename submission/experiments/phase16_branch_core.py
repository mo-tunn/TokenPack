"""Faz16 gold-blind branch-complete evidence packing under exact token cap.

Uses identical Faz15 graph plans, MS MARCO candidates, initial and rescue
verifier decisions. Adds PROVISIONAL candidates for uncovered edges; these
ARE NOT verified evidence. Enumerates <=4 graph edges, <=3 alternatives each.
"""
import itertools
from submission.experiments.phase16_protocol import EDGE_ALTERNATIVES
from submission.experiments.phase9_crossdoc_core import render_atomic,rank_rescale
from submission.experiments.phase13_graph_core import (
    bridge_link,EDGE_FILL_WEIGHT,ORIGINAL_FILL_WEIGHT,
    LINKED_FILL_BONUS,DOC_REPEAT_PENALTY
)

def branch_locked_select(case,initial,rescue,counter):
    atoms=case["atoms"]
    edges=[e for b in case["branches"] for e in b["edges"]]
    assert 1<=len(edges)<=4
    assert len(case["candidate_pools"])==len(case["expanded_pools"])==len(edges)
    limit=case["budget"]
    confirmed=dict(initial["decision_ids"])
    for edgeid,cid in rescue["rescue_decisions"].items():
        assert edgeid in confirmed and confirmed[edgeid] is None
        if cid is not None:confirmed[edgeid]=cid
    edge_to_branch={}
    e=0
    for bi,b in enumerate(case["branches"]):
        for _ in b["edges"]:
            edge_to_branch[e]=bi;e+=1
    assert e==len(edges)
    scores=case["edge_scores"]
    norm_scores=[rank_rescale(v) for v in scores]
    orig=rank_rescale(case["original_scores"])
    choices=[]
    for j in range(len(edges)):
        pool=case["candidate_pools"][j]+case["expanded_pools"][j]
        seen=set();uniq=[]
        for p in pool:
            i=p["atom_index"]
            if i in seen:continue
            seen.add(i);uniq.append(p)
        positive=confirmed[f"e{j}"]
        if positive is not None:
            found=[p for p in uniq if p["candidate_id"]==positive]
            assert len(found)==1
            # Verified positives must not be displaced by a merely provisional
            # candidate for that same edge. Missing edge allowed if budget.
            edge_choices=[(found[0]["atom_index"],True)]
        else:
            # Two from the original top4 + the highest-ranked expansion
            # (if present), all are PROVISIONAL and never called verified.
            selected=case["candidate_pools"][j][:2]
            if case["expanded_pools"][j]:
                selected+=case["expanded_pools"][j][:1]
            ids=set();edge_choices=[]
            for c in selected:
                if c["atom_index"] not in ids:
                    ids.add(c["atom_index"])
                    edge_choices.append((c["atom_index"],False))
            edge_choices=edge_choices[:EDGE_ALTERNATIVES]
        assert edge_choices
        choices.append([None]+edge_choices)
    best=None;chosen_mapping={}
    test_n=0
    for options in itertools.product(*choices):
        mapping={j:v for j,v in enumerate(options) if v is not None}
        indices=sorted({v[0] for v in mapping.values()})
        text=render_atomic(indices,atoms)
        tokens=counter.count(text)
        test_n+=1
        if tokens>limit:continue
        complete=sum(all(j in mapping for j,bi in edge_to_branch.items() if bi==b)
                     for b in range(len(case["branches"])))
        edge_n=len(mapping)
        verified_n=sum(v[1] for v in mapping.values())
        score=sum(norm_scores[j][v[0]] for j,v in mapping.items())
        rank=(complete,edge_n,verified_n,score,-tokens)
        if best is None or rank>best:
            best=rank;chosen_mapping=mapping
    assert best is not None,"empty selection must always be feasible"
    selected=sorted({v[0] for v in chosen_mapping.values()})
    used=set(selected)
    base_by_doc={}
    for i in selected:
        d=atoms[i]["article"];base_by_doc[d]=base_by_doc.get(d,0)+1
    render_checks=test_n
    def try_add(i):
        nonlocal render_checks
        if i in used:return False
        render_checks+=1
        if counter.count(render_atomic([*selected,i],atoms))>limit:return False
        selected.append(i);used.add(i)
        d=atoms[i]["article"];base_by_doc[d]=base_by_doc.get(d,0)+1
        return True
    def utility(i):
        link=max((bridge_link(atoms[i],atoms[j])
            for j in selected if atoms[i]["article"]!=atoms[j]["article"]),default=0.)
        return (EDGE_FILL_WEIGHT*max(v[i] for v in norm_scores)+
                ORIGINAL_FILL_WEIGHT*orig[i]+LINKED_FILL_BONUS*link-
                DOC_REPEAT_PENALTY*min(3,base_by_doc.get(atoms[i]["article"],0)))
    # Strictly keep edge anchors; only fill remaining context capacity.
    remaining=set(range(len(atoms)))-used
    while remaining:
        order=sorted(remaining,key=lambda i:(-utility(i),
            atoms[i]["article"],atoms[i]["sent_id"],i))
        added=None
        for i in order:
            if try_add(i):added=i;break
        if added is None:break
        remaining.remove(added)
    final=render_atomic(selected,atoms)
    n=counter.count(final)
    assert n<=limit
    anchors=chosen_mapping
    covered_branches=sum(all(j in anchors for j,bi in edge_to_branch.items() if bi==b)
                for b in range(len(case["branches"])))
    return {"text":final,"tokens":n,"budget":limit,
        "selected_sentence_indexes":sorted(used),
        "selected_source_idxs":sorted({atoms[i]["original_paragraph_idx"] for i in used}),
        "selected_paragraph_count":len({atoms[i]["article"] for i in used}),
        "selected_sentence_count":len(used),
        "edge_hypotheses":len(edges),"branch_hypotheses":len(case["branches"]),
        "anchored_edge_count":len(anchors),
        "fully_anchored_branches":covered_branches,
        "positive_verified_edge_anchors":sum(v[1] for v in anchors.values()),
        "provisional_unverified_edge_anchors":sum(not v[1] for v in anchors.values()),
        "missing_hypothetical_edge_anchors":len(edges)-len(anchors),
        "anchor_candidate_indexes":{f"e{j}":v[0] for j,v in anchors.items()},
        "anchor_verification_status":{f"e{j}":("verified" if v[1] else "provisional")
                                        for j,v in anchors.items()},
        "render_checks":render_checks,
        "semantics_warning":"Provisional candidate only proves index selection, not relation entailment."}
