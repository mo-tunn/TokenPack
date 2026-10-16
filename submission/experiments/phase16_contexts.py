"""Faz16 fixed gold-blind pair: unchanged Phase15 conditional-rescue vs
hard branch anchoring (verified if found, otherwise provisional) under SAME
actual rendered cl100k_base cap and identical initial/rescue Qwen judgments.
"""
from __future__ import annotations
import copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase16_register import OUT
from submission.experiments.phase16_protocol import ARMS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase14_selection import edge_verified_select
from submission.experiments.phase16_branch_core import branch_locked_select
from tokenpack.tokenization import TokenCounter
POOLS=OUT/"scored_candidate_pools.jsonl"
INITIAL=OUT/"verifier_responses.jsonl"
RESCUE=OUT/"rescue_verifier_responses.jsonl"
CTX=OUT/"gold_blind_contexts.jsonl"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
def main():
    record=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    a=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    b=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    c=json.loads((OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(POOLS)==a["candidate_sha256"]
    assert sha_file(INITIAL)==b["responses_sha256"]
    assert sha_file(RESCUE)==c["responses_sha256"]
    cases=[json.loads(s) for s in POOLS.open(encoding="utf8") if s.strip()]
    initial=[json.loads(s) for s in INITIAL.open(encoding="utf8") if s.strip()]
    rescue=[json.loads(s) for s in RESCUE.open(encoding="utf8") if s.strip()]
    assert [r["id"] for r in cases]==[r["id"] for r in initial]==[r["id"] for r in rescue]==record["pilot_ids"]
    if CTX.exists() or FROZEN.exists():raise FileExistsError("Preserve existing context output")
    metadata={
        "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "branch_code_sha256":sha_file(ROOT/"submission/experiments/phase16_branch_core.py"),
        "source_code_sha256":sha_file(Path(__file__)),
        "old_selection_code_sha256":sha_file(ROOT/"submission/experiments/phase14_selection.py"),
        "candidate_sha256":sha_file(POOLS),
        "initial_verifier_sha256":sha_file(INITIAL),
        "rescue_verifier_sha256":sha_file(RESCUE),
        "arms":list(ARMS),"gold_opened":False}
    new_json(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json",metadata)
    counter=TokenCounter()
    assert counter._encoding is not None
    stats={"n":0,"cap_violations":0,"different_contexts":0,
           "complete_graph_branch_controls":0,
           "complete_graph_branch_new":0,
           "hard_provisional_edge_anchors":0,
           "hard_verified_edge_anchors":0,
           "full_branch_questions_new":0,
           "full_branch_questions_control_anchor_proxy":0,
           "verified_positive_final_loss_new":0}
    started=time.monotonic()
    with CTX.open("x",encoding="utf8") as f:
        for case,v1,v2 in zip(cases,initial,rescue):
            combined=dict(v1["decision_ids"])
            with_expanded=copy.copy(case)
            with_expanded["candidate_pools"]=[p+e for p,e in zip(case["candidate_pools"],case["expanded_pools"])]
            for k,val in v2["rescue_decisions"].items():
                assert k in combined and combined[k] is None
                if val is not None:
                    j=int(k[1:])
                    assert val in {p["candidate_id"] for p in case["expanded_pools"][j]}
                    combined[k]=val
            control=edge_verified_select(with_expanded,{"decision_ids":combined},counter)
            new=branch_locked_select(case,v1,v2,counter)
            for ctx in (control,new):
                assert ctx["tokens"]==counter.count(ctx["text"])<=ctx["budget"]==case["budget"]
                ctx["selected_paragraph_count"]=len({
                    case["atoms"][j]["article"] for j in ctx["selected_sentence_indexes"]})
            verified_indices={}
            for j,pool in enumerate(with_expanded["candidate_pools"]):
                if combined[f"e{j}"] is not None:
                    hit=[r["atom_index"] for r in pool if r["candidate_id"]==combined[f"e{j}"]]
                    assert len(hit)==1
                    verified_indices[j]=hit[0]
            lost=sum(idx not in new["selected_sentence_indexes"] for idx in verified_indices.values())
            stats["n"]+=1
            stats["different_contexts"]+=int(control["text"]!=new["text"])
            stats["hard_provisional_edge_anchors"]+=new["provisional_unverified_edge_anchors"]
            stats["hard_verified_edge_anchors"]+=new["positive_verified_edge_anchors"]
            stats["complete_graph_branch_new"]+=new["fully_anchored_branches"]
            stats["full_branch_questions_new"]+=int(new["fully_anchored_branches"]==new["branch_hypotheses"])
            stats["verified_positive_final_loss_new"]+=lost
            # Count only edge anchors that coincide with a verified/provisional
            # pool candidate, not just coincidentally present source titles.
            control_ix=set(control["selected_sentence_indexes"])
            proxy=[]
            for j,pool in enumerate(with_expanded["candidate_pools"]):
                proxy.append(any(v["atom_index"] in control_ix for v in pool))
            edge_to_branch=[]
            for branch in case["branches"]:
                edge_to_branch.append(len(branch["edges"]))
            pos=0;branch_complete=0
            for k in edge_to_branch:
                branch_complete+=int(all(proxy[pos:pos+k]));pos+=k
            stats["complete_graph_branch_controls"]+=branch_complete
            stats["full_branch_questions_control_anchor_proxy"]+=int(branch_complete==len(edge_to_branch))
            output={
                "id":case["id"],"question":case["question"],
                "budget":case["budget"],"source_token_sum":case["source_token_sum"],
                "source_context":case["source_context"],
                "candidate_map":[{"title":r["title"],"article":r["article"],
                    "sent_id":r["sent_id"]} for r in case["atoms"]],
                "graph_branches":case["branches"],
                "edge_queries":case["edge_queries"],
                "verified_positive_count":len(verified_indices),
                "verified_positive_missed_in_branch_lock":lost,
                "contexts":{"conditional-rescue":control,"branch-lock":new}}
            f.write(json.dumps(output,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            if stats["n"]%8==0:
                print("PHASE16_CONTEXTS_PROGRESS",stats["n"],"/",len(cases),
                     "elapsed",round(time.monotonic()-started,2),flush=True)
    new_json(FROZEN,{"n":32,"sha256":sha_file(CTX),
        "protocol_sha256":sha_file(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"),
        "stats":stats,"gold_opened":False,
        "wall_seconds":round(time.monotonic()-started,1)})
    print("PHASE16_CONTEXTS_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
