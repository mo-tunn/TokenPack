"""Faz17 immutable gold-blind paired contexts:
control Phase16 branch-lock vs cross-edge coherence-gated original citation pool.
"""
from __future__ import annotations
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase17_register import OUT
from submission.experiments.phase17_protocol import ARMS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase16_branch_core import branch_locked_select
from submission.experiments.phase17_coherence_core import coherence_select
from tokenpack.tokenization import TokenCounter
POOLS=OUT/"scored_candidate_pools.jsonl"
INITIAL=OUT/"verifier_responses.jsonl"
RESCUE=OUT/"rescue_verifier_responses.jsonl"
COHERENCE=OUT/"coherence_verifier_responses.jsonl"
CTX=OUT/"gold_blind_contexts.jsonl"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
def main():
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    a=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    b=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    c=json.loads((OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    d=json.loads((OUT/"COHERENCE_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(POOLS)==a["candidate_sha256"]
    assert sha_file(INITIAL)==b["responses_sha256"]
    assert sha_file(RESCUE)==c["responses_sha256"]
    assert sha_file(COHERENCE)==d["responses_sha256"]
    cases=[json.loads(x) for x in POOLS.open(encoding="utf8") if x.strip()]
    first=[json.loads(x) for x in INITIAL.open(encoding="utf8") if x.strip()]
    second=[json.loads(x) for x in RESCUE.open(encoding="utf8") if x.strip()]
    judged=[json.loads(x) for x in COHERENCE.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in cases]==[x["id"] for x in first]==[x["id"] for x in second]==[x["id"] for x in judged]==manifest["pilot_ids"]
    if CTX.exists() or FROZEN.exists():raise FileExistsError("Existing contexts, preserve")
    frozen_protocol={
        "pilot_ids_sha256":manifest["pilot_id_sha256"],
        "candidate_sha256":a["candidate_sha256"],
        "initial_sha256":b["responses_sha256"],
        "rescue_sha256":c["responses_sha256"],
        "coherence_sha256":d["responses_sha256"],
        "core_sha256":sha_file(ROOT/"submission/experiments/phase17_coherence_core.py"),
        "control_core_sha256":sha_file(ROOT/"submission/experiments/phase16_branch_core.py"),
        "this_script_sha256":sha_file(Path(__file__)),
        "methods":list(ARMS),"gold_opened":False,
        "budget":"25% sum each source block cl100k_base, exact final render cap"
    }
    new_json(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json",frozen_protocol)
    counter=TokenCounter()
    assert counter._encoding is not None
    stats={"n":0,"token_cap_violations":0,
        "contexts_changed":0,"coherence_abstained_edges":0,
        "coherence_positive_anchors_retained":0,
        "coherence_positive_edges":0,
        "model_positive_branches_fully_anchored":0,
        "model_positive_questions_all_edges":0,
        "control_provisional_anchors":0}
    started=time.monotonic()
    with CTX.open("x",encoding="utf8") as f:
        for case,i,r,j in zip(cases,first,second,judged):
            assert case["id"]==i["id"]==r["id"]==j["id"]
            control=branch_locked_select(case,i,r,counter)
            new=coherence_select(case,j,counter)
            assert control["tokens"]==counter.count(control["text"])<=case["budget"]==control["budget"]
            assert new["tokens"]==counter.count(new["text"])<=case["budget"]==new["budget"]
            stats["n"]+=1
            stats["contexts_changed"]+=int(control["text"]!=new["text"])
            stats["coherence_abstained_edges"]+=new["abstained_edge_n"]
            stats["coherence_positive_edges"]+=new["coherence_approved_edge_n"]
            stats["coherence_positive_anchors_retained"]+=new["coherence_approved_anchors_retained_n"]
            stats["model_positive_branches_fully_anchored"]+=new["coherence_approved_complete_branches"]
            stats["model_positive_questions_all_edges"]+=int(new["coherence_approved_edge_n"]==len(j["decision_ids"]))
            stats["control_provisional_anchors"]+=control["provisional_unverified_edge_anchors"]
            result={"id":case["id"],"question":case["question"],
                "budget":case["budget"],"source_token_sum":case["source_token_sum"],
                "source_context":case["source_context"],
                "candidate_map":[{"title":a["title"],"article":a["article"],
                    "sent_id":a["sent_id"]} for a in case["atoms"]],
                "graph_branches":case["branches"],
                "edge_queries":case["edge_queries"],
                "coherence_decisions":j["decision_ids"],
                "coherence_parse_status":j["parse_status"],
                "contexts":{"branch-lock":control,"coherence-gated":new}}
            f.write(json.dumps(result,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            if stats["n"]%8==0:
                print("PHASE17_CONTEXTS_PROGRESS",stats["n"],32,
                    "elapsed",round(time.monotonic()-started,1),flush=True)
    new_json(FROZEN,{"n":32,"sha256":sha_file(CTX),
         "pre_gold_sha256":sha_file(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"),
         "stats":stats,"gold_opened":False,
         "wall_seconds":round(time.monotonic()-started,1)})
    print("PHASE17_CONTEXTS_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
