"""Gold-blind Phase15 paired contexts: initial top4 vs conditional verifier rescue.
Fixed same 25% rendered token cap and exact same ranking/graph fill.
No gold inputs, no Qwen calls; immutable on first creation.
"""
from __future__ import annotations
import copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase15_register import OUT
from submission.experiments.phase15_protocol import METHODS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase14_selection import edge_verified_select
from tokenpack.tokenization import TokenCounter
POOLS=OUT/"scored_candidate_pools.jsonl"
INITIAL=OUT/"verifier_responses.jsonl"
RESCUE=OUT/"rescue_verifier_responses.jsonl"
CTX=OUT/"gold_blind_contexts.jsonl"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
def main():
    m=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    a=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    b=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    c=json.loads((OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(POOLS)==a["candidate_sha256"]
    assert sha_file(INITIAL)==b["responses_sha256"]
    assert sha_file(RESCUE)==c["responses_sha256"]
    cases=[json.loads(s) for s in POOLS.open(encoding="utf8") if s.strip()]
    old=[json.loads(s) for s in INITIAL.open(encoding="utf8") if s.strip()]
    new=[json.loads(s) for s in RESCUE.open(encoding="utf8") if s.strip()]
    assert [r["id"] for r in cases]==[r["id"] for r in old]==[r["id"] for r in new]==m["pilot_ids"]
    if CTX.exists() or FROZEN.exists():raise FileExistsError("Partial/frozen contexts already exist")
    meta={
        "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "source_sha256":sha_file(Path(__file__)),
        "selection_sha256":sha_file(
            ROOT/"submission/experiments/phase14_selection.py"),
        "initial_decisions_sha256":sha_file(INITIAL),
        "rescue_decisions_sha256":sha_file(RESCUE),
        "candidate_sha256":sha_file(POOLS),
        "arms":list(METHODS),"gold_opened":False}
    new_json(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json",meta)
    counter=TokenCounter()
    assert counter._encoding is not None
    total={"n":0,"violations":0,"changed_contexts":0,
        "initial_positives":0,"rescue_positives":0,
        "new_verifier_positives_survived_context":0,
        "realized_positive_rescue_nontrivial":0}
    start=time.monotonic()
    with CTX.open("x",encoding="utf8") as f:
        for case,v1,v2 in zip(cases,old,new):
            oldverdict={"decision_ids":v1["decision_ids"]}
            c1=edge_verified_select(case,oldverdict,counter)
            combined=dict(v1["decision_ids"])
            expanded=copy.copy(case)
            expanded["candidate_pools"]=[a+b for a,b in zip(
                case["candidate_pools"],case["expanded_pools"])]
            rescue_candidates={}
            for k,val in v2["rescue_decisions"].items():
                assert k in combined and combined[k] is None
                if val is not None:
                    # Verify every promoted id existed in that SAME edge's
                    # (new) expansion, not merely any edge.
                    j=int(k[1:])
                    allowed={z["candidate_id"] for z in case["expanded_pools"][j]}
                    assert val in allowed
                    combined[k]=val
                    rescue_candidates[k]=val
            c2=edge_verified_select(expanded,{"decision_ids":combined},counter)
            for item in (c1,c2):
                assert item["budget"]==case["budget"] and item["tokens"]==counter.count(item["text"])<=case["budget"]
                item["selected_paragraph_count"]=len({
                    case["atoms"][i]["article"] for i in item["selected_sentence_indexes"]})
            promoted_idx=[]
            for edge_id,val in rescue_candidates.items():
                j=int(edge_id[1:])
                x=next(z["atom_index"] for z in case["expanded_pools"][j]
                       if z["candidate_id"]==val)
                promoted_idx.append(x)
            promoted_survived=sum(i in c2["selected_sentence_indexes"] for i in promoted_idx)
            assert promoted_survived==len(promoted_idx) or c2["verified_edge_selected"]<len(combined)
            total["n"]+=1;total["initial_positives"]+=sum(v is not None for v in v1["decision_ids"].values())
            total["rescue_positives"]+=len(promoted_idx)
            total["new_verifier_positives_survived_context"]+=promoted_survived
            total["changed_contexts"]+=int(c1["text"]!=c2["text"])
            total["realized_positive_rescue_nontrivial"]+=int(
                bool(promoted_idx) and c1["text"]!=c2["text"])
            result={"id":case["id"],"question":case["question"],
                "budget":case["budget"],"source_token_sum":case["source_token_sum"],
                "source_context":case["source_context"],
                "graph_branches":case["branches"],"edge_queries":case["edge_queries"],
                "candidate_map":[{"title":a["title"],
                  "sent_id":a["sent_id"],"article":a["article"]}
                    for a in case["atoms"]],
                "rescue_promoted_count":len(promoted_idx),
                "verifier_initial":v1["decision_ids"],
                "verifier_rescued":v2["rescue_decisions"],
                "contexts":{"top4-verified":c1,"conditional-rescue":c2}}
            f.write(json.dumps(result,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            if total["n"]%8==0:
                print("PHASE15_CONTEXTS_PROGRESS",total["n"],32,
                    "elapsed_s",round(time.monotonic()-start,1),flush=True)
    new_json(FROZEN,{"n":32,"sha256":sha_file(CTX),
            "protocol_sha256":sha_file(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"),
            "stats":total,"gold_opened":False,
            "wall_seconds":round(time.monotonic()-start,1)})
    print("PHASE15_CONTEXTS_FROZEN",json.dumps(total),flush=True)
if __name__=="__main__":main()
