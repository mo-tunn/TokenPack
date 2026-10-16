"""Faz19 frozen gold-blind context pair: branch-lock vs citation-closed.

Both share Phase17 Qwen graph, two rounds of verification, Qwen coherence
decisions, all exact source sentences and cl100k_base %25 token cap.
"""
from __future__ import annotations
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase19_register import OUT
from submission.experiments.phase19_protocol import ARMS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase16_branch_core import branch_locked_select
from submission.experiments.phase19_soft_bridge_core import select
from tokenpack.tokenization import TokenCounter
DATASETS={
    "candidates":("scored_candidate_pools.jsonl","CANDIDATES_FROZEN.json","candidate_sha256"),
    "first":("verifier_responses.jsonl","VERIFIER_RESPONSES_FROZEN.json","responses_sha256"),
    "second":("rescue_verifier_responses.jsonl","RESCUE_VERIFIER_RESPONSES_FROZEN.json","responses_sha256"),
    "coherence":("coherence_verifier_responses.jsonl","COHERENCE_RESPONSES_FROZEN.json","responses_sha256")
}
CONTEXTS=OUT/"gold_blind_contexts.jsonl"
FREEZE=OUT/"CONTEXTS_FROZEN.json"
def main():
    prereg=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    records={}
    filehashes={}
    for k,(fname,meta,hashkey) in DATASETS.items():
        f=OUT/fname
        frozen=json.loads((OUT/meta).read_text(encoding="utf8"))
        assert sha_file(f)==frozen[hashkey]
        records[k]=[json.loads(x) for x in f.open(encoding="utf8") if x.strip()]
        assert len(records[k])==32 and [r["id"] for r in records[k]]==prereg["pilot_ids"]
        filehashes[k]=sha_file(f)
    if CONTEXTS.exists() or FREEZE.exists():raise FileExistsError("Do not overwrite")
    meta={"pilot_sha256":prereg["pilot_sha256"],
        "input_sha256":filehashes,
        "source_sha256":sha_file(Path(__file__)),
        "control_sha256":sha_file(ROOT/"submission/experiments/phase16_branch_core.py"),
        "new_sha256":sha_file(ROOT/"submission/experiments/phase19_soft_bridge_core.py"),
        "arms":list(ARMS),"data_gold_unopened":True}
    new_json(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json",meta)
    tok=TokenCounter();assert tok._encoding is not None
    stats={"n":0,"token_violations":0,"different_rendered_contexts":0,"parity_errors":0,"max_abs_token_delta":0,
        "no_model_positive_fallback":0,
        "citation_anchors":0,"bridge_source_total":0,
        "mean_token_branch":0,"mean_token_citation":0,
        "coherence_abstained_edges":0}
    start=time.monotonic()
    with CONTEXTS.open("x",encoding="utf8") as writer:
        for i in range(32):
            case=records["candidates"][i]
            first=records["first"][i];second=records["second"][i]
            coherence=records["coherence"][i]
            control=branch_locked_select(case,first,second,tok)
            intervention=select(case,coherence,control,tok)
            for c in (control,intervention):
                assert c["tokens"]==tok.count(c["text"])<=case["budget"]==c["budget"]
            out={"id":case["id"],"question":case["question"],
                "budget":case["budget"],"source_token_sum":case["source_token_sum"],
                "source_context":case["source_context"],
                "graph_branches":case["branches"],
                "candidate_map":[{"title":a["title"],"article":a["article"],
                    "sent_id":a["sent_id"]} for a in case["atoms"]],
                "coherence_verdicts":coherence["decision_ids"],
                "coherence_parse_status":coherence["parse_status"],
                "contexts":{"branch-lock":control,"soft-bridge-matched":intervention}}
            writer.write(json.dumps(out,ensure_ascii=False)+"\n")
            writer.flush();os.fsync(writer.fileno())
            stats["n"]+=1
            stats["different_rendered_contexts"]+=int(control["text"]!=intervention["text"])
            stats["no_model_positive_fallback"]+=int(intervention["soft_fallback_reason"]=="no_cross_edge_positive_citation")
            stats["citation_anchors"]+=intervention["positive_edge_anchors"]
            stats["bridge_source_total"]+=intervention["bridge_exception_sources"]
            stats["mean_token_branch"]+=control["tokens"]/32
            stats["mean_token_citation"]+=intervention["tokens"]/32
            stats["max_abs_token_delta"]=max(stats["max_abs_token_delta"],abs(control["tokens"]-intervention["tokens"]))
            stats["parity_errors"]+=int(abs(control["tokens"]-intervention["tokens"])>intervention["token_parity_tolerance"])
            assert stats["parity_errors"]==0
            stats["coherence_abstained_edges"]+=len(coherence["decision_ids"])-sum(
                v is not None for v in coherence["decision_ids"].values())
            if stats["n"]%8==0:
                print("PHASE19_CONTEXTS_PROGRESS",stats["n"],32,
                    "seconds",round(time.monotonic()-start,1),flush=True)
    new_json(FREEZE,{"n":32,"sha256":sha_file(CONTEXTS),
        "pre_gold_manifest_sha256":sha_file(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"),
        "stats":stats,"gold_unopened":True,"seconds":round(time.monotonic()-start,1)})
    print("PHASE19_CONTEXTS_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
