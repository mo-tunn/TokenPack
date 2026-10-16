"""Freeze SAME Faz14 edge-verified source context for both Phase22 readers.

Crucial: unlike recent policies, no packing policy differs between arms.
Gold never read. Per-original-source exact rendered cl100k_base 25% limit.
"""
from __future__ import annotations
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase22_reader_register import OUT
from submission.experiments.phase22_reader_protocol import ARMS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase14_selection import edge_verified_select
from tokenpack.tokenization import TokenCounter

def main():
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    cmeta=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    vmeta=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    scored=OUT/"scored_candidate_pools.jsonl"
    verified=OUT/"verifier_responses.jsonl"
    result=OUT/"gold_blind_contexts.jsonl"
    frozen=OUT/"CONTEXTS_FROZEN.json"
    assert sha_file(scored)==cmeta["candidate_sha256"]
    assert sha_file(verified)==vmeta["responses_sha256"]
    cases=[json.loads(line) for line in scored.open(encoding="utf8") if line.strip()]
    votes=[json.loads(line) for line in verified.open(encoding="utf8") if line.strip()]
    assert [x["id"] for x in cases]==[x["id"] for x in votes]==manifest["pilot_ids"]
    if result.exists() or frozen.exists():raise FileExistsError("Already frozen")
    new_json(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json",{
       "code_sha256":sha_file(Path(__file__)),
       "selector_core_sha256":sha_file(ROOT/"submission/experiments/phase14_selection.py"),
       "candidate_sha256":cmeta["candidate_sha256"],
       "verifier_sha256":vmeta["responses_sha256"],
       "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
       "arms":list(ARMS),"only_one_shared_context_per_question":True,
       "gold_columns_read":False})
    tok=TokenCounter();assert tok._encoding is not None
    stats={"n":0,"cap_violations":0,"mean_tokens":0,
           "verified_edge_positive":0,"verified_edge_selected":0,
           "verifier_no_positive_fallback":0}
    with result.open("x",encoding="utf8") as f:
        for case,vote in zip(cases,votes):
            ctx=edge_verified_select(case,vote,tok)
            assert ctx["tokens"]==tok.count(ctx["text"])<=ctx["budget"]==case["budget"]
            ctx["selected_paragraph_count"]=len({
                case["atoms"][i]["article"] for i in ctx["selected_sentence_indexes"]})
            shared={"id":case["id"],"question":case["question"],
                "budget":case["budget"],"source_token_sum":case["source_token_sum"],
                "source_context":case["source_context"],
                "candidate_map":[{"title":a["title"],"sent_id":a["sent_id"],
                                  "article":a["article"]} for a in case["atoms"]],
                "graph_branches":case["branches"],
                "verification_status":vote["status"],
                "contexts":{arm:dict(ctx) for arm in ARMS}}
            assert shared["contexts"][ARMS[0]]["text"]==shared["contexts"][ARMS[1]]["text"]
            f.write(json.dumps(shared,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            stats["n"]+=1
            stats["mean_tokens"]+=ctx["tokens"]/32
            stats["verified_edge_positive"]+=ctx["verified_edge_pick_count"]
            stats["verified_edge_selected"]+=ctx["verified_edge_selected"]
            stats["verifier_no_positive_fallback"]+=int(ctx["verification_fallback_to_graph"])
    new_json(frozen,{"n":32,"sha256":sha_file(result),
         "gold_unopened":True,"arms_share_identical_context_text":True,
         "pre_gold_protocol_sha256":sha_file(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"),
         "stats":stats})
    print("PHASE22_CONTEXTS_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
