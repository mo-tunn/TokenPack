"""Phase14 final gold-blind frozen 3-arm contexts from fixed scores + decisions.
No gold loaded. All arms use the SAME actual cl100k_base 25% rendered cap.
"""
from __future__ import annotations
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase14_register import OUT
from submission.experiments.phase14_protocol import ARMS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase13_graph_core import graph_select
from submission.experiments.phase11_contexts import reranked_selection
from submission.experiments.phase14_selection import edge_verified_select
from tokenpack.tokenization import TokenCounter
POOLS=OUT/"scored_candidate_pools.jsonl"
VERIFIER=OUT/"verifier_responses.jsonl"
CONTEXTS=OUT/"gold_blind_contexts.jsonl"
FROZEN=OUT/"CONTEXTS_FROZEN.json"

def main():
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    candidates=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    verified=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(POOLS)==candidates["candidate_sha256"]
    assert sha_file(VERIFIER)==verified["responses_sha256"]
    rows=[json.loads(s) for s in POOLS.open(encoding="utf8") if s.strip()]
    verdicts=[json.loads(s) for s in VERIFIER.open(encoding="utf8") if s.strip()]
    ids=protocol["pilot_ids"]
    assert [r["id"] for r in rows]==[r["id"] for r in verdicts]==ids
    if CONTEXTS.exists() or FROZEN.exists():raise FileExistsError("Immutable output exists")
    prereg={
       "pilot_id_sha256":protocol["pilot_id_sha256"],
       "data_sha256":protocol["data_sha256"],
       "candidate_sha256":candidates["candidate_sha256"],
       "verifier_sha256":verified["responses_sha256"],
       "selector_code_sha256":sha_file(
           ROOT/"submission/experiments/phase14_selection.py"),
       "graph_code_sha256":sha_file(
           ROOT/"submission/experiments/phase13_graph_core.py"),
       "flat_code_sha256":sha_file(
           ROOT/"submission/experiments/phase11_contexts.py"),
       "this_code_sha256":sha_file(Path(__file__)),
       "max_ratio":.25,"arms":list(ARMS),
       "gold_opened":False}
    new_json(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json",prereg)
    counter=TokenCounter()
    assert counter._encoding is not None
    stats={"n":0,"token_violations":0,"changed_verified_vs_graph":0,
       "changed_verified_vs_flat":0,
       "verified_picks":0,"verified_selected":0,
       "fallback_to_graph":0,"infeasible_verified":0}
    start=time.monotonic()
    with CONTEXTS.open("x",encoding="utf8") as f:
        for row,verdict in zip(rows,verdicts):
            assert verdict["id"]==row["id"]
            atoms=row["atoms"]
            scores=row["edge_scores"]
            budget=row["budget"]
            graph=graph_select(atoms,row["original_scores"],
                scores,row["branches"],budget,counter)
            flat=reranked_selection(atoms,row["original_scores"],scores,budget,counter)
            final=edge_verified_select(row,verdict,counter)
            arms={"flat-edge":flat,"graph-path":graph,"edge-verified":final}
            assert set(arms)==set(ARMS)
            for c in arms.values():
                assert c["tokens"]==counter.count(c["text"])<=budget==c["budget"]
                # The Phase11 flat selector's bookkeeping can count unselected
                # docs. Recompute ONLY metadata from actual indices, never change
                # selected text/token/semantic retrieval decision.
                c["selected_paragraph_count"]=len({
                    atoms[i]["article"] for i in c["selected_sentence_indexes"]})
            out={"id":row["id"],"question":row["question"],
                "budget":budget,"source_token_sum":row["source_token_sum"],
                "source_context":row["source_context"],
                "candidate_map":[{"title":a["title"],"sent_id":a["sent_id"],
                                  "article":a["article"]} for a in atoms],
                "edge_queries":row["edge_queries"],
                "graph_branches":row["branches"],
                "verification_status":verdict["status"],
                "verification_decision_ids":verdict["decision_ids"],
                "contexts":arms}
            f.write(json.dumps(out,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            stats["n"]+=1
            stats["changed_verified_vs_graph"]+=int(
                final["text"]!=graph["text"])
            stats["changed_verified_vs_flat"]+=int(
                final["text"]!=flat["text"])
            stats["verified_picks"]+=final["verified_edge_pick_count"]
            stats["verified_selected"]+=final["verified_edge_selected"]
            stats["fallback_to_graph"]+=int(final["verification_fallback_to_graph"])
            stats["infeasible_verified"]+=final["rejected_verified_due_budget"]
            if stats["n"]%8==0:
                print("PHASE14_GOLDBLIND_CONTEXTS_PROGRESS",stats["n"],len(rows),
                      "elapsed_s",round(time.monotonic()-start,1),flush=True)
    new_json(FROZEN,{"n":len(rows),"sha256":sha_file(CONTEXTS),
          "plan_sha256":sha_file(OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"),
          "stats":stats,"gold_opened":False,
          "production_unchanged":True,
          "elapsed_seconds":round(time.monotonic()-start,1)})
    print("PHASE14_GOLDBLIND_CONTEXTS_FROZEN",json.dumps(stats),flush=True)

if __name__=="__main__":main()
