"""Immutable Phase15 public 2Wiki pilot and reserved holdout registration.
Exclude ALL Phase12+Phase14 touched and reserved ids. Phase13 subset Phase12.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import (
    OUT as P12, DATA,verify_data,sha_file,new_json)
from submission.experiments.phase13_register import OUT as P13
from submission.experiments.phase14_register import OUT as P14
from submission.experiments.phase15_protocol import *
OUT=ROOT/"submission/results/phase15_2wiki_targeted_edge_rescue_20261012"
def main():
    if OUT.exists():raise FileExistsError("Phase15 prereg exists")
    a=json.loads((P12/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    b=json.loads((P13/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    c=json.loads((P14/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert json.loads((P14/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))["decision"]=="STOP_NO_VALIDATION"
    prior=set(a["ids"])|set(a["reserved_ids"])|set(b["pilot_ids"])|set(b["validation_ids"])|set(c["pilot_ids"])|set(c["validation_ids"])
    assert len(prior)==320
    assert sha_file(DATA)==c["data_sha256"]
    rows=verify_data().read(columns=["id"]).to_pylist()
    ids=[r["id"] for r in rows]
    assert len(ids)==len(set(ids))==12576
    unused=sorted(set(ids)-prior,key=lambda x:(hashlib.sha256((SALT+"|"+x).encode()).hexdigest(),x))
    pilot=unused[:PILOT_N];hold=unused[PILOT_N:PILOT_N+HOLDOUT_N]
    assert len(pilot)==32 and len(hold)==96 and not (set(pilot)|set(hold))&prior
    exp=ROOT/"submission/experiments"
    frozen=["phase15_protocol.py","phase15_register.py",
            "phase13_planner.py","phase13_graph_core.py",
            "phase14_verifier.py","phase14_selection.py",
            "phase14_reader.py","phase13_graph_modal_relay_source.py",
            "phase7_modal27b_matched8.py",
            "phase11_contexts.py","phase12_contexts.py"]
    record={
        "date":"2026-10-12","study":"Phase15 conditional second-pass gold-blind edge evidence candidate rescue",
        "dataset":"2WikiMultiHopQA PUBLIC validation, not hidden test",
        "dataset_parquet_sha256":c["data_sha256"],
        "prior_used_reserved_unique_ids":len(prior),
        "pilot_ids":pilot,"validation_ids":hold,
        "sample_seed":SALT,"sample_method":"SHA256(seed|id) over complement all Phase12,13,14 prior IDs",
        "pilot_n":len(pilot),"holdout_n":len(hold),
        "pilot_hash":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
        "holdout_hash":hashlib.sha256("\n".join(hold).encode()).hexdigest(),
        "control":"top4-verified: Phase14 edge verifier on top4 original MS MARCO scores",
        "new":"conditional-rescue: same top4 decisions; for abstained edges only, Qwen same verification prompt on 4 fresh source sentences; added positive decisions anchored, same remainder scorer and cap",
        "planner":"unchanged Phase13 question-only graph Qwen27B 1-2 branches, max 4 edges",
        "initial_verifier":"unchanged Phase14 candidate verification prompt and parser, 4 originals per edge",
        "expansion_algorithm":"second batch of 4 not in original top4: rank remaining exact source sentences using 0.60 normalized original edge-query MS MARCO, 0.25 normalized (subject relation object) MS MARCO, 0.15 original-question MS MARCO, choose distinct document titles first where possible then fill by composite score",
        "rescue_verifier":"exact same prompt and Qwen model, only for initially abstained edge IDs; null on invalid JSON",
        "arms":list(METHODS),"max_edges":MAX_EDGE_COUNT,
        "top_k_initial":EDGE_TOP_K,"top_k_expansion":EXPANSION_TOP_K,
        "actual_cl100k_budget_ratio":TOKEN_CAP_RATIO,
        "anchor_share":ANCHOR_SHARE,
        "reader":"same Phase14 short answer Qwen27B prompt at temperature 0",
        "gold_primary":"all annotated exact (Wikipedia title,sentence_id) source sentences selected in final text",
        "gold_secondary":"mean support recall, pool top4 vs expanded top8 oracle accessibility, types exploratory",
        "reader_primary":"paired EM conditional-rescue minus top4-verified on same 32",
        "pilot_go":"at least +3 net EM correct of 32, zero cl100k violations; else NO HOLDOUT",
        "go_net":GATE_NET_EM,
        "no_gold_in_candidate_rank_verifier_reader":True,
        "max_real_calls":{"planner":MAX_PLANNER,"initial_verifier":MAX_INITIAL_VERIFY,
             "rescue_verifier":MAX_RESCUE_VERIFY,"reader":MAX_READER},
        "per_session_estimated_usd":MAX_EST_SESSION_USD,
        "estimated_hour_usd":ESTIMATED_HOUR_USD,"real_invoice_not_hard_cap":True,
        "qasper_sealed_test_touched":False,"production_untouched":True,
        "frozen_dependencies_sha256":{f:sha_file(exp/f) for f in frozen},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",record)
    print("PHASE15_REGISTERED",json.dumps({"excluded":len(prior),
          "pilot":len(pilot),"holdout":len(hold),
          "pilot_hash":record["pilot_hash"]}),flush=True)
if __name__=="__main__":main()
