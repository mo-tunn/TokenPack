"""Faz19 one-time gold-blind public 2Wiki validation preregistration.

Use fresh IDs excluding every used/reserved 2Wiki phase 12-18 ID; do not
reopen previous official QASPER test, overwrite results or touch production.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import (
    OUT as P12,DATA,verify_data,sha_file,new_json)
from submission.experiments.phase13_register import OUT as P13
from submission.experiments.phase14_register import OUT as P14
from submission.experiments.phase15_register import OUT as P15
from submission.experiments.phase16_register import OUT as P16
from submission.experiments.phase17_register import OUT as P17
from submission.experiments.phase18_register import OUT as P18
from submission.experiments.phase19_protocol import *
OUT=ROOT/"submission/results/phase19_2wiki_soft_bridge_matched_20261012"

def main():
    if OUT.exists():raise FileExistsError("Existing Phase19 prereg. Never overwrite.")
    predecessors=[P12,P13,P14,P15,P16,P17,P18]
    p=[json.loads((d/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
        for d in predecessors]
    assert p[-1]["phase"]==18
    assert json.loads((P18/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))["decision"]=="STOP_NO_VALIDATION"
    assert sha_file(DATA)==p[-1]["parquet_sha256"]
    seen=set(p[0]["ids"])|set(p[0]["reserved_ids"])
    for r in p[1:]:
        seen|=set(r["pilot_ids"])|set(r["validation_ids"])
    assert len(seen)==832
    ids=[r["id"] for r in verify_data().read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    ranked=sorted(set(ids)-seen,key=lambda v:(
        hashlib.sha256((SEED+"|"+v).encode()).hexdigest(),v))
    pilot=ranked[:PILOT_N]
    holdout=ranked[PILOT_N:PILOT_N+HOLDOUT_N]
    assert len(pilot)==32 and len(holdout)==96
    assert not(set(pilot)|set(holdout))&seen
    src=ROOT/"submission/experiments"
    srcfiles=[
        "phase19_protocol.py","phase19_register.py",
        "phase19_soft_bridge_core.py","phase19_planner.py",
        "phase19_candidates.py","phase19_initial_verifier.py",
        "phase19_rescue_verifier.py","phase19_coherence_verifier.py",
        "phase19_reader.py","phase19_evaluate.py",
        "phase16_branch_core.py",
        "phase13_graph_core.py","phase17_coherence_core.py",
        "phase13_graph_modal_relay_source.py",
        "phase7_modal27b_matched8.py",
        "phase9_crossdoc_core.py"
    ]
    manifest={
        "phase":19,"date":"2026-10-12",
        "study":"soft cross-document bridge exceptions with ACTUAL rendered token parity",
        "registered_before_planner_and_gold":True,
        "dataset":"2WikiMultiHopQA PUBLIC validation (NOT sealed test)",
        "parquet_sha256":sha_file(DATA),
        "all_prior_used_or_reserved_excluded_n":len(seen),
        "seed":SEED,
        "sample_rule":"SHA256(seed|id), no question/gold conditioning",
        "pilot_ids":pilot,"validation_ids":holdout,
        "pilot_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
        "validation_sha256":hashlib.sha256("\n".join(holdout).encode()).hexdigest(),
        "methods":list(ARMS),
        "source_projection_before_gold":["id","question","context.title","context.sentences"],
        "control":"unchanged Phase16 branch-lock applied to new Phase19 frozen gold-blind inputs",
        "new":"Qwen coherence-positive original citations pinned first; up to 2 extra original-source bridge documents; Qwen-abstained candidate penalty, not veto; fill from baseline+all source paragraphs with relevance and lexical crossdoc links",
        "positive_citation_is_not_verified_entailment":True,
        "per_question_actual_token_parity":"abs(new_actual_cl100k_tokens-control_actual)<=max(8,ceil(.05*control_actual)); otherwise unchanged control fallback",
        "token_ratio_tolerance":TOKEN_RATIO_TOLERANCE,
        "token_abs_tolerance":TOKEN_ABSOLUTE_TOLERANCE,
        "max_extra_bridge_documents":MAX_BRIDGE_DOCS,
        "max_render_cap":BUDGET_RATIO,
        "pre_reader_structural_gate":{"changed_reader_payloads_min":STRUCTURAL_CHANGED_MIN,
           "token_matching_all_32":True,"no_25pct_cap_violations":True},
        "gate_failure":"STOP_NO_READER_NO_GOLD, retain all 96 unused IDs",
        "primary_answer":"paired normalized EM new minus control, 32",
        "reader_pilot_gate_net_correct":GATE_NET_EM,
        "success":"if reader net gain >=3 and zero cap violations, only then 96 validation eligible",
        "secondary_gold_post_reader":"all exact (title,sentence id) supporting sentences, mean support recall, type stratification, token and distinct source matching",
        "gold_never_in_planner_ranker_or_Qwen":True,
        "reader_prompt":"Phase7/18 Qwen27B temp0 short answer, exact per-case prompt re-use",
        "max_paid_calls":{"planner":MAX_PLANNER_CALLS,"first_verifier":MAX_INITIAL_VERIFY_CALLS,
           "rescue":MAX_RESCUE_CALLS,"coherence":MAX_COHERENCE_CALLS,"reader":MAX_READER_CALLS},
        "modal_secret":"qwen-modal-secrets via Modal, no local secret read",
        "estimated_hourly_USD":ESTIMATED_HOURLY_USD,
        "session_estimated_USD_guard":PER_SESSION_EST_USD_GUARD,
        "real_modal_invoice_unverified":True,
        "production_unchanged":True,
        "qasper_sealed_test_never_used":True,
        "source_sha256":{name:sha_file(src/name) for name in srcfiles},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",manifest)
    print("PHASE19_PREREG",json.dumps({
        "excluded":len(seen),"pilot":len(pilot),"validation":len(holdout),
        "pilot_hash":manifest["pilot_sha256"]}),flush=True)
if __name__=="__main__":main()
