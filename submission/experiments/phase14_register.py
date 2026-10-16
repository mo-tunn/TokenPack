"""Faz14 sample freeze: untouched public 2Wiki validation IDs, gold-blind.
Never reuse Phase12/13 used/reserved IDs. Production & QASPER untouched.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import (
    OUT as P12,DATA,verify_data,sha_file,new_json)
from submission.experiments.phase13_register import OUT as P13
from submission.experiments.phase14_protocol import *
OUT=ROOT/"submission/results/phase14_2wiki_edge_verification_20261012"

def main():
    if OUT.exists():raise FileExistsError("Never overwrite preregistration")
    a=json.loads((P12/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    b=json.loads((P13/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert b["parent_phase12_protocol_sha256"]==sha_file(P12/"PRE_GOLD_PROTOCOL.json")
    assert json.loads((P13/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))["decision"]=="STOP_NO_VALIDATION"
    excluded=set(a["ids"])|set(a["reserved_ids"])|set(b["pilot_ids"])|set(b["validation_ids"])
    assert len(excluded)==192 and len(a["ids"])==64
    assert a["parquet_sha256"]==b["data_sha256"]==sha_file(DATA)
    ids=[r["id"] for r in verify_data().read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    order=sorted(set(ids)-excluded,key=lambda v:(hashlib.sha256((SALT+"|"+v).encode()).hexdigest(),v))
    pilot=order[:PILOT_N];validation=order[PILOT_N:PILOT_N+VALIDATION_N]
    assert len(pilot)==32 and len(validation)==96 and not (set(pilot)|set(validation))&excluded
    src=ROOT/"submission/experiments"
    files=["phase14_protocol.py","phase14_register.py",
           "phase13_graph_core.py","phase13_graph_modal_relay_source.py",
           "phase11_contexts.py","phase9_crossdoc_core.py",
           "phase12_contexts.py","phase7_modal27b_matched8.py"]
    info={
       "study":"Faz14 gold-blind Qwen sentence relation verification vs graph-path & flat-edge",
       "registered_before_gold_source_or_model_planning":True,
       "dataset":"2WikiMultiHopQA public validation, NOT hidden test",
       "data_sha256":sha_file(DATA),
       "phase12_excluded_used_and_reserved":len(set(a["ids"])|set(a["reserved_ids"])),
       "phase13_overlap_with_phase12_reserved":len(set(a["reserved_ids"])&set(b["pilot_ids"])),
       "excluded_unique_ids":len(excluded),
       "selection":"SHA256(seed|id) over complement; no answer/type/gold conditioning",
       "seed":SALT,"pilot_ids":pilot,"validation_ids":validation,
       "pilot_id_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
       "validation_id_sha256":hashlib.sha256("\n".join(validation).encode()).hexdigest(),
       "arms":list(ARMS),
       "planner":"Phase13 question-only 1-2 branches, 1-2 relations per branch",
       "verifier":"Qwen27B uses original question, graph edge, and <=4 scored ORIGINAL sentences PER edge; judges whether candidate directly supports relation, returns ID or null",
       "verifier_is_model_judgment_not_formal_entailment":True,
       "gold_free_projection":["id","question","context.title","context.sentences"],
       "max_candidates_per_edge":EDGE_TOP_K,
       "verifier_output_tokens":MAX_VERIFIER_OUTPUT_TOKENS,
       "graph_edges_max":EDGE_MAX_COUNT,
       "token_ratio":BUDGET_RATIO,
       "edge_anchor_share":EDGE_ANCHOR_SHARE,
       "selection":"First one verified candidate per distinct relation, by deterministic covered edges and real final token cap, then original graph fill; missing/invalid verification falls back to unverified graph-path without dropping question",
       "fairness":"flat-edge and graph-path share exact same phase14 graph edge queries/reranker scores as verifier candidates",
       "paired_gold_primary":"all exact supporting (source title, original sentence-index) pairs in rendered selection",
       "reader_primary":"graph-path vs edge-verified Qwen27B normalized EM on same 32 questions",
       "reader_secondary":"F1 and flat-edge comparator",
       "reader_gate":"edge-verified minus graph-path >= +3 net EM correct out of 32, zero budget violation; else 96 untouched",
       "reader_gate_net_correct":GATE_NET_EM,
       "max_paid_calls":{"planner":32,"verifier":32,"reader":96},
       "budget_estimate_usd_per_hour":ESTIMATED_HOUR_USD,
       "per_session_estimate_guard_usd":MAX_SESSION_ESTIMATED_USD,
       "invoice_not_hard_cost_cap":True,
       "potential_multicomparison_bias":True,
       "QASPER_official_test_never_opened":True,
       "production_untouched":True,
       "source_sha256":{f:sha_file(src/f) for f in files}
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",info)
    print("PHASE14_REGISTERED",json.dumps({"pilot":32,"validation":96,
        "excluded":len(excluded),"pilot_hash":info["pilot_id_sha256"]}),flush=True)
if __name__=="__main__":main()
