"""One-time Faz16 prereg: 32 public validation pilot + 96 reserved,
excluding every ID touched or reserved in Faz12-15. Gold never read here.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import (
    OUT as OUT12,DATA,verify_data,sha_file,new_json)
from submission.experiments.phase13_register import OUT as OUT13
from submission.experiments.phase14_register import OUT as OUT14
from submission.experiments.phase15_register import OUT as OUT15
from submission.experiments.phase16_register import OUT as OUT16
from submission.experiments.phase17_protocol import *
OUT=ROOT/"submission/results/phase17_2wiki_cross_edge_coherence_20261012"
def main():
    if OUT.exists():raise FileExistsError("Already preregistered, do not overwrite")
    infos=[json.loads((p/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
            for p in [OUT12,OUT13,OUT14,OUT15,OUT16]]
    a,b,c,d,e=infos
    assert d["dataset_parquet_sha256"]==e["parquet_sha256"]==sha_file(DATA)
    assert json.loads((OUT16/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))["decision"]=="STOP_NO_VALIDATION"
    excluded=set(a["ids"])|set(a["reserved_ids"])
    for x in (b,c,d,e):excluded|=set(x["pilot_ids"])|set(x["validation_ids"])
    assert len(excluded)==576
    ids=[r["id"] for r in verify_data().read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    unused=set(ids)-excluded
    order=sorted(unused,key=lambda x:(hashlib.sha256((SEED+"|"+x).encode()).hexdigest(),x))
    pilot=order[:PILOT_N]
    held=order[PILOT_N:PILOT_N+HOLDOUT_N]
    assert len(pilot)==32 and len(held)==96 and not (set(pilot)|set(held))&excluded
    src=ROOT/"submission/experiments"
    deps=["phase17_protocol.py","phase17_register.py","phase16_protocol.py",
          "phase16_planner.py","phase16_candidates.py","phase16_initial_verifier.py",
          "phase16_rescue_verifier.py","phase16_branch_core.py",
          "phase14_selection.py","phase16_reader.py",
          "phase9_crossdoc_core.py","phase13_graph_core.py",
          "phase13_graph_modal_relay_source.py",
          "phase17_coherence_core.py"]
    record={
      "phase":17,"registered_before_model_inference_or_gold":True,
      "benchmark":"2WikiMultiHopQA public validation, not hidden independent test",
      "parquet_sha256":sha_file(DATA),
      "preexisting_used_reserved_id_n":len(excluded),
      "seed":SEED,
      "sample_rule":"SHA256(seed|id) on complement of all previous used/reserved ids",
      "pilot_ids":pilot,"validation_ids":held,
      "pilot_id_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
      "heldout_id_sha256":hashlib.sha256("\n".join(held).encode()).hexdigest(),
      "arms":list(ARMS),
      "control":"Phase16 branch-lock: same planner, same original/expanded candidate pool and source verification, same branch-locked packing",
      "new":"coherence-gated: same Phase16 upstream plus one question-level Qwen cross-edge consistency pass over at most 3 candidate original sentences per graph edge; use Qwen coherence picks as priority anchors, null means no forced evidence for that edge, then same fixed final rendered cap and relevance filler",
      "semantic_caution":"The consistency LLM is not a formal entailment checker; no supported candidate means abstain, not gold-known false. Branch coverage != gold support completeness.",
      "priorities":["coherence model-selected original candidate for each edge, if any; exact token budget",
                    "preserve any initial/rescue verified positives not contradicted by coherence selection",
                    "MS MARCO relevance fill after anchors"],
      "max_alternatives_per_edge":TOP_CANDIDATES_PER_EDGE,
      "max_graph_edges":MAX_GRAPH_EDGES,
      "coherence_system_prompt":COHERENCE_SYSTEM,
      "cap_ratio":CAP_RATIO,
      "hard_no_budget_violation":True,
      "primary_reader":"paired normalized Exact Match coherence-gated minus branch-lock on same fixed 32",
      "gate":"coherence-gated +3 or more net EM correct vs branch-lock, zero cap breaches -> validation; otherwise 96 validation untouched",
      "pilot_go_net":GATE_NET_EM,
      "secondary":["all gold exact (title,sentence-index) pairs","mean exact gold sentence recall",
                   "branch anchors supported vs provisional","gold-aware feasibility purely after freeze"],
      "projection_before_gold":["id","question","context.title","context.sentences"],
      "gold_eligible_only_post_reader":["answer","supporting_facts","type","evidences"],
      "per_session_cost_estimate_usd":MAX_ESTIMATED_USD,
      "estimated_hour_usd":ESTIMATED_HOURLY_USD,
      "cost_estimate_not_invoice":True,
      "remote_call_guards":{"planner":32,"initial_verifier":32,"rescue_verifier":32,"coherence":32,"reader":64},
      "no_qasper_official_test":True,"production_untouched":True,
      "code_sha256":{p:sha_file(src/p) for p in deps},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",record)
    print("PHASE17_PRE_REGISTERED",json.dumps({
       "pilot":len(pilot),"validation":len(held),
       "excluded":len(excluded),"pilot_hash":record["pilot_id_sha256"]}),flush=True)
if __name__=="__main__":main()
