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
from submission.experiments.phase17_register import OUT as OUT17
from submission.experiments.phase18_protocol import *
OUT=ROOT/"submission/results/phase18_2wiki_citation_closed_20261012"
def main():
    if OUT.exists():raise FileExistsError("Already preregistered, do not overwrite")
    infos=[json.loads((p/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
            for p in [OUT12,OUT13,OUT14,OUT15,OUT16,OUT17]]
    a,b,c,d,e,f=infos
    assert d["dataset_parquet_sha256"]==e["parquet_sha256"]==sha_file(DATA)
    assert json.loads((OUT17/"FUTILITY_STOP_PRE_GOLD_AUDIT.json").read_text(encoding="utf8"))["decision"].startswith("STOP_FUTILITY")
    excluded=set(a["ids"])|set(a["reserved_ids"])
    for x in (b,c,d,e,f):excluded|=set(x["pilot_ids"])|set(x["validation_ids"])
    assert len(excluded)==704
    ids=[r["id"] for r in verify_data().read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    unused=set(ids)-excluded
    order=sorted(unused,key=lambda x:(hashlib.sha256((SEED+"|"+x).encode()).hexdigest(),x))
    pilot=order[:PILOT_N]
    held=order[PILOT_N:PILOT_N+VALIDATION_N]
    assert len(pilot)==32 and len(held)==96 and not (set(pilot)|set(held))&excluded
    src=ROOT/"submission/experiments"
    deps=["phase18_protocol.py","phase18_register.py","phase17_protocol.py",
          "phase18_planner.py","phase18_candidates.py","phase18_initial_verifier.py",
          "phase18_rescue_verifier.py","phase18_coherence_verifier.py",
          "phase18_reader.py","phase18_evaluate.py",
          "phase17_planner.py","phase17_candidates.py","phase17_initial_verifier.py",
          "phase17_rescue_verifier.py","phase16_branch_core.py",
          "phase14_selection.py","phase17_coherence_core.py","phase17_reader.py",
          "phase9_crossdoc_core.py","phase13_graph_core.py",
          "phase13_graph_modal_relay_source.py",
          "phase18_citation_core.py"]
    record={
      "phase":18,"registered_before_model_inference_or_gold":True,
      "benchmark":"2WikiMultiHopQA public validation, not hidden independent test",
      "parquet_sha256":sha_file(DATA),
      "preexisting_used_reserved_id_n":len(excluded),
      "seed":SEED,
      "sample_rule":"SHA256(seed|id) on complement of all previous used/reserved ids",
      "pilot_ids":pilot,"validation_ids":held,
      "pilot_id_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
      "heldout_id_sha256":hashlib.sha256("\n".join(held).encode()).hexdigest(),
      "arms":list(ARMS),
      "structural_gate":{"at_least_N_distinct_final_reader_prompts":STRUCTURAL_DIFFERENT_MIN,
           "zero_token_violations":True,"before_paid_reader":True},
      "pre_reader_decision":"if structural gate fails STOP_NO_READER_NO_GOLD; do not weaken gate posthoc",
      "control":"Phase16 branch-lock unchanged on exact same graph, reranker scores and Qwen edge decisions",
      "new":"citation-closed: same question-level Qwen coherence picks, mandatory positive citation anchors under exact token cap; banned unapproved candidates of Qwen-abstained edges; filler restricted to Wikipedia articles with one positively cited anchor; zero-positive fallback to branch-lock",
      "semantic_caution":"The consistency LLM is not a formal entailment checker; no supported candidate means abstain, not gold-known false. Branch coverage != gold support completeness.",
      "priorities":["Qwen-positive citation anchors under exact rendered cap",
                     "veto rejected edge candidate sentences unless cited by a positive edge",
                     "filler ONLY from positively cited Wikipedia article IDs",
                     "no positive citations -> deterministic unchanged branch-lock fallback"],
      "max_alternatives_per_edge":3,
      "max_graph_edges":4,
      "coherence_system_prompt_sha256":sha_file(src/"phase17_protocol.py"),
      "cap_ratio":CAP_RATIO,
      "hard_no_budget_violation":True,
      "primary_reader":"paired normalized Exact Match citation-closed minus branch-lock, conditional on preregistered structural gate",
      "gate":"first at least 8 different paired reader payloads with 0 cap violations; then citation-closed >=+3 net EM correct vs branch-lock to unlock 96; otherwise 96 untouched",
      "pilot_go_net":GATE_NET_EM,
      "secondary":["all gold exact (title,sentence-index) pairs","mean exact gold sentence recall",
                   "branch anchors supported vs provisional","gold-aware feasibility purely after freeze"],
      "projection_before_gold":["id","question","context.title","context.sentences"],
      "gold_eligible_only_post_reader":["answer","supporting_facts","type","evidences"],
      "per_session_cost_estimate_usd":SESSION_ESTIMATED_USD_GUARD,
      "estimated_hour_usd":ESTIMATED_USD_PER_HOUR,
      "cost_estimate_not_invoice":True,
      "remote_call_guards":{"planner":32,"initial_verifier":32,"rescue_verifier":32,"coherence":32,"reader":64},
      "no_qasper_official_test":True,"production_untouched":True,
      "code_sha256":{p:sha_file(src/p) for p in deps},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",record)
    print("PHASE18_PRE_REGISTERED",json.dumps({
       "pilot":len(pilot),"validation":len(held),
       "excluded":len(excluded),"pilot_hash":record["pilot_id_sha256"]}),flush=True)
if __name__=="__main__":main()
