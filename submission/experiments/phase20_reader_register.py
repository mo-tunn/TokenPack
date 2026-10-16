"""Phase20B preregistration: randomized-by-ID 2Wiki public validation pilot.

Builds on Phase14 edge-verified original source sentence selector.
No earlier pilot IDs, including all reserved holdouts from Faz12-19, are reused.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import (
    OUT as P12,DATA,verify_data,sha_file,new_json)
from submission.experiments.phase13_register import OUT as P13
from submission.experiments.phase14_register import OUT as P14
from submission.experiments.phase15_register import OUT as P15
from submission.experiments.phase16_register import OUT as P16
from submission.experiments.phase17_register import OUT as P17
from submission.experiments.phase18_register import OUT as P18
from submission.experiments.phase19_register import OUT as P19
from submission.experiments.phase20_reader_protocol import *
OUT=ROOT/"submission/results/phase20_evidence_first_reader_20261012"
TAXONOMY=ROOT/"submission/results/phase20_error_taxonomy_20261012"
def main():
    if OUT.exists():raise FileExistsError("Do not overwrite preregistered output")
    paths=[P12,P13,P14,P15,P16,P17,P18,P19]
    registers=[json.loads((d/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
               for d in paths]
    excluded=set(registers[0]["ids"])|set(registers[0]["reserved_ids"])
    for info in registers[1:]:
        excluded|=set(info["pilot_ids"])|set(info["validation_ids"])
    assert len(excluded)==960
    assert sha_file(DATA)==registers[-1]["parquet_sha256"]
    tax=TAXONOMY/"RETROSPECTIVE_TAXONOMY_SUMMARY.json"
    assert tax.exists()
    summary=json.loads(tax.read_text(encoding="utf8"))
    assert summary["n"]==160 and summary["cross_phase_accuracy_is_not_head_to_head"]
    assert summary["reader_error_despite_gold_complete"]==21
    ids=[x["id"] for x in verify_data().read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    sampled=sorted(set(ids)-excluded,
      key=lambda id:(hashlib.sha256((SEED+"|"+id).encode()).hexdigest(),id))
    pilot=sampled[:PILOT_N];hold=sampled[PILOT_N:PILOT_N+VALIDATION_N]
    assert len(pilot)==32 and len(hold)==96
    assert not(set(pilot)|set(hold))&excluded
    root=ROOT/"submission/experiments"
    codes=["phase20_reader_protocol.py","phase20_reader_register.py",
           "phase20_reader_planner.py","phase20_reader_candidates.py",
           "phase20_reader_verifier.py","phase20_reader_contexts.py",
           "phase20_reader_run.py","phase14_selection.py",
           "phase13_graph_core.py","phase13_planner.py",
           "phase7_modal27b_matched8.py"]
    prereg={
      "study":"Phase20B same-context Qwen27B explicit source-quote multi-hop reasoning reader",
      "data":"2WikiMultiHopQA PUBLIC validation, NOT independent held-out test",
      "data_sha256":sha_file(DATA),
      "phase20A_taxonomy_summary_sha256":sha_file(tax),
      "historical_evidence_context":"Phase14 edge-verified 19/32; Phase18 branch-lock 19/32 on DIFFERENT ID samples; winner NOT established; Phase14 chosen as interpretable source-verified starting point",
      "excluded_prior_used_reserved_ids":len(excluded),
      "sample_method":"SHA256(seed|id) only, no gold/type conditioning",
      "sample_seed":SEED,
      "pilot_ids":pilot,"validation_ids":hold,
      "pilot_id_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
      "validation_id_sha256":hashlib.sha256("\n".join(hold).encode()).hexdigest(),
      "arms":list(ARMS),
      "common_goldblind_context_policy":"byte-for-byte unchanged Phase14 edge_verified_select on same Phase14 question-only graph + zero-shot MiniLM + top4 original source sentences verified by Qwen",
      "reader_control":"unchanged Phase14/Phase7 short direct answer Qwen27B reader",
      "reader_new":"same exact question and FINAL rendered original evidence text; Qwen evidence-first structured intermediate facts + direct supporting original quote + answer JSON",
      "reader_inference":"temperature 0, Qwen27B fixed, new reader max_tokens 480, baseline prompt max_tokens unchanged Phase7; extra generation budget a genuine confound of treatment",
      "empty_or_invalid_JSON":"record raw output, robustly parse answer, missing/invalid output yields empty prediction and EM0; never drop case or retry selectively",
      "gold_columns_visible_before_freeze":["id","question","context.title","context.sentences"],
      "gold_never_sent_to_modal":True,
      "final_render_cl100k_max_ratio":.25,
      "primary_metric":"paired normalized EM difference: verified-evidence-first minus verified-direct on same 32",
      "secondary_metrics":["answer F1","exact original gold source sentence support","JSON validity","quote actually found in provided source"],
      "go_to_96_rule":"at least +3 net EM correct and zero actual context-cap violations; else stop and DO NOT open 96",
      "net_EM_gate":GATE_NET_EM,
      "no_reader_tuning_on_pilot_gold":True,
      "estimated_inference_call_guard":{"planner":32,"verifier":32,"two_readers":64},
      "cost_guards_estimated_not_invoice":True,
      "QASPER_official_sealed_test_touched":False,
      "production_code_touched":False,
      "source_sha256":{p:sha_file(root/p) for p in codes},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",prereg)
    print("PHASE20B_PREREGISTERED",json.dumps({
       "excluded":len(excluded),"pilot":len(pilot),"validation":len(hold),
       "pilot_hash":prereg["pilot_id_sha256"]}),flush=True)
if __name__=="__main__":main()
