"""Register ONE new, not-yet-executed hypothesis on untouched 2Wiki IDs.

Faz23 research labels are posthoc: reported as motivation only. They are
NOT fed into model, retrieval, prompt examples or sample selection.
This script reads ONLY the dataset id column and old manifests.
"""
from __future__ import annotations
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
from submission.experiments.phase20_reader_register import OUT as P20
from submission.experiments.phase21_reader_register import OUT as P21
from submission.experiments.phase22_reader_register import OUT as P22
from submission.experiments.phase24_answer_span_core import SYSTEM,USER,needs_review
OUT=ROOT/"submission/results/phase24_answer_span_prereg_20261012"
SEED="TokenPack-phase24-source-anchored-answer-span-20261012-v1"
PILOT_N=64
VALIDATION_N=192
def h(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("Already preregistered: do not overwrite")
    paths=[P12,P13,P14,P15,P16,P17,P18,P19,P20,P21,P22]
    old=[json.loads((d/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
         for d in paths]
    seen=set(old[0]["ids"])|set(old[0]["reserved_ids"])
    for x in old[1:]:
        seen|=set(x["pilot_ids"])|set(x["validation_ids"])
    assert len(seen)==1344
    assert sha_file(DATA)==old[-1]["data_sha256"]
    assert json.loads((P22/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))[
        "decision"]=="STOP_NO_VALIDATION"
    ids=[r["id"] for r in verify_data().read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    order=sorted(set(ids)-seen,key=lambda key:(h(SEED+"|"+key),key))
    pilot=order[:PILOT_N];validate=order[PILOT_N:PILOT_N+VALIDATION_N]
    assert len(pilot)==64 and len(validate)==192
    assert set(pilot).isdisjoint(validate)
    assert not (set(pilot)|set(validate))&seen
    core=ROOT/"submission/experiments/phase24_answer_span_core.py"
    diagnostics=ROOT/"submission/results/phase23_semantic_failure_audit_20261012"
    taxa=diagnostics/"MANUAL_SEMANTIC_TAXONOMY.json"
    info=json.loads(taxa.read_text(encoding="utf8"))
    assert info["summary"]["taxonomy"]["ANSWER_SPAN_OVER_SPECIFIED"]==5
    assert info["summary"]["n_full_gold_but_hybrid_wrong"]==8
    protocol={
      "phase":24,
      "status":"PRE_REGISTERED_ONLY; no model calls, no gold opened for pilot/validation",
      "basis":"Faz23 local retrospective analysis: 5/8 gold-complete EM failures have contiguous gold string in longer model answer; not certified semantic equivalence; single unblinded reviewer",
      "hypothesis":"A strict source-grounded answer-only contiguous substring contractor improves normalized Exact Match versus unchanged Phase14 edge-verified plus short direct reader on unseen 2Wiki questions WITHOUT unacceptable meaning drift.",
      "single_modification":"Postprocessing existing direct reader answer with one Qwen27B call only for eligible verbose answers; extraction strictly contiguous and whole-word-bounded; fail-closed fallback to original answer",
      "baseline":"Phase14 frozen edge_verified_select original source sentences, exact cl100k 25% cap; original Qwen27B short direct answer via full Phase13 relay",
      "experimental":"Same exact source context, identical first Qwen answer and inference parameters; if answer contains >=3 whitespace words OR a comma OR parenthesis, fixed extra Qwen span contract prompt (max 60 tokens), accepted only if exact source-answer substring at whole-word bounds; otherwise original answer.",
      "no_reranking_or_extra_retrieval":True,
      "dataset":"2WikiMultiHopQA PUBLIC validation, NOT independent hidden test",
      "dataset_parquet_sha256":sha_file(DATA),
      "prior_pilot_and_reserved_ids_excluded":len(seen),
      "source_id_only_hash_sampling":SEED,
      "pilot_ids":pilot,"validation_ids":validate,
      "pilot_id_sha256":h("\n".join(pilot)),
      "validation_id_sha256":h("\n".join(validate)),
      "gold_schema_not_read_pre_execution":True,
      "model_gold_answer_never_sent":True,
      "preemptive_model_code_sha256":sha_file(core),
      "model_prompt_system_sha256":h(SYSTEM),
      "model_prompt_user_template_sha256":h(USER),
      "previous_phase14_context_selector_sha256":sha_file(
          ROOT/"submission/experiments/phase14_selection.py"),
      "modal_full_relay_sha256":sha_file(
          ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
      "phase23_diagnostic_sha256":sha_file(taxa),
      "pairing":"Same 64 questions, same original source context, same original direct reader output; only postprocessor output differs",
      "main_metric":"paired normalized EM new minus direct across all 64, including skipped/failed contractions; no per-case filtering",
      "secondary_metrics":["paired answer F1","accepted contraction count","token cost",
            "model invalid/non-substring return frequency",
            "blinded semantic-fidelity judgments on ALL actually changed answers",
            "factual equivalence and answer specificity, not only gold substring hit"],
      "semantic_noninferiority_requirement":"Changed predictions must undergo independently blinded semantic review against same question/original source without gold. Any unreviewed set cannot be called confirmatory. Predeclared severe meaning drift tolerance=0 (or stop). Two independent annotators should adjudicate; if unavailable, report as exploratory only.",
      "pilot_success_gate":{
           "min_net_new_correct":6,
           "paired_two_sided_exact_mcnemar_max_p":0.05,
           "cap_violations_allowed":0,
           "confirmed_severe_semantic_regressions_allowed":0,
           "semantic_review_required_for_confirmatory_claim":True},
      "failure_decision":"STOP_NO_VALIDATION; do not run or reveal gold for 192",
      "validation_limit":"192 separately reserved public validation IDs used ONLY after passing all pilot gates; public validation is not an independent secret test",
      "cost_estimates":"64 planner +64 initial source-verifier +64 baseline reader + <=64 contraction calls pilot; 192 new validation only after gate. Estimated USD NOT actual invoice.",
      "leak_prevention":"Model only question, original gold-blind source context and initial predicted answer. Never gold targets, gold support labels, phase23 manually reviewed examples. Freeze all outputs before gold scoring.",
      "qasper_phase3D_official_test_must_not_reopen":True,
      "production_untouched":True,
      "no_inference_requested_in_this_preregistration":True,
      "no_hypothesis_change_after_gold":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",protocol)
    print("PHASE24_PREREG_ONLY",json.dumps({
      "excluded":len(seen),"pilot":len(pilot),"validation_reserved":len(validate),
      "pilot_sha256":protocol["pilot_id_sha256"],
      "model_calls":0,"new_gold_opened":False},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
