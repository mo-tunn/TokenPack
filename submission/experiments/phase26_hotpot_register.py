"""Phase26 preregistered sentence-level sufficiency DIAGNOSTIC on HotpotQA.

No new selection method. Existing Phase8 pretrained sentence rank unchanged.
Only ID projection before preregistration. Exclude all Phase6 512 and
Phase7/8 questions even if subset, plus smoke8 if distinct.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase6_hotpot_eval import DATA,PROVENANCE
from submission.experiments.phase5b_pretrained_rerank_eval import (
    REVISION,WEIGHTS_SHA256)
import pyarrow.parquet as pq
OUT=ROOT/"submission/results/phase26_hotpot_sentence_sufficiency_20261012"
SEED="TokenPack-phase26-HotpotQA-original-sentence-sufficiency-v1-20261012"
PILOT_N=96
RESERVED_N=192
def h(x):return hashlib.sha256(x.encode("utf8")).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("No overwriting prereg")
    provenance=json.loads(PROVENANCE.read_text(encoding="utf8"))
    assert sha_file(DATA)==provenance["sha256"]
    f=pq.ParquetFile(DATA)
    ids=[r["id"] for r in f.read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))
    earlier=ROOT/"submission/results"
    old6=[s.strip() for s in (earlier/"phase6_hotpotqa_preregistered512_20261011/question_ids.txt").read_text(encoding="utf8").splitlines() if s.strip()]
    assert len(old6)==len(set(old6))==512
    old7=[json.loads(s)["id"] for s in (
        earlier/"phase7_hotpotqa_reader256_20261012/gold_blind_contexts.jsonl").read_text(encoding="utf8").splitlines() if s.strip()]
    assert len(old7)==len(set(old7))==256
    old8=[json.loads(s)["id"] for s in (
        earlier/"phase8_atomic_goldblind152_20261012/gold_blind_atomic_contexts.jsonl").read_text(encoding="utf8").splitlines() if s.strip()]
    assert len(old8)==len(set(old8))==152
    # Phase6 smoke8 shares the Phase6 pinned set. It must be explicitly
    # audited, not silently assumed.
    smoke=earlier/"phase6_hotpot_smoke8_20261011/question_ids.txt"
    oldsmoke=[s.strip() for s in smoke.read_text(encoding="utf8").splitlines() if s.strip()] if smoke.exists() else []
    exclude=set(old6)|set(old7)|set(old8)|set(oldsmoke)
    assert set(old7)<=set(old6) and set(old8)<=set(old7)
    assert exclude<=set(ids)
    samples=sorted(set(ids)-exclude,key=lambda k:(h(SEED+"|"+k),k))
    pilot=samples[:PILOT_N];hold=samples[PILOT_N:PILOT_N+RESERVED_N]
    assert len(pilot)==96 and len(hold)==192 and not (set(pilot)|set(hold))&exclude
    codes=ROOT/"submission/experiments"
    protocol={
      "phase":26,"kind":"Ablation-free pre-registered evidence sufficiency taxonomy, no policy optimization",
      "benchmark":"HotpotQA distractor PUBLIC validation; seen benchmark family Phase6-8, but new disjoint IDs; NOT independent hidden test",
      "dataset_revision":provenance["revision"],"dataset_sha256":provenance["sha256"],
      "total_dataset_ids":len(ids),
      "excluded_all_phase6_ids":512,"excluded_phase7_subset":256,
      "excluded_phase8_subset":152,"excluded_smoke_count":len(oldsmoke),
      "total_unique_excluded_ids":len(exclude),
      "sample_seed":SEED,"sample_rule":"SHA256(seed|id) excludes every prior Hotpot selected ID",
      "pilot_ids":pilot,"reserved_validation_ids":hold,
      "pilot_sha256":h("\n".join(pilot)),
      "reserved_validation_sha256":h("\n".join(hold)),
      "pre_inference_fields":["id","question","context.title","context.sentences"],
      "gold_never_used_before_all_reader_outputs_frozen":True,
      "gold_fields_after_freeze":["answer","supporting_facts.title","supporting_facts.sent_id","type"],
      "source_policy":"Unmodified Phase8 sentence_candidates, encode_scores, choose_atoms with bridge=False; pretrained frozen MiniLM MS MARCO; same compact whole-sentence renderer and 25% real token cap",
      "model_revision":REVISION,"model_weights_sha256":WEIGHTS_SHA256,
      "direct_reader":"Unmodified Phase7 short Qwen27B answer prompt through existing full-length Phase13 relay; 96 maximum actual paid inference calls",
      "readout":"Gold ORIGINAL supporting sentence title+sentence-index pairs. Three stages: gold missing from source candidates, gold in candidates but missing in final, all gold final with wrong reader EM; also final gold correct.",
      "pre_registered_primary_counts":["all_gold_source_available","all_gold_final_selected","all_gold_final_reader_correct","all_gold_final_reader_wrong","gold_available_but_not_final","gold_not_available_in_source"],
      "secondary":["per-question recall and original source tokens","question type strata","reader normalized EM/F1","answer string present in final"],
      "unannotated_alternatives":"Gold supporting annotation overlap is NOT semantic entailment precision or proof the annotation is exhaustive",
      "no_method_search_or_postgold_retuning":True,
      "success_definition":"Diagnostic measurement complete for all 96, final contexts/outputs hash-frozen before gold; NO model improvement claim or automatic validation gate",
      "validation_rule":"192 reserved ID NEVER touched in this descriptive pilot",
      "model_calls_expected":96,
      "human_semantic_evaluation_required_for_claim":True,
      "existing_core_sha256":{name:sha_file(codes/name) for name in [
         "phase8_atomic_contexts.py","phase8_atomic_protocol.py",
         "phase7_modal27b_matched8.py","phase13_graph_modal_relay_source.py"]},
      "previous_phase6_id_file_sha256":sha_file(earlier/"phase6_hotpotqa_preregistered512_20261011/question_ids.txt"),
      "previous_phase7_context_sha256":sha_file(earlier/"phase7_hotpotqa_reader256_20261012/gold_blind_contexts.jsonl"),
      "previous_phase8_context_sha256":sha_file(earlier/"phase8_atomic_goldblind152_20261012/gold_blind_atomic_contexts.jsonl"),
      "qasper_official_test_untouched":True,"no_production_edits":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",protocol)
    print("PHASE26_PREREG",json.dumps({
        "dataset":len(ids),"excluded":len(exclude),"pilot":len(pilot),
        "reserve":len(hold),"pilot_sha256":protocol["pilot_sha256"]}),flush=True)
if __name__=="__main__":main()
