"""Preregister HotpotQA Faz27 fixed cross-article evidence pair pilot.

Before gold or model calls. Exclude all previous phase6/7/8 and phase26
used AND reserved IDs. No official test, no public validation leakage to
selection. Gold only after BOTH matched reader answers frozen.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase6_hotpot_eval import DATA,PROVENANCE
from submission.experiments.phase26_hotpot_register import OUT as P26
from submission.experiments.phase5b_pretrained_rerank_eval import (
    REVISION,WEIGHTS_SHA256)
import pyarrow.parquet as pq
OUT=ROOT/"submission/results/phase27_hotpot_joint_chain_20261012"
SEED="TokenPack-phase27-HotpotQA-cross-article-title-chain-seed-20261012-v1"
PILOT_N=48
RESERVED_N=96
def digest(x):return hashlib.sha256(x.encode("utf8")).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("Prereg immutable")
    prov=json.loads(PROVENANCE.read_text(encoding="utf8"))
    assert sha_file(DATA)==prov["sha256"]
    rows=pq.ParquetFile(DATA).read(columns=["id"]).to_pylist()
    ids=[x["id"] for x in rows]
    assert len(ids)==len(set(ids))==7405
    results=ROOT/"submission/results"
    old6=[x.strip() for x in (results/"phase6_hotpotqa_preregistered512_20261011/question_ids.txt").read_text(encoding="utf8").splitlines() if x.strip()]
    assert len(old6)==512
    p26=json.loads((P26/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert len(p26["pilot_ids"])==96 and len(p26["reserved_validation_ids"])==192
    excluded=set(old6)|set(p26["pilot_ids"])|set(p26["reserved_validation_ids"])
    assert len(excluded)==512+96+192==800
    assert p26["dataset_sha256"]==prov["sha256"]
    sampled=sorted(set(ids)-excluded,key=lambda i:(digest(SEED+"|"+i),i))
    pilot=sampled[:PILOT_N]
    reserve=sampled[PILOT_N:PILOT_N+RESERVED_N]
    assert len(pilot)==48 and len(reserve)==96 and not set(pilot)&set(reserve)
    assert not (set(pilot)|set(reserve))&excluded
    d=ROOT/"submission/experiments"
    core=d/"phase27_joint_chain_core.py"
    protocol={
      "study":"Phase27 SAME 25pct original sentence cap: baseline unmodified Faz8 sentence rank vs gold-blind two-article title-link seed with identical greedy fill",
      "nature":"new 48-question matched pilot on HotpotQA distractor PUBLIC validation; not heldout independent official test",
      "benchmark_rev":prov["revision"],"dataset_sha256":prov["sha256"],
      "prior_used_and_reserved_excluded":len(excluded),
      "exclude_manifest":"Phase6 512 including Phase7/8, Phase26 96 pilot +192 RESERVED",
      "seed":SEED,"pilot_ids":pilot,"reserved_ids":reserve,
      "pilot_sha256":digest("\n".join(pilot)),
      "reserve_sha256":digest("\n".join(reserve)),
      "only_pre_gold_inference_columns":["id","question","context.title","context.sentences"],
      "gold_untouched_until_all_reader_responses_frozen":True,
      "arms":["atomic-sentence-rank","joint-title-link-seed"],
      "baseline":"Unmodified Phase8 choose_atoms(...,bridge=False), MS MARCO MiniLM-L6-v2 pretrained zero-shot",
      "treatment":"Pretrained same per-original-sentence scores; among TOP24 ranked original sentences A choose best fitting two-article seed where A contains exact normalized title of another source B, paired with best-ranked original sentence B; maximize scoreA+scoreB. Then greedily fill original source sentences by unchanged baseline rank. If none fits, baseline exactly. BOTH originals printed verbatim, no source transformation.",
      "reader":"Same Phase7 Qwen27B short direct answer prompt, Phase13 full output relay, temp0. Two matched tasks per pilot = 96 successful Modal reader outputs maximum.",
      "mechanistic_primary":"paired difference in all original gold supporting_facts title+sentence-index recall: joint minus baseline, on all 48 questions; descriptive bootstrap and exact discordance",
      "co_primary_answer":"paired normalized answer EM: joint minus baseline on same 48 reader questions; two-sided exact McNemar plus bootstrap",
      "secondary":["token cap compliance","contexts differing","source document count","average annotated support sentence recall","paired F1","gold source vs final access","reader costs"],
      "pilot_gate":"Go to reserved 96 ONLY if joint has at least +5 NET questions with all original gold supporting sentence pairs AND at least +3 NET answer EM correct, zero cap violations; otherwise STOP_NO_VALIDATION.",
      "support_net_gate":5,"answer_net_gate":3,"zero_cap_violations_required":True,
      "same_sample_not_cross_phase_cherry_picking":True,
      "no_gold_planner_no_gold_model":True,
      "do_not_use_prior_gold_labels_to_pick_policy":True,
      "documented_assumptions":"Title occurrence is only lexical link candidate; it does not establish true multi-hop entailment. Full gold supporting_fact coverage may not guarantee correctness or logical necessity.",
      "source_code_sha256":{
         name:sha_file(d/name) for name in [
           "phase27_joint_chain_core.py",
           "phase8_atomic_contexts.py",
           "phase7_modal27b_matched8.py",
           "phase13_graph_modal_relay_source.py"]},
      "model_revision":REVISION,"weights_sha256":WEIGHTS_SHA256,
      "reserved_96_never_open_unless_pilot_gate_pass":True,
      "qasper_official_sealed_test_untouched":True,
      "production_code_unmodified":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",protocol)
    print("PHASE27_PREREG",json.dumps({
       "excluded":800,"pilot":48,"reserved":96,
       "pilot_hash":protocol["pilot_sha256"]}),flush=True)
if __name__=="__main__":main()
