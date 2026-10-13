"""Preregister new HotpotQA Phase28 source-grounded joint citation verifier.

Exclude every prior HotpotQA used/reserved ID, including Phase27 reserved.
Before ANY model call, gold access, selection of cases or reader scores.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase6_hotpot_eval import DATA,PROVENANCE
from submission.experiments.phase26_hotpot_register import OUT as P26
from submission.experiments.phase27_joint_chain_register import OUT as P27
from submission.experiments.phase28_semantic_core import (
 SYSTEM,USER,MAX_TOP,MAX_CANDIDATES,MAX_OUTPUT_TOKENS)
from submission.experiments.phase5b_pretrained_rerank_eval import REVISION,WEIGHTS_SHA256
import pyarrow.parquet as pq
OUT=ROOT/"submission/results/phase28_hotpot_semantic_citation_chain_20261013"
SEED="TokenPack-phase28-HotpotQA-original-semantic-citation-chain-20261013-v1"
PILOT_N=48
RESERVED_N=96
def digest(x):return hashlib.sha256(x.encode("utf8")).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("Prereg already exists")
    provenance=json.loads(PROVENANCE.read_text(encoding="utf8"))
    assert sha_file(DATA)==provenance["sha256"]
    ids=[r["id"] for r in pq.ParquetFile(DATA).read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==7405
    parent=ROOT/"submission/results"
    original=set(x.strip() for x in (
       parent/"phase6_hotpotqa_preregistered512_20261011/question_ids.txt"
       ).read_text(encoding="utf8").splitlines() if x.strip())
    assert len(original)==512
    p26=json.loads((P26/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    p27=json.loads((P27/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert p26["dataset_sha256"]==p27["dataset_sha256"]==provenance["sha256"]
    exclude=original|set(p26["pilot_ids"])|set(p26["reserved_validation_ids"])|set(p27["pilot_ids"])|set(p27["reserved_ids"])
    assert len(exclude)==512+96+192+48+96==944
    sampled=sorted(set(ids)-exclude,key=lambda x:(digest(SEED+"|"+x),x))
    pilot=sampled[:PILOT_N]
    reserved=sampled[PILOT_N:PILOT_N+RESERVED_N]
    assert len(pilot)==48 and len(reserved)==96
    assert not(set(pilot)|set(reserved))&exclude
    files=ROOT/"submission/experiments"
    prereg={
      "phase":28,
      "dataset":"HotpotQA distractor PUBLIC validation, separate IDs, not hidden test",
      "data_revision":provenance["revision"],
      "data_sha256":provenance["sha256"],
      "excluded_prior_used_and_reserved":len(exclude),
      "pilot_ids":pilot,"reserved_ids":reserved,
      "seed":SEED,"pilot_id_sha256":digest("\n".join(pilot)),
      "reserve_id_sha256":digest("\n".join(reserved)),
      "data_allowed_before_gold":["id","question","context.title","context.sentences"],
      "treatment":"One Qwen source-grounded semantic verifier Q per question; given original sentence candidates, choose 2-3 original sentence IDs from >=2 different input docs jointly supporting intermediate entity and target relation; source-only citations must fit cap; seed then fill same MiniLM rank; if invalid/abstain/too-long, revert EXACT baseline.",
      "baseline":"Phase8 unchanged sentence rank on SAME zero-shot MiniLM scores and 25pct original source token cap",
      "candidate_policy":"top24 MiniLM sentences plus best original sentence in each otherwise-unrepresented input source article, total <=36, Qwen receives only original source citation text and question",
      "semantic_verifier_system_sha256":digest(SYSTEM),
      "semantic_verifier_user_template_sha256":digest(USER),
      "max_verifier_output_tokens":MAX_OUTPUT_TOKENS,
      "reader":"Same Phase7 short Qwen27B prompt via Phase13 4000-char response relay, temp0, paired 96 maximum reads",
      "max_verifier_calls":48,"max_reader_calls":96,
      "joint_selected_original_source_hard_cap":0.25,
      "primary_mechanistic":"paired full annotated original supporting_facts title/sentence-index coverage",
      "primary_answer":"paired exact normalized EM after reader",
      "secondary":["original support recall","F1","context-difference rate","citation acceptance/abstention","actual cap","paid model calls"],
      "mechanistic_go_gate_net_full_support":5,
      "answer_go_gate_net_EM_correct":3,
      "zero_cap_violations_required":True,
      "pre_gold_structural_futility":"If fewer than 5 new original sentence sets differ, max possible net full gold support gain <5. STOP BEFORE READER and BEFORE GOLD.",
      "post_gold_gate":"Advance to reserved 96 only when >=5 net full support questions AND >=3 net EM correct, zero cap violation. Otherwise STOP_NO_VALIDATION.",
      "never_optimize_on_pilot_gold":True,
      "gold_fields_never_sent_to_model":True,
      "source_text_not_necessarily_entailment":"Qwen citation judgement is a fallible model prediction, not semantic proof",
      "code_hashes":{name:sha_file(files/name) for name in (
        "phase28_semantic_core.py","phase8_atomic_contexts.py",
        "phase27_joint_chain_core.py",
        "phase7_modal27b_matched8.py","phase13_graph_modal_relay_source.py")},
      "local_model_sha256":WEIGHTS_SHA256,
      "qasper_sealed_test_untouched":True,"production_unchanged":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",prereg)
    print("PHASE28_PREREG",json.dumps({
      "excluded":len(exclude),"pilot":len(pilot),"reserved":len(reserved),
      "hash":prereg["pilot_id_sha256"]}),flush=True)
if __name__=="__main__":main()
