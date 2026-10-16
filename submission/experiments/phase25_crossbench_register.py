"""Faz25 independent *benchmark-family* granularity audit, MuSiQue public validation.

New MuSiQue IDs excluded from all previously used/reserved Phase9/10/11.
Gold answer and aliases are never opened until all 64 Qwen answers are
persisted and hash-frozen. No new methods are being tuned here.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase9_musique_protocol import (
    DATA_SHA256,DATA_REV,EXPECTED_DATASET_ROWS,SAMPLE_SEED,N_QUESTIONS,
    BUDGET_RATIO,MAX_SCORE_INPUT_TOKENS,RANK_BATCH_SIZE)
from submission.experiments.phase9_musique_contexts import (
    DATA,PROV,OUTPUT as P9)
from submission.experiments.phase5b_pretrained_rerank_eval import (
    WEIGHTS_SHA256,REVISION)
import pyarrow.parquet as pq
OUT=ROOT/"submission/results/phase25_crossbench_musique_granularity_20261012"
SEED="TokenPack-phase25-MuSiQue-independent-family-lexical-calibration-20261012-v1"
PILOT_N=64
RESERVED_N=128
def hh(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def main():
    if OUT.exists():raise FileExistsError("Prior preregistration exists")
    assert sha_file(DATA)==DATA_SHA256
    provenance=json.loads(PROV.read_text(encoding="utf8"))
    assert provenance["rev"]==DATA_REV and provenance["sha256"]==DATA_SHA256
    p9=json.loads((P9/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    p10dir=ROOT/"submission/results/phase10_musique_neighbor_pilot_20261012"
    p11dir=ROOT/"submission/results/phase11_musique_hop_planner_20261012"
    p10=json.loads((p10dir/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    p11=json.loads((p11dir/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert p9["data_sha256"]==p10["source_data_sha256"]==p11["dataset_sha256"]==DATA_SHA256
    pf=pq.ParquetFile(DATA)
    assert pf.metadata.num_rows==EXPECTED_DATASET_ROWS==2417
    ids=[x["id"] for x in pf.read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))
    phase9=sorted(ids,key=lambda x:(hh(SAMPLE_SEED+"|"+x),x))[:N_QUESTIONS]
    assert hh("\n".join(sorted(phase9)))==p9["sample_ids_sha256"]
    sampled_contexts=[json.loads(x) for x in
       (P9/"gold_blind_contexts.jsonl").read_text(encoding="utf8").splitlines() if x]
    assert set(phase9)==set(x["id"] for x in sampled_contexts)
    excluded=set(phase9)|set(p10["samples"]["pilot"])|set(p10["samples"]["validation"])
    excluded|=set(p11["pilot_ids"])|set(p11["validation_ids"])
    assert len(excluded)==256+128+128
    ranking=sorted(set(ids)-excluded,key=lambda x:(hh(SEED+"|"+x),x))
    pilot=ranking[:PILOT_N]
    validation=ranking[PILOT_N:PILOT_N+RESERVED_N]
    assert len(pilot)==64 and len(validation)==128
    assert not (set(pilot)|set(validation))&excluded
    source=ROOT/"submission/experiments"
    prereg={
      "study":"Phase25 cross-benchmark answer specificity and alias-aware metrics; not a treatment optimization",
      "dataset":"awinml/musique PUBLIC validation (different family from 2Wiki; dataset previously used Phase9-11, so NOT new hidden test)",
      "revision":DATA_REV,"data_sha256":DATA_SHA256,
      "phase9_prior_ids":len(phase9),"phase10_prior_pilot_and_reserved":128,
      "phase11_prior_pilot_and_reserved":128,
      "all_prior_musique_used_or_reserved_excluded":len(excluded),
      "seed":SEED,"pilot_ids":pilot,"reserved_validation_ids":validation,
      "pilot_sha256":hh("\n".join(pilot)),
      "reserved_sha256":hh("\n".join(validation)),
      "data_columns_for_inference":["id","question","paragraphs.idx","paragraphs.title","paragraphs.paragraph_text"],
      "gold_fields_excluded_pre_inference":["answer","answer_aliases","question_decomposition","is_supporting","answerable"],
      "baseline_evidence_selector":"Unchanged Phase9 MuSiQue zero-shot pretrained MiniLM MS MARCO crossdoc-bridge, unmodified original sentences, 25pct source cap",
      "reader":"Unchanged Phase7 direct short Qwen27B answer via Phase13 FULL 4000-char Modal relay, temperature 0, same original prompt",
      "n":64,"reserved_n":128,
      "research_question":"Do lexical answer granularity mismatches in Faz23/24 2Wiki pilot also arise in unused MuSiQue questions? Quantify EM misses with alias and whole-boundary gold substring.",
      "primary_descriptive_measure":"fraction of 64 cases EM incorrect but gold answer OR one provided gold alias occurs as contiguous whole-word-boundary substring in frozen prediction. This is an OBJECTIVE STRING signal, not semantic correctness.",
      "secondary_descriptive_measure":["literal EM","alias-aware EM using normalized gold and answer_aliases","answer F1","word-boundary substring hits","question-only expected answer-type strata","actual prompt answer lengths"],
      "specificity_candidate_gate":"At least 6/64 EM misses with gold/alias whole-boundary substring AND at least 4/64 alias-aware EM successes above strict EM would support further blinded semantic study. Gate does NOT authorize validation inference or automatic recoding correctness.",
      "no_reader_method_tuning":True,
      "no_paid_planner_or_verifier":True,
      "max_reader_calls":64,
      "no_gold_read_before_reader_hash_freeze":True,
      "no_model_call_with_gold":True,
      "no_independent_semantic_human_raters_available":"lexical analysis exploratory only, never certify semantic equivalence",
      "nonclaim":"Already-tested public validation MuSiQue benchmark is a different family, but the answer is NOT an external blind test",
      "safety":"never QASPER Phase3D official sealed test, no production edits, no prior archives overwritten",
      "file_sha256":{x:sha_file(source/x) for x in [
          "phase9_musique_contexts.py","phase9_crossdoc_core.py",
          "phase9_musique_protocol.py","phase7_modal27b_matched8.py",
          "phase13_graph_modal_relay_source.py"]},
      "model_revision":REVISION,"minilm_model_sha256":WEIGHTS_SHA256,
      "preregistered_before_new_models":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",prereg)
    print("PHASE25_PRE_GOLD_PREREG",json.dumps({
       "excluded":len(excluded),"pilot":len(pilot),
       "reserve":len(validation),"pilot_hash":prereg["pilot_sha256"],
       "model_calls":0}),flush=True)
if __name__=="__main__":main()
