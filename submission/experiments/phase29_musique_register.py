"""Phase29 preregistration: frozen Phase28 selector on fresh MuSiQue public IDs."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase9_musique_protocol import (
    DATA_REV, DATA_SHA256, EXPECTED_DATASET_ROWS, SAMPLE_SEED as PHASE9_SEED)
from submission.experiments.phase5b_pretrained_rerank_eval import WEIGHTS_SHA256, REVISION
from submission.experiments.phase28_semantic_core import SYSTEM, USER
from submission.experiments.phase12_register import sha_file
DATA=ROOT/"submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"
PROV=DATA.parent.parent/"PROVENANCE.json"
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
SEED="TokenPack-phase29-MuSiQue-semantic-citation-crossbench-20261013-v1"
PILOT_N=48
RESERVED_N=96

def digest(s): return hashlib.sha256(s.encode("utf8")).hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding="utf8"))
def new_json(path,obj):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())

def main():
    if OUT.exists(): raise FileExistsError("Phase29 is immutable once registered.")
    prov=read_json(PROV)
    assert prov["rev"]==DATA_REV and prov["sha256"]==DATA_SHA256
    assert sha_file(DATA)==DATA_SHA256
    pf=pq.ParquetFile(DATA)
    assert pf.metadata.num_rows==EXPECTED_DATASET_ROWS==2417
    ids=[r["id"] for r in pf.read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==2417

    # Reconstruct Phase9's deterministic 256-ID sample using IDs only.
    phase9=set(sorted(ids,key=lambda x:(digest(PHASE9_SEED+"|"+x),x))[:256])
    results=ROOT/"submission/results"
    p10=read_json(results/"phase10_musique_neighbor_pilot_20261012/PRE_GOLD_PROTOCOL.json")
    p11=read_json(results/"phase11_musique_hop_planner_20261012/PRE_GOLD_PROTOCOL.json")
    p25=read_json(results/"phase25_crossbench_musique_granularity_20261012/PRE_GOLD_PROTOCOL.json")
    phase10=set(p10["samples"]["pilot"])|set(p10["samples"]["validation"])
    phase11=set(p11["pilot_ids"])|set(p11["validation_ids"])
    phase25=set(p25["pilot_ids"])|set(p25["reserved_validation_ids"])
    groups={"phase9":phase9,"phase10":phase10,"phase11":phase11,"phase25":phase25}
    assert {k:len(v) for k,v in groups.items()}=={
        "phase9":256,"phase10":128,"phase11":128,"phase25":192}
    assert sum(map(len,groups.values()))==len(set().union(*groups.values()))==704
    assert all(v<=set(ids) for v in groups.values())
    excluded=set().union(*groups.values())
    available=sorted(set(ids)-excluded,key=lambda x:(digest(SEED+"|"+x),x))
    pilot=available[:PILOT_N]
    reserved=available[PILOT_N:PILOT_N+RESERVED_N]
    assert len(pilot)==48 and len(reserved)==96
    assert not set(pilot)&set(reserved)
    assert not (set(pilot)|set(reserved))&excluded

    exp=ROOT/"submission/experiments"
    code_files=[
        "phase29_musique_register.py","phase29_musique_candidates.py",
        "phase29_musique_verifier.py","phase29_musique_contexts.py",
        "phase29_musique_structural_gate.py","phase29_musique_reader.py",
        "phase29_musique_score.py","phase29_musique_integrity_audit.py",
        "phase28_semantic_core.py","phase27_joint_chain_core.py",
        "phase8_atomic_contexts.py","phase8_atomic_protocol.py",
        "phase9_musique_protocol.py","phase9_musique_contexts.py",
        "phase7_modal27b_matched8.py","phase13_graph_modal_relay_source.py",
        "phase5b_pretrained_rerank_eval.py"]
    hashes={name:sha_file(exp/name) for name in code_files}
    manifest={
      "study":"Phase29 frozen Phase28 semantic-joint-citations cross-benchmark replication",
      "phase":29,
      "benchmark":"MuSiQue PUBLIC validation; not hidden/sealed test",
      "benchmark_independence":"Different benchmark family from Phase28 HotpotQA; public validation had prior TokenPack experiments in Phases 9-11 and 25. All their used/reserved IDs are excluded, but public-label researcher-adaptation risk remains.",
      "dataset_repo":"awinml/musique",
      "dataset_revision":DATA_REV,
      "data_sha256":DATA_SHA256,
      "row_count":2417,
      "prior_id_exclusions":{k:len(v) for k,v in groups.items()},
      "prior_excluded_union_n":len(excluded),
      "available_after_exclusions_n":len(available),
      "seed":SEED,
      "pilot_ids":pilot,
      "reserved_ids":reserved,
      "pilot_ids_sha256":digest("\n".join(pilot)),
      "reserved_ids_sha256":digest("\n".join(reserved)),
      "sampling_rule":"Sort every nonexcluded validation ID by SHA256(seed|id), tie-break by ID; first 48 pilot, next 96 reserved. No gold/type filtering.",
      "goldblind_columns":["id","question","paragraphs.idx","paragraphs.title","paragraphs.paragraph_text"],
      "gold_fields_never_in_pre_reader_projection":["answer","answer_aliases","question_decomposition","paragraphs.is_supporting","answerable"],
      "baseline":"Unmodified Phase8/27 atomic-sentence-rank over same original source sentences and frozen pretrained MS MARCO MiniLM scores.",
      "treatment":"Exact frozen Phase28 semantic-joint-citations: same top-24-plus-best-per-source candidate policy; Qwen selects 2-3 original source sentence IDs from at least two source paragraphs; identical fallback on invalid/abstain/over-cap; fill remainder by same frozen relevance rank.",
      "selector_prompt_system":SYSTEM,
      "selector_prompt_user_template":USER,
      "selector_system_sha256":digest(SYSTEM),
      "selector_user_template_sha256":digest(USER),
      "reader":"Same frozen Phase7 short-answer Qwen27B payload, temperature 0, identical prompt on both arms.",
      "model":"qwen3.8-27b-orcarouter-q4km",
      "reranker":"cross-encoder/ms-marco-MiniLM-L6-v2",
      "reranker_revision":REVISION,
      "reranker_weights_sha256":WEIGHTS_SHA256,
      "sentence_segmentation":"Exact Phase9 MuSiQue split_sentences regex; each selected sentence is an unchanged substring of its original source paragraph.",
      "budget_ratio":0.25,
      "budget":"floor(0.25 * sum cl100k_base(title + newline + full original paragraph)); both arms' actual final rendered context must be <= this cap.",
      "pilot_n":PILOT_N,
      "reserved_confirmation_n":RESERVED_N,
      "co_primary_endpoints":[
        "Canonical-answer normalized exact match (EM), paired by question.",
        "Full annotated support-paragraph coverage: every gold is_supporting paragraph idx has at least one selected original sentence from that paragraph."
      ],
      "support_metric_caveat":"MuSiQue labels supporting paragraphs, not original gold sentence indices. Full paragraph coverage is NOT sentence-level entailment and is not identical to Phase28's Hotpot supporting-sentence endpoint.",
      "pilot_gates":{
        "pre_reader_structural_minimum_changed_source_sentence_sets":5,
        "post_reader_net_full_support_paragraph_coverage_gain_minimum":5,
        "post_reader_net_EM_correct_gain_minimum":3,
        "zero_render_cap_violations_required":True,
        "failure_action":"STOP; never run reserved 96 if either post-gold gate fails."
      },
      "confirmation_gate":{
        "both_co_primary_net_positive":True,
        "exact_two_sided_mcnemar_p_each_max":0.025,
        "zero_render_cap_violations_required":True,
        "bootstrap_replicates_secondary":10000,
        "failure_action":"VALIDATION_NOT_CONFIRMED; no post-hoc tuning."
      },
      "gold_open_only_after_all_reader_responses_frozen":True,
      "gold_or_answers_sent_to_remote_model":False,
      "qasper_official_sealed_test_untouched":True,
      "production_code_unchanged":True,
      "code_sha256":hashes
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",manifest)
    print("PHASE29_PREREG_FROZEN",json.dumps({
      "excluded":len(excluded),"available":len(available),
      "pilot_n":len(pilot),"reserved_n":len(reserved),
      "pilot_sha256":manifest["pilot_ids_sha256"],
      "reserved_sha256":manifest["reserved_ids_sha256"],
      "data_sha256":DATA_SHA256,"gold_columns_read":["id"]},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
