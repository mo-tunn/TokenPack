"""Faz12 strictly preregistered 2Wiki validation IDs; gold read only after freeze."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
DATA_DIR=ROOT/"submission/data/phase12_2wiki_validation_20261012"
DATA=DATA_DIR/"data/validation-00000-of-00001.parquet"
OUT=ROOT/"submission/results/phase12_2wiki_relation_diagnostic_20261012"
SEED="TokenPack_phase12_2Wiki_reasoning_relation_analysis_20261012_v1"
N_RETRIEVAL=64
N_READER=32
N_RESERVED=128
EXPECTED_SHA="408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59"
EXPECTED_REV="fe713bfbd1afbca1a65246741a75890405d56a3a"
def sha_file(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def new_json(path,obj):
    with path.open("x",encoding="utf8") as fh:
        json.dump(obj,fh,indent=2,ensure_ascii=False);fh.write("\n");fh.flush();os.fsync(fh.fileno())

def verify_data():
    prov=json.loads((DATA_DIR/"PROVENANCE.json").read_text(encoding="utf8"))
    assert prov["revision"]==EXPECTED_REV and prov["sha256"]==EXPECTED_SHA
    assert sha_file(DATA)==EXPECTED_SHA
    pf=pq.ParquetFile(DATA)
    assert pf.metadata.num_rows==12576
    return pf

def main():
    if OUT.exists():raise FileExistsError("Phase12 already registered; immutable.")
    pf=verify_data()
    ids=[r["id"] for r in pf.read(columns=["id"]).to_pylist()]
    assert len(ids)==len(set(ids))==12576
    ordered=sorted(ids,key=lambda id:(hashlib.sha256((SEED+"|"+id).encode()).hexdigest(),id))
    pilot=ordered[:N_RETRIEVAL]; reserved=ordered[N_RETRIEVAL:N_RETRIEVAL+N_RESERVED]
    assert len(pilot)==64 and len(reserved)==128 and not set(pilot)&set(reserved)
    scripts=["phase12_register.py","phase11_planner.py","phase11_contexts.py",
             "phase9_crossdoc_core.py","phase11_musique_protocol.py","phase5b_pretrained_rerank_eval.py"]
    base=ROOT/"submission/experiments"
    manifest={
        "phase":"12","preregistered":"Before dataset gold/reasoning labels or planner inference",
        "dataset":"2WikiMultiHopQA public validation (new benchmark for TokenPack research)",
        "data_repo":"framolfese/2WikiMultihopQA",
        "dataset_revision":EXPECTED_REV,"parquet_sha256":EXPECTED_SHA,
        "sample_seed":SEED,"selection":"SHA256(seed|id) only, no type or gold filtering",
        "goldblind_sample_n":N_RETRIEVAL,"reader_first_n":N_READER,
        "reserved_untouched_n":N_RESERVED,"ids":pilot,"reserved_ids":reserved,
        "primary_gold_metric":"all annotated supporting (title, sentence-id) pairs present in exact selected original sentence indices; OR if title occurs multiple times",
        "secondary_gold_metrics":["fraction of all required supporting sentence pairs",
             "subject-object intermediate bridge entity retention",
             "relation triple endpoint-mention proxy; NOT correctness of relation assertion",
             "per-question-type (compositional, inference, comparison, bridge_comparison) exploratory"],
        "downstream_reader":"32 predetermined ID-only prefix, 2 arms, Qwen27B zero-temp short answer; paired normalized EM/F1",
        "methods":["crossdoc-bridge","hop-planner"],
        "goldblind_selection":"Phase9 crossdoc bridge vs Phase11 Qwen question-only 3-query zero-shot MS MARCO selection; unchanged parameter constants and render",
        "source_candidates":"all original 2Wiki context sentences, title prepended for reranker",
        "max_real_context_cap":"floor(.25 * sum(cl100k(title+newline+sentence-concatenated-paragraph))) each, final render cl100k <=cap",
        "model":"MS MARCO MiniLM-L6-v2 zero-shot and Qwen27B Q4KM for question-only planning/reading",
        "primary_reader_report":"paired answer EM difference in 32, CI/McNemar exploratory",
        "planner_gold_columns_read":False,"planner_source_uploaded":False,
        "inference_projection":["id","question","context.title","context.sentences"],
        "gold_columns_only_postfreeze":["supporting_facts","evidences","answer","type"],
        "test_split_unused":True,"QASPER_frozen_test_unused":True,
        "production_changed":False,
        "budget_guard":{"max_planner_calls":64,"max_reader_calls":64,
                        "session_minutes_limit":30,"approx_session_usd_guard":1.35,
                        "invoice_not_a_hard_limit":True},
        "code_sha256":{s:sha_file(base/s) for s in scripts},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",manifest)
    print("PHASE12_PRE_REGISTERED",json.dumps({
        "n":len(pilot),"reader":N_READER,"reserved":N_RESERVED,
        "ids_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
        "data_sha256":EXPECTED_SHA}),flush=True)
if __name__=="__main__":main()
