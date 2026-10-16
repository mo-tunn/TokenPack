"""One-time Phase11 sample preregistration and immutable public validation split.
No MuSiQue gold fields are ever read.
"""
import hashlib,json,os,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase9_musique_contexts import DATA,PROV
from submission.experiments.phase9_musique_protocol import DATA_REV,DATA_SHA256,EXPECTED_DATASET_ROWS
from submission.experiments.phase11_musique_protocol import *

OUT=ROOT/"submission/results/phase11_musique_hop_planner_20261012"
PHASE9=ROOT/"submission/results/phase9_musique_bridge_new256_20261012"
PHASE10=ROOT/"submission/results/phase10_musique_neighbor_pilot_20261012"

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_new(path,data):
    with path.open("x",encoding="utf8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.write("\n")
        f.flush();os.fsync(f.fileno())

def read_ids():
    assert digest(DATA)==DATA_SHA256
    assert json.loads(PROV.read_text(encoding="utf8"))["rev"]==DATA_REV
    pf=pq.ParquetFile(DATA)
    assert pf.metadata.num_rows==EXPECTED_DATASET_ROWS
    rows=pf.read(columns=["id"]).to_pylist()
    ids=[x["id"] for x in rows]
    assert len(ids)==len(set(ids))==EXPECTED_DATASET_ROWS
    return ids

def main():
    if OUT.exists():
        raise FileExistsError("Phase11 exists; do not re-register or overwrite")
    prior9=json.loads((PHASE9/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert digest(PHASE9/"gold_blind_contexts.jsonl")==prior9["case_file_sha256"]
    ids9={json.loads(x)["id"] for x in (PHASE9/"gold_blind_contexts.jsonl").open(encoding="utf8") if x.strip()}
    prior10=json.loads((PHASE10/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    ids10=set(prior10["samples"]["pilot"])|set(prior10["samples"]["validation"])
    assert len(ids9)==256 and len(ids10)==128 and not ids9&ids10
    population=set(read_ids())
    assert ids9|ids10<=population
    candidates=population-ids9-ids10
    ordered=sorted(candidates,key=lambda x:(hashlib.sha256((DATA_SEED+"|"+x).encode("utf8")).hexdigest(),x))
    pilot=ordered[:PILOT_N]
    validation=ordered[PILOT_N:PILOT_N+VALIDATION_N]
    assert len(pilot)==PILOT_N and len(validation)==VALIDATION_N and not set(pilot)&set(validation)
    srcs={
        "phase11_register":__file__,
        "phase11_protocol":str(ROOT/"submission/experiments/phase11_musique_protocol.py"),
        "phase9_crossdoc_core":str(ROOT/"submission/experiments/phase9_crossdoc_core.py"),
    }
    record={
        "preregistration":"Before Phase11 planner responses, retrieved contexts or gold inspection",
        "benchmark":"MuSiQue PUBLIC validation, disjoint question IDs but same public benchmark",
        "dataset_sha256":DATA_SHA256,"dataset_revision":DATA_REV,
        "phase9_exclusion_ids":len(ids9),"phase10_exclusion_ids":len(ids10),
        "phase9_context_sha256":prior9["case_file_sha256"],
        "phase10_manifest_sha256":digest(PHASE10/"PRE_GOLD_PROTOCOL.json"),
        "sample_seed":DATA_SEED,
        "sampling_rule":"SHA256(seed|id) smallest on complement of all 384 Phase9/10 IDs",
        "pilot_ids":pilot,"validation_ids":validation,
        "pilot_id_sha256":hashlib.sha256("\n".join(pilot).encode()).hexdigest(),
        "validation_id_sha256":hashlib.sha256("\n".join(validation).encode()).hexdigest(),
        "method_parameters":{
            "methods":METHODS,"budget_ratio":BUDGET_RATIO,"query_weight":QUERY_WEIGHT,
            "original_weight":ORIGINAL_WEIGHT,"anchor_budget_share":ANCHOR_BUDGET_SHARE,
            "doc_repeat_penalty":DOC_REPEAT_PENALTY,"crossdoc_link_bonus":CROSSDOC_LINK_BONUS,
            "max_planned_queries":MAX_PLANNED_QUERIES,"max_model_score_tokens":MAX_CANDIDATE_SCORE_INPUT,
        },
        "planner":"Question-only Qwen27B, 3 short evidence search queries; no documents, answer or gold supplied",
        "reader":"Same Qwen27B temperature=0 short-answer reader and same prompt, paired 32 pilot cases",
        "pilot_gate":"At least +3 net EM correct hop-planner vs crossdoc-bridge among all 32 questions; 0 budget breaches. If not, do not use 96 validation.",
        "primary_metric":"paired Qwen normalized answer exact match",
        "secondary":"paired answer F1, gold supporting PARAGRAPH access proxy after response freeze",
        "max_pilot_planner_requests":PLANNER_MAX_REMOTE_CALLS,
        "max_pilot_reader_requests":READER_PILOT_MAX_REMOTE_CALLS,
        "no_gold_in_model_requests":True,"qasper_test_reused":False,"production_untouched":True,
        "code_sha256":{k:digest(v) for k,v in srcs.items()},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    save_new(OUT/"PRE_GOLD_PROTOCOL.json",record)
    print("PHASE11_PRE_REGISTERED",json.dumps({
        "pilot_n":len(pilot),"validation_n":len(validation),
        "excluded_n":len(ids9|ids10),"remaining":len(candidates),
        "pilot_digest":record["pilot_id_sha256"],"validation_digest":record["validation_id_sha256"]}),flush=True)
if __name__=="__main__":main()
