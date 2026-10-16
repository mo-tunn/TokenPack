"""Build frozen 3-case x 8-arm Phase-2 benchmark solely on local CPU.

Cases were selected before inference from LongBench v2's cached original
503-case parquet, excluding all recorded prior 14B/27B pilot case IDs.
No network, Modal or model inference. Never overwrites existing artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"

from submission.longbench_eval.client import build_tasks
from submission.longbench_eval.eval_utils import write_jsonl

CACHE=ROOT/".tokenpack/cache/longbench_v2/longbench_v2.parquet"
RESULTS=ROOT/"submission/results"
OTHER=Path.home()/"Desktop/TokenPack-Blind-Benchmark-MCP/runs/longbench-v2-clean60-gpt6luna-medium-20261009/private/manifest.jsonl"
OUT=RESULTS/"phase2_qwen27b_matched3_20261011"
CASE_IDS=[
    "66ee8bab821e116aacb21e44", # multi-document QA academic
    "66fcfb5fbb02136c067c93ae", # code repository QA
    "671b080abb02136c067d4dde", # long-dialogue history QA
]

def previously_seen()->set[str]:
    seen=set()
    for f in RESULTS.glob("longbench_v2_modal*/longbench_generation_tasks.jsonl"):
        with f.open(encoding="utf8",errors="replace") as stream:
            for ln in stream:
                if not ln.strip():continue
                try:x=json.loads(ln)
                except json.JSONDecodeError:continue
                if x.get("case_id"):seen.add(str(x["case_id"]))
    if OTHER.exists():
        for ln in OTHER.read_text(encoding="utf8").splitlines():
            if ln.strip():seen.add(str(json.loads(ln)["original_task"]["case_id"]))
    return seen

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    seen=previously_seen()
    overlap=set(CASE_IDS)&seen
    if overlap:raise RuntimeError(f"Selected cases overlap historical development/pilot set: {overlap}")
    raw=pd.read_parquet(CACHE)
    cases=raw[raw["_id"].isin(CASE_IDS)].to_dict("records")
    by_id={x["_id"]:x for x in cases}
    if len(by_id)!=3 or set(by_id)!=set(CASE_IDS):raise RuntimeError("Missing selected cases")
    cases=[by_id[case] for case in CASE_IDS]
    if any(x.get("length")!="short" for x in cases):raise RuntimeError("Not a short-context matched set")
    print(json.dumps({"mode":"PREPARE_OFFLINE" if args.execute else "DRY_RUN",
        "case_ids":CASE_IDS,"domains":[c["domain"] for c in cases],
        "context_chars":[len(c["context"]) for c in cases],
        "historically_seen_cases":len(seen),"overlap":len(overlap)},ensure_ascii=False),flush=True)
    if not args.execute:return
    OUT.mkdir(parents=True,exist_ok=True)
    cases_path=OUT/"source_cases_private.json"
    tasks_path=OUT/"phase2_tasks_private.jsonl"
    report_path=OUT/"phase2_task_report.json"
    for p in [cases_path,tasks_path,report_path]:
        if p.exists():raise FileExistsError(f"Refusing to overwrite {p}")
    cases_path.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    ns=argparse.Namespace(
        data_file=str(cases_path),limit=3,max_scanned=3,
        source_min_tokens=1000,source_max_tokens=1000000,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_allow_download=False,chunker="structure-aware",
        scoring="evidence-hybrid",selection_strategy="budget-top-k",
        budget_ratio=.50,context_order="score",candidate_pool=300,
        target_tokens=650,min_tokens=120,max_tokens=900,
        reranker="none",skip_compression=True,
        phase2_ablation=True,phase2_neighbor_fraction=.15,
    )
    tasks,report=build_tasks(ns)
    methods=sorted({t["method"] for t in tasks})
    if len(tasks)!=24 or len(methods)!=8:
        raise RuntimeError("Expected 3 cases x 8 methods")
    over=[(x["case_id"],x["method"],x["context_tokens"],x["budget_tokens"]) for x in tasks if ("-p2-" in x["method"] and x["context_tokens"]>x["budget_tokens"])]
    if over:raise RuntimeError(f"Output budget failures: {over}")
    report["source_parquet_sha256"]=hashlib.sha256(CACHE.read_bytes()).hexdigest()
    report["selected_case_ids"]=CASE_IDS
    report["nonoverlap_prior_development"]=True
    report["phase2_gold_never_in_prompt"]=True
    report["model_requests"]=0
    write_jsonl(tasks,tasks_path)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"ready":len(tasks),"methods":methods,"output":str(OUT),
        "per_case":[{"id":case,"counts":{t["method"]:t["context_tokens"] for t in tasks if t["case_id"]==case}} for case in CASE_IDS]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
