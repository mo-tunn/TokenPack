"""Offline Phase-2 real-archive smoke test; no model inference or web access.

Derives two *source-only* LongBench v2 cases from archived Qwen14B
full-context tasks and builds seven contexts using locally cached MiniLM.
Reports contexts' measured token budgets without judging answers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUT = ROOT / "submission" / "results" / "longbench_v2_modal_hybrid_greedy_83_latency" / "longbench_generation_tasks.jsonl"
OUTPUT = ROOT / "submission" / "results" / "phase2_offline_smoke_v2_20261011"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from submission.longbench_eval.client import build_tasks  # noqa: E402


def original_cases(n: int) -> list[dict]:
    cases: list[dict] = []
    with INPUT.open("r", encoding="utf8") as source:
        for line in source:
            if not line.strip():
                continue
            row=json.loads(line)
            if row["method"]!="full-context":
                continue
            cases.append({
                "_id":row["case_id"],
                "domain":row["domain"],"sub_domain":row["sub_domain"],
                "difficulty":row["difficulty"],"length":row["length"],
                "question":row["question"],
                "choice_A":row["choice_A"],"choice_B":row["choice_B"],
                "choice_C":row["choice_C"],"choice_D":row["choice_D"],
                "answer":row["answer"],"context":row["context"],
            })
            if len(cases)>=n:
                break
    if len(cases)!=n:
        raise RuntimeError(f"Found only {len(cases)} full-context cases")
    return cases


def main() -> int:
    parser=argparse.ArgumentParser(description="Local-only Phase2 selection and rendering smoke.")
    parser.add_argument("--cases",type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.cases<=3:
        raise ValueError("Offline smoke is limited to 1..3 archived cases")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    outfile=OUTPUT/"token_budget_report.json"
    if outfile.exists():
        raise FileExistsError("Smoke report already exists; not overwriting")
    source=OUTPUT/"source_cases.json"
    if source.exists():
        raise FileExistsError("Intermediate source_cases.json exists; not overwriting")
    source.write_text(json.dumps(original_cases(args.cases),ensure_ascii=False)+"\n",encoding="utf8")
    ns=argparse.Namespace(
        data_file=str(source),limit=args.cases,max_scanned=args.cases,
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
    if len(tasks)!=8*args.cases:
        raise RuntimeError("Expected exactly eight methods per case")
    by_case={}
    for t in tasks:
        row={
            "method":t["method"],
            "source_tokens":t["source_tokens"],"context_tokens":t["context_tokens"],
            "budget_tokens":t["budget_tokens"],
            "within_budget":t["context_tokens"]<=t["budget_tokens"],
        }
        if t.get("phase2_ablation"):
            row.update({
                "seed_chunks":t["phase2_seed_chunk_count"],
                "retained_chunks":t["phase2_kept_chunk_count"],
                "overhead_trimmed_chunks":t["phase2_trimmed_header_overhead"],
            })
            if not row["within_budget"]:
                raise RuntimeError("Phase2 context exceeded rendered token budget")
        by_case.setdefault(t["case_id"],[]).append(row)
    output={
        "note":"OFFLINE ONLY. Real archived source contexts, locally cached MiniLM, no inference.",
        "cases":by_case,"methods":report["methods"],
    }
    outfile.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
