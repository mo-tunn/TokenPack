"""Finish Phase4A from already-computed 5,334 immutable case JSONL records.

The original run completed all gold-aware context computations and then hit
an accounting-only assertion (expected 1,778 instead of 3,556 baseline
control arm rows). THIS TOOL DOES NOT RE-RUN ANY ORACLE SEARCH OR EMBEDDING.
It verifies every historical manual baseline from original Phase3C CSV.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import pathlib
import statistics
import sys

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase4a_qasper_multires_oracle import (
    SCALES,RATIOS,read_reference,VALIDATION_SHA,
)
from submission.experiments.phase3b_full_qasper_train import DEFAULT_DATA
from submission.experiments.phase3b_native_qasper import load_local_qasper_rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",type=pathlib.Path,required=True)
    args=parser.parse_args()
    folder=args.dir
    if (folder/"summary.json").exists():
        raise FileExistsError("Do not overwrite existing experiment summaries")
    file=folder/"cases.jsonl"
    records=[json.loads(s) for s in file.read_text(encoding="utf8").splitlines()
             if s.strip()]
    if len(records)!=5334:raise AssertionError("Incomplete saved case records")
    ids={(r["case_id"],r["scale"],r["budget_ratio"]) for r in records}
    if len(ids)!=len(records):raise AssertionError("Duplicate case-scale-budget")
    questions={r["case_id"] for r in records}
    if len(questions)!=889:raise AssertionError("Wrong QASPER validation denominator")
    original=DEFAULT_DATA/"validation-00000-of-00001.parquet"
    if hashlib.sha256(original.read_bytes()).hexdigest()!=VALIDATION_SHA:
        raise ValueError("Validation dataset SHA256 mismatch")
    docs=load_local_qasper_rows(original)
    if len(docs)!=281:raise AssertionError("Unexpected paper count")
    reference=read_reference()
    verified=0
    for r in records:
        if r["scale"]!=650:continue
        for method,field in (("budget-top-k","topk"),("render-value","render")):
            old=reference[(r["case_id"],r["budget_ratio"],method)]
            now=r[field]
            if abs(float(old["gold_5gram_recall"])-now["recall"])>2e-7:
                raise AssertionError("Frozen Phase3C recall mismatch")
            if int(old["rendered_tokens"])!=now["real_tokens"]:
                raise AssertionError("Frozen Phase3C real token mismatch")
            verified+=1
    if verified!=3556:raise AssertionError("Wrong verified baseline count")
    stats=[]
    for ratio in RATIOS:
        for scale in SCALES:
            rows=[r for r in records
                  if r["budget_ratio"]==ratio and r["scale"]==scale]
            if len(rows)!=889:raise AssertionError("Unpaired scale")
            for r in rows:
                if max(r["topk"]["real_tokens"],r["render"]["real_tokens"],
                       r["oracle"]["oracle_used_tokens"])>r["budget_tokens"]:
                    raise AssertionError("Gold or blind context exceeds actual budget")
            stats.append({
                "scale":scale,"ratio":ratio,"questions":len(rows),
                "manual_topk_recall":round(statistics.mean(
                    r["topk"]["recall"] for r in rows),6),
                "manual_render_recall":round(statistics.mean(
                    r["render"]["recall"] for r in rows),6),
                "gold_feasible_oracle_lower":round(statistics.mean(
                    r["best_feasible_rendered_recall"] for r in rows),6),
                "gold_pool_internal_upper_no_budget":round(statistics.mean(
                    r["oracle"]["pool_internal_upper"] for r in rows),6),
                "oracle_feasible_minus_manual_render":round(statistics.mean(
                    r["feasible_oracle_gap_vs_render"] for r in rows),6),
                "manual_topk_zero":sum(r["topk"]["recall"]<1e-9 for r in rows),
                "manual_render_zero":sum(r["render"]["recall"]<1e-9 for r in rows),
                "oracle_lower_zero":sum(r["best_feasible_rendered_recall"]<1e-9 for r in rows),
                "oracle_all_group_certified":sum(
                    r["oracle"]["all_annotation_objectives_certified"] for r in rows),
                "oracle_heuristic_cases":sum(
                    not r["oracle"]["all_annotation_objectives_certified"] for r in rows),
                "mean_chunks_per_question":round(statistics.mean(
                    r["n_chunks"] for r in rows),2),
                "rendered_budget_breaches":0,
            })
    audit={
        "papers_in_native_validation":281,
        "papers_with_evaluable_questions":len({r["paper_id"] for r in records}),
        "original_questions":sum(len(x["qas"]["question"]) for x in docs),
        "evaluable_questions":len(questions),
        "non_evaluable_questions":sum(len(x["qas"]["question"]) for x in docs)-len(questions),
        "saved_case_records":len(records),
        "baseline_context_pairs_reproduced":verified,
        "all_cases_retrieved_from_saved_jsonl":True,
        "oracle_or_embedding_rerun":False,
        "finalization_only":"original run miscounted 3,556 baseline controls as 1,778",
    }
    output={
        "experiment":"Phase4A QASPER validation-only multires evidence ceiling",
        "dataset":"Native QASPER validation, previously used for development",
        "test_split_used":False,
        "scales":SCALES,"ratios":RATIOS,
        "oracle_uses_gold":True,"gold_blind_manual_scoring":True,
        "gold_objective":"chunk-internal gold 5gram multiset",
        "gold_annotation_policy":"each annotator group separately; OR alternatives",
        "exact_max_relevant":9,
        "n_validation_papers":281,
        "n_development_questions":889,
        "audit":audit,
        "per_scale":stats,
        "limits":[
            "Feasible oracle is a realized lower bound where heuristic; not a general exact oracle.",
            "Certified exactness is for internal objective, not necessarily final rendered recall.",
            "Bütçesiz havuz sınırı sadece chunk içi gold 5gram amacına üst sınırdır.",
            "Gold is used in oracle only; no oracle improvement claimed for real inference.",
            "Output budgets were validated using full cl100k_base final rendering.",
            "Validation was previously seen; official QASPER test was never used in this analysis.",
            "No model downloads, API, Modal, GPU, generative LLM or production code edits.",
        ],
    }
    (folder/"summary.json").write_text(
        json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    (folder/"FINALIZATION_AUDIT.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({"verified":verified,"audits":audit,"per_scale":stats},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
