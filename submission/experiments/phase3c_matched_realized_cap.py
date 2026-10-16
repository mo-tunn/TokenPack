"""Phase 3C posthoc sensitivity: use ORIGINAL TokenPack's realized tokens as cap.

This is NOT exact equality of realized tokens, because chunks are indivisible.
Instead, every competing method has the same per-case cap equal to the
archived manual Budget-Top-K realized length. Original contexts/results are
frozen and reproduced from the archived paired CSV (never overwritten).

No new model training, no gold in selection, no QASPER test split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))

import lightgbm as lgb
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,parse_input,prepare_split_batched,
)
from submission.experiments.phase3b_learned_scoring import make_learned_candidates
from submission.experiments.phase3c_qasper_validation import (
    PREVIOUS,choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter

METHODS=(
    "manual-original",
    "manual-budget-top-k-capped",
    "lightgbm-budget-top-k-capped",
    "manual-render-value-capped",
    "lightgbm-render-value-capped",
    "manual-set-aware-capped",
    "lightgbm-set-aware-capped",
)
PAIRS=(
    ("manual-render-value-capped","manual-original"),
    ("lightgbm-render-value-capped","manual-original"),
    ("lightgbm-render-value-capped","manual-render-value-capped"),
    ("manual-set-aware-capped","manual-original"),
    ("lightgbm-set-aware-capped","manual-original"),
    ("manual-budget-top-k-capped","manual-original"),
    ("lightgbm-budget-top-k-capped","manual-original"),
    ("manual-set-aware-capped","manual-render-value-capped"),
    ("lightgbm-set-aware-capped","lightgbm-render-value-capped"),
)


def load_old_rows(path:Path):
    result={}
    with (path/"validation_paired_contexts.csv").open(encoding="utf8") as file:
        for row in csv.DictReader(file):
            key=(row["case_id"],float(row["budget_ratio"]),row["method"])
            if key in result:
                raise ValueError("Duplicate previous result")
            result[key]=row
    return result


def empty_choice():
    return {"context":"","selected":[],"used_tokens":0,"latency_s":0.,
            "checks":0,"trimmed":0}


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite experiment directories")
    old=load_old_rows(args.previous_dir)
    split_rows,splits=parse_input(args.data_dir)
    if splits!={"train":888,"validation":281}:
        raise ValueError("Not the pinned native QASPER splits")
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("Real cl100k_base missing")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    questions,_=prepare_split_batched(
        "validation",split_rows["validation"],embedder,counter,args.cache_dir,
        max_papers=args.validation_papers or None)
    model=lgb.Booster(model_file=str(
        args.previous_dir/"full_train_lambdamart.model.txt"))
    active_methods=(
        ("manual-original","manual-render-value-capped",
         "lightgbm-render-value-capped")
        if args.focus_render else METHODS
    )
    active_pairs=[(new,base) for new,base in PAIRS
                  if new in active_methods and base in active_methods]
    print(f"MATCHED CAP: questions={len(questions)}, policies={len(active_methods)}",flush=True)
    total_start=time.perf_counter()
    records=[]
    selected_audit=[]
    for idx,q in enumerate(questions,1):
        predictions=model.predict(q.feature_matrix,num_threads=2)
        learned=make_learned_candidates(q.scored,predictions)
        singleton_cost_cache={}
        req=None
        coverage_cache={}
        for ratio in (.25,.5):
            baseline_key=(q.case_id,ratio,"manual-budget-top-k")
            if baseline_key not in old:
                raise ValueError("Question missing from official Phase3B paired archive")
            frozen=old[baseline_key]
            cap=int(frozen["rendered_tokens"])
            if cap<0:raise ValueError("Invalid manual used tokens")
            # Freeze the observed control, rather than needlessly repacking it
            # under a new, tighter *selector weight* cap.
            common={
                "case_id":q.case_id,"paper_id":q.paper_id,
                "budget_ratio":ratio,"manual_realized_token_cap":cap,
                "original_nominal_budget":int(frozen["budget_tokens"]),
            }
            records.append({
                **common,"method":"manual-original",
                "actual_tokens":cap,
                "gold_5gram_recall":float(frozen["gold_5gram_recall"]),
                "gold_token_recall":float(frozen["gold_token_recall"]),
                "any_annotator_group_complete":
                    str(frozen["any_annotator_group_complete"]).lower()=="true",
                "selected_chunks":int(frozen["selected_chunks"]),
                "selector_ms":0.,
                "zero_cap_case":cap==0,
            })
            choices=[]
            for scorer,items in (("manual",q.scored),("lightgbm",learned)):
                for selector in (("render-value",) if args.focus_render
                                 else ("budget-top-k","render-value","set-aware")):
                    method=f"{scorer}-{selector}-capped"
                    if cap==0:
                        chosen=empty_choice()
                    else:
                        chosen=choose(
                            items,query=q.text,budget=cap,counter=counter,
                            policy=selector,
                            singleton_cost_cache=singleton_cost_cache,
                        )
                    if chosen["used_tokens"]>cap:
                        raise AssertionError("Matched realized cap exceeded!")
                    choices.append((method,chosen))
            # Require no annotator gold access until ALL candidates selected.
            if req is None:req=compile_gold_groups(q.evidence_groups)
            for method,chosen in choices:
                content=chosen["context"]
                if content not in coverage_cache:
                    coverage_cache[content]=fast_group_coverage(content,req)
                tok,five,complete=coverage_cache[content]
                records.append({
                    **common,"method":method,
                    "actual_tokens":chosen["used_tokens"],
                    "gold_5gram_recall":five,
                    "gold_token_recall":tok,
                    "any_annotator_group_complete":complete,
                    "selected_chunks":len(chosen["selected"]),
                    "selector_ms":round(1000*chosen["latency_s"],4),
                    "zero_cap_case":cap==0,
                })
                selected_audit.append({
                    "case_id":q.case_id,"budget_ratio":ratio,
                    "method":method,
                    "selected_chunk_ids":[x.chunk.id for x in chosen["selected"]],
                })
        if idx%100==0 or idx==len(questions):
            print(f"MATCHED-CAP {idx}/{len(questions)} questions evaluated",flush=True)
    if len(records)!=len(questions)*2*len(active_methods):
        raise AssertionError("Missing matched-cap evaluation arms")
    result=[]
    for ratio in (.25,.5):
        subset=[r for r in records if r["budget_ratio"]==ratio]
        by_case=defaultdict(dict)
        for r in subset:by_case[r["case_id"]][r["method"]]=r
        if any(set(x)!=set(active_methods) for x in by_case.values()):
            raise AssertionError("Incomplete paired case")
        arms={}
        for method in active_methods:
            data=[x[method] for x in by_case.values()]
            n=len(data)
            arms[method]={
                "questions":n,
                "mean_gold_5gram_recall":round(sum(x["gold_5gram_recall"] for x in data)/n,6),
                "mean_gold_token_recall":round(sum(x["gold_token_recall"] for x in data)/n,6),
                "all_spans_in_any_annotator_group":sum(
                    x["any_annotator_group_complete"] for x in data),
                "average_realized_tokens":round(sum(x["actual_tokens"] for x in data)/n,2),
                "mean_token_shortfall_vs_cap":round(
                    sum(x["manual_realized_token_cap"]-x["actual_tokens"] for x in data)/n,2),
                "fraction_realized_within_5pct_of_manual_cap":
                    round(sum(
                        x["manual_realized_token_cap"]==0 or
                        x["actual_tokens"]>=.95*x["manual_realized_token_cap"]
                        for x in data)/n,6),
                "exact_realized_token_matches":sum(
                    x["manual_realized_token_cap"]==x["actual_tokens"] for x in data),
                "mean_selector_ms":round(sum(x["selector_ms"] for x in data)/n,3),
                "budget_violations":sum(
                    x["actual_tokens"]>x["manual_realized_token_cap"] for x in data),
            }
        contrasts={}
        for newer,older in active_pairs:
            diffs={
                cid:{
                    "control":{"paper_id":items[older]["paper_id"],"delta":0.},
                    "new":{"paper_id":items[newer]["paper_id"],
                           "delta":items[newer]["gold_5gram_recall"]-
                                   items[older]["gold_5gram_recall"]},
                } for cid,items in by_case.items()
            }
            values=[x["new"]["delta"] for x in diffs.values()]
            contrasts[f"{newer} minus {older}"]={
                "mean_difference":round(sum(values)/len(values),6),
                "wins":sum(v>1e-10 for v in values),
                "losses":sum(v< -1e-10 for v in values),
                "ties":sum(abs(v)<=1e-10 for v in values),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    diffs,"delta","new",replicates=4000),
            }
        result.append({
            "ratio":ratio,"n":len(by_case),
            "manual_realized_cap_zero_cases":sum(
                items["manual-original"]["manual_realized_token_cap"]==0
                for items in by_case.values()),
            "arms":arms,"contrasts":contrasts,
        })
    out={
        "experiment":"QASPER Phase3C token-used-matched-CAP sensitivity (NOT exact token equality)",
        "dataset":"Official QASPER validation previously used for exploratory development",
        "question_count":len(questions),
        "paper_count":len({q.paper_id for q in questions}),
        "methods":list(active_methods),
        "original_baseline_frozen":True,
        "shared_cap":"Original manual TokenPack actually rendered tokens PER CASE, not original source-ratio maximum",
        "budget_rule":"Each competing selector may realize fewer, but NEVER more, tokens than frozen manual control",
        "zero_cap_handling":"All methods empty for zero-token baseline cases; cases included in mean",
        "training_performed":False,
        "gold_blind_selection":True,
        "test_split_used":False,
        "results":result,
        "elapsed_seconds":round(time.perf_counter()-total_start,3),
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"paired_realized_caps.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
    with (args.output_dir/"selected_chunks.jsonl").open("w",encoding="utf8") as f:
        for x in selected_audit:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "questions":len(questions),"output":str(args.output_dir),
        "elapsed_s":out["elapsed_seconds"],
        "results":result,
    },ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    ap.add_argument("--previous-dir",type=Path,default=PREVIOUS)
    ap.add_argument("--validation-papers",type=int,default=0)
    ap.add_argument("--focus-render",action="store_true",
                    help="Primary three arms only: original baseline, manual and learned render-aware")
    ap.add_argument("--output-dir",type=Path,required=True)
    run(ap.parse_args())


if __name__=="__main__":
    main()
