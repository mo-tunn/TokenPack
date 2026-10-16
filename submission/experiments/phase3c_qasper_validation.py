"""Phase 3C: frozen LightGBM scorer and predeclared selectors on native validation.

No training, no hyperparameter optimization, no gold access inside selector.
The official test split is NOT read. Entire run is local/offline and opt-in.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.evidence_failure_audit import tokens
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENPACK_HF_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import lightgbm as lgb
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,parse_input,prepare_split_batched,
    best_annotator_coverage,
)
from submission.experiments.phase3b_learned_scoring import make_learned_candidates
from submission.experiments.phase3c_set_aware import select_render_aware
from submission.experiments.phase2_qasper_diagnostic_v2 import render_and_fit
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.export import render_context
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter

PREVIOUS=ROOT/"submission/results/phase3b_qasper_full_train_validation_20261011"
POLICIES=("budget-top-k","render-value","set-aware")
SCORERS=("manual","lightgbm")


def compile_gold_groups(groups:list[list[str]]):
    """Post-selection only: cache immutable annotator requirements per query."""
    compiled=[]
    for group in groups:
        spans=[tokens(x) for x in group if isinstance(x,str) and x.strip()]
        spans=[x for x in spans if x]
        words=Counter(w for span in spans for w in span)
        grams=Counter(tuple(span[i:i+5])
                      for span in spans for i in range(max(0,len(span)-4)))
        if not grams:continue
        span_five=[Counter(tuple(span[i:i+5])
                           for i in range(max(0,len(span)-4)))
                   for span in spans]
        compiled.append((words,grams,span_five))
    return compiled


def fast_group_coverage(context:str,compiled):
    """Exactly match Phase 3B best-annotator policy with one context tokenization.

    The legacy evaluator re-tokenizes the full context for every annotator
    group AND evidence span. Use the same rounded counter/5-gram operation,
    doing the context part once for this frozen set of selections.
    """
    ctx=tokens(context)
    ctx_words=Counter(ctx)
    ctx_five=Counter(tuple(ctx[i:i+5]) for i in range(max(0,len(ctx)-4)))
    options=[]
    any_complete=False
    for group_words,group_5grams,span_five in compiled:
        token_recall=round(sum(min(n,ctx_words.get(k,0)) for k,n in group_words.items())/
                           max(1,sum(group_words.values())),6)
        gram_recall=round(sum(min(n,ctx_five.get(k,0)) for k,n in group_5grams.items())/
                          sum(group_5grams.values()),6)
        options.append((token_recall,gram_recall))
        if not any_complete:
            span_ratios=[]
            for g in span_five:
                span_ratios.append(
                    round(sum(min(n,ctx_five.get(k,0)) for k,n in g.items())/
                          sum(g.values()),6) if g else None
                )
            if span_ratios and all(x is not None and x>=.5 for x in span_ratios):
                any_complete=True
    if not options:return 0.,0.,False
    return (*max(options,key=lambda p:(p[1],p[0])),any_complete)

def choose(scored,*,query,budget,counter,policy,singleton_cost_cache=None):
    render=lambda items:render_context([x.chunk for x in items])
    if policy=="budget-top-k":
        start=time.perf_counter()
        result=select_chunks(scored,strategy="budget-top-k",
                             budget=budget,candidate_pool=300)
        context,kept,trimmed,used,_=render_and_fit(result.selected,render,budget,counter)
        return {
            "context":context,"selected":kept,"used_tokens":used,
            "latency_s":time.perf_counter()-start,
            "checks":0,"trimmed":trimmed,
        }
    selection=select_render_aware(
        scored,query=query,budget=budget,counter=counter,
        render=render,policy=policy,candidate_pool=300,
        singleton_cost_cache=singleton_cost_cache)
    return {
        "context":selection.context,"selected":selection.selected,
        "used_tokens":selection.real_tokens,
        "latency_s":selection.elapsed_seconds,
        "checks":selection.candidates_checked,
        "trimmed":0,
    }


def make_records(questions,model,counter,*,limit=None):
    records=[]
    audits=[]
    for i,q in enumerate(questions[:limit] if limit else questions,1):
        pred=model.predict(q.feature_matrix,num_threads=2)
        learned=make_learned_candidates(q.scored,pred)
        singleton_cost_cache={}
        # Select and render ALL arms with NO gold access.
        choices=[]
        for ratio in (.25,.50):
            budget=max(1,math.floor(q.source_tokens*ratio))
            for scorer,items in (("manual",q.scored),("lightgbm",learned)):
                for policy in POLICIES:
                    selected=choose(items,query=q.text,budget=budget,
                                    counter=counter,policy=policy,
                                    singleton_cost_cache=singleton_cost_cache)
                    if counter.count(selected["context"])>budget:
                        raise AssertionError("Real rendered token budget violation")
                    choices.append({
                        "ratio":ratio,"budget":budget,
                        "scorer":scorer,"selector":policy,"chosen":selected,
                    })
        # Gold annotations are consulted ONLY after all selections are frozen.
        requirements=compile_gold_groups(q.evidence_groups)
        coverage_cache={}
        for choice in choices:
            sel=choice["chosen"]
            content=sel["context"]
            if content not in coverage_cache:
                coverage_cache[content]=fast_group_coverage(content,requirements)
            tok,five,full=coverage_cache[content]
            records.append({
                "case_id":q.case_id,"paper_id":q.paper_id,
                "budget_ratio":choice["ratio"],
                "budget_tokens":choice["budget"],
                "scorer":choice["scorer"],"selector":choice["selector"],
                "method":f"{choice['scorer']}-{choice['selector']}",
                "gold_5gram_recall":round(five,8),
                "gold_token_recall":round(tok,8),
                "any_annotator_group_complete":full,
                "rendered_tokens":sel["used_tokens"],
                "chosen_chunks":len(sel["selected"]),
                "selector_time_seconds":round(sel["latency_s"],6),
                "exact_budget_checks":sel["checks"],
                "discarded_chunks_after_render":sel["trimmed"],
            })
            audits.append({
                "case_id":q.case_id,"budget_ratio":choice["ratio"],
                "method":f"{choice['scorer']}-{choice['selector']}",
                "selected_chunk_ids":[x.chunk.id for x in sel["selected"]],
            })
        if i%100==0 or i==len(questions):
            print(f"PHASE3C validation {i}/{len(questions)} "
                  f"questions, {len(records)} arm-contexts",flush=True)
    return records,audits


def verify_previous_phase3b_control(rows,previous:Path):
    """Raise on any invalid baseline reproduction, not silently proceed."""
    archive={}
    with (previous/"validation_paired_contexts.csv").open(encoding="utf8") as file:
        for old in csv.DictReader(file):
            who="manual" if old["method"]=="manual-budget-top-k" else "lightgbm"
            archive[(old["case_id"],float(old["budget_ratio"]),who)]=old
    seen=0
    for row in rows:
        if row["selector"]!="budget-top-k":continue
        key=(row["case_id"],row["budget_ratio"],row["scorer"])
        old=archive[key]
        if abs(row["gold_5gram_recall"]-float(old["gold_5gram_recall"]))>2e-7:
            raise AssertionError(f"Phase3B baseline evidence mismatch: {key}")
        if row["rendered_tokens"]!=int(old["rendered_tokens"]):
            raise AssertionError(f"Phase3B baseline token mismatch: {key}")
        seen+=1
    if seen!=len(archive):
        raise AssertionError(f"Missing baseline cases: {seen} vs {len(archive)}")
    return seen


def summary_by_paper(rows):
    results=[]
    for ratio in (.25,.5):
        subset=[r for r in rows if r["budget_ratio"]==ratio]
        by_case=defaultdict(dict)
        for row in subset:by_case[row["case_id"]][row["method"]]=row
        methods=[f"{s}-{p}" for s in SCORERS for p in POLICIES]
        for case,found in by_case.items():
            if set(found)!=set(methods):
                raise AssertionError("Unpaired experimental case "+case)
        arms={}
        for method in methods:
            arm=[pair[method] for pair in by_case.values()]
            n=len(arm)
            arms[method]={
                "n":n,
                "gold_5gram_mean":round(sum(x["gold_5gram_recall"] for x in arm)/n,6),
                "gold_token_mean":round(sum(x["gold_token_recall"] for x in arm)/n,6),
                "any_annotator_group_complete":sum(x["any_annotator_group_complete"] for x in arm),
                "avg_real_tokens":round(sum(x["rendered_tokens"] for x in arm)/n,2),
                "avg_selected_chunks":round(sum(x["chosen_chunks"] for x in arm)/n,3),
                "avg_selector_time_ms":round(1000*sum(x["selector_time_seconds"] for x in arm)/n,3),
                "actual_budget_violations":sum(x["rendered_tokens"]>x["budget_tokens"] for x in arm),
            }
        comparisons={}
        comparison_pairs=[
            ("manual-render-value","manual-budget-top-k"),
            ("manual-set-aware","manual-budget-top-k"),
            ("manual-set-aware","manual-render-value"),
            ("lightgbm-render-value","lightgbm-budget-top-k"),
            ("lightgbm-set-aware","lightgbm-budget-top-k"),
            ("lightgbm-set-aware","lightgbm-render-value"),
            ("lightgbm-set-aware","manual-budget-top-k"),
            ("lightgbm-set-aware","manual-set-aware"),
        ]
        for new,base in comparison_pairs:
            deltas={
                cid:{
                    "control":{"paper_id":pair[base]["paper_id"],"delta":0.0},
                    "new":{"paper_id":pair[new]["paper_id"],
                           "delta":pair[new]["gold_5gram_recall"]-pair[base]["gold_5gram_recall"]}
                } for cid,pair in by_case.items()
            }
            diffs=[x["new"]["delta"] for x in deltas.values()]
            comparisons[f"{new} minus {base}"]={
                "mean_difference":round(sum(diffs)/len(diffs),6),
                "wins":sum(d>1e-10 for d in diffs),
                "losses":sum(d< -1e-10 for d in diffs),
                "ties":sum(abs(d)<=1e-10 for d in diffs),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    deltas,"delta","new",replicates=4000),
            }
        results.append({"budget_ratio":ratio,
                        "case_count":len(by_case),
                        "arms":arms,"comparisons":comparisons})
    return results


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Historical or existing experiment directory must not be overwritten")
    model_file=args.previous_dir/"full_train_lambdamart.model.txt"
    if not model_file.is_file():
        raise FileNotFoundError("Frozen Phase3B model required")
    # Load ONLY official validation: train and test are not touched.
    _,meta=parse_input(args.data_dir)
    if meta!={"train":888,"validation":281}:
        raise ValueError("Unexpected QASPER split")
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("True cl100k_base required")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    from submission.experiments.phase3b_native_qasper import load_local_qasper_rows
    validation=load_local_qasper_rows(
        args.data_dir/"validation-00000-of-00001.parquet")
    questions,audit=prepare_split_batched(
        "validation",validation,embedder,counter,args.cache_dir,
        max_papers=args.validation_papers or None)
    model=lgb.Booster(model_file=str(model_file))
    print(f"PHASE3C frozen model; questions={len(questions)}; "
          f"policies={POLICIES}; scorer arms={SCORERS}",flush=True)
    start=time.perf_counter()
    rows,audits=make_records(questions,model,counter)
    replicated=None
    if not args.validation_papers:
        replicated=verify_previous_phase3b_control(rows,args.previous_dir)
    results=summary_by_paper(rows)
    duration=round(time.perf_counter()-start,3)
    payload={
        "experiment":"Phase 3C frozen scorer x selector paired 2x3 factorial",
        "dataset":"Official QASPER validation (already used in Phase3B); NOT independent holdout",
        "cases":len(questions),"papers":len({q.paper_id for q in questions}),
        "scorers":list(SCORERS),"selectors":list(POLICIES),
        "source_model_path":str(model_file),
        "production_modified":False,
        "fixed_predeclared_weights":{
            "normalization":"per-query .1-.999 normalized relevance",
            "set_aware_novelty":"max(.15,1-.50*max_jaccard-.15*max_embedding_cosine)",
            "set_aware_utility":"(relevance*(.78+.22*novelty)+.22*new_query_terms)/(singleton_rendered_token_cost)**.30; additive singleton planning, exact final-render correction and slack fill",
            "render_value_utility":"relevance; additive singleton rendered-cost planning, exact final-render correction and slack fill",
        },
        "selection_gold_blind":True,
        "test_split_used":False,
        "network_calls":0,
        "all_budgets_strict_final_render":True,
        "baseline_contexts_reproduced":replicated,
        "validation_data_audit":audit,
        "duration_seconds":duration,
        "results":results,
        "limitations":[
            "Native QASPER validation already used to measure Phase3B; any selection improvement is exploratory, not a sealed final test.",
            "Gold 5gram recall and span-group completion are proxies, not final answer accuracy.",
            "Set-aware novelty is heuristic and fixed ex ante; no model tuning performed on this run.",
        ],
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"paired_contexts.csv").open("w",encoding="utf8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with (args.output_dir/"chunk_selection_audit.jsonl").open("w",encoding="utf8") as file:
        for item in audits:file.write(json.dumps(item,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "output":str(args.output_dir),"cases":len(questions),
        "baseline_verified":replicated,
        "duration_s":duration,
        "results":results,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    ap.add_argument("--previous-dir",type=Path,default=PREVIOUS)
    ap.add_argument("--validation-papers",type=int,default=0,
                    help="ONLY for smoke: first N native validation papers")
    ap.add_argument("--output-dir",type=Path,required=True)
    run(ap.parse_args())
