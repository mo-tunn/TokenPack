"""Phase 3D: ONE frozen, offline evaluation on never-used native QASPER test.

Only four predeclared arms at source-relative maximum token budgets 25/50%.
NO tuning, training, LLM answer generation, GPU, API calls or network access.
All inference is gold-blind. Gold 5-gram metrics are computed after every
candidate context is frozen. Existing production and validation never change.

Test can be downloaded separately from the same fixed Hugging Face commit.
For provenance, check PREDECLARED_SOURCE_SHA256 below before test evaluation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENPACK_HF_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import lightgbm as lgb
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,EXPECTED_HASHES,prepare_split_batched,
)
from submission.experiments.phase3b_native_qasper import (
    load_local_qasper_rows,validate_native_splits,
)
from submission.experiments.phase3b_learned_scoring import (
    make_learned_candidates,ndcg_at,
)
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter

COMMIT="96fd84d8f3c9d9129403f06ddc9ad72042d81a20"
MODEL=ROOT/"submission/results/phase3b_qasper_full_train_validation_20261011/full_train_lambdamart.model.txt"
MODEL_SHA256="7492db71ec8a9113ee81e5e607e4392427d248a0b3bda7933b494bd33f6fcf01"
SOURCES={
    "submission/experiments/phase3b_full_qasper_train.py":
        "7415e46a63db059185cba673b5399f83e6f55d492c4bb08f605487fdfd587d59",
    "submission/experiments/phase3b_native_qasper.py":
        "3d371d986d1f6c73c39a63d46854e1487674b08903111948e9c0940167b51e77",
    "submission/experiments/phase3c_set_aware.py":
        "cabafb49cf03c70a1ae75d272df7dffa1fb89c05938b6c88c85c5369ce7d6928",
    "submission/experiments/phase3c_qasper_validation.py":
        "4cad3bcb3620d7f5ecb9fa0713da49cb3c07673f6d07b70b8973434c540d3e61",
    "submission/experiments/phase3b_learned_scoring.py":
        "6ab09eae153667ec918795cb49743c7ae3b328079a2949519bff7a5f9d438772",
}
ARMS=(
    "manual-budget-top-k",
    "manual-render-value",
    "lightgbm-budget-top-k",
    "lightgbm-render-value",
)
PAIRS=(
    ("manual-render-value","manual-budget-top-k"),
    ("lightgbm-render-value","lightgbm-budget-top-k"),
    ("lightgbm-render-value","manual-budget-top-k"),
    ("lightgbm-render-value","manual-render-value"),
    ("lightgbm-budget-top-k","manual-budget-top-k"),
)
RATIOS=(.25,.5)


def sha256_file(file:Path)->str:
    h=hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_sources():
    for rel,digest in SOURCES.items():
        file=ROOT/rel
        if sha256_file(file)!=digest:
            raise RuntimeError("Frozen Phase3D dependency changed: "+rel)
    if sha256_file(MODEL)!=MODEL_SHA256:
        raise RuntimeError("Frozen LightGBM model hash mismatch")


def load_untouched_test(test_path:Path,data_dir:Path):
    if test_path.name!="test-00000-of-00001.parquet":
        raise ValueError("Official pinned test parquet filename required")
    if not test_path.is_file() or test_path.stat().st_size<10000:
        raise FileNotFoundError("Official pinned QASPER test not present")
    with test_path.open("rb") as f:
        if f.read(4)!=b"PAR1":
            raise ValueError("Bad Parquet magic")
    old={}
    for name in ("train","validation"):
        path=data_dir/f"{name}-00000-of-00001.parquet"
        if sha256_file(path)!=EXPECTED_HASHES[name]:
            raise RuntimeError("Previous native split SHA256 mismatch: "+name)
        old[name]=load_local_qasper_rows(path)
    test=load_local_qasper_rows(test_path)
    disjoint=validate_native_splits(old["train"],old["validation"],test)
    return test,disjoint,sha256_file(test_path)


def fixed_budget(source_tokens:int,ratio:float)->int:
    """Match Phase 3C nominal budget, not realized-token posthoc variant."""
    if source_tokens<=0 or ratio not in RATIOS:
        raise ValueError("Bad source tokens/ratio")
    return max(1,math.floor(source_tokens*ratio))


def paired_stats(records):
    results=[]
    for ratio in RATIOS:
        by_case=defaultdict(dict)
        for row in records:
            if row["budget_ratio"]==ratio:
                by_case[row["case_id"]][row["method"]]=row
        if not by_case or any(set(items)!=set(ARMS) for items in by_case.values()):
            raise AssertionError("Missing/duplicate paired arms")
        arms={}
        for name in ARMS:
            samples=[c[name] for c in by_case.values()]
            n=len(samples)
            arms[name]={
                "n":n,
                "gold_5gram_mean":round(sum(x["gold_5gram_recall"] for x in samples)/n,6),
                "gold_token_mean":round(sum(x["gold_token_recall"] for x in samples)/n,6),
                "any_annotator_group_complete":sum(x["any_annotator_group_complete"] for x in samples),
                "mean_realized_tokens":round(sum(x["actual_tokens"] for x in samples)/n,2),
                "mean_selected_chunks":round(sum(x["selected_chunks"] for x in samples)/n,3),
                "mean_selector_ms":round(sum(x["selector_ms"] for x in samples)/n,3),
                "budget_violations":sum(x["actual_tokens"]>x["budget_tokens"] for x in samples),
            }
        comparisons={}
        for new,base in PAIRS:
            values={
                cid:{
                    "control":{"paper_id":x[base]["paper_id"],"delta":0.},
                    "new":{"paper_id":x[new]["paper_id"],
                           "delta":x[new]["gold_5gram_recall"]-x[base]["gold_5gram_recall"]},
                } for cid,x in by_case.items()
            }
            diffs=[x["new"]["delta"] for x in values.values()]
            comparisons[f"{new} minus {base}"]={
                "mean_difference":round(sum(diffs)/len(diffs),6),
                "wins":sum(x>1e-10 for x in diffs),
                "losses":sum(x< -1e-10 for x in diffs),
                "ties":sum(abs(x)<=1e-10 for x in diffs),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    values,"delta","new",replicates=6000),
            }
        results.append({"ratio":ratio,"n":len(by_case),
                        "arms":arms,"comparisons":comparisons})
    return results


def rank_metrics(questions,model):
    # Secondary (gold-based) measurement only, NOT used to re-rank.
    rows=[]
    for q in questions:
        pred=model.predict(q.feature_matrix,num_threads=2)
        rows.append({
            "case_id":q.case_id,"paper_id":q.paper_id,
            "chunks":len(q.scored),
            "annotation_groups":len(q.evidence_groups),
            "manual_ndcg_at5":round(ndcg_at(q.labels,[x.value for x in q.scored]),6),
            "lightgbm_ndcg_at5":round(ndcg_at(q.labels,pred),6),
        })
    return rows


def select_before_gold(questions,model,counter):
    all_rows=[]
    audit=[]
    rankings=[]
    from submission.experiments.phase3b_learned_scoring import ndcg_at
    for index,q in enumerate(questions,1):
        pred=model.predict(q.feature_matrix,num_threads=2)
        learned=make_learned_candidates(q.scored,pred)
        choices=[]
        cache={}
        for ratio in RATIOS:
            budget=fixed_budget(q.source_tokens,ratio)
            for scorer,items in (("manual",q.scored),("lightgbm",learned)):
                for selector in ("budget-top-k","render-value"):
                    method=f"{scorer}-{selector}"
                    selected=choose(items,query=q.text,budget=budget,
                                    counter=counter,policy=selector,
                                    singleton_cost_cache=cache)
                    context=selected["context"]
                    actual=counter.count(context)
                    if actual!=selected["used_tokens"] or actual>budget:
                        raise AssertionError("Final exact-render token mismatch")
                    choices.append((ratio,budget,method,selected))
        # Gold is opened for evaluation only AFTER all selections are frozen.
        requirements=compile_gold_groups(q.evidence_groups)
        measured={}
        for ratio,budget,method,choice in choices:
            context=choice["context"]
            if context not in measured:
                measured[context]=fast_group_coverage(context,requirements)
            token,five,complete=measured[context]
            all_rows.append({
                "case_id":q.case_id,"paper_id":q.paper_id,
                "budget_ratio":ratio,"budget_tokens":budget,
                "method":method,"actual_tokens":choice["used_tokens"],
                "gold_5gram_recall":round(five,8),
                "gold_token_recall":round(token,8),
                "any_annotator_group_complete":complete,
                "selected_chunks":len(choice["selected"]),
                "selector_ms":round(choice["latency_s"]*1000,4),
            })
            audit.append({
                "case_id":q.case_id,"ratio":ratio,"method":method,
                "selected_ids":[x.chunk.id for x in choice["selected"]],
            })
        rankings.append({
            "case_id":q.case_id,"paper_id":q.paper_id,
            "num_candidates":len(q.scored),
            "num_alternative_annotator_groups":len(q.evidence_groups),
            "manual_ndcg_at5":round(ndcg_at(q.labels,[x.value for x in q.scored]),6),
            "lightgbm_ndcg_at5":round(ndcg_at(q.labels,pred),6),
        })
        if index%100==0 or index==len(questions):
            print(f"PHASE3D TEST {index}/{len(questions)} questions, {len(all_rows)} contexts",flush=True)
    return all_rows,audit,rankings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test-parquet",required=True,type=Path)
    ap.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--cache-dir",type=Path,default=ROOT/".tokenpack/cache/phase3d_qasper_test_frozen_20261011")
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite prior test result")
    verify_frozen_sources()
    paper_rows,isolation,test_hash=load_untouched_test(args.test_parquet,args.data_dir)
    print(f"FROZEN sources and model OK; dataset SHA256={test_hash}; paper splits={isolation}",flush=True)
    if isolation!={"train":888,"validation":281,"test":416}:
        raise ValueError("Unexpected native QASPER splits, abort")
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("cl100k_base tokenizer required")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    model=lgb.Booster(model_file=str(MODEL))
    elapsed=time.perf_counter()
    questions,data_audit=prepare_split_batched(
        "test",paper_rows,embedder,counter,args.cache_dir)
    if not questions:
        raise ValueError("No evaluable native test questions")
    rows,audits,ranking=select_before_gold(questions,model,counter)
    if len(rows)!=len(questions)*len(RATIOS)*len(ARMS):
        raise AssertionError("Not all frozen test combinations were evaluated")
    results=paired_stats(rows)
    summary={
        "experiment":"Faz 3D one-shot QASPER official held-out test, frozen 4-arm protocol",
        "test_dataset_release_commit":COMMIT,
        "test_file_sha256":test_hash,
        "prior_train_validation_papers":{"train":isolation["train"],"validation":isolation["validation"]},
        "num_test_papers_all":isolation["test"],
        "num_test_questions_all":sum(len(row["qas"]["question"]) for row in paper_rows),
        "num_evaluable_test_questions":len(questions),
        "num_test_papers_evaluable":len({x.paper_id for x in questions}),
        "test_data_audit":data_audit,
        "frozen_model_sha256":MODEL_SHA256,
        "frozen_source_sha256":SOURCES,
        "scores":{"manual":"evidence-hybrid","lightgbm":"fixed QASPER train-trained LambdaMART"},
        "selector_policies":["existing budget-top-k","frozen render-value singleton planning with exact final token check"],
        "budgets":list(RATIOS),
        "identical_token_maxima":True,
        "note":"Equal token upper bounds, NOT necessarily equal actual used tokens",
        "annotations":"Separate QASPER evidence alternative groups; group-max 5gram recall proxy",
        "test_predictions_used_for_tuning":False,
        "test_split_used_for_training":False,
        "generative_inference_calls":0,"paid_api_calls":0,"gpu_calls":0,
        "mean_manual_ndcg_at5":round(sum(x["manual_ndcg_at5"] for x in ranking)/len(ranking),6),
        "mean_lightgbm_ndcg_at5":round(sum(x["lightgbm_ndcg_at5"] for x in ranking)/len(ranking),6),
        "results":results,
        "elapsed_seconds":round(time.perf_counter()-elapsed,3),
        "limitations":[
            "Gold evidence lexical 5-gram proxy is NOT answer accuracy.",
            "Excluded unanswerable / absent / too-short evidence questions from evidence-recall denominator.",
            "Alternative evidence annotations are respected as OR groups, not pooled into impossible unions.",
            "Compared equal maximum final rendered token budgets; actual consumption can differ.",
            "LightGBM was developed using QASPER validation; test is a single held-out check with no test-driven tuning.",
        ],
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    def write_csv(file,items):
        with (args.output_dir/file).open("w",encoding="utf8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]))
            w.writeheader();w.writerows(items)
    write_csv("paired_test_contexts.csv",rows)
    write_csv("test_ranking.csv",ranking)
    with (args.output_dir/"selection_audit.jsonl").open("w",encoding="utf8") as f:
        for item in audits:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "output":str(args.output_dir),
        "num_evaluable_test_questions":len(questions),
        "test_papers_evaluable":summary["num_test_papers_evaluable"],
        "mean_manual_ndcg_at5":summary["mean_manual_ndcg_at5"],
        "mean_lightgbm_ndcg_at5":summary["mean_lightgbm_ndcg_at5"],
        "results":results,
        "seconds":summary["elapsed_seconds"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
