"""Phase 4C: CPU-only marginal evidence-gain regression and greedy selection.

TRAIN labels use evidence from original QASPER train ONLY; validation is used
only for evaluation. Held-out QASPER test is NOT read, scored, or retuned.

Controlled 650-token candidate pool, identical to native Phase 3B/3C.
Controls: frozen manual+render, frozen LambdaMART+render, learned conditional
marginal model with an empty-state (static) versus updated-state greedy policy.
No production default changes.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics
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

import numpy as np
import lightgbm as lgb

from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,parse_input,prepare_split_batched,
)
from submission.experiments.phase3a_oracle_core import gram_counts
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase4c_marginal_core import (
    feature_rows,marginal_gain_targets,training_states,NAMES,RATIOS,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.export import render_context
from tokenpack.tokenization import TokenCounter

BASELINE=ROOT/"submission/results/phase3c_qasper_full889_optimized_20261011/paired_contexts.csv"
ARMS=("manual-render","lambdamart-render",
      "marginal-static","marginal-conditional")
PAIRS=(
    ("marginal-static","manual-render"),
    ("marginal-conditional","manual-render"),
    ("marginal-conditional","lambdamart-render"),
    ("marginal-conditional","marginal-static"),
)
PARAMS={
    "n_estimators":150,"learning_rate":.045,
    "num_leaves":15,"max_depth":5,"min_child_samples":35,
    "reg_lambda":2.0,"random_state":20261011,
    "n_jobs":2,"verbosity":-1,"deterministic":True,
    "force_col_wise":True,
}


def make_training_matrix(train):
    X=[]
    y=[]
    papers=Counter()
    stats=Counter()
    for question in train:
        ids=tuple(range(len(question.scored)))
        if not ids:continue
        grams=[gram_counts(x.chunk.text) for x in question.scored]
        for ratio in RATIOS:
            budget=max(1,math.floor(question.source_tokens*ratio))
            for selected in training_states(question):
                candidates=[i for i in ids if i not in selected]
                if not candidates:continue
                block=feature_rows(question,candidates,selected,budget,ratio,
                                   precomputed=grams)
                gains=marginal_gain_targets(
                    question,candidates,selected,precomputed=grams)
                X.append(block)
                y.append(gains)
                stats["states"]+=1
                stats["candidate_labels"]+=len(gains)
                stats["positive_marginal_labels"]+=int(np.sum(gains>1e-9))
        papers[question.paper_id]+=1
        stats["questions"]+=1
    if not X:raise ValueError("No marginal training samples")
    return np.vstack(X),np.concatenate(y),{
        **stats,"papers":len(papers),
    }


def minmax(scores):
    scores=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(scores).all():raise ValueError("Nonfinite learned scores")
    low=float(np.min(scores))
    hi=float(np.max(scores))
    if hi==low:return np.full_like(scores,.5)
    return .01+.99*(scores-low)/(hi-low)


def dynamic_choose(q,model,counter,budget,ratio,*,conditional:bool):
    """Gold-blind iterative learned marginal prediction, exact render cap.

    Features and normalized learned gain are recomputed when selected set
    changes for conditional arm. Static arm fixes empty-set predictions.
    Normalized score + 20% static relevance fallback is predeclared.
    """
    all_candidates=tuple(range(len(q.scored)))
    if not all_candidates:return {"context":"","used_tokens":0,"selected":[]}
    grams=[gram_counts(x.chunk.text) for x in q.scored]
    render=lambda ids:render_context([
        q.scored[i].chunk for i in sorted(
            ids,key=lambda i:(q.scored[i].chunk.order_key,
                              q.scored[i].chunk.id))])
    empty=render([])
    base_cost=counter.count(empty)
    if base_cost>budget:raise AssertionError("Empty render exceeds cap")
    singleton={
        i:max(1,counter.count(render([i]))-base_cost)
        for i in all_candidates
    }
    selected=[]
    seen=set()
    predicted_static=None
    manual_prior=minmax([x.value for x in q.scored])
    estimated=base_cost
    if not conditional:
        predicted_static=np.asarray(model.predict(
            feature_rows(q,list(all_candidates),(),budget,ratio,grams)),
            dtype=np.float64)
    # Greedy group selection, no gold and no answer string accesses.
    while len(seen)<len(all_candidates):
        remaining=[i for i in all_candidates if i not in seen]
        if conditional:
            prediction=model.predict(
                feature_rows(q,remaining,selected,budget,ratio,grams))
        else:
            prediction=[predicted_static[i] for i in remaining]
        scaled=minmax(prediction)
        ranked=sorted(
            zip(remaining,scaled,strict=True),
            key=lambda i:(-(.8*i[1]+.2*manual_prior[i[0]])/
                          max(1,singleton[i[0]])**.20,
                          q.scored[i[0]].chunk.order_key,i[0]),
        )
        choice=next((i for i,_ in ranked
                     if estimated+singleton[i]<=budget),None)
        if choice is None:break
        selected.append(choice)
        seen.add(choice)
        estimated+=singleton[choice]
    context=render(selected)
    actual=counter.count(context)
    while actual>budget and selected:
        selected.pop()
        context=render(selected)
        actual=counter.count(context)
    # Greedy exact-cap slack fill from predictions with selected-state features
    seen=set(selected)
    remaining=[i for i in all_candidates if i not in seen]
    if remaining:
        predictions=model.predict(feature_rows(
            q,remaining,selected,budget,ratio,grams)) if conditional else [
            predicted_static[i] for i in remaining]
        scaled=minmax(predictions)
        for i,_ in sorted(zip(remaining,scaled,strict=True),
                          key=lambda p:(-p[1],q.scored[p[0]].chunk.order_key)):
            next_context=render([*selected,i])
            next_used=counter.count(next_context)
            if next_used<=budget:
                selected.append(i)
                context,actual=next_context,next_used
    if counter.count(context)!=actual or actual>budget:
        raise AssertionError("True final-render budget violation")
    return {
        "context":context,"used_tokens":actual,
        "selected":[q.scored[i] for i in selected],
    }


def previous_baselines():
    baselines={}
    with BASELINE.open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["method"] in ("manual-render-value","lightgbm-render-value"):
                key=(r["case_id"],float(r["budget_ratio"]),r["method"])
                baselines[key]=r
    return baselines


def pairwise(rows):
    result=[]
    for ratio in RATIOS:
        grouped=defaultdict(dict)
        for r in rows:
            if r["ratio"]==ratio:
                grouped[r["case_id"]][r["method"]]=r
        if any(set(x)!=set(ARMS) for x in grouped.values()):
            raise AssertionError("Incomplete paired marginal score evaluations")
        means={}
        for method in ARMS:
            arm=[x[method] for x in grouped.values()]
            means[method]={
                "n":len(arm),"gold_5gram_mean":round(
                    statistics.mean(x["gold_5gram_recall"] for x in arm),6),
                "mean_realized_tokens":round(
                    statistics.mean(x["used_tokens"] for x in arm),2),
                "zero_gold_cases":sum(x["gold_5gram_recall"]<1e-9 for x in arm),
                "budget_violations":sum(
                    x["used_tokens"]>x["budget"] for x in arm),
            }
        comparisons={}
        for new,old in PAIRS:
            items={
                cid:{
                    "control":{"paper_id":x[old]["paper_id"],"delta":0.},
                    "new":{"paper_id":x[new]["paper_id"],
                           "delta":x[new]["gold_5gram_recall"]-
                                   x[old]["gold_5gram_recall"]},
                } for cid,x in grouped.items()
            }
            diffs=[x["new"]["delta"] for x in items.values()]
            comparisons[f"{new} minus {old}"]={
                "mean_difference":round(statistics.mean(diffs),6),
                "wins":sum(v>1e-9 for v in diffs),
                "losses":sum(v< -1e-9 for v in diffs),
                "ties":sum(abs(v)<=1e-9 for v in diffs),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    items,"delta","new",replicates=5000),
            }
        result.append({"ratio":ratio,"n":len(grouped),
                       "arms":means,"comparisons":comparisons})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--train-papers",type=int,default=0)
    parser.add_argument("--validation-papers",type=int,default=0)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Experiment output exists")
    rows,isolation=parse_input(DEFAULT_DATA)
    if isolation!={"train":888,"validation":281}:
        raise ValueError("Unexpected QASPER native train+validation papers")
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("cl100k_base required")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    start=time.perf_counter()
    train,train_stats=prepare_split_batched(
        "train",rows["train"],embedder,counter,DEFAULT_CACHE,
        args.train_papers or None)
    X,y,labels_stats=make_training_matrix(train)
    print("MARGINAL TRAIN",X.shape,"positive=",labels_stats["positive_marginal_labels"],
          "label_avg=",round(float(y.mean()),5),flush=True)
    model=lgb.LGBMRegressor(**PARAMS)
    weights=np.where(y>1e-9,2.0,1.0)
    model.fit(X,y,sample_weight=weights)
    # Validation selection is performed with a frozen fitted model; no
    # hyperparameter search or retraining based on evaluation is allowed.
    train=[]
    val,val_stats=prepare_split_batched(
        "validation",rows["validation"],embedder,counter,DEFAULT_CACHE,
        args.validation_papers or None)
    previous=previous_baselines()
    records=[]
    selection=[]
    for k,q in enumerate(val,1):
        required=compile_gold_groups(q.evidence_groups)
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            for method,conditional in (("marginal-static",False),
                                       ("marginal-conditional",True)):
                result=dynamic_choose(q,model.booster_,counter,budget,ratio,
                                      conditional=conditional)
                t,f,done=fast_group_coverage(result["context"],required)
                records.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"budget":budget,"method":method,
                    "gold_5gram_recall":round(f,8),
                    "used_tokens":result["used_tokens"],
                    "selected_chunks":len(result["selected"]),
                })
                selection.append({
                    "case_id":q.case_id,"ratio":ratio,"method":method,
                    "chunks":[x.chunk.id for x in result["selected"]],
                })
            for method,prevname in (("manual-render","manual-render-value"),
                                    ("lambdamart-render","lightgbm-render-value")):
                old=previous[(q.case_id,ratio,prevname)]
                records.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"budget":budget,"method":method,
                    "gold_5gram_recall":float(old["gold_5gram_recall"]),
                    "used_tokens":int(old["rendered_tokens"]),
                    "selected_chunks":int(old["chosen_chunks"]),
                })
        if k%100==0 or k==len(val):
            print(f"MARGINAL VALIDATION {k}/{len(val)} questions",flush=True)
    if not args.train_papers and labels_stats["questions"]!=2102:
        raise AssertionError("Expected full native QASPER train 2102 questions")
    if not args.validation_papers and len(val)!=889:
        raise AssertionError("Expected full QASPER validation 889")
    report={
        "experiment":"Phase 4C marginal gold evidence increment learned on QASPER train; validation-only evaluation",
        "train_papers_seen":train_stats["papers_seen"],
        "train_questions_evaluable":train_stats["questions_evaluated"],
        "training_marginal_label_audit":labels_stats,
        "train_feature_shape":list(X.shape),
        "train_positive_label_fraction":round(float(np.mean(y>1e-9)),6),
        "train_mean_label":round(float(y.mean()),6),
        "validation_questions":len(val),
        "validation_papers":len({x.paper_id for x in val}),
        "features":list(NAMES),"params":PARAMS,
        "model_feature_importances":{
            x:int(v) for x,v in zip(NAMES,model.feature_importances_,strict=True)},
        "gold_used_in_training_labels_only":True,
        "train_val_paper_disjoint":True,
        "test_split_used":False,
        "production_code_modified":False,
        "not_independent_heldout_test":True,
        "limits":[
            "Development validation already inspected in Phase3; Phase4C is exploratory.",
            "This pilot uses 650-token original native chunks, not yet the merged three-scale pool.",
            "Selection uses frozen trained model; gold evidence is measured after selection.",
            "Chunk-internal lexical marginal 5grams are only a proxy, not semantic answer sufficiency.",
            "No generative LLM, paid API, GPU or network calls.",
        ],
        "duration_seconds":round(time.perf_counter()-start,2),
        "results":pairwise(records),
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model.booster_.save_model(str(args.output_dir/"marginal_regressor.model.txt"))
    with (args.output_dir/"paired_contexts.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (args.output_dir/"selected_chunks.jsonl").open("w",encoding="utf8") as f:
        for item in selection:f.write(json.dumps(item,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "output":str(args.output_dir),"duration_s":report["duration_seconds"],
        "train":labels_stats,"validation":len(val),"results":report["results"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
