"""Phase 4C3: offline CPU real-budget rollout-state conditional training.

TRAIN: native paper-disjoint QASPER train (888 papers, 2,102 usable QAs).
VALIDATE: 889 usable QASPER validation questions, previously inspected.
NO QASPER test access; no web, GPU, generative model, production-code edit.

Ablation versus frozen Phase 4C original conditional regression, frozen
LambdaMART + render-aware and manual + render-aware; same 650-token chunk
pool, same feature set/LightGBM hyperparameters and identical Phase 4C
dynamic inference selector. Only train state distribution + correct
best-alternative-annotator label are different.
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
os.environ["OMP_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"

import lightgbm as lgb
import numpy as np
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,parse_input,prepare_split_batched,
)
from submission.experiments.phase3a_oracle_core import gram_counts
from submission.experiments.phase3c_qasper_validation import (
    compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase4c_marginal_core import (
    feature_rows,NAMES,RATIOS,
)
from submission.experiments.phase4c2_aligned_core import (
    aligned_marginal_gain_targets,
)
from submission.experiments.phase4c_train_marginal import (
    PARAMS,dynamic_choose,
)
from submission.experiments.phase4c3_rollout_core import (
    STATE_COUNTS,manual_budget_rollout,candidates_with_remaining_budget,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter

OLD=ROOT/"submission/results/phase4c_marginal_full_train_validation889_20261011/paired_contexts.csv"
METHODS=("manual-render","lambdamart-render",
         "old-marginal-conditional","rollout-marginal-static",
         "rollout-marginal-conditional")
CONTRASTS=(
    ("rollout-marginal-conditional","old-marginal-conditional"),
    ("rollout-marginal-conditional","lambdamart-render"),
    ("rollout-marginal-conditional","manual-render"),
    ("rollout-marginal-conditional","rollout-marginal-static"),
)


class BoundedBooster:
    """Numerically identical LightGBM predictions with bounded CPU threads.

    Older Phase4C dynamic_choose calls model.predict(...) without passing a
    thread count. For a raw Booster this may spawn dozens of OpenMP threads
    *for every single greedy step*. Only scheduling is changed.
    """
    def __init__(self,booster):
        self.booster=booster

    def predict(self,matrix):
        return self.booster.predict(matrix,num_threads=2)


def build_training_matrix(examples,counter):
    features=[]
    labels=[]
    counter_audit=Counter()
    states_by_size=Counter()
    for q in examples:
        n=len(q.scored)
        grams=[gram_counts(item.chunk.text) for item in q.scored]
        counter_audit["questions"]+=1
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            trace=manual_budget_rollout(q,counter,budget)
            counter_audit["rollout_count"]+=1
            counter_audit["budget_count"]+=budget
            counter_audit["num_rollout_selected"]+=len(trace.ordered)
            counter_audit["rollout_exact_checks"]+=trace.checks
            for chosen in trace.states:
                remaining=candidates_with_remaining_budget(
                    q,trace,chosen,budget,counter)
                states_by_size[len(chosen)]+=1
                if not remaining:
                    counter_audit["states_no_feasible_actions"]+=1
                    continue
                x=feature_rows(q,remaining,chosen,budget,ratio,grams)
                y=aligned_marginal_gain_targets(q,remaining,chosen,grams)
                if len(x)!=len(y):
                    raise AssertionError("Feature/target rows misaligned")
                features.append(x)
                labels.append(y)
                counter_audit["useful_states"]+=1
                counter_audit["training_action_rows"]+=len(y)
                counter_audit["positive_action_labels"]+=int(np.sum(y>1e-9))
                counter_audit["positive_state_labels"]+=int(np.any(y>1e-9))
    if not features:raise RuntimeError("No budgeted states")
    X=np.vstack(features)
    y=np.concatenate(labels)
    counter_audit["n_features"]=len(NAMES)
    return X,y,dict(counter_audit),dict(sorted(states_by_size.items()))


def frozen_baselines():
    result={}
    with OLD.open(encoding="utf8") as f:
        for record in csv.DictReader(f):
            if record["method"] not in (
                "manual-render","lambdamart-render","marginal-conditional"):
                continue
            result[record["case_id"],float(record["ratio"]),record["method"]]=record
    return result


def pairwise(records):
    output=[]
    for ratio in RATIOS:
        groups=defaultdict(dict)
        for row in records:
            if row["ratio"]==ratio:
                if row["method"] in groups[row["case_id"]]:
                    raise AssertionError("Duplicate candidate results")
                groups[row["case_id"]][row["method"]]=row
        if not groups or any(set(g)!=set(METHODS) for g in groups.values()):
            raise AssertionError("Unpaired or missing method results")
        arms={}
        for method in METHODS:
            xs=[g[method] for g in groups.values()]
            arms[method]={
                "n":len(xs),
                "gold_5gram_mean":round(statistics.mean(
                    x["gold_5gram_recall"] for x in xs),6),
                "mean_actual_tokens":round(statistics.mean(
                    x["actual_tokens"] for x in xs),2),
                "mean_selected_chunks":round(statistics.mean(
                    x["selected_chunks"] for x in xs),2),
                "mean_selector_ms":round(statistics.mean(
                    x["selector_ms"] for x in xs),2),
                "zero_evidence":sum(x["gold_5gram_recall"]<1e-9 for x in xs),
                "budget_violations":sum(
                    x["actual_tokens"]>x["budget"] for x in xs),
            }
        contrasts={}
        for newer,older in CONTRASTS:
            data={
                cid:{
                    "control":{"paper_id":g[older]["paper_id"],"delta":0.},
                    "new":{"paper_id":g[newer]["paper_id"],
                           "delta":g[newer]["gold_5gram_recall"]-
                                   g[older]["gold_5gram_recall"]},
                } for cid,g in groups.items()
            }
            differences=[x["new"]["delta"] for x in data.values()]
            contrasts[f"{newer} minus {older}"]={
                "mean_difference":round(statistics.mean(differences),6),
                "wins":sum(x>1e-10 for x in differences),
                "losses":sum(x< -1e-10 for x in differences),
                "ties":sum(abs(x)<=1e-10 for x in differences),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    data,"delta","new",replicates=5000),
            }
        output.append({"ratio":ratio,"questions":len(groups),
                       "arms":arms,"contrasts":contrasts})
    return output


def run(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite experiment results")
    splits,isolation=parse_input(DEFAULT_DATA)
    if isolation!={"train":888,"validation":281}:
        raise ValueError("Official QASPER paper split mismatch")
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("Require genuine cl100k_base")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    start=time.perf_counter()
    train,train_audit=prepare_split_batched(
        "train",splits["train"],embedder,counter,DEFAULT_CACHE,
        args.train_papers or None)
    X,y,states_audit,by_count=build_training_matrix(train,counter)
    print("PHASE4C3 TRAIN MATRIX",X.shape,"LABELS",len(y),
          "POSITIVE",states_audit["positive_action_labels"],
          "STATES",by_count,flush=True)
    if not args.train_papers and len(train)!=2102:
        raise AssertionError("Expected 2102 usable train questions")
    model=lgb.LGBMRegressor(**PARAMS)
    model.fit(X,y,sample_weight=np.where(y>1e-9,2.0,1.0))
    importance=dict(zip(NAMES,map(int,model.feature_importances_),strict=True))
    booster=model.booster_
    bounded=BoundedBooster(booster)
    # Persist trained weights BEFORE expensive validation, so inference
    # interruption cannot destroy a complete train-only experiment.
    args.output_dir.mkdir(parents=True,exist_ok=False)
    booster.save_model(str(args.output_dir/"rollout_marginal_regressor.model.txt"))
    (args.output_dir/"TRAIN_CHECKPOINT.json").write_text(
        json.dumps({
            "train_questions":states_audit["questions"],
            "train_audit":states_audit,
            "state_counts":by_count,
            "feature_importances":importance,
            "model_params":PARAMS,
            "note":"Train-only model saved before validation; not independently tested.",
        },ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("PHASE4C3 TRAIN CHECKPOINT SAVED",args.output_dir,flush=True)
    train_paper_ids={q.paper_id for q in train}
    train=[];X=None;y=None
    validation,val_audit=prepare_split_batched(
        "validation",splits["validation"],embedder,counter,DEFAULT_CACHE,
        args.validation_papers or None)
    control=frozen_baselines()
    records=[]
    selections=[]
    for num,q in enumerate(validation,1):
        if q.paper_id in train_paper_ids:
            raise AssertionError("Train/validation paper overlap")
        plans=[]
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            for name,conditional in (
                ("rollout-marginal-static",False),
                ("rollout-marginal-conditional",True),
            ):
                t=time.perf_counter()
                picked=dynamic_choose(
                    q,bounded,counter,budget,ratio,conditional=conditional)
                duration=(time.perf_counter()-t)*1000
                plans.append((ratio,budget,name,picked,duration))
                selections.append({
                    "case_id":q.case_id,"ratio":ratio,"method":name,
                    "ids":[s.chunk.id for s in picked["selected"]],
                })
        # Only read gold evidence during EVALUATION, after all context decisions.
        requirements=compile_gold_groups(q.evidence_groups)
        for ratio,budget,name,picked,ms in plans:
            _,recall,_=fast_group_coverage(picked["context"],requirements)
            records.append({
                "case_id":q.case_id,"paper_id":q.paper_id,
                "ratio":ratio,"budget":budget,"method":name,
                "gold_5gram_recall":round(recall,8),
                "actual_tokens":picked["used_tokens"],
                "selected_chunks":len(picked["selected"]),
                "selector_ms":round(ms,2),
            })
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            for arm,oldname in (
                ("manual-render","manual-render"),
                ("lambdamart-render","lambdamart-render"),
                ("old-marginal-conditional","marginal-conditional"),
            ):
                r=control[(q.case_id,ratio,oldname)]
                records.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"budget":budget,"method":arm,
                    "gold_5gram_recall":float(r["gold_5gram_recall"]),
                    "actual_tokens":int(r["used_tokens"]),
                    "selected_chunks":int(r["selected_chunks"]),
                    "selector_ms":0.,  # frozen archive has no per-arm timing
                })
        if num%100==0 or num==len(validation):
            print(f"PHASE4C3 VALIDATION {num}/{len(validation)}",flush=True)
    if not args.validation_papers and len(validation)!=889:
        raise AssertionError("Expected 889 usable validation questions")
    if any(r["actual_tokens"]>r["budget"] for r in records):
        raise AssertionError("Real rendered token budget overshoot")
    summary={
        "experiment":"Phase4C3 budget-rollout aligned-marginal learning",
        "split":"Official QASPER train fit, paper-disjoint validation eval; test NEVER used",
        "train_questions":states_audit["questions"],
        "train_papers":len(train_paper_ids),
        "validation_questions":len(validation),
        "validation_papers":len({q.paper_id for q in validation}),
        "train_rollout_states_by_selected_count":by_count,
        "train_action_audit":states_audit,
        "fixed_state_counts":list(STATE_COUNTS),
        "same_24_features_as_phase4c":list(NAMES),
        "model_parameters":PARAMS,
        "feature_importances":importance,
        "old_result_read_only":True,
        "train_gold_labels_aligned_to_best_annotator_metric":True,
        "gold_blind_train_rollouts":True,
        "gold_blind_validation_inference":True,
        "max_true_rendered_output_budget":True,
        "test_split_used":False,
        "no_network_gpu_generation_paid_api":True,
        "limitations":[
            "Exploratory: QASPER validation is NOT new independent hold-out.",
            "Both aligned label and state distribution differ from original Phase4C.",
            "CPU time for old archive arms is not re-measured; do not compare zero placeholders.",
            "Gold evidence lexical 5-gram is not answer accuracy or entailment.",
            "Train rollouts are gold-blind manual greedy, not true learned-policy on-policy DAgger.",
            "Training candidate fit uses singleton-additive estimated cost; final selection checks actual render.",
        ],
        "duration_seconds":round(time.perf_counter()-start,2),
        "results":pairwise(records),
    }
    (args.output_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with (args.output_dir/"paired_contexts.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (args.output_dir/"selected_chunks.jsonl").open("w",encoding="utf8") as f:
        for item in selections:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    print("PHASE4C3 RESULT "+json.dumps({
        "output":str(args.output_dir),"time":summary["duration_seconds"],
        "train":states_audit,"by_count":by_count,
        "results":summary["results"],
    },ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--train-papers",type=int,default=0)
    p.add_argument("--validation-papers",type=int,default=0)
    run(p.parse_args())


if __name__=="__main__":
    main()
