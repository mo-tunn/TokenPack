"""Phase 4C2: original vs metric-aligned marginal target, all else fixed.

Offline CPU native QASPER train only for fit; validation-only ablation.
Previous Phase4C model/result is FROZEN baseline. Never reads QASPER test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

import lightgbm as lgb
import numpy as np
from submission.experiments.phase3a_oracle_core import gram_counts
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,parse_input,prepare_split_batched,
)
from submission.experiments.phase3c_qasper_validation import (
    compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase4c_marginal_core import (
    feature_rows,training_states,marginal_gain_targets,NAMES,RATIOS,
)
from submission.experiments.phase4c2_aligned_core import (
    aligned_marginal_gain_targets,
)
from submission.experiments.phase4c_train_marginal import (
    PARAMS,dynamic_choose,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter

OLD=ROOT/"submission/results/phase4c_marginal_full_train_validation889_20261011"
ARMS=("old-marginal-conditional","aligned-marginal-conditional",
      "old-marginal-static","aligned-marginal-static")
PAIRS=(
    ("aligned-marginal-conditional","old-marginal-conditional"),
    ("aligned-marginal-static","old-marginal-static"),
    ("aligned-marginal-conditional","aligned-marginal-static"),
)


def load_old_records():
    result={}
    with (OLD/"paired_contexts.csv").open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            key=(r["case_id"],float(r["ratio"]),r["method"])
            result[key]=r
    return result


def training_matrix(train):
    """Fixed manual prefixes, original features, only corrected target label."""
    mats=[]
    labels=[]
    audit={"questions":0,"states":0,"candidate_labels":0,
           "original_vs_aligned_changed":0,"original_positive":0,
           "aligned_positive":0}
    for q in train:
        indices=tuple(range(len(q.scored)))
        grams=[gram_counts(item.chunk.text) for item in q.scored]
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            for selected in training_states(q):
                rest=[i for i in indices if i not in selected]
                if not rest:continue
                feats=feature_rows(q,rest,selected,budget,ratio,
                                   precomputed=grams)
                old_labels=marginal_gain_targets(q,rest,selected,grams)
                new_labels=aligned_marginal_gain_targets(q,rest,selected,grams)
                mats.append(feats)
                labels.append(new_labels)
                audit["states"]+=1
                audit["candidate_labels"]+=len(new_labels)
                audit["original_vs_aligned_changed"]+=int(
                    np.sum(np.abs(new_labels-old_labels)>1e-9))
                audit["original_positive"]+=int(np.sum(old_labels>1e-9))
                audit["aligned_positive"]+=int(np.sum(new_labels>1e-9))
        audit["questions"]+=1
    return np.vstack(mats),np.concatenate(labels),audit


def pairwise(records):
    outcomes=[]
    for ratio in RATIOS:
        grouped=defaultdict(dict)
        for r in records:
            if r["ratio"]==ratio:
                grouped[r["case_id"]][r["method"]]=r
        if any(set(x)!=set(ARMS) for x in grouped.values()):
            raise AssertionError("Incomplete label-alignment ablation")
        arms={}
        for key in ARMS:
            data=[x[key] for x in grouped.values()]
            arms[key]={
                "mean_gold_5gram_recall":round(
                    statistics.mean(x["recall"] for x in data),6),
                "mean_actual_tokens":round(
                    statistics.mean(x["actual_tokens"] for x in data),2),
                "zero_recall_cases":sum(x["recall"]<=1e-9 for x in data),
                "budget_violations":sum(
                    x["actual_tokens"]>x["budget"] for x in data),
            }
        contrasts={}
        for newer,older in PAIRS:
            values={
                cid:{
                    "control":{"paper_id":d[older]["paper_id"],"delta":0.},
                    "new":{"paper_id":d[newer]["paper_id"],
                           "delta":d[newer]["recall"]-d[older]["recall"]},
                } for cid,d in grouped.items()
            }
            differences=[x["new"]["delta"] for x in values.values()]
            contrasts[f"{newer} minus {older}"]={
                "mean_difference":round(statistics.mean(differences),6),
                "wins":sum(x>1e-10 for x in differences),
                "losses":sum(x< -1e-10 for x in differences),
                "ties":sum(abs(x)<=1e-10 for x in differences),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    values,"delta","new",replicates=4000),
            }
        outcomes.append({"ratio":ratio,"n":len(grouped),
                         "arms":arms,"comparisons":contrasts})
    return outcomes


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--train-papers",type=int,default=0)
    p.add_argument("--validation-papers",type=int,default=0)
    p.add_argument("--audit-only",action="store_true",
                   help="Count train-label alignment differences without refitting a model")
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite previous experiment")
    splits,ids=parse_input(DEFAULT_DATA)
    if ids!={"train":888,"validation":281}:
        raise ValueError("Unexpected native paper-disjoint splits")
    started=time.perf_counter()
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("Need real cl100k_base")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    train,_=prepare_split_batched(
        "train",splits["train"],embedder,counter,DEFAULT_CACHE,
        args.train_papers or None)
    X,y,audit=training_matrix(train)
    print("ALIGNED TRAIN",X.shape,"label diffs",audit,flush=True)
    if args.audit_only:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        (args.output_dir/"LABEL_ALIGNMENT_AUDIT.json").write_text(
            json.dumps({
                "questions":audit["questions"],
                "train_papers_requested":args.train_papers or 888,
                "label_alignment":audit,
                "label_changed_fraction":audit["original_vs_aligned_changed"]/
                    max(1,audit["candidate_labels"]),
                "test_split_used":False,"model_trained":False,
            },ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        print("LABEL-ONLY AUDIT FINISHED",flush=True)
        return
    model=lgb.LGBMRegressor(**PARAMS)
    model.fit(X,y,sample_weight=np.where(y>1e-9,2.0,1.0))
    model_booster=model.booster_
    X=None;y=None;train=[]
    validation,_=prepare_split_batched(
        "validation",splits["validation"],embedder,counter,DEFAULT_CACHE,
        args.validation_papers or None)
    old=load_old_records()
    records=[]
    chosen=[]
    for j,q in enumerate(validation,1):
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            newchoices=[]
            for policy,conditional in (
                ("aligned-marginal-static",False),
                ("aligned-marginal-conditional",True),
            ):
                selection=dynamic_choose(
                    q,model_booster,counter,budget,ratio,
                    conditional=conditional)
                newchoices.append((policy,selection))
                chosen.append({
                    "case_id":q.case_id,"ratio":ratio,"method":policy,
                    "selected_chunks":[x.chunk.id for x in selection["selected"]],
                })
            # Gold only AFTER both new contexts have been selected.
            requirements=compile_gold_groups(q.evidence_groups)
            for name,selection in newchoices:
                _,five,_=fast_group_coverage(selection["context"],requirements)
                records.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"method":name,"budget":budget,
                    "actual_tokens":selection["used_tokens"],
                    "recall":round(five,8),
                })
            for name,oldname in (
                ("old-marginal-static","marginal-static"),
                ("old-marginal-conditional","marginal-conditional"),
            ):
                row=old[(q.case_id,ratio,oldname)]
                records.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"method":name,"budget":budget,
                    "actual_tokens":int(row["used_tokens"]),
                    "recall":float(row["gold_5gram_recall"]),
                })
        if j%100==0 or j==len(validation):
            print(f"ALIGNED VAL {j}/{len(validation)}",flush=True)
    result={
        "experiment":"Faz4C2 aligned marginal train label A/B on QASPER validation",
        "paper_disjoint_train_validation":True,
        "num_training_questions":audit["questions"],
        "num_validation_questions":len(validation),
        "train_audit":audit,
        "original_target":"max_g[coverage_g(S+c)-coverage_g(S)]",
        "aligned_target":"max_g coverage_g(S+c)-max_g coverage_g(S)",
        "model_params_same_as_first_phase4c":True,
        "state_features_and_selector_fixed":True,
        "test_split_used":False,
        "generative_llm_gpu_network_paid_api":False,
        "notes":[
            "Exploratory on previously inspected validation; not independent test.",
            "Gold targets use distinct alternative annotations, never union.",
            "Original Phase4C model/evidence is read-only.",
            "Equal nominal strict final rendered-token budget, realized tokens may differ.",
        ],
        "elapsed_seconds":round(time.perf_counter()-started,2),
        "results":pairwise(records),
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    model_booster.save_model(
        str(args.output_dir/"aligned_marginal_regressor.model.txt"))
    with (args.output_dir/"paired_contexts.csv").open(
            "w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (args.output_dir/"selected_chunks.jsonl").open(
            "w",encoding="utf8") as f:
        for selection in chosen:
            f.write(json.dumps(selection,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RESULT "+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
