"""Offline QASPER Phase 3B: paper-grouped out-of-fold learning-to-rank pilot.

Safety/validity contract:
* All embeddings are LOCAL and Hugging Face is strictly offline.
* No generative model, network, Modal, or paid API.
* No gold in feature extraction, inference, or deployment selectors.
* Gold relevance labels ONLY on the training side of a GroupKFold paper split.
* Source is the previously used 200-question archived QASPER development set,
  not a new QASPER train/validation or independent test split.
* Strict equal final rendered-token limits and an unchanged chunk pool.
* Historical artifacts and production selectors are not changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TOKENPACK_HF_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from submission.experiments.phase2_qasper_diagnostic_v2 import (
    CACHE, blocks_from_document, prepare_cases, render_and_fit
)
from submission.experiments.evidence_failure_audit import evidence_coverage
from submission.experiments.phase3a_oracle_core import (
    gold_counts, gram_counts, matched_count,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.export import render_context
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter

FEATURES=(
    "hybrid_score", "raw_embedding_similarity",
    "cosine_normalized", "bm25_normalized", "query_coverage",
    "structural_prior", "neighbor_coherence",
    "log_chunk_tokens", "chunk_fraction_of_document",
    "relative_chunk_position", "chunk_count",
    "query_length_log", "filtered_query_term_coverage",
)
METHODS=("manual-budget-top-k","lightgbm-budget-top-k")
OUTPUT=ROOT/"submission/results/phase3b_qasper_oof200_20261011"
TERM_RE=re.compile(r"(?u)\b[^\W\d_]\w{2,}\b")
STOP=frozenset({
    "the","and","for","from","with","what","which","where","when","how",
    "this","that","have","has","were","was","are","can","does","about",
    "into","their","our","its","these","those","such","also","they",
})


def _terms(text: str) -> set[str]:
    return {t.casefold() for t in TERM_RE.findall(text)
            if t.casefold() not in STOP}


def features_for_question(scored, question: str) -> np.ndarray:
    """Compute gold-blind numeric features using scored candidates only."""
    qterms=_terms(question)
    total=sum(max(1,x.weight) for x in scored)
    n=max(1,len(scored))
    result=[]
    for position,item in enumerate(scored):
        c=item.score_components
        overlap=len(qterms&_terms(item.chunk.text))/max(1,len(qterms))
        result.append([
            item.value,item.raw_similarity,
            *[float(c.get(k,0.0)) for k in
              ("cosine","bm25","query_coverage","structural_prior","neighbor_coherence")],
            math.log1p(item.weight),
            item.weight/max(1,total),
            position/max(1,n-1),
            float(n),math.log1p(len(question.split())),
            overlap,
        ])
    return np.asarray(result,dtype=np.float64).reshape(-1,len(FEATURES))


def relevance_label(covered_fraction: float) -> int:
    """Four-level per-question relevance, derived from chunk-internal gold 5grams."""
    if covered_fraction<=0:return 0
    if covered_fraction<.10:return 1
    if covered_fraction<.5:return 2
    return 3


def relevance_labels(scored,evidence: list[str]) -> np.ndarray:
    gold,_=gold_counts(evidence)
    denom=sum(gold.values())
    if denom==0:
        raise ValueError("No evaluable 5gram gold evidence")
    return np.asarray([
        relevance_label(matched_count(gram_counts(x.chunk.text),gold)/denom)
        for x in scored],dtype=np.int32)


def positive_rescale(raw_scores) -> list[float]:
    """Rank-preserving within-query rescale to keep every candidate eligible."""
    if len(raw_scores)==0:return []
    vals=[float(x) for x in raw_scores]
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("Non-finite model prediction")
    low,high=min(vals),max(vals)
    if high==low:return [0.5]*len(vals)
    return [.01+.99*(x-low)/(high-low) for x in vals]


def make_learned_candidates(scored,raw_scores):
    values=positive_rescale(raw_scores)
    if len(scored)!=len(values):raise ValueError("Unmatched candidate scores")
    return [replace(x,value=v,score_components={**x.score_components,
                    "phase3b_learned_rank_score":v}) for x,v in zip(scored,values)]


def ndcg_at(labels,values,k:int=5)->float:
    if not len(labels):return 0.
    def dcg(rels):
        return sum((2**int(x)-1)/math.log2(i+2) for i,x in enumerate(rels))
    order=sorted(range(len(labels)),key=lambda i:-float(values[i]))[:k]
    ideal=sorted([int(x) for x in labels],reverse=True)[:k]
    denominator=dcg(ideal)
    return dcg([labels[i] for i in order])/denominator if denominator else 0.


@dataclass
class Question:
    case_id: str
    paper_id: str
    source_tokens: int
    archived_half_budget: int
    evidence: list[str]
    scored: list
    X: np.ndarray
    y: np.ndarray


def build_questions(max_cases:int)->list[Question]:
    cases=prepare_cases(max_cases)
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("cl100k_base required")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    paper_pool={}
    questions=[]
    for n,case in enumerate(cases,1):
        full=case["full-document"]
        paper,source=full["paper_id"],full["context"]
        digest=hashlib.sha256(source.encode()).hexdigest()
        if paper not in paper_pool:
            blocks=blocks_from_document(paper,source,counter)
            cache=EmbeddingCache(CACHE/(paper.replace("/","_")+".embeddings.json"))
            bv=cache.get_or_embed([b.text for b in blocks],embedder)
            chunks=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=counter,block_embeddings=bv,semantic_threshold=.35
            ).chunk(blocks)
            cv=cache.get_or_embed([c.text for c in chunks],embedder)
            paper_pool[paper]=(chunks,cv,digest)
        chunks,cv,stored=paper_pool[paper]
        if stored!=digest:raise ValueError("Inconsistent paper source "+paper)
        question=str(full["question"])
        scored=score_chunks(embedder.embed([question])[0],chunks,cv,
                            scoring="evidence-hybrid",query_text=question)
        evidence=full.get("evidence_texts") or []
        X=features_for_question(scored,question)
        # Label extraction follows gold-blind embedding/scoring/features.
        y=relevance_labels(scored,evidence)
        questions.append(Question(
            case_id=full["case_id"],paper_id=paper,
            source_tokens=int(full["source_tokens"]),
            archived_half_budget=int(full["budget_tokens"]),
            evidence=evidence,scored=scored,X=X,y=y))
        if n%50==0:print(f"Prepared {n}/{len(cases)} questions",flush=True)
    return questions


def data_for_ranker(questions,indices):
    ordered=sorted(indices)
    X=np.vstack([questions[i].X for i in ordered])
    y=np.concatenate([questions[i].y for i in ordered])
    group=[len(questions[i].scored) for i in ordered]
    assert sum(group)==len(y)==len(X)
    return X,y,group


def check_group_isolation(questions,train_indices,test_indices):
    train={questions[i].paper_id for i in train_indices}
    test={questions[i].paper_id for i in test_indices}
    if not train or not test or train&test:
        raise ValueError("Empty or overlapping paper folds")
    train_case={questions[i].case_id for i in train_indices}
    test_case={questions[i].case_id for i in test_indices}
    if train_case&test_case:
        raise ValueError("Repeated question identifiers across folds")
    return len(train),len(test)


def write_csv(path,rows):
    with path.open("w",encoding="utf8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_cv(questions,out:Path,folds:int=5):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Never overwrite old experiment artifacts")
    if len(questions)<folds:raise ValueError("Not enough cases")
    papers={q.paper_id for q in questions}
    if len(papers)<folds:raise ValueError("Not enough papers")
    # Only predeclared fixed hyperparameters. No tuning on OOF held-out folds.
    params={
        "objective":"lambdarank","metric":"ndcg",
        "n_estimators":110,"learning_rate":.05,
        "num_leaves":15,"max_depth":5,"min_child_samples":8,
        "reg_lambda":2.0,"random_state":20261011,
        "n_jobs":2,"verbosity":-1,
        "deterministic":True,"force_col_wise":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    folds_info=[]
    predictions={}
    feature_importances=[]
    group_labels=[q.paper_id for q in questions]
    splitter=GroupKFold(n_splits=folds)
    for fold,(train_index,test_index) in enumerate(
        splitter.split(np.arange(len(questions)),groups=group_labels),1
    ):
        ptrain,ptest=check_group_isolation(questions,train_index,test_index)
        X_train,y_train,groups=data_for_ranker(questions,train_index)
        assert X_train.shape[1]==len(FEATURES)
        model=lgb.LGBMRanker(**params)
        start=time.perf_counter()
        model.fit(X_train,y_train,group=groups,feature_name=list(FEATURES))
        training_seconds=time.perf_counter()-start
        model.booster_.save_model(str(out/f"paper_fold{fold}.model.txt"))
        importance=dict(zip(FEATURES,[int(v) for v in model.feature_importances_]))
        feature_importances.append(importance)
        for idx in test_index:
            q=questions[int(idx)]
            # Booster prediction accepts the same strictly ordered feature
            # matrix without sklearn's feature-name validation warnings.
            raw=model.booster_.predict(q.X,num_threads=2)
            predictions[q.case_id]=make_learned_candidates(q.scored,raw)
            records.append({
                "case_id":q.case_id,"paper_id":q.paper_id,
                "fold":fold,
                "n_chunks":len(q.scored),
                "positive_chunks":int(sum(q.y>0)),
                "manual_ndcg_at5":round(ndcg_at(q.y,[x.value for x in q.scored]),6),
                "learned_ndcg_at5":round(ndcg_at(q.y,raw),6),
                "manual_top1_relevant":int(q.y[
                    int(np.argmax([x.value for x in q.scored]))]>0),
                "learned_top1_relevant":int(q.y[int(np.argmax(raw))]>0),
                "positive_gold_label_fraction":round(float(np.mean(q.y>0)),6),
            })
        folds_info.append({
            "fold":fold,"train_questions":len(train_index),
            "test_questions":len(test_index),
            "train_papers":ptrain,"test_papers":ptest,
            "training_seconds":round(training_seconds,3),
            "train_relevance_distribution":dict(
                sorted((int(k),int(v)) for k,v in Counter(y_train.tolist()).items())
            ),
            "feature_importance_split":importance,
        })
        print(f"Fold {fold}/{folds}: train papers={ptrain}, "
              f"held-out papers={ptest}, train={training_seconds:.2f}s",flush=True)
    if len(predictions)!=len(questions):
        raise AssertionError("Incomplete out-of-fold predictions")
    assert len(records)==len(questions)
    counter=TokenCounter()
    render=lambda items:render_context([x.chunk for x in items])
    cases=[]
    for q in questions:
        for ratio in (.25,.50):
            budget=q.archived_half_budget if ratio==.5 else math.floor(q.source_tokens*ratio)
            for method,candidates in (
                ("manual-budget-top-k",q.scored),
                ("lightgbm-budget-top-k",predictions[q.case_id]),
            ):
                chosen=select_chunks(candidates,strategy="budget-top-k",
                                     budget=budget,candidate_pool=300)
                text,kept,trimmed,used,_=render_and_fit(
                    chosen.selected,render,budget,counter)
                token,five=evidence_coverage(text,q.evidence)
                span_scores=[evidence_coverage(text,[e])[1] for e in q.evidence if e.strip()]
                span_scores=[x for x in span_scores if x is not None]
                cases.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "budget_ratio":ratio,"budget_tokens":budget,
                    "method":method,"rendered_tokens":used,
                    "gold_5gram_recall":five,
                    "gold_token_recall":token,
                    "all_gold_spans_present":bool(span_scores) and all(x>=.5 for x in span_scores),
                    "n_chunks":len(kept),"header_trimmed":trimmed,
                })
    comparisons={}
    for ratio in (.25,.50):
        rows=[x for x in cases if x["budget_ratio"]==ratio]
        by_case=defaultdict(dict)
        for row in rows:by_case[row["case_id"]][row["method"]]=row
        if any(set(data)!=set(METHODS) for data in by_case.values()):
            raise AssertionError("Unmatched evaluation")
        overview=[]
        for method in METHODS:
            arm=[row for row in rows if row["method"]==method]
            overview.append({
                "method":method,"n":len(arm),
                "mean_gold_5gram_recall":round(sum(r["gold_5gram_recall"] for r in arm)/len(arm),6),
                "mean_gold_token_recall":round(sum(r["gold_token_recall"] for r in arm)/len(arm),6),
                "all_gold_spans_cases":sum(r["all_gold_spans_present"] for r in arm),
                "mean_real_render_tokens":round(sum(r["rendered_tokens"] for r in arm)/len(arm),3),
                "budget_breaches":sum(r["rendered_tokens"]>r["budget_tokens"] for r in arm),
            })
        deltas={
            cid:{
                "control":{"paper_id":r["manual-budget-top-k"]["paper_id"],"delta":0.},
                "learned":{"paper_id":r["lightgbm-budget-top-k"]["paper_id"],
                           "delta":r["lightgbm-budget-top-k"]["gold_5gram_recall"]-
                                   r["manual-budget-top-k"]["gold_5gram_recall"]}
            } for cid,r in by_case.items()
        }
        gains=[x["learned"]["delta"] for x in deltas.values()]
        comparisons[str(ratio)]={
            "methods":overview,
            "paired_gold_5gram":{
                "mean_difference":round(sum(gains)/len(gains),6),
                "wins":sum(x>1e-10 for x in gains),
                "losses":sum(x< -1e-10 for x in gains),
                "ties":sum(abs(x)<=1e-10 for x in gains),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    deltas,"delta","learned",replicates=6000,
                ),
            },
        }
    by_case_ranks={r["case_id"]:r for r in records}
    result={
        "experiment":"Faz 3B paper-grouped 5-fold OOF learned scoring (EXPLORATORY)",
        "dataset":"reused archived QASPER 200 questions, NOT original train/validation and NOT independent holdout",
        "questions":len(questions),"papers":len(papers),"folds":folds,
        "no_generating_llm":True,"network_calls":0,
        "training":"LightGBM LambdaMART CPU; pretrained local MiniLM embeddings used for features",
        "same_chunk_pool":True,"same_selector":"budget-top-k",
        "same_render_token_budget":True,
        "gold_in_feature_or_prediction":False,
        "gold_only_in_training_labels_and_posthoc_eval":True,
        "relevance_label":"chunk-internal gold 5gram overlap: 0, (0,.10)->1, [.10,.5)->2, [.5,1]->3",
        "hyperparameters":params,
        "features":list(FEATURES),
        "fold_info":folds_info,
        "mean_ndcg_at5":{
            "manual":round(sum(r["manual_ndcg_at5"] for r in records)/len(records),6),
            "lightgbm":round(sum(r["learned_ndcg_at5"] for r in records)/len(records),6),
        },
        "top1_relevant_counts":{
            "manual":sum(x["manual_top1_relevant"] for x in records),
            "lightgbm":sum(x["learned_top1_relevant"] for x in records),
        },
        "budget_comparisons":comparisons,
        "limitations":[
            "Previous development archive; OOF by paper does not constitute a sealed independent benchmark.",
            "Gold 5gram lexical labels can favor lexical matching and do not ensure answer entailment.",
            "Alternative QASPER annotator evidence sets are not retained in this archive.",
            "The old adapter reconstructs paragraphs without original QASPER section provenance.",
            "No LLM answer-generation accuracy measured.",
        ],
    }
    write_csv(out/"ranking_oof.csv",records)
    write_csv(out/"paired_contexts.csv",cases)
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=int,choices=[50,200],default=200)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing output overwrite")
    start=time.perf_counter()
    questions=build_questions(args.cases)
    result=evaluate_cv(questions,args.output_dir)
    print("RESULT "+json.dumps({
        "output":str(args.output_dir),
        "papers":result["papers"],
        "questions":result["questions"],
        "ndcg":result["mean_ndcg_at5"],
        "top1":result["top1_relevant_counts"],
        "budgets":result["budget_comparisons"],
        "duration_seconds":round(time.perf_counter()-start,2),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
