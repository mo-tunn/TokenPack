"""Train on the *native* QASPER train split, evaluate on native validation.

Public dataset download is performed separately and never from this file.
Purely local embedding + CPU LightGBM; no generative model/network/Modal.
Test split is NOT read. Results in new output folder; production untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass

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

from submission.experiments.phase3b_native_qasper import (
    load_local_qasper_rows,native_questions_from_qasper_row,
    native_blocks_with_section_hints,validate_native_splits,
)
from submission.experiments.phase3b_learned_scoring import (
    FEATURES,features_for_question,make_learned_candidates,
    relevance_label,ndcg_at,
)
from submission.experiments.phase3a_oracle_core import (
    gold_counts,gram_counts,matched_count,
)
from submission.experiments.evidence_failure_audit import evidence_coverage
from submission.experiments.phase2_qasper_diagnostic_v2 import render_and_fit
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.export import render_context
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter

DEFAULT_DATA=ROOT/"submission/data/qasper_hf_pinned_96fd84d_train_validation"
DEFAULT_CACHE=ROOT/".tokenpack/cache/phase3b_native_qasper_full_20261011"
EXPECTED_HASHES={
    "train":"68a20057dcccc1ddd42e26910ff87bef276c912808c0426a7d884a0873963e09",
    "validation":"6a1105448de2c14528f4e5781198cbc82d8f3b29fe9c5c34b452f7739731ada3",
}
PARAMS={
    "objective":"lambdarank","metric":"ndcg","n_estimators":110,
    "learning_rate":.05,"num_leaves":15,"max_depth":5,
    "min_child_samples":8,"reg_lambda":2.0,
    "random_state":20261011,"n_jobs":2,"verbosity":-1,
    "deterministic":True,"force_col_wise":True,
}


@dataclass
class Question:
    paper_id: str
    case_id: str
    text: str
    source_tokens: int
    evidence_groups: list[list[str]]
    scored: list
    feature_matrix: np.ndarray
    labels: np.ndarray


def groups_with_ngrams(evidence_groups:list[list[str]]):
    """Never merge evidence annotations from distinct annotators."""
    return [group for group in evidence_groups if sum(gold_counts(group)[0].values())>0]


def alt_relevance_labels(scored,evidence_groups) -> np.ndarray:
    """Per candidate, best single-annotation relevance (never a merged union)."""
    valid=groups_with_ngrams(evidence_groups)
    if not valid:raise ValueError("No evaluable gold 5grams")
    gold_options=[gold_counts(group)[0] for group in valid]
    out=[]
    for item in scored:
        grams=gram_counts(item.chunk.text)
        best=max(matched_count(grams,gold)/max(1,sum(gold.values()))
                 for gold in gold_options)
        out.append(relevance_label(best))
    return np.asarray(out,dtype=np.int32)


def best_annotator_coverage(context:str,groups:list[list[str]]) -> tuple[float,float,bool]:
    """Max completed *one* annotation; no impossible all-annotator union."""
    options=[]
    for group in groups_with_ngrams(groups):
        tok,gram=evidence_coverage(context,group)
        if tok is None or gram is None:continue
        options.append((float(tok),float(gram)))
    if not options:return 0.,0.,False
    by_gram=max(options,key=lambda p:(p[1],p[0]))
    complete=False
    for group in groups_with_ngrams(groups):
        spanrec=[evidence_coverage(context,[span])[1] for span in group]
        if spanrec and all(r is not None and r>=.5 for r in spanrec):
            complete=True
            break
    return by_gram[0],by_gram[1],complete


def parse_input(data_dir:Path):
    rows={}
    for split in ("train","validation"):
        file=data_dir/f"{split}-00000-of-00001.parquet"
        if not file.exists():
            raise FileNotFoundError("Data missing (run offline-safe downloader first): "+str(file))
        digest=hashlib.sha256(file.read_bytes()).hexdigest()
        if digest!=EXPECTED_HASHES[split]:
            raise ValueError("Pinned QASPER parquet SHA256 mismatch: "+split)
        rows[split]=load_local_qasper_rows(file)
    isolation=validate_native_splits(rows["train"],rows["validation"])
    return rows,isolation


def prepare_split(split:str,rows:list,embedder,counter,cache_dir:Path,
                  max_papers:int|None=None) -> tuple[list[Question],dict]:
    examples=[]
    stats=Counter()
    t0=time.perf_counter()
    if max_papers is not None:rows=rows[:max_papers]
    for paper_index,row in enumerate(rows):
        paper_id=str(row["id"])
        stats["papers_seen"]+=1
        native_q,exclusions=native_questions_from_qasper_row(row)
        stats["questions_total"]+=len(row["qas"]["question"])
        stats["questions_without_usable_gold"]+=sum(exclusions.values())
        qs=[(q,groups_with_ngrams(q.evidence_groups))
            for q in native_q]
        qs=[(q,groups) for q,groups in qs if groups]
        stats["questions_not_5gram_evaluable"]+=len(native_q)-len(qs)
        if not qs:continue
        blocks=native_blocks_with_section_hints(row,counter)
        if not blocks:
            stats["questions_without_blocks"]+=len(qs)
            continue
        token_count=sum(counter.count(b.text) for b in blocks)
        cache=EmbeddingCache(cache_dir/f"{paper_id}.embeddings.json")
        bv=cache.get_or_embed([b.text for b in blocks],embedder)
        chunks=StructureAwareChunker(
            target_tokens=650,min_tokens=120,max_tokens=900,
            token_counter=counter,block_embeddings=bv,
            semantic_threshold=.35,
        ).chunk(blocks)
        if not chunks:continue
        cv=cache.get_or_embed([x.text for x in chunks],embedder)
        qv=cache.get_or_embed([q.question for q,g in qs],embedder)
        stats["chunks_total"]+=len(chunks)
        stats["native_blocks"]+=len(blocks)
        stats["chunks_with_section_hint"]+=sum(
            bool(c.metadata.get("section_hint")) for c in chunks)
        for (q,groups),embedding in zip(qs,qv,strict=True):
            scored=score_chunks(embedding,chunks,cv,
                                scoring="evidence-hybrid",query_text=q.question)
            labels=alt_relevance_labels(scored,groups)
            stats["questions_evaluated"]+=1
            stats["questions_with_multi_annotator_groups"]+=int(len(groups)>1)
            stats["questions_with_no_positive_chunk"]+=int(not any(labels))
            stats["positive_chunk_labels"]+=int(np.sum(labels>0))
            examples.append(Question(
                paper_id=paper_id,
                case_id=f"{paper_id}:{q.question_id}",
                text=q.question,source_tokens=token_count,
                evidence_groups=groups,
                scored=scored,feature_matrix=features_for_question(scored,q.question),
                labels=labels,
            ))
        if (paper_index+1)%25==0 or paper_index+1==len(rows):
            print(f"{split.upper()} {paper_index+1}/{len(rows)} papers; "
                  f"questions={len(examples)}; elapsed={time.perf_counter()-t0:.1f}s",
                  flush=True)
    if len({x.case_id for x in examples})!=len(examples):
        raise ValueError("Duplicate native question id")
    stats["elapsed_seconds"]=round(time.perf_counter()-t0,2)
    return examples,dict(stats)


def batch_vectors(requests: list[tuple[EmbeddingCache,list[str]]],embedder,
                  encode_batch:int=128) -> list[list[list[float]]]:
    """Batch offline MiniLM encode over many papers, preserving per-paper caches.

    Batching changes no feature, paper split or model. Cache checks use the
    same SHA256 keys as EmbeddingCache so interrupted runs can resume.
    """
    missing={}
    requested=[]
    dirty=[]
    for cache,texts in requests:
        keys=[cache._key(t,embedder.model_name) for t in texts]
        needs=False
        for k,t in zip(keys,texts):
            if k not in cache._records:
                missing[k]=t
                needs=True
        requested.append((cache,keys))
        dirty.append(needs)
    if missing:
        keys=list(missing)
        result={}
        for offset in range(0,len(keys),encode_batch):
            sub=keys[offset:offset+encode_batch]
            vecs=embedder.embed([missing[k] for k in sub])
            if len(vecs)!=len(sub):raise RuntimeError("Incomplete local embedding batch")
            result.update(zip(sub,vecs,strict=True))
        for (cache,keys),was_dirty in zip(requested,dirty,strict=True):
            if was_dirty:
                cache._records.update({k:result[k] for k in keys if k in result})
                cache.save()
    return [[cache._records[k] for k in keys] for cache,keys in requested]


def prepare_split_batched(split:str,rows:list,embedder,counter,cache_dir:Path,
                          max_papers:int|None=None,paper_batch:int=12):
    """Equivalent to prepare_split, with streaming paper batches for CPU speed."""
    if max_papers is not None:rows=rows[:max_papers]
    examples=[]
    stats=Counter()
    start=time.perf_counter()
    for begin in range(0,len(rows),paper_batch):
        specs=[]
        for row in rows[begin:begin+paper_batch]:
            paper_id=str(row["id"])
            stats["papers_seen"]+=1
            native_q,exclusions=native_questions_from_qasper_row(row)
            stats["questions_total"]+=len(row["qas"]["question"])
            stats["questions_without_usable_gold"]+=sum(exclusions.values())
            qs=[(q,groups_with_ngrams(q.evidence_groups)) for q in native_q]
            qs=[(q,groups) for q,groups in qs if groups]
            stats["questions_not_5gram_evaluable"]+=len(native_q)-len(qs)
            if not qs:continue
            blocks=native_blocks_with_section_hints(row,counter)
            if not blocks:
                stats["questions_without_blocks"]+=len(qs)
                continue
            stats["native_blocks"]+=len(blocks)
            specs.append({
                "paper_id":paper_id,"blocks":blocks,"questions":qs,
                "source_tokens":sum(counter.count(b.text) for b in blocks),
                "cache":EmbeddingCache(cache_dir/f"{paper_id}.embeddings.json"),
            })
        if not specs:continue
        bv=batch_vectors([(s["cache"],[b.text for b in s["blocks"]])
                          for s in specs],embedder)
        for s,blocks_vecs in zip(specs,bv,strict=True):
            s["chunks"]=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=counter,block_embeddings=blocks_vecs,
                semantic_threshold=.35,
            ).chunk(s["blocks"])
            stats["chunks_total"]+=len(s["chunks"])
            stats["chunks_with_section_hint"]+=sum(
                bool(c.metadata.get("section_hint")) for c in s["chunks"])
        cv=batch_vectors([(s["cache"],[c.text for c in s["chunks"]])
                          for s in specs],embedder)
        qv=batch_vectors([(s["cache"],[q.question for q,_ in s["questions"]])
                          for s in specs],embedder)
        for s,vchunks,vquestions in zip(specs,cv,qv,strict=True):
            for (q,groups),embedding in zip(s["questions"],vquestions,strict=True):
                scored=score_chunks(embedding,s["chunks"],vchunks,
                                    scoring="evidence-hybrid",query_text=q.question)
                labels=alt_relevance_labels(scored,groups)
                stats["questions_evaluated"]+=1
                stats["questions_with_multi_annotator_groups"]+=int(len(groups)>1)
                stats["questions_with_no_positive_chunk"]+=int(not any(labels))
                stats["positive_chunk_labels"]+=int(np.sum(labels>0))
                examples.append(Question(
                    paper_id=s["paper_id"],
                    case_id=f"{s['paper_id']}:{q.question_id}",
                    text=q.question,source_tokens=s["source_tokens"],
                    evidence_groups=groups,scored=scored,
                    feature_matrix=features_for_question(scored,q.question),
                    labels=labels,
                ))
        done=min(begin+paper_batch,len(rows))
        if done%24==0 or done==len(rows):
            print(f"{split.upper()} {done}/{len(rows)} papers; "
                  f"questions={len(examples)}; elapsed={time.perf_counter()-start:.1f}s",
                  flush=True)
    if len({x.case_id for x in examples})!=len(examples):
        raise ValueError("Duplicate native question id")
    stats["elapsed_seconds"]=round(time.perf_counter()-start,2)
    return examples,dict(stats)


def train_and_compare(train:list[Question],validation:list[Question],out:Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Never overwrite experiment results")
    papers_train={q.paper_id for q in train}
    papers_val={q.paper_id for q in validation}
    if papers_train & papers_val:
        raise ValueError("Paper leakage train vs validation")
    if not train or not validation:
        raise ValueError("No usable labeled train/validation")
    out.mkdir(parents=True,exist_ok=False)
    X=np.vstack([q.feature_matrix for q in train])
    y=np.concatenate([q.labels for q in train])
    groups=[len(q.labels) for q in train]
    assert len(y)==len(X)==sum(groups)
    model=lgb.LGBMRanker(**PARAMS)
    start=time.perf_counter()
    model.fit(X,y,group=groups,feature_name=list(FEATURES))
    train_time=time.perf_counter()-start
    model.booster_.save_model(str(out/"full_train_lambdamart.model.txt"))
    counter=TokenCounter()
    render=lambda selected:render_context([i.chunk for i in selected])
    rows=[]
    rankings=[]
    for i,q in enumerate(validation,1):
        pred=model.booster_.predict(q.feature_matrix,num_threads=2)
        learned=make_learned_candidates(q.scored,pred)
        rankings.append({
            "case_id":q.case_id,"paper_id":q.paper_id,
            "n_chunks":len(q.scored),
            "annotation_alternatives":len(q.evidence_groups),
            "manual_ndcg_at5":round(ndcg_at(q.labels,[x.value for x in q.scored]),6),
            "learned_ndcg_at5":round(ndcg_at(q.labels,pred),6),
            "manual_top1_relevant":int(q.labels[int(np.argmax(
                [x.value for x in q.scored]))]>0),
            "learned_top1_relevant":int(q.labels[int(np.argmax(pred))]>0),
        })
        for ratio in (.25,.5):
            budget=max(1,math.floor(q.source_tokens*ratio))
            for name,scored in (
                ("manual-budget-top-k",q.scored),
                ("lightgbm-budget-top-k",learned),
            ):
                selected=select_chunks(
                    scored,strategy="budget-top-k",budget=budget,
                    candidate_pool=300)
                text,kept,trimmed,used,_=render_and_fit(
                    selected.selected,render,budget,counter)
                tok,five,full=best_annotator_coverage(text,q.evidence_groups)
                if used>budget:raise AssertionError("Exceeded actual rendered budget")
                rows.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "budget_ratio":ratio,"method":name,
                    "budget_tokens":budget,"rendered_tokens":used,
                    "gold_5gram_recall":round(five,8),
                    "gold_token_recall":round(tok,8),
                    "any_annotator_group_complete":full,
                    "selected_chunks":len(kept),
                    "trimmed":trimmed,
                })
        if i%250==0 or i==len(validation):
            print(f"VALIDATION {i}/{len(validation)} questions packed",flush=True)
    summary={
        "scope":"QASPER official train -> official validation, test split UNTOUCHED",
        "learned_model":"LightGBM LambdaMART CPU, fixed hyperparameters from exploratory pilot",
        "training_labeled_questions":len(train),
        "validation_labeled_questions":len(validation),
        "train_papers":len(papers_train),"validation_papers":len(papers_val),
        "train_candidate_rows":len(y),
        "train_positive_chunks":int(sum(y>0)),
        "train_seconds":round(train_time,3),
        "features":list(FEATURES),"parameters":PARAMS,
        "mean_ndcg_at5":{
            name:round(sum(r[name+"_ndcg_at5"] for r in rankings)/len(rankings),6)
            for name in ("manual","learned")
        },
        "top1_relevant":{
            name:sum(r[name+"_top1_relevant"] for r in rankings)
            for name in ("manual","learned")
        },
        "annotation_policy":"Each annotation is a separate valid evidence group; train labels maximize per-chunk coverage of one alternative; validation recall maximizes separately evaluated group scores; never flatten union.",
        "train_validation_paper_overlap":0,
        "budget_comparisons":{},
    }
    for ratio in (.25,.5):
        cases=defaultdict(dict)
        for row in rows:
            if row["budget_ratio"]==ratio:
                cases[row["case_id"]][row["method"]]=row
        if any(len(row)!=2 for row in cases.values()):
            raise AssertionError("Unpaired validation arms")
        arms={}
        for name in ("manual-budget-top-k","lightgbm-budget-top-k"):
            one=[c[name] for c in cases.values()]
            arms[name]={
                "n":len(one),
                "mean_gold_5gram_recall":round(sum(r["gold_5gram_recall"] for r in one)/len(one),6),
                "mean_gold_token_recall":round(sum(r["gold_token_recall"] for r in one)/len(one),6),
                "any_annotator_group_complete_count":sum(r["any_annotator_group_complete"] for r in one),
                "average_real_render_tokens":round(sum(r["rendered_tokens"] for r in one)/len(one),2),
                "budget_violations":sum(r["rendered_tokens"]>r["budget_tokens"] for r in one),
            }
        d={
            case:{
                "control":{"paper_id":pair["manual-budget-top-k"]["paper_id"],"delta":0},
                "learned":{"paper_id":pair["lightgbm-budget-top-k"]["paper_id"],
                    "delta":pair["lightgbm-budget-top-k"]["gold_5gram_recall"]-
                            pair["manual-budget-top-k"]["gold_5gram_recall"]}
            } for case,pair in cases.items()
        }
        delta=[x["learned"]["delta"] for x in d.values()]
        summary["budget_comparisons"][str(ratio)]={
            "methods":arms,
            "paired_gold_5gram":{
                "mean_delta":round(sum(delta)/len(delta),6),
                "wins":sum(x>1e-10 for x in delta),
                "losses":sum(x< -1e-10 for x in delta),
                "ties":sum(abs(x)<=1e-10 for x in delta),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    d,"delta","learned",replicates=6000),
            },
        }
    def save_csv(name,records):
        with (out/name).open("w",newline="",encoding="utf8") as file:
            w=csv.DictWriter(file,fieldnames=list(records[0]))
            w.writeheader();w.writerows(records)
    save_csv("validation_paired_contexts.csv",rows)
    save_csv("validation_ranking.csv",rankings)
    summary["feature_importance_split"]=dict(zip(
        FEATURES,[int(x) for x in model.feature_importances_]))
    (out/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "train":len(train),"validation":len(validation),"train_papers":len(papers_train),
        "validation_papers":len(papers_val),
        "train_candidate_rows":len(y),
        "ndcg":summary["mean_ndcg_at5"],
        "budget":summary["budget_comparisons"],
        "output":str(out),
    },ensure_ascii=False),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,default=DEFAULT_DATA)
    ap.add_argument("--cache-dir",type=Path,default=DEFAULT_CACHE)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--train-papers",type=int,default=0,
                    help="Only for smoke; 0 means full 888 papers")
    ap.add_argument("--validation-papers",type=int,default=0,
                    help="Only for smoke; 0 means full 281 papers")
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output exists")
    rows,isolation=parse_input(args.data_dir)
    print(f"NATIVE QASPER: {isolation}, SHA256 OK, test not accessed",flush=True)
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("cl100k_base not available")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    train,trainstats=prepare_split_batched(
        "train",rows["train"],embedder,counter,args.cache_dir,
        args.train_papers or None)
    val,valstats=prepare_split_batched(
        "validation",rows["validation"],embedder,counter,args.cache_dir,
        args.validation_papers or None)
    result=train_and_compare(train,val,args.output_dir)
    (args.output_dir/"native_data_audit.json").write_text(
        json.dumps({"train":trainstats,"validation":valstats,
                    "original_paper_isolation":isolation},indent=2),
        encoding="utf8")
    print("AUDIT "+json.dumps({"train":trainstats,"validation":valstats}),flush=True)

if __name__=="__main__":
    main()
