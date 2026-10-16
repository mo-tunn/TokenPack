"""Phase 5B: pinned pretrained MS MARCO cross-encoder, ZERO-SHOT QASPER.

Paper-disjoint official QASPER *validation only*. No training, no labels
until after gold-blind selection, no test use or production code edits.
Offline HF cache snapshot pinned + weight SHA256 validated at startup.
The historical manual and LambdaMART controls use identical native chunks,
real final cl100k_base budget, same render-aware selection policy.
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
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"

import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,parse_input,prepare_split_batched,
)
from submission.experiments.phase3b_learned_scoring import (
    make_learned_candidates,
)
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter

REVISION="233902d25c440f23af6f7d6e94d2946bac0bee0a"
WEIGHTS_SHA256="821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
MODEL_ID="cross-encoder/ms-marco-MiniLM-L6-v2"
FOLDER=Path.home()/".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots"/REVISION
PRIOR=ROOT/"submission/results/phase3c_qasper_full889_optimized_20261011/paired_contexts.csv"
RATIOS=(.25,.5)
METHODS=("manual-render","lambdamart-render","msmarco-zero-shot")


def verify_local_model():
    file=FOLDER/"model.safetensors"
    if not file.is_file():raise FileNotFoundError(f"Missing pinned offline weights {file}")
    hh=hashlib.sha256()
    with file.open("rb") as f:
        for block in iter(lambda:f.read(262144),b""):hh.update(block)
    if hh.hexdigest()!=WEIGHTS_SHA256:
        raise AssertionError("Downloaded model SHA256 mismatch")
    return file.stat().st_size


def old_results():
    result={}
    with PRIOR.open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["method"] in ("manual-render-value","lightgbm-render-value"):
                key=(row["case_id"],float(row["budget_ratio"]),row["method"])
                result[key]=row
    return result


@torch.inference_mode()
def score(model,tokenizer,question,chunks,device,max_length,batch_size):
    model.eval()
    scores=[]
    for lo in range(0,len(chunks),batch_size):
        batch=chunks[lo:lo+batch_size]
        inp=tokenizer(
            [question]*len(batch),[x.chunk.text for x in batch],
            padding=True,truncation="only_second",
            max_length=max_length,return_tensors="pt")
        inp={k:v.to(device) for k,v in inp.items()}
        scores.extend(model(**inp).logits.squeeze(-1).float().cpu().tolist())
    if len(scores)!=len(chunks) or not all(math.isfinite(s) for s in scores):
        raise AssertionError("Incomplete/nonfinite cross-encoder scores")
    return scores


def paired(records,bootstrap=4000):
    stats=[]
    for ratio in RATIOS:
        cases=defaultdict(dict)
        for row in records:
            if row["ratio"]==ratio:
                cases[row["case_id"]][row["method"]]=row
        if any(set(x)!=set(METHODS) for x in cases.values()):
            raise AssertionError("Paired QASPER controls missing")
        arms={}
        for name in METHODS:
            rr=[x[name] for x in cases.values()]
            arms[name]={
                "n":len(rr),
                "mean_gold_5gram_recall":round(statistics.mean(
                    x["gold_5gram_recall"] for x in rr),7),
                "mean_actual_tokens":round(statistics.mean(
                    x["actual_tokens"] for x in rr),2),
                "zero_gold_cases":sum(x["gold_5gram_recall"]<1e-9 for x in rr),
                "budget_violations":sum(
                    x["actual_tokens"]>x["budget"] for x in rr),
            }
        diffs={}
        for new,old in (
            ("msmarco-zero-shot","lambdamart-render"),
            ("msmarco-zero-shot","manual-render"),
        ):
            sample={
                case:{
                    "control":{"paper_id":x[old]["paper_id"],"delta":0.0},
                    "new":{"paper_id":x[new]["paper_id"],
                           "delta":x[new]["gold_5gram_recall"]-
                                   x[old]["gold_5gram_recall"]},
                } for case,x in cases.items()
            }
            values=[v["new"]["delta"] for v in sample.values()]
            diffs[f"{new} minus {old}"]={
                "mean_difference":round(statistics.mean(values),7),
                "wins":sum(v>1e-9 for v in values),
                "losses":sum(v< -1e-9 for v in values),
                "ties":sum(abs(v)<=1e-9 for v in values),
                "paper_bootstrap":paper_cluster_ci(
                    sample,"delta","new",replicates=bootstrap),
            }
        stats.append({"ratio":ratio,"questions":len(cases),
                      "arms":arms,"contrasts":diffs})
    return stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--validation-papers",type=int,default=281)
    ap.add_argument("--batch-size",type=int,default=12)
    ap.add_argument("--max-length",type=int,default=512)
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--bootstrap",type=int,default=4000)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite an existing experiment")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    start=time.perf_counter()
    model_size=verify_local_model()
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to(args.device).eval()
    print("MS MARCO MODEL LOADED",MODEL_ID,
          "revision",REVISION,"params",sum(p.numel() for p in model.parameters()),
          "device",args.device,flush=True)
    if args.max_length>model.config.max_position_embeddings:
        raise ValueError("BERT maximum position exceeded")
    splits,counts=parse_input(DEFAULT_DATA)
    if counts!={"train":888,"validation":281}:
        raise AssertionError("Official paper counts mismatch")
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("Missing real token counter")
    embedder=make_embedder(
        "sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    questions,stats=prepare_split_batched(
        "validation",splits["validation"],embedder,counter,
        DEFAULT_CACHE,args.validation_papers)
    old=old_results()
    rows=[]
    selections=[]
    infer_seconds=0.0
    for n,q in enumerate(questions,1):
        tick=time.perf_counter()
        logits=score(model,tokenizer,q.text,q.scored,
                     args.device,args.max_length,args.batch_size)
        infer_seconds+=time.perf_counter()-tick
        learned=make_learned_candidates(q.scored,logits)
        plans=[]
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            choice=choose(learned,query=q.text,budget=budget,
                          counter=counter,policy="render-value")
            plans.append((ratio,budget,choice))
        gold=compile_gold_groups(q.evidence_groups)
        for ratio,budget,choice in plans:
            _,score5,_=fast_group_coverage(choice["context"],gold)
            rows.append({
                "case_id":q.case_id,"paper_id":q.paper_id,
                "ratio":ratio,"method":"msmarco-zero-shot",
                "budget":budget,"actual_tokens":choice["used_tokens"],
                "gold_5gram_recall":round(score5,8),
            })
            selections.append({
                "case_id":q.case_id,"ratio":ratio,
                "selected_chunk_ids":[x.chunk.id for x in choice["selected"]],
            })
            for method,archive_name in (
                ("manual-render","manual-render-value"),
                ("lambdamart-render","lightgbm-render-value"),
            ):
                archived=old[(q.case_id,ratio,archive_name)]
                if int(archived["budget_tokens"])!=budget:
                    raise AssertionError("Mismatched true output budget")
                rows.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"method":method,
                    "budget":budget,
                    "actual_tokens":int(archived["rendered_tokens"]),
                    "gold_5gram_recall":float(archived["gold_5gram_recall"]),
                })
        if n%100==0 or n==len(questions):
            print(f"ZERO-SHOT {n}/{len(questions)} validation questions "
                  f"GPU_inference_sec={infer_seconds:.1f} "
                  f"elapsed={time.perf_counter()-start:.1f}",flush=True)
    if args.validation_papers==281 and len(questions)!=889:
        raise AssertionError("Expected all 889 usable validation questions")
    if any(r["actual_tokens"]>r["budget"] for r in rows):
        raise AssertionError("Actual rendered token-budget violation")
    outcome={
        "experiment":"Phase5B pretrained MS MARCO MiniLM crossencoder, zero-shot",
        "pinned_model":MODEL_ID,
        "revision":REVISION,"model_sha256":WEIGHTS_SHA256,
        "weights_bytes":model_size,"parameters":sum(
            p.numel() for p in model.parameters()),
        "zero_shot_no_qasper_training":True,
        "validation_papers_requested":args.validation_papers,
        "validation_papers_evaluable":len({q.paper_id for q in questions}),
        "validation_questions_evaluable":len(questions),
        "device":args.device,"batch_size":args.batch_size,
        "max_wordpiece_length":args.max_length,
        "pretrained_ranking_supervision":"MS MARCO",
        "gold_blind_inference":True,
        "test_split_used":False,
        "same_native_650_chunk_and_render_aware_selector":True,
        "no_generation_or_paid_api":True,
        "limits":[
            "Previously seen QASPER validation, not a new independent holdout.",
            "WordPiece max_length=512 truncates some 650-token chunks.",
            "MS MARCO pretrained ranking does not guarantee domain transfer to QASPER.",
            "Gold lexical evidence 5gram is not answer correctness.",
            "Equal token output ceilings, not necessarily equal token consumption.",
            "Production code and historical outputs are untouched.",
        ],
        "gpu_inference_seconds":round(infer_seconds,2),
        "wall_seconds":round(time.perf_counter()-start,2),
        "results":paired(rows,bootstrap=args.bootstrap),
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"paired_contexts.csv").open(
            "w",encoding="utf8",newline="") as f:
        cw=csv.DictWriter(f,fieldnames=list(rows[0]))
        cw.writeheader();cw.writerows(rows)
    with (args.output_dir/"selected_chunks.jsonl").open(
            "w",encoding="utf8") as f:
        for row in selections:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FINAL_PHASE5B "+json.dumps(
        {"output":str(args.output_dir),"questions":len(questions),
         "results":outcome["results"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
