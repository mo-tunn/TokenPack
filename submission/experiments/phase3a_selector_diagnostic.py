"""Phase 3A companion ablation: fixed scoring, change ONLY selector.

Gold never enters scoring or selection. This discriminates an existing
greedy-packing deficiency from learned relevance issues, without claiming
to prove causal root causes or running generative LLM inference.
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
from submission.experiments.phase2_qasper_diagnostic_v2 import (
    CACHE,blocks_from_document,prepare_cases,render_and_fit
)
from submission.experiments.phase3a_oracle_core import gold_counts,pool_coverage
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache,make_embedder
from tokenpack.export import render_context
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter
METHODS=("budget-top-k","knapsack","greedy-density")

def run(out,number,ratios):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Existing run is immutable")
    groups=prepare_cases(number)
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("Tokenizer unavailable")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    cache_by_paper={}
    records=[]
    render=lambda selected:render_context([x.chunk for x in selected])
    t0=time.perf_counter()
    for n,g in enumerate(groups,1):
        f=g["full-document"]
        paper,source=f["paper_id"],f["context"]
        if paper not in cache_by_paper:
            blocks=blocks_from_document(paper,source,counter)
            cache=EmbeddingCache(CACHE/(paper.replace("/","_")+".embeddings.json"))
            bv=cache.get_or_embed([b.text for b in blocks],embedder)
            chunks=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=counter,block_embeddings=bv,semantic_threshold=.35
            ).chunk(blocks)
            cv=cache.get_or_embed([c.text for c in chunks],embedder)
            cache_by_paper[paper]=(chunks,cv,hashlib.sha256(source.encode()).hexdigest())
        chunks,cv,digest=cache_by_paper[paper]
        assert hashlib.sha256(source.encode()).hexdigest()==digest
        scored=score_chunks(embedder.embed([f["question"]])[0],chunks,cv,
                            scoring="evidence-hybrid",query_text=f["question"])
        for ratio in ratios:
            budget=(int(f["budget_tokens"]) if ratio==.5
                    else math.floor(int(f["source_tokens"])*ratio))
            choices={}
            for method in METHODS:
                t1=time.perf_counter()
                selected=select_chunks(scored,strategy=method,budget=budget,
                                       candidate_pool=300)
                text,kept,trimmed,used,_=render_and_fit(
                    selected.selected,render,budget,counter)
                choices[method]=(kept,used,trimmed,time.perf_counter()-t1)
            gold,spans=gold_counts(f.get("evidence_texts") or [])
            for method,(kept,used,trimmed,elapsed) in choices.items():
                cov=pool_coverage(kept,gold,spans)
                records.append({
                    "case_id":f["case_id"],"paper_id":paper,"ratio":ratio,
                    "method":method,"budget_tokens":budget,"rendered_tokens":used,
                    "gold_fivegram_recall":cov["gold_5gram_recall"],
                    "all_gold_spans":cov["all_spans_present"],
                    "kept_chunks":len(kept),"header_trimmed":trimmed,
                    "elapsed_s":round(elapsed,6),
                })
        if n%50==0:print(f"{n}/{number} scored; {time.perf_counter()-t0:.1f}s",flush=True)
    per_ratio=[]
    for ratio in ratios:
        subset=[r for r in records if r["ratio"]==ratio]
        by_case=defaultdict(dict)
        for row in subset:by_case[row["case_id"]][row["method"]]=row
        if any(set(case)!=set(METHODS) for case in by_case.values()):
            raise ValueError("Unmatched")
        arms=[]
        for method in METHODS:
            x=[r for r in subset if r["method"]==method]
            arms.append({
                "method":method,"n":len(x),
                "avg_gold_fivegram_recall":round(sum(r["gold_fivegram_recall"] for r in x)/len(x),6),
                "all_gold_spans_cases":sum(r["all_gold_spans"] for r in x),
                "avg_real_tokens":round(sum(r["rendered_tokens"] for r in x)/len(x),3),
                "avg_selection_render_seconds":round(sum(r["elapsed_s"] for r in x)/len(x),6),
                "budget_violations":sum(r["rendered_tokens"]>r["budget_tokens"] for r in x),
            })
        comparison={}
        for m in METHODS[1:]:
            by_case_pair={
                cid:{
                    "control":{"paper_id":x["budget-top-k"]["paper_id"],"delta":0.},
                    "new":{"paper_id":x[m]["paper_id"],
                        "delta":x[m]["gold_fivegram_recall"]-x["budget-top-k"]["gold_fivegram_recall"]}
                } for cid,x in by_case.items()
            }
            comparison[m]={
                "paper_cluster_ci":paper_cluster_ci(by_case_pair,"delta","new",replicates=3000),
                "wins":sum(x[m]["gold_fivegram_recall"]>x["budget-top-k"]["gold_fivegram_recall"] for x in by_case.values()),
                "losses":sum(x[m]["gold_fivegram_recall"]<x["budget-top-k"]["gold_fivegram_recall"] for x in by_case.values()),
                "ties":sum(x[m]["gold_fivegram_recall"]==x["budget-top-k"]["gold_fivegram_recall"] for x in by_case.values()),
            }
        per_ratio.append({"ratio":ratio,"methods":arms,"comparisons":comparison})
    result={"scope":"Offline fixed-score selector ablation; gold used only after selection",
            "cases":number,"papers":len(cache_by_paper),"model_calls":0,
            "shared_same_embeddings":True,"shared_same_scoring":True,
            "strict_render_token_budget":True,"results":per_ratio,
            "elapsed_seconds":round(time.perf_counter()-t0,2)}
    out.mkdir(parents=True,exist_ok=True)
    with (out/"cases.csv").open("w",encoding="utf8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=int,choices=[50,200],default=200)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()
    run(args.output_dir,args.cases,[.25,.50])
