"""Phase5B controlled 300-token zero-shot MS MARCO cross-encoder ablation.

Only native QASPER VALIDATION. Frozen 650-token MS MARCO results are read
ONLY as controls. Frozen Phase4A 300-manual validation controls are reused.
Same true source-relative cl100k_base output cap and Phase3C render selector.
All gold is read strictly after score/selection of each question's contexts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict,Counter
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
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,MODEL_ID,WEIGHTS_SHA256,verify_local_model,
    score,paired,
)
from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,groups_with_ngrams,
)
from submission.experiments.phase3b_native_qasper import (
    native_questions_from_qasper_row,native_blocks_with_section_hints,
    load_local_qasper_rows,
)
from submission.experiments.phase4a_qasper_multires_oracle import (
    SCALES,VALIDATION_SHA,make_cached,
)
from submission.experiments.phase3b_learned_scoring import make_learned_candidates
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import make_embedder
from tokenpack.scoring import score_chunks
from tokenpack.tokenization import TokenCounter

REFERENCE_300=ROOT/"submission/results/phase4a_multires_validation889_20261011/cases.jsonl"
REFERENCE_650=ROOT/"submission/results/phase5b_msmarco_zeroshot_validation889_20261011/paired_contexts.csv"
RATIOS=(.25,.5)
METHODS=("manual-300","msmarco-650","msmarco-300","lambdamart-650")


def references():
    previous={}
    for line in REFERENCE_300.read_text(encoding="utf8").splitlines():
        if not line:continue
        row=json.loads(line)
        if row["scale"]==300:
            previous[row["case_id"],row["budget_ratio"],"manual-300"]=({
                "budget":row["budget_tokens"],"used":row["render"]["real_tokens"],
                "recall":row["render"]["recall"]})
    with REFERENCE_650.open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["method"] in ("msmarco-zero-shot","lambdamart-render"):
                name="msmarco-650" if r["method"]=="msmarco-zero-shot" else "lambdamart-650"
                previous[r["case_id"],float(r["ratio"]),name]=({
                    "budget":int(r["budget"]),
                    "used":int(r["actual_tokens"]),
                    "recall":float(r["gold_5gram_recall"]),
                })
    return previous


def comparisons(records,bootstrap=5000):
    out=[]
    for ratio in RATIOS:
        cases=defaultdict(dict)
        for r in records:
            if r["ratio"]==ratio:
                cases[r["case_id"]][r["method"]]=r
        if any(set(x)!=set(METHODS) for x in cases.values()):
            raise AssertionError("Unpaired multiresolution comparison")
        arms={}
        for method in METHODS:
            values=[x[method] for x in cases.values()]
            arms[method]={
                "mean_gold_5gram_recall":round(statistics.mean(
                    r["recall"] for r in values),7),
                "mean_actual_tokens":round(statistics.mean(
                    r["actual_tokens"] for r in values),2),
                "zero_gold_questions":sum(r["recall"]<=1e-9 for r in values),
                "budget_violations":sum(
                    r["actual_tokens"]>r["budget"] for r in values),
            }
        diffs={}
        for new,old in (
            ("msmarco-300","msmarco-650"),
            ("msmarco-300","lambdamart-650"),
            ("msmarco-300","manual-300"),
        ):
            vals={
                cid:{
                    "control":{"paper_id":x[old]["paper_id"],"delta":0.},
                    "new":{"paper_id":x[new]["paper_id"],
                           "delta":x[new]["recall"]-x[old]["recall"]},
                } for cid,x in cases.items()
            }
            ds=[v["new"]["delta"] for v in vals.values()]
            diffs[f"{new} minus {old}"]={
                "delta":round(statistics.mean(ds),7),
                "wins":sum(v>1e-9 for v in ds),
                "losses":sum(v< -1e-9 for v in ds),
                "ties":sum(abs(v)<=1e-9 for v in ds),
                "paper_bootstrap":paper_cluster_ci(vals,"delta","new",
                                                   replicates=bootstrap),
            }
        out.append({"ratio":ratio,"n":len(cases),"arms":arms,"contrasts":diffs})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--validation-papers",type=int,default=281)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--checkpoint-model-dir",type=Path,default=None,
                    help="Optional local QASPER-train-only model; default pinned MS MARCO zero-shot")
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite an experiment")
    file=DEFAULT_DATA/"validation-00000-of-00001.parquet"
    if hashlib.sha256(file.read_bytes()).hexdigest()!=VALIDATION_SHA:
        raise AssertionError("Native QASPER validation dataset changed")
    docs=load_local_qasper_rows(file)
    if len(docs)!=281:raise AssertionError("Wrong validation paper count")
    docs=docs[:args.validation_papers]
    verify_local_model()
    model_source=args.checkpoint_model_dir or FOLDER
    if not (model_source/"model.safetensors").is_file():
        raise FileNotFoundError("Local checkpoint lacks model.safetensors")
    model=AutoModelForSequenceClassification.from_pretrained(
        model_source,local_files_only=True).to("cuda").eval()
    tokenizer=AutoTokenizer.from_pretrained(model_source,local_files_only=True)
    counter=TokenCounter()
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    history=references()
    stats=Counter()
    records=[]
    selected=[]
    start=time.perf_counter()
    for paper_index,paper in enumerate(docs,1):
        pid=str(paper["id"])
        questions,_=native_questions_from_qasper_row(paper)
        pairs=[(q,groups_with_ngrams(q.evidence_groups)) for q in questions]
        pairs=[(q,groups) for q,groups in pairs if groups]
        if not pairs:continue
        blocks=native_blocks_with_section_hints(paper,counter)
        source_tokens=sum(counter.count(b.text) for b in blocks)
        cache=make_cached(
            ROOT/".tokenpack/cache/phase4a_multires_20261011",
            DEFAULT_CACHE,pid)
        block_vec=cache.get_or_embed([b.text for b in blocks],embedder)
        chunks=StructureAwareChunker(
            **SCALES[300],token_counter=counter,
            block_embeddings=block_vec,semantic_threshold=.35).chunk(blocks)
        chunk_vec=cache.get_or_embed([c.text for c in chunks],embedder)
        q_vec=cache.get_or_embed([q.question for q,_ in pairs],embedder)
        stats["papers"]+=1
        stats["chunks"]+=len(chunks)
        for (q,groups),v in zip(pairs,q_vec,strict=True):
            cid=f"{pid}:{q.question_id}"
            scored=score_chunks(v,chunks,chunk_vec,
                                scoring="evidence-hybrid",
                                query_text=q.question)
            raw=score(model,tokenizer,q.question,scored,"cuda",512,
                      args.batch_size)
            learned=make_learned_candidates(scored,raw)
            plans=[]
            for ratio in RATIOS:
                cap=max(1,math.floor(source_tokens*ratio))
                selected_choice=choose(learned,query=q.question,budget=cap,
                                       counter=counter,policy="render-value")
                plans.append((ratio,cap,selected_choice))
            # GOLD NOT ACCESSED UNTIL INFERENCE HAS SELECTED BOTH CONTEXTS
            annotated=compile_gold_groups(groups)
            for ratio,cap,choice in plans:
                _,five,_=fast_group_coverage(choice["context"],annotated)
                records.append({
                    "case_id":cid,"paper_id":pid,"ratio":ratio,
                    "method":"msmarco-300","budget":cap,
                    "actual_tokens":choice["used_tokens"],
                    "recall":round(five,8),
                })
                selected.append({
                    "case_id":cid,"ratio":ratio,"method":"msmarco-300",
                    "selected_ids":[x.chunk.id for x in choice["selected"]],
                })
                for method in ("msmarco-650","lambdamart-650","manual-300"):
                    prior=history[(cid,ratio,method)]
                    if prior["budget"]!=cap:raise AssertionError("Budget changed")
                    records.append({
                        "case_id":cid,"paper_id":pid,"ratio":ratio,
                        "method":method,"budget":cap,
                        "actual_tokens":prior["used"],
                        "recall":prior["recall"],
                    })
            stats["questions"]+=1
        if paper_index%40==0 or paper_index==len(docs):
            print("PHASE5B-300",paper_index,len(docs),
                  "questions",stats["questions"],
                  "elapsed",round(time.perf_counter()-start,1),flush=True)
    if args.validation_papers==281 and stats["questions"]!=889:
        raise AssertionError("Expected 889 usable native validation questions")
    if any(r["actual_tokens"]>r["budget"] for r in records):
        raise AssertionError("Rendered token budget exceeded")
    summary={
        "experiment":("Phase5C QASPER-train-adapted MS MARCO native300 validation"
                      if args.checkpoint_model_dir else
                      "Phase5B native QASPER validation pretrained MS MARCO at 300-token chunks"),
        "msmarco_model_id":MODEL_ID,"revision":REVISION,
        "model_sha256":WEIGHTS_SHA256,
        "fine_tuned_on_qasper_train":bool(args.checkpoint_model_dir),
        "local_checkpoint_directory":str(model_source),
        "baseline_650_pinned_zero_shot":True,
        "same_real_final_rendered_token_cap":True,
        "native_source_300_chunk_params":SCALES[300],
        "gold_blind_inference":True,
        "test_split_used":False,
        "exploratory_not_independent_holdout":True,
        "limits":[
            "Same pretrained MS MARCO model, different candidate chunk resolution.",
            "Previously used QASPER validation: do not claim new held-out generalization.",
            "Lexical gold 5gram, not answer correctness.",
            "The source-based maximum token cap matches; realized token counts differ.",
            "The full 300-token reranker scores all native candidates, no gold access.",
        ],
        "audit":dict(stats),"seconds":round(time.perf_counter()-start,2),
        "results":comparisons(records),
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with (args.output_dir/"paired_contexts.csv").open(
            "w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
    with (args.output_dir/"selected_chunks.jsonl").open(
            "w",encoding="utf8") as f:
        for row in selected:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("RESULT_PHASE5B300",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
