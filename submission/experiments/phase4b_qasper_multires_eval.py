"""Phase 4B: gold-blind multiresolution (150/300/650) candidate pool.

All experiments occur on native QASPER VALIDATION, previously used for
development. DO NOT read/test/tune with QASPER held-out official test split.

Two fixed multi-res ablations:
 - joint-pooled-render: deduplicate exact equivalent source spans and jointly
   score all resolution candidates, with Phase3C render-aware selector;
 - joint-source-overlap: identical scoring but apply fixed 80% source span
   near-containment prefilter before the SAME render-aware selector.

Single-scale manual render-aware controls are copied (not recomputed) from
Phase4A's gold-blind validation cases. The multi-res pool and its inference
scoring NEVER consume gold, labels, evidence groups, or oracle results.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
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
os.environ["TOKENPACK_HF_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]=""

from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,groups_with_ngrams,
)
from submission.experiments.phase3b_native_qasper import (
    native_blocks_with_section_hints,native_questions_from_qasper_row,
    load_local_qasper_rows,
)
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase4a_qasper_multires_oracle import (
    SCALES,RATIOS,VALIDATION_SHA,make_cached,
)
from submission.experiments.phase4b_multires_core import (
    construct_multiresolution,filter_heavy_overlap,count_selected_scale,
    OVERLAP_THRESHOLD,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import make_embedder
from tokenpack.scoring import score_chunks
from tokenpack.tokenization import TokenCounter

PHASE4A=ROOT/"submission/results/phase4a_multires_validation889_20261011/cases.jsonl"
METHODS=(
    "manual650",
    "manual300",
    "manual150",
    "joint-pooled-render",
    "joint-source-overlap",
)
CONTRASTS=(
    ("joint-pooled-render","manual650"),
    ("joint-source-overlap","manual650"),
    ("joint-pooled-render","manual300"),
    ("joint-source-overlap","manual300"),
    ("joint-source-overlap","joint-pooled-render"),
)


def baseline_index():
    output=defaultdict(dict)
    with PHASE4A.open(encoding="utf8") as file:
        for line in file:
            item=json.loads(line)
            key=(item["case_id"],item["budget_ratio"])
            output[key][item["scale"]]=item
    if len(output)!=889*2 or any(set(x)!=set(SCALES) for x in output.values()):
        raise AssertionError("Phase4A immutable validation controls incomplete")
    return output


def paired_by_paper(rows):
    stats=[]
    for ratio in RATIOS:
        items=defaultdict(dict)
        for r in rows:
            if r["ratio"]==ratio:
                if r["method"] in items[r["case_id"]]:
                    raise AssertionError("Duplicate method case")
                items[r["case_id"]][r["method"]]=r
        if any(set(x)!=set(METHODS) for x in items.values()):
            raise AssertionError("Incomplete paired multires ablation")
        arms={}
        for method in METHODS:
            arm=[x[method] for x in items.values()]
            arms[method]={
                "n":len(arm),
                "mean_gold_5gram_recall":round(statistics.mean(x["recall"] for x in arm),6),
                "mean_gold_token_recall":round(statistics.mean(x["token_recall"] for x in arm),6),
                "complete_gold_annotation":sum(x["complete"] for x in arm),
                "zero_5gram_cases":sum(x["recall"]<1e-9 for x in arm),
                "mean_actual_tokens":round(statistics.mean(x["used_tokens"] for x in arm),2),
                "mean_selected_chunks":round(statistics.mean(x["selected_chunks"] for x in arm),2),
                "budget_violations":sum(x["used_tokens"]>x["budget"] for x in arm),
            }
        paired={}
        for newer,older in CONTRASTS:
            diffs={
                cid:{
                    "control":{"paper_id":arms_by_method[older]["paper_id"],"delta":0.},
                    "new":{
                        "paper_id":arms_by_method[newer]["paper_id"],
                        "delta":arms_by_method[newer]["recall"]-
                                arms_by_method[older]["recall"],
                    },
                } for cid,arms_by_method in items.items()
            }
            d=[x["new"]["delta"] for x in diffs.values()]
            paired[f"{newer} minus {older}"]={
                "mean_difference":round(statistics.mean(d),6),
                "wins":sum(x>1e-10 for x in d),
                "losses":sum(x< -1e-10 for x in d),
                "ties":sum(abs(x)<=1e-10 for x in d),
                "paper_cluster_bootstrap":paper_cluster_ci(
                    diffs,"delta","new",replicates=5000),
            }
        stats.append({"ratio":ratio,"questions":len(items),
                      "arms":arms,"comparisons":paired})
    return stats


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path,default=
                        ROOT/".tokenpack/cache/phase4a_multires_20261011")
    parser.add_argument("--paper-limit",type=int,default=0)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite existing experiment output")
    source=DEFAULT_DATA/"validation-00000-of-00001.parquet"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=VALIDATION_SHA:
        raise ValueError("Official validation checksum mismatch")
    papers=load_local_qasper_rows(source)
    if len(papers)!=281:raise ValueError("Not official validation")
    if args.paper_limit:papers=papers[:args.paper_limit]
    base=baseline_index()
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("Real cl100k_base required")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    audit=Counter()
    rows=[]
    started=time.perf_counter()
    file=args.output_dir/"paired_cases.jsonl"
    audit_file=args.output_dir/"selected_chunks.jsonl"
    with file.open("x",encoding="utf8") as writer, audit_file.open("x",encoding="utf8") as selected_writer:
        for paperidx,paper in enumerate(papers,1):
            paper_id=str(paper["id"])
            questions,_=native_questions_from_qasper_row(paper)
            q=[(item,groups_with_ngrams(item.evidence_groups))
               for item in questions]
            q=[(item,groups) for item,groups in q if groups]
            if not q:continue
            blocks=native_blocks_with_section_hints(paper,counter)
            if not blocks:continue
            source_tokens=sum(counter.count(b.text) for b in blocks)
            cache=make_cached(args.cache_dir,DEFAULT_CACHE,paper_id)
            vectors=cache.get_or_embed([b.text for b in blocks],embedder)
            questions_vectors=cache.get_or_embed([x.question for x,_ in q],embedder)
            chunks_by_scale={}
            vecs_by_scale={}
            for scale,params in SCALES.items():
                chunks=StructureAwareChunker(
                    **params,token_counter=counter,block_embeddings=vectors,
                    semantic_threshold=.35).chunk(blocks)
                chunks_by_scale[scale]=chunks
                vecs_by_scale[scale]=cache.get_or_embed(
                    [x.text for x in chunks],embedder)
                audit[f"generated_{scale}_chunks"]+=len(chunks)
            merged,merged_vectors,scale_lookup=construct_multiresolution(
                chunks_by_scale,vecs_by_scale)
            merged=sorted(zip(merged,merged_vectors,strict=True),
                          key=lambda p:(p[0].order_key,
                                        p[0].metadata["research_resolution"],
                                        p[0].id))
            merged_chunks=[x for x,_ in merged]
            merged_vectors=[x for _,x in merged]
            audit["unique_multires_chunks"]+=len(merged_chunks)
            audit["raw_multires_chunks"]+=sum(len(x) for x in chunks_by_scale.values())
            for index,(question,groups) in enumerate(q):
                cid=f"{paper_id}:{question.question_id}"
                scored=score_chunks(
                    questions_vectors[index],merged_chunks,merged_vectors,
                    query_text=question.question,scoring="evidence-hybrid")
                filtered,filter_audit=filter_heavy_overlap(
                    scored,threshold=OVERLAP_THRESHOLD)
                audit["overlap_rejections"]+=filter_audit["duplicates_removed"]
                audit["questions_seen"]+=1
                selected=[]
                for ratio in RATIOS:
                    budget=max(1,math.floor(source_tokens*ratio))
                    pool_choice=choose(scored,query=question.question,budget=budget,
                                       counter=counter,policy="render-value")
                    overlap_choice=choose(filtered,query=question.question,budget=budget,
                                          counter=counter,policy="render-value")
                    selected.extend([
                        (ratio,"joint-pooled-render",pool_choice),
                        (ratio,"joint-source-overlap",overlap_choice),
                    ])
                # All inference is now complete; first gold annotation access:
                gold=compile_gold_groups(groups)
                metrics={}
                for ratio,method,choice in selected:
                    if choice["context"] not in metrics:
                        metrics[choice["context"]]=fast_group_coverage(
                            choice["context"],gold)
                    tok,five,complete=metrics[choice["context"]]
                    record={
                        "case_id":cid,"paper_id":paper_id,
                        "ratio":ratio,"method":method,
                        "recall":round(five,8),
                        "token_recall":round(tok,8),"complete":complete,
                        "budget":max(1,math.floor(source_tokens*ratio)),
                        "used_tokens":choice["used_tokens"],
                        "selected_chunks":len(choice["selected"]),
                        "resolution_selection":count_selected_scale(
                            choice["selected"]),
                        "candidate_count":len(scored) if method=="joint-pooled-render" else len(filtered),
                        "redundant_candidates_removed":filter_audit["duplicates_removed"],
                    }
                    if record["used_tokens"]>record["budget"]:
                        raise AssertionError("Final-render cap violated")
                    rows.append(record)
                    writer.write(json.dumps(record,ensure_ascii=False)+"\n")
                    selected_writer.write(json.dumps({
                        "case_id":cid,"ratio":ratio,"method":method,
                        "chosen":[x.chunk.id for x in choice["selected"]],
                    },ensure_ascii=False)+"\n")
                    audit["comparison_arm_records"]+=1
                # Source control is frozen Phase4A manual render scorer.
                for ratio in RATIOS:
                    previous=base[(cid,ratio)]
                    for scale in SCALES:
                        old=previous[scale]
                        control=old["render"]
                        rows.append({
                            "case_id":cid,"paper_id":paper_id,"ratio":ratio,
                            "method":f"manual{scale}",
                            "recall":control["recall"],
                            "token_recall":control["token_recall"],
                            "complete":control["complete"],
                            "budget":old["budget_tokens"],
                            "used_tokens":control["real_tokens"],
                            "selected_chunks":control["chunks"],
                            "resolution_selection":{str(scale):control["chunks"]},
                            "candidate_count":old["n_chunks"],
                            "redundant_candidates_removed":0,
                        })
                        audit["frozen_baseline_records"]+=1
            if paperidx%24==0 or paperidx==len(papers):
                writer.flush();selected_writer.flush()
                print(f"PHASE4B {paperidx}/{len(papers)} papers "
                      f"{audit['questions_seen']} questions, "
                      f"{audit['comparison_arm_records']} multires contexts "
                      f"elapsed={time.perf_counter()-started:.1f}s",flush=True)
    if not args.paper_limit:
        if len({x["case_id"] for x in rows})!=889:
            raise AssertionError("Wrong validation question count")
        if audit["frozen_baseline_records"]!=889*2*3:
            raise AssertionError("Missing preserved Phase4A baseline cases")
    results=paired_by_paper(rows)
    result={
        "experiment":"Phase4B multi-resolution merged gold-blind scoring and overlap ablation",
        "dataset":"Official QASPER validation previously used for research, NOT held-out test",
        "n_papers_total":len(papers),
        "n_evaluable_questions":len(set(x["case_id"] for x in rows)),
        "methods":METHODS,
        "resolutions":list(SCALES),
        "overlap_threshold_predeclared":OVERLAP_THRESHOLD,
        "merged_joint_scoring":"One evidence-hybrid normalization over source-ordered mixed chunks",
        "chunks_identical_source_text_deduplicated":True,
        "gold_blind_scoring_and_selecting":True,
        "test_split_opened":False,
        "budget_true_final_render":True,
        "notes":[
            "Comparisons are exploratory: validation already used in Phase4A, Phase3B and Phase3C.",
            "Joint-pooled scoring recomputes BM25/cosine normalization, so benefit cannot be uniquely ascribed to chunk-resolution mixing.",
            "Near-containment is based on annotated source char offsets, not semantic evidence similarity.",
            "Overlap prefilter may reject helpful context and has no learned marginal objective.",
            "All single-resolution baselines are prior frozen Phase4A cases; no new gold-aware oracle trained/used.",
            "These are lexical evidence coverage proxies, not generated-answer correctness.",
        ],
        "audit":dict(audit),
        "duration_seconds":round(time.perf_counter()-started,2),
        "results":results,
    }
    (args.output_dir/"summary.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "output":str(args.output_dir),
        "questions":result["n_evaluable_questions"],
        "audits":result["audit"],
        "results":results,
        "elapsed":result["duration_seconds"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
