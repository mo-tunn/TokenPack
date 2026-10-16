"""Phase 4A: multi-resolution gold-aware attainable-frontier diagnostic.

DEVELOPMENT DATA ONLY: original native QASPER VALIDATION, not held-out test.
Three structural chunk resolutions (150/300/650) with fixed source-relative
25% and 50% true final-render token budgets. No model tuning or training.

Report separately:
 - gold-blind manual scorer + existing budget-top-k
 - gold-blind manual scorer + frozen Phase3C render-aware selector
 - gold-aware oracle feasible objective (attainable LOWER bound if heuristic)
 - gold-aware candidate-pool 5gram representation upper bound, ignoring budget

The oracle learns no deployable ranker. Exactness applies ONLY to the declared
chunk-internal 5-gram objective, never semantic entailment or answer accuracy.
Alternative annotator evidence lists remain separate OR groups.
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
import shutil
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
    DEFAULT_DATA,DEFAULT_CACHE,EXPECTED_HASHES,groups_with_ngrams,
)
from submission.experiments.phase3b_native_qasper import (
    native_blocks_with_section_hints,native_questions_from_qasper_row,
    load_local_qasper_rows,
)
from submission.experiments.phase3a_oracle_core import gram_counts,matched_count
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase4a_oracle_core import (
    solve_group,as_grams,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache,make_embedder
from tokenpack.export import render_context
from tokenpack.scoring import score_chunks
from tokenpack.tokenization import TokenCounter

SCALES={
    150:{"target_tokens":150,"min_tokens":40,"max_tokens":220},
    300:{"target_tokens":300,"min_tokens":70,"max_tokens":420},
    650:{"target_tokens":650,"min_tokens":120,"max_tokens":900},
}
RATIOS=(.25,.50)
EXACT_RELEVANT_LIMIT=9
VALIDATION_SHA=EXPECTED_HASHES["validation"]
REFERENCE=ROOT/"submission/results/phase3c_qasper_full889_optimized_20261011/paired_contexts.csv"


def read_reference():
    controls={}
    with REFERENCE.open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["scorer"]=="manual" and r["selector"] in ("budget-top-k","render-value"):
                key=(r["case_id"],float(r["budget_ratio"]),r["selector"])
                controls[key]=r
    return controls


def choice_quality(context,gold_groups):
    return fast_group_coverage(context,compile_gold_groups(gold_groups))


def oracle_for_alternatives(scored,groups,budget,counter):
    """Representational upper bound + feasible, gold-privileged oracle.

    For each *single* alternate annotation, solve a separate chunk-internal
    lexical 5gram problem; never union annotator alternatives.
    The final selected context is assessed with the SAME final-render
    5gram metric as gold-blind selections.
    """
    ordered_groups=groups_with_ngrams(groups)
    if not ordered_groups:raise ValueError("Gold groups have no usable 5grams")
    per_chunk=[gram_counts(x.chunk.text) for x in scored]
    render=lambda selected:render_context([x.chunk for x in selected])
    results=[]
    for group in ordered_groups:
        gold=as_grams(group)
        solved=solve_group(
            scored,gold,budget,counter,render,
            max_exact=EXACT_RELEVANT_LIMIT,precomputed=per_chunk,
        )
        chosen=[scored[i] for i in solved.chosen_indices]
        context=render(sorted(chosen,key=lambda x:(x.chunk.order_key,x.chunk.id)))
        used=counter.count(context)
        if used>budget:raise AssertionError("Privileged oracle budget breach")
        # This metric is the official best-of-alternatives final render proxy,
        # not the internal 5gram objective optimized by the search.
        tok,five,complete=choice_quality(context,ordered_groups)
        results.append((five,-used,solved,tok,complete))
    # Select the best rendered recall achieved across separate annotation
    # objective runs; never conflate with mathematically exact rendered oracle.
    best=max(results,key=lambda x:(x[0],x[1]))
    feasible,best_used,solution,tok,complete=best
    # Candidate-pool ceiling is for CHUNK-INTERNAL gold 5gram objective only.
    # This is an upper bound on oracle *internal*, not necessarily on
    # final-rendered recall which can create boundary/header 5grams.
    pool_upper=max(r[2].pool_upper_recall for r in results)
    all_exact=all(r[2].certified_optimal for r in results)
    return {
        "oracle_rendered_5gram":round(feasible,8),
        "oracle_rendered_token":round(tok,8),
        "oracle_any_group_complete":complete,
        "oracle_used_tokens":-best_used,
        "oracle_selected_chunks":len(solution.chosen_indices),
        "pool_internal_upper":round(pool_upper,8),
        "chosen_group_internal_oracle":round(solution.recall,8),
        "best_annotation_objective_mode":solution.mode,
        "all_annotation_objectives_certified":all_exact,
        "num_alternatives":len(results),
        "max_gold_relevant_chunks":max(r[2].relevant_chunks for r in results),
        "oracle_render_budget_checks":sum(r[2].checks for r in results),
    }


def make_cached(cache_dir,old_cache_dir,paper_id):
    dst=cache_dir/f"{paper_id}.embeddings.json"
    if not dst.exists():
        old=old_cache_dir/f"{paper_id}.embeddings.json"
        if old.is_file():
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old,dst)
    return EmbeddingCache(dst)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--cache-dir",type=Path,
                    default=ROOT/".tokenpack/cache/phase4a_multires_20261011")
    ap.add_argument("--paper-limit",type=int,default=0,
                    help="Development smoke only, no test split used")
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite historic experiment results")
    validation_path=DEFAULT_DATA/"validation-00000-of-00001.parquet"
    if hashlib.sha256(validation_path.read_bytes()).hexdigest()!=VALIDATION_SHA:
        raise ValueError("QASPER validation SHA256 mismatch")
    docs=load_local_qasper_rows(validation_path)
    if len(docs)!=281:raise ValueError("Unexpected official validation paper count")
    if args.paper_limit:docs=docs[:args.paper_limit]
    refer=read_reference()
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("cl100k_base required")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    outfile=args.output_dir/"cases.jsonl"
    records=[]
    audit=Counter()
    started=time.perf_counter()
    for paperidx,paper in enumerate(docs,1):
        paper_id=str(paper["id"])
        native,excluded=native_questions_from_qasper_row(paper)
        audit["original_questions"]+=len(paper["qas"]["question"])
        audit["without_evidence"]+=sum(excluded.values())
        pairs=[(q,groups_with_ngrams(q.evidence_groups)) for q in native]
        pairs=[(q,groups) for q,groups in pairs if groups]
        audit["too_short_evidence"]+=len(native)-len(pairs)
        if not pairs:continue
        blocks=native_blocks_with_section_hints(paper,counter)
        if not blocks:continue
        source_tokens=sum(counter.count(b.text) for b in blocks)
        cache=make_cached(args.cache_dir,DEFAULT_CACHE,paper_id)
        block_vectors=cache.get_or_embed([b.text for b in blocks],embedder)
        q_vectors=cache.get_or_embed([q.question for q,groups in pairs],embedder)
        chunks_by_scale={}
        scores_by_scale={}
        for target,params in SCALES.items():
            chunks=StructureAwareChunker(
                **params,token_counter=counter,
                block_embeddings=block_vectors,
                semantic_threshold=.35,
            ).chunk(blocks)
            if not chunks:raise ValueError("No chunks after structure splitting")
            vectors=cache.get_or_embed([c.text for c in chunks],embedder)
            chunks_by_scale[target]=chunks
            scores_by_scale[target]=[
                score_chunks(qvec,chunks,vectors,scoring="evidence-hybrid",
                             query_text=q.question)
                for (q,groups),qvec in zip(pairs,q_vectors,strict=True)
            ]
            audit[f"chunks_{target}"]+=len(chunks)
            audit[f"chunks_over_max_{target}"]+=sum(
                x.token_count>params["max_tokens"] for x in chunks)
        cache=None  # Free embedding records per paper.
        with outfile.open("a",encoding="utf8") as f:
            for qi,(q,groups) in enumerate(pairs):
                case_id=f"{paper_id}:{q.question_id}"
                for target,scored_by_question in scores_by_scale.items():
                    scored=scored_by_question[qi]
                    for ratio in RATIOS:
                        cap=max(1,math.floor(source_tokens*ratio))
                        top=choose(scored,query=q.question,budget=cap,
                                   counter=counter,policy="budget-top-k")
                        render_aware=choose(scored,query=q.question,budget=cap,
                                            counter=counter,policy="render-value")
                        # The two DEPLOYABLE arms are fully gold-blind.
                        def evaluate_selected(x):
                            tok,gr,comp=choice_quality(x["context"],groups)
                            return {"recall":gr,"token_recall":tok,"complete":comp,
                                    "real_tokens":x["used_tokens"],
                                    "chunks":len(x["selected"])}
                        top_metric=evaluate_selected(top)
                        render_metric=evaluate_selected(render_aware)
                        oracle=oracle_for_alternatives(
                            scored,groups,cap,counter)
                        # Report achieved oracle proxy as the best available
                        # FEASIBLE context, including learned-blind arms. This
                        # guards heuristic lower bound against fake inversions.
                        attained=max(oracle["oracle_rendered_5gram"],
                                     top_metric["recall"],render_metric["recall"])
                        rec={
                            "paper_id":paper_id,"case_id":case_id,
                            "scale":target,"budget_ratio":ratio,
                            "source_tokens":source_tokens,"budget_tokens":cap,
                            "n_chunks":len(scored),
                            "gold_annotation_groups":len(groups),
                            "topk":top_metric,"render":render_metric,
                            "oracle":oracle,
                            "best_feasible_rendered_recall":round(attained,8),
                            "feasible_oracle_gap_vs_render":
                                round(attained-render_metric["recall"],8),
                            "feasible_oracle_gap_vs_topk":
                                round(attained-top_metric["recall"],8),
                        }
                        if target==650 and not args.paper_limit:
                            for name,current in (("budget-top-k",top_metric),
                                                 ("render-value",render_metric)):
                                old=refer[(case_id,ratio,name)]
                                if abs(float(old["gold_5gram_recall"])-
                                       current["recall"])>2e-7:
                                    raise AssertionError(
                                        "Frozen 650-token baseline recall mismatch")
                                if int(old["rendered_tokens"])!=current["real_tokens"]:
                                    raise AssertionError(
                                        "Frozen 650-token baseline token mismatch")
                            audit["baseline_context_pairs_reproduced"]+=2
                        records.append(rec)
                        f.write(json.dumps(rec,ensure_ascii=False)+"\n")
            f.flush()
        audit["papers_processed"]+=1
        if paperidx%12==0 or paperidx==len(docs):
            print(f"PHASE4A {paperidx}/{len(docs)} development papers "
                  f"{len(records)} case-scale-budget rows "
                  f"elapsed={time.perf_counter()-started:.1f}s",flush=True)
    if not records:raise ValueError("No questions")
    stats=[]
    for ratio in RATIOS:
        for scale in SCALES:
            selected=[r for r in records
                      if r["budget_ratio"]==ratio and r["scale"]==scale]
            if not selected:continue
            n=len(selected)
            summary={
                "scale":scale,"ratio":ratio,"questions":n,
                "manual_topk_recall":round(statistics.mean(
                    r["topk"]["recall"] for r in selected),6),
                "manual_render_recall":round(statistics.mean(
                    r["render"]["recall"] for r in selected),6),
                "gold_feasible_oracle_lower":round(statistics.mean(
                    r["best_feasible_rendered_recall"] for r in selected),6),
                "gold_pool_internal_upper_no_budget":round(statistics.mean(
                    r["oracle"]["pool_internal_upper"] for r in selected),6),
                "oracle_feasible_minus_manual_render":round(statistics.mean(
                    r["feasible_oracle_gap_vs_render"] for r in selected),6),
                "manual_topk_zero":sum(r["topk"]["recall"]<=1e-9 for r in selected),
                "manual_render_zero":sum(r["render"]["recall"]<=1e-9 for r in selected),
                "oracle_lower_zero":sum(r["best_feasible_rendered_recall"]<=1e-9 for r in selected),
                "oracle_all_group_certified":sum(
                    r["oracle"]["all_annotation_objectives_certified"] for r in selected),
                "oracle_heuristic_cases":sum(
                    not r["oracle"]["all_annotation_objectives_certified"]
                    for r in selected),
                "mean_chunks_per_question":round(statistics.mean(
                    r["n_chunks"] for r in selected),2),
                "rendered_budget_breaches":sum(
                    max(r["topk"]["real_tokens"],r["render"]["real_tokens"],
                        r["oracle"]["oracle_used_tokens"])>r["budget_tokens"]
                    for r in selected),
            }
            stats.append(summary)
    summary={
        "experiment":"Phase4A validation-only 150/300/650 multires evidence ceiling",
        "dataset":"Official QASPER validation, previously inspected, NOT independent holdout",
        "test_split_used":False,
        "scales":SCALES,"ratios":RATIOS,
        "oracle_uses_gold":True,
        "gold_blind_manual_scoring":True,
        "gold_objective":"chunk-internal gold 5gram multiset",
        "gold_annotation_policy":"one alternative group at a time; maximize chosen feasible result",
        "exact_cap_relevant_candidates":EXACT_RELEVANT_LIMIT,
        "limits":[
            "Pool upper is budget-free chunk-internal 5gram bound, not a feasible solution.",
            "When >9 gold-relevant chunks, oracle feasible is heuristic LOWER bound, not certified optimum.",
            "Oracle reported rendered-recall is not generally exact optimum for the final-render metric.",
            "Gold-aware oracle cannot be deployed or claimed as blind retrieval improvement.",
            "Source-based token budget and full final-render cl100k_base token counting identical at all scales.",
            "Validation already used in prior development; no untouched test leakage.",
            "No generative LLM, GPU, paid API, or network access.",
        ],
        "n_validation_papers":len(docs),
        "n_development_questions":len(set(r["case_id"] for r in records)),
        "audit":dict(audit),
        "elapsed_seconds":round(time.perf_counter()-started,2),
        "per_scale":stats,
    }
    if not args.paper_limit:
        expected=889*len(SCALES)*len(RATIOS)
        if len(records)!=expected or audit["baseline_context_pairs_reproduced"]!=889*2*2:
            raise AssertionError("Missing full QASPER validation rows/control")
    (args.output_dir/"summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({
        "output":str(args.output_dir),
        "questions":summary["n_development_questions"],
        "audits":summary["audit"],
        "per_scale":stats,
        "elapsed_seconds":summary["elapsed_seconds"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
