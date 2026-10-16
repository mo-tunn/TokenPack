"""Phase 2, pure offline QASPER evidence-retention evaluation.

No generative LLM, Modal, network, gold-aware selection, or answer scoring.
Rebuilds a common chunk pool from archived full-document contexts. The pool
is not identical to the original QASPER paper-section index, so regenerated
arms are paired with one another, not claimed to reproduce historical arms.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENPACK_HF_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.evidence_grouping import fit_render_to_token_budget, render_source_evidence_groups
from tokenpack.evidence_neighbors import select_neighbor_augmented
from tokenpack.export import render_context
from tokenpack.models import TextBlock
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter
from submission.experiments.evidence_failure_audit import evidence_coverage
from submission.modal_generation_eval.client import _render_production_rag_context

ARCHIVE = ROOT / "submission/results/modal_generation_eval_qasper_full200/qasper_generation_tasks.jsonl"
OUTPUT = ROOT / "submission/results/phase2_qasper_evidence_offline_20261011"
CACHE = ROOT / ".tokenpack/cache/phase2_qasper_evidence_offline_20261011"
METHODS = (
    "archived-tokenpack-50",
    "archived-budget-top-k-50",
    "regenerated-tokenpack-budgeted",
    "p2-grouped",
    "p2-neighbor",
    "p2-neighbor-grouped",
    "p2-rag-budgeted",
)

def blocks_from_document(paper: str, source: str, counter: TokenCounter) -> list[TextBlock]:
    # Reconstructed spans, not the original QASPER section segmentation.
    blocks=[]
    cursor=0
    for n,text in enumerate(source.split("\n\n")):
        start=cursor
        end=start+len(text)
        cursor=end+2
        if not text.strip():continue
        blocks.append(TextBlock(
            text=text.strip(),source_path="qasper:"+paper,document_index=0,
            paragraph_index=n,char_start=start,char_end=end,
            metadata={"content_type":"text","token_count":counter.count(text)},
        ))
    return blocks

def span_stats(context: str, gold: list[str], full_document: str) -> dict:
    # A span counts as available when at least half its contiguous 5-grams
    # survived. This is a PROXY; wording variations may lower the score.
    full_hits=[]
    selected_hits=[]
    for span in gold:
        if not span.strip():continue
        _,full5=evidence_coverage(full_document,[span])
        _,selection5=evidence_coverage(context,[span])
        if full5 is None:continue
        full_hits.append(full5>=.50)
        selected_hits.append(selection5 is not None and selection5>=.50)
    _,five=evidence_coverage(context,gold)
    _,full_five=evidence_coverage(full_document,gold)
    eligible=[chosen for source,chosen in zip(full_hits,selected_hits) if source]
    return {
        "gold_fivegram_recall":five,
        "full_document_fivegram_recall":full_five,
        "eligible_gold_spans":len(eligible),
        "total_gold_spans":len(full_hits),
        "supported_span_fraction":sum(eligible)/len(eligible) if eligible else None,
        "all_eligible_spans_present":all(eligible) if eligible else None,
    }

def prepare_cases(max_cases: int)->list[dict]:
    methods=defaultdict(dict)
    with ARCHIVE.open(encoding="utf8") as f:
        for ln in f:
            if not ln.strip():continue
            rec=json.loads(ln)
            cid=rec["case_id"]
            methods[cid][rec["method"]]=rec
    cases=[]
    for cid,groups in methods.items():
        for name in ("full-document","tokenpack-50","budget-top-k-50"):
            if name not in groups:raise ValueError(f"Missing {name} for {cid}")
        cases.append(groups)
        if len(cases)>=max_cases:break
    return cases

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=int,default=50,choices=[5,50,200])
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()
    OUTPUT.mkdir(exist_ok=True,parents=True)
    report=OUTPUT/f"cases_{args.cases}.csv"
    summary_path=OUTPUT/f"summary_{args.cases}.json"
    if args.execute and (report.exists() or summary_path.exists()):
        raise FileExistsError("Existing analysis is immutable; refuse overwrite")
    cases=prepare_cases(args.cases)
    print(json.dumps({
        "mode":"OFFLINE_EXECUTE" if args.execute else "DRY_RUN",
        "cases":len(cases),"papers":len({x["full-document"]["paper_id"] for x in cases}),
        "model_calls":0,"remote_calls":0,
        "gold_is_only_for_scoring":True,"output":str(OUTPUT)
    },ensure_ascii=False),flush=True)
    if not args.execute:return

    t0=time.perf_counter()
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    counter=TokenCounter()
    CACHE.mkdir(exist_ok=True,parents=True)
    pool={}
    rows=[]
    for index,group in enumerate(cases):
        full=group["full-document"]
        paper=full["paper_id"]
        original=full["context"]
        if paper not in pool:
            blocks=blocks_from_document(paper,original,counter)
            cache=EmbeddingCache(CACHE/(paper.replace("/","_")+".embeddings.json"))
            block_vecs=cache.get_or_embed([b.text for b in blocks],embedder)
            chunker=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=counter,block_embeddings=block_vecs,
                semantic_threshold=.35,
            )
            chunks=chunker.chunk(blocks)
            chunk_vecs=cache.get_or_embed([c.text for c in chunks],embedder)
            pool[paper]=(chunks,chunk_vecs,hashlib.sha256(original.encode()).hexdigest())
        chunks,chunk_vecs,sha=pool[paper]
        if hashlib.sha256(original.encode()).hexdigest()!=sha:
            raise RuntimeError(f"Same paper has mismatched archived source: {paper}")
        question=str(full["question"])
        scored=score_chunks(
            embedder.embed([question])[0],chunks,chunk_vecs,
            scoring="evidence-hybrid",query_text=question,
        )
        budget=int(full["budget_tokens"])
        base=select_chunks(scored,strategy="budget-top-k",budget=budget,candidate_pool=300)
        neigh=select_neighbor_augmented(scored,budget,candidate_pool=300,neighbor_fraction=.15)
        rag=select_chunks(scored,strategy="production-rag",budget=budget,candidate_pool=300)
        ctx_specs=[
            ("archived-tokenpack-50",group["tokenpack-50"]["context"],None),
            ("archived-budget-top-k-50",group["budget-top-k-50"]["context"],None),
        ]
        for method, selected, renderer, priority in [
            ("regenerated-tokenpack-budgeted",base.selected,
             lambda items: render_context([i.chunk for i in items]),None),
            ("p2-grouped",base.selected,render_source_evidence_groups,None),
            ("p2-neighbor",neigh.selected,
             lambda items: render_context([i.chunk for i in items]),None),
            ("p2-neighbor-grouped",neigh.selected,render_source_evidence_groups,None),
            ("p2-rag-budgeted",rag.selected,_render_production_rag_context,
             lambda x:x.raw_similarity),
        ]:
            rendered,kept,trimmed=fit_render_to_token_budget(
                selected,render=renderer,token_counter=counter,budget=budget,
                drop_priority=priority
            )
            if counter.count(rendered)>budget:raise AssertionError("Budget overflow")
            ctx_specs.append((method,rendered,{"kept":len(kept),"trimmed":trimmed,
                                                "selected_ids":[i.chunk.id for i in kept]}))
        ids={method:set(extra["selected_ids"]) for method,_,extra in ctx_specs if extra}
        old_tp=ids["regenerated-tokenpack-budgeted"]
        new_tp=ids["p2-neighbor"]
        evidence=full.get("evidence_texts") or []
        for method,context,extras in ctx_specs:
            metrics=span_stats(context,evidence,original)
            rows.append({
                "case_id":full["case_id"],"paper_id":paper,"method":method,
                "budget_tokens":budget,"rendered_tokens":counter.count(context),
                "under_budget":counter.count(context)<=budget,
                "new_chunk_added":len(new_tp-old_tp) if method.startswith("p2-") else 0,
                "kept_chunks":extras.get("kept") if extras else None,
                "trimmed_for_headers":extras.get("trimmed") if extras else None,
                **metrics,
            })
        print(json.dumps({"case":index+1,"total":len(cases),
            "paper":paper,"elapsed_s":round(time.perf_counter()-t0,1),
            "new_neighbor_chunks":len(new_tp-old_tp),
            "gold_spans":len(evidence)},ensure_ascii=False),flush=True)

    with report.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    overview=[]
    for method in METHODS:
        subset=[r for r in rows if r["method"]==method]
        def mean(key):
            a=[float(r[key]) for r in subset if r[key] is not None]
            return round(sum(a)/len(a),4) if a else None
        overview.append({
            "method":method,"cases":len(subset),
            "avg_gold_fivegram_recall":mean("gold_fivegram_recall"),
            "avg_supported_span_fraction":mean("supported_span_fraction"),
            "all_spans_present_cases":sum(r["all_eligible_spans_present"] is True for r in subset),
            "all_spans_evaluable_cases":sum(r["all_eligible_spans_present"] is not None for r in subset),
            "avg_context_tokens":mean("rendered_tokens"),
            "budget_violations":sum(not r["under_budget"] for r in subset),
        })
    result={
        "scope":"OFFLINE evidence selection proxy; not model answer accuracy",
        "cases":len(cases),"papers":len(pool),
        "model":"local MiniLM embedding only; no generative LLM calls",
        "provenance_caveat":"Reconstructed paragraph index from archived full-document text; not original QASPER section blocks. Regenerated methods compared with themselves; archived context reported for orientation only.",
        "gold_inference_leak":False,
        "methods":overview,"elapsed_s":round(time.perf_counter()-t0,2),
        "new_artifacts":[str(report),str(summary_path)],
    }
    summary_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RESULT "+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
