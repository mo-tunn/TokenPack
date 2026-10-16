"""Matched and fully offline QASPER Phase 2b: use spare output tokens only.

The original selector, historical output files, and gold-free production code
are unchanged. Model access: local embedding MiniLM only, strict HF offline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
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
os.environ["TOKENIZERS_PARALLELISM"]="false"

from submission.experiments.phase2_qasper_diagnostic_v2 import (
    CACHE, blocks_from_document, paired_bootstrap, prepare_cases,
    render_and_fit,
)
from submission.experiments.evidence_failure_audit import evidence_coverage
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.evidence_slack import fill_render_slack
from tokenpack.export import render_context
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter

METHODS=("control","slack-global","slack-neighbor")


def paired_and_cluster(rows,field,method):
    cases=defaultdict(dict)
    for row in rows:
        cases[row["case_id"]][row["method"]]=row
    if any(set(methods)!=set(METHODS) for methods in cases.values()):
        raise ValueError("Missing paired method")
    a=[case[method][field] for case in cases.values()]
    b=[case["control"][field] for case in cases.values()]
    return {
        "question_bootstrap":paired_bootstrap(a,b),
        "paper_cluster_bootstrap":paper_cluster_ci(cases,field,method),
    }


def write_csv(path,rows):
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=int,choices=[50,200],default=50)
    ap.add_argument("--output-dir",required=True,type=Path)
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Existing result directory is immutable")
    cases=prepare_cases(args.cases)
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("tiktoken cl100k_base required; fallback disallowed")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    pool={}
    rows=[]
    provenance=[]
    start=time.perf_counter()
    render=lambda selected:render_context([item.chunk for item in selected])
    print(f"OFFLINE Phase2b: {args.cases} matched cases",flush=True)
    for n,group in enumerate(cases,1):
        full=group["full-document"]
        paper,source=full["paper_id"],full["context"]
        if paper not in pool:
            blocks=blocks_from_document(paper,source,counter)
            cache=EmbeddingCache(CACHE/(paper.replace("/","_")+".embeddings.json"))
            bv=cache.get_or_embed([b.text for b in blocks],embedder)
            chunks=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=counter,block_embeddings=bv,semantic_threshold=.35
            ).chunk(blocks)
            cv=cache.get_or_embed([c.text for c in chunks],embedder)
            pool[paper]=(chunks,cv,hashlib.sha256(source.encode()).hexdigest())
        chunks,cv,digest=pool[paper]
        if digest!=hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("Mismatched paper source "+paper)
        budget=int(full["budget_tokens"])
        question=full["question"]
        query=embedder.embed([question])[0]
        scored=score_chunks(query,chunks,cv,scoring="evidence-hybrid",
                            query_text=question)
        seeds=select_chunks(scored,"budget-top-k",budget,candidate_pool=300)
        original,kept,dropped,used,_=render_and_fit(
            seeds.selected,render,budget,counter)
        # Complete ALL candidate selections before loading the gold spans.
        contexts={
            "control":(original,kept,{
                "added_chunks":0,"displaced_baseline_chunks":0,
                "source_tokens_used_before":used,
                "rendered_tokens":used,"candidate_checks":0}),
        }
        for method,only_neighbors in (
            ("slack-global",False),("slack-neighbor",True)
        ):
            text,selected,meta=fill_render_slack(
                scored,kept,question=question,budget=budget,render=render,
                counter=counter,only_neighbors=only_neighbors,
                relative_neighbor_score=.40,
            )
            contexts[method]=(text,selected,meta)

        evidence=full.get("evidence_texts") or []
        base_ids=set(item.chunk.id for item in kept)
        for method in METHODS:
            text,selected,meta=contexts[method]
            selected_ids={item.chunk.id for item in selected}
            if not base_ids.issubset(selected_ids):
                raise AssertionError("Phase2b displaced baseline")
            actual=counter.count(text)
            if actual>budget:
                raise AssertionError("Budget violation")
            token,five=evidence_coverage(text,evidence)
            spans=[
                evidence_coverage(text,[span])[1]
                for span in evidence if span.strip()
            ]
            spans=[x for x in spans if x is not None]
            rows.append({
                "case_id":full["case_id"],"paper_id":paper,
                "method":method,"budget_tokens":budget,
                "rendered_tokens":actual,"token_recall":token,
                "gold_fivegram_recall":five,
                "all_gold_spans_present":all(v>=.5 for v in spans) if spans else None,
                "spans_evaluable":len(spans),
                "base_kept_chunks":len(kept),
                "selected_chunks":len(selected),
                "added_chunks":meta["added_chunks"],
                "displaced_chunks":len(base_ids-selected_ids),
                "candidate_checks":meta["candidate_checks"],
                "trimmed_original_headers":dropped,
            })
            provenance.append({
                "case_id":full["case_id"],"method":method,
                "selected_ids":sorted(selected_ids),
                "added_ids":sorted(selected_ids-base_ids),
            })
        if n%25==0:
            print(f"{n}/{len(cases)}, elapsed={time.perf_counter()-start:.1f}s",
                  flush=True)
    def avg(method,key):
        values=[row[key] for row in rows
                if row["method"]==method and row[key] is not None]
        return round(sum(values)/len(values),6) if values else None
    methods=[]
    for method in METHODS:
        r=[row for row in rows if row["method"]==method]
        methods.append({
            "method":method,"cases":len(r),
            "mean_gold_5gram_recall":avg(method,"gold_fivegram_recall"),
            "mean_gold_token_recall":avg(method,"token_recall"),
            "all_gold_spans_present":sum(x["all_gold_spans_present"] is True for x in r),
            "mean_output_tokens":avg(method,"rendered_tokens"),
            "mean_added_chunks":avg(method,"added_chunks"),
            "cases_with_additions":sum(x["added_chunks"]>0 for x in r),
            "total_baseline_displacements":sum(x["displaced_chunks"] for x in r),
            "budget_violations":sum(x["rendered_tokens"]>x["budget_tokens"] for x in r),
        })
    comparisons={
        m:{
            "fivegram":paired_and_cluster(rows,"gold_fivegram_recall",m),
            "token":paired_and_cluster(rows,"token_recall",m),
        } for m in METHODS if m!="control"
    }
    per_case=defaultdict(dict)
    for row in rows:per_case[row["case_id"]][row["method"]]=row
    deltas=[{
        "case_id":cid,
        "control":data["control"]["gold_fivegram_recall"],
        "neighbor":data["slack-neighbor"]["gold_fivegram_recall"],
        "global":data["slack-global"]["gold_fivegram_recall"],
    } for cid,data in per_case.items()
       if data["slack-neighbor"]["gold_fivegram_recall"]!=data["control"]["gold_fivegram_recall"]]
    result={
        "purpose":"Exploratory gold-blind add-only Phase2b selection",
        "cases":len(per_case),"papers":len(pool),
        "generative_llm_calls":0,"remote_calls":0,
        "no_gold_used_for_selection":True,
        "source":"Archived QASPER full-document contexts; common regenerated chunk pool",
        "tokenizer":"cl100k_base",
        "parameters":{"relative_neighbor_score":.40,
                      "only_add_in_render_slack":True,
                      "query_lexical_relevance_required_for_neighbors":True},
        "methods":methods,"paired_vs_control":comparisons,
        "changed_neighbor_cases":deltas,
        "elapsed_seconds":round(time.perf_counter()-start,3),
        "limitations":[
            "All data are from previous development archives, not independent holdout.",
            "Only lexical recall proxy, not semantic answer accuracy.",
            "Source chunk index was reconstructed from archived full-document contexts.",
            "Parameters were not chosen on a sealed test set.",
        ],
    }
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(args.output_dir/"cases.csv",rows)
    with (args.output_dir/"chunk_ids.jsonl").open("w",encoding="utf8") as f:
        for rec in provenance:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    (args.output_dir/"summary.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    lines=[
        f"# QASPER Phase 2b — {len(per_case)} soru / {len(pool)} makale",
        "",
        "Tamamen çevrimdışı. Hiçbir orijinal chunk çıkarılmadı; gold yalnızca seçimden sonra ölçüldü.",
        "İki ekleme kolu: bütçedeki boşluğu normal skorlu adaylarla doldurmak ve yalnızca kaynak komşularıyla doldurmak.",
        "",
        "| Kol | 5-gram | Token recall | Bütün spanlar | Ortalama ek chunk | Ek yapılan soru |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in methods:
        lines.append(f"| {x['method']} | {100*x['mean_gold_5gram_recall']:.2f}% | "
            f"{100*x['mean_gold_token_recall']:.2f}% | "
            f"{x['all_gold_spans_present']}/{x['cases']} | "
            f"{x['mean_added_chunks']:.3f} | {x['cases_with_additions']} |")
    lines.extend(["","## Kontrole karşı eşleştirilmiş 5-gram farkı","",
                  "| Kol | Fark (yp) | Makale-kümeli bootstrap %95 GA (yp) | K/E/B |",
                  "|---|---:|---:|---:|"])
    for method in METHODS[1:]:
        comparison=comparisons[method]["fivegram"]
        q=comparison["question_bootstrap"]
        c=comparison["paper_cluster_bootstrap"]
        low,high=c["cluster_bootstrap_ci95"]
        lines.append(f"| {method} | {100*q['mean_difference']:+.2f} | "
            f"[{100*low:+.2f}; {100*high:+.2f}] | "
            f"{q['wins']}/{q['losses']}/{q['ties']} |")
    lines.extend(["","Kaynakta bitişik olmak semantik kanıt tamamlamak anlamına gelmez.",
                  "200 soru da dahil olmak üzere arşiv geçmişte kullanıldı; bu bağımsız doğrulama değildir.",
                  "Cevap doğruluğu ölçülmemiştir; model inference yapılmamıştır.",""])
    (args.output_dir/"README.md").write_text("\n".join(lines),encoding="utf8")
    print(json.dumps({
        "output":str(args.output_dir),
        "methods":methods,"paired":comparisons,
        "changed_neighbor_cases":len(deltas),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
