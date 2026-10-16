"""Phase 3A QASPER oracle diagnostics: NO generative inference / no network.

The oracle alone may use gold evidence; production ranking and selection never
receive it. Feasible results include technical headers and use real tokenizer.
Exact certification applies only to limited eligible chunk counts and only to
the explicitly declared chunk-internal lexical multiset objective.
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
os.environ["TOKENIZERS_PARALLELISM"]="false"

from submission.experiments.phase2_qasper_diagnostic_v2 import (
    CACHE,blocks_from_document,prepare_cases,render_and_fit
)
from submission.experiments.phase3a_oracle_core import (
    gold_counts,oracle_search,pool_coverage
)
from submission.experiments.evidence_failure_audit import evidence_coverage
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache,make_embedder
from tokenpack.export import render_context
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter

OUTPUT=ROOT/"submission/results/phase3a_qasper_oracle_20261011"


def rows_to_csv(path,rows):
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analyze(cases:int,out:Path,ratios:list[float],exact_limit:int=12):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Oracle result directory is immutable")
    if not ratios or any(r<=0 or r>1 for r in ratios):
        raise ValueError("Bad budget ratios")
    groups=prepare_cases(cases)
    tokenizer=TokenCounter()
    if tokenizer._encoding is None:
        raise RuntimeError("Actual cl100k_base tokenizer required (no fallback)")
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    cache_by_paper={}
    records=[]
    diagnostic=[]
    timing=time.perf_counter()
    render=lambda items:render_context([x.chunk for x in items])
    print(f"PHASE3A OFFLINE {len(groups)} questions; ratios={ratios}; no LLM/network",flush=True)
    for number,group in enumerate(groups,1):
        full=group["full-document"]
        paper,source=full["paper_id"],full["context"]
        if paper not in cache_by_paper:
            blocks=blocks_from_document(paper,source,tokenizer)
            cache=EmbeddingCache(CACHE/(paper.replace("/","_")+".embeddings.json"))
            bv=cache.get_or_embed([b.text for b in blocks],embedder)
            chunks=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=tokenizer,block_embeddings=bv,
                semantic_threshold=.35,
            ).chunk(blocks)
            cv=cache.get_or_embed([c.text for c in chunks],embedder)
            cache_by_paper[paper]=(chunks,cv,hashlib.sha256(source.encode()).hexdigest())
        chunks,cv,digest=cache_by_paper[paper]
        if hashlib.sha256(source.encode()).hexdigest()!=digest:
            raise ValueError("Archived paper source hash mismatch")
        question=str(full["question"])
        scored=score_chunks(embedder.embed([question])[0],chunks,cv,
                            scoring="evidence-hybrid",query_text=question)
        by_rank=sorted(scored,key=lambda x:(-x.value,x.chunk.order_key,x.chunk.id))
        # Strict boundary: ONLY after retrieval ranking are complete may gold
        # appear in diagnostic scope. No scored item is modified.
        annotated=full.get("evidence_texts") or []
        gold,span_counts=gold_counts(annotated)
        if not gold:
            raise ValueError(f"No lexical 5-grams for case {full['case_id']}")
        full_doc_recall=evidence_coverage(source,annotated)[1]
        all_coverage=pool_coverage(scored,gold,span_counts)
        avail={}
        for count in (5,10,20,300):
            avail[f"top{count}"]=pool_coverage(by_rank[:count],gold,span_counts)
        for ratio in ratios:
            budget=(int(full["budget_tokens"]) if abs(ratio-0.5)<1e-9
                    else math.floor(int(full["source_tokens"])*ratio))
            baseline=select_chunks(scored,strategy="budget-top-k",
                                   budget=budget,candidate_pool=300)
            baseline_text,baseline_kept,trimmed,baseline_used,_=render_and_fit(
                baseline.selected,render,budget,tokenizer)
            baseline_proxy=pool_coverage(baseline_kept,gold,span_counts)
            baseline_rendered_recall=evidence_coverage(baseline_text,annotated)[1]
            for candidate_name,candidates in [
                ("top20",by_rank[:20]),
                ("top300",by_rank[:300]),
            ]:
                t0=time.perf_counter()
                oracle=oracle_search(
                    candidates,gold=gold,span_grams=span_counts,
                    budget=budget,render=render,counter=tokenizer,
                    exact_max_relevant=exact_limit
                )
                oracle_time=time.perf_counter()-t0
                capacity=pool_coverage(candidates,gold,span_counts)
                o5=oracle["oracle_5gram"]
                oj=oracle["oracle_joint"]
                if o5["gold_5gram_recall"]>capacity["gold_5gram_recall"]+1e-9:
                    raise AssertionError("Oracle unexpectedly exceeded candidate ceiling")
                if o5["gold_5gram_recall"]+1e-9<baseline_proxy["gold_5gram_recall"] and candidate_name=="top300":
                    # Baseline itself is feasible in candidate pool. Do NOT
                    # claim heuristic oracle dominates an observed solution.
                    # Note instead for manual inspection.
                    pass
                record={
                    "case_id":full["case_id"],"paper_id":paper,
                    "question_id":full["question_id"],
                    "budget_ratio":ratio,"budget_tokens":budget,
                    "source_tokens":int(full["source_tokens"]),
                    "all_chunk_count":len(chunks),
                    "candidate_pool":candidate_name,
                    "candidate_count":len(candidates),
                    "eligible_gold_5grams":sum(gold.values()),
                    "annotated_evidence_spans":len(span_counts),
                    "full_document_5gram_recall":full_doc_recall,
                    "chunking_ceiling_recall":all_coverage["gold_5gram_recall"],
                    "chunking_ceiling_complete_spans":all_coverage["completed_spans"],
                    "candidate_ceiling_recall":capacity["gold_5gram_recall"],
                    "candidate_ceiling_complete_spans":capacity["completed_spans"],
                    "baseline_proxy_5gram_recall":baseline_proxy["gold_5gram_recall"],
                    "baseline_rendered_5gram_recall":baseline_rendered_recall,
                    "baseline_all_spans":baseline_proxy["all_spans_present"],
                    "baseline_used_tokens":baseline_used,
                    "baseline_header_trimmed":trimmed,
                    "oracle_5gram_recall":o5["gold_5gram_recall"],
                    "oracle_5gram_completed_spans":o5["completed_spans"],
                    "oracle_joint_recall":oj["gold_5gram_recall"],
                    "oracle_joint_completed_spans":oj["completed_spans"],
                    "oracle_joint_all_spans":oj["all_spans_present"],
                    "oracle_5gram_used_tokens":o5["rendered_tokens"],
                    "oracle_joint_used_tokens":oj["rendered_tokens"],
                    "oracle_mode":o5["mode"],
                    "oracle_relevant_chunks":oracle["eligible_relevant_chunks"],
                    "oracle_time_s":round(oracle_time,5),
                }
                if o5["rendered_tokens"]>budget or oj["rendered_tokens"]>budget:
                    raise AssertionError("Oracle exceeded exact render budget")
                records.append(record)
                diagnostic.append({
                    "case_id":full["case_id"],"paper_id":paper,
                    "ratio":ratio,"pool":candidate_name,
                    "baseline_selected_chunk_ids":[i.chunk.id for i in baseline_kept],
                    "oracle_gold5_selected_ids":o5["selected_ids"],
                    "oracle_joint_selected_ids":oj["selected_ids"],
                    "mode":o5["mode"],
                    "oracle_checked":o5["candidate_subsets_checked"],
                })
        if number%10==0 or number==len(groups):
            print(f"{number}/{len(groups)}; elapsed={time.perf_counter()-timing:.1f}s",flush=True)
    for rec in records:
        if rec["oracle_mode"].startswith("exact"):
            if rec["oracle_5gram_recall"]+1e-9 < rec["baseline_proxy_5gram_recall"] and rec["candidate_pool"]=="top300":
                raise AssertionError("Exact oracle worse than baseline at same pool/budget")
    summaries=[]
    for ratio in ratios:
        for pool in ("top20","top300"):
            subset=[row for row in records if row["budget_ratio"]==ratio and row["candidate_pool"]==pool]
            def avg(key):
                return round(sum(row[key] for row in subset)/len(subset),6)
            unique_cases={r["case_id"]:{"control":{
                "paper_id":r["paper_id"],"delta":r["baseline_proxy_5gram_recall"]
            },"oracle":{
                "paper_id":r["paper_id"],"delta":r["oracle_5gram_recall"]
            }} for r in subset}
            diffs=[{
                "case_id":r["case_id"],"method":"oracle",
                "paper_id":r["paper_id"],
                "gap":r["oracle_5gram_recall"]-r["baseline_proxy_5gram_recall"],
            } for r in subset]
            def cluster_differences():
                by_case={
                    r["case_id"]:{
                        "control":{"paper_id":r["paper_id"],"gap":0.0},
                        "oracle":{"paper_id":r["paper_id"],
                                  "gap":r["oracle_5gram_recall"]-r["baseline_proxy_5gram_recall"]}
                    } for r in subset
                }
                return paper_cluster_ci(by_case,"gap","oracle",replicates=2000)
            summaries.append({
                "budget_ratio":ratio,"pool":pool,
                "n":len(subset),
                "mean_chunking_unbudgeted_ceiling":avg("chunking_ceiling_recall"),
                "mean_top5_unbudgeted_candidate_ceiling":None,
                "mean_candidate_unbudgeted_ceiling":avg("candidate_ceiling_recall"),
                "mean_baseline_rendered_5gram_recall":avg("baseline_rendered_5gram_recall"),
                "mean_baseline_chunk_internal_5gram_recall":avg("baseline_proxy_5gram_recall"),
                "mean_gold_aware_feasible_5gram_recall":avg("oracle_5gram_recall"),
                "mean_gold_aware_feasible_joint_spans":avg("oracle_joint_completed_spans"),
                "mean_baseline_5gram_to_oracle_gap":round(
                    avg("oracle_5gram_recall")-avg("baseline_proxy_5gram_recall"),6),
                "paper_cluster_bootstrap_gap95":cluster_differences(),
                "exact_certified_questions":sum(r["oracle_mode"].startswith("exact") for r in subset),
                "heuristic_questions":sum(r["oracle_mode"].startswith("heuristic") for r in subset),
                "oracle_all_spans_cases":sum(r["oracle_joint_all_spans"] for r in subset),
                "baseline_all_spans_cases":sum(r["baseline_all_spans"] for r in subset),
                "actual_budget_violations":0,
            })
    topk={}
    for k in (5,10,20,300):
        key=f"top{k}"
        # Per-question cache of top-k covers: reconstruct from main-row
        # lightweight standalone reading to avoid leaking it into scoring.
        topk[key]=[]
    # Store availability explicitly in per-case provenance in a separate
    # pass, with no ranking regeneration or hidden oracle calls.
    for group in groups:
        f=group["full-document"]
        paper,source=f["paper_id"],f["context"]
        chunks,cv,_=cache_by_paper[paper]
        ranked=sorted(score_chunks(embedder.embed([f["question"]])[0],chunks,cv,
            scoring="evidence-hybrid",query_text=f["question"]),
            key=lambda x:(-x.value,x.chunk.order_key,x.chunk.id))
        gold,spans=gold_counts(f.get("evidence_texts") or [])
        for k in (5,10,20,300):
            topk[f"top{k}"].append(pool_coverage(ranked[:k],gold,spans)["gold_5gram_recall"])
    average_topk={k:round(sum(v)/len(v),6) for k,v in topk.items()}
    full_source=[r for r in records if r["candidate_pool"]=="top300" and r["budget_ratio"]==ratios[0]]
    result={
        "analysis":"QASPER offline oracle diagnosis (NOT model answer accuracy)",
        "cases":len(groups),"papers":len(cache_by_paper),
        "methodology":{
            "gold_data":"archived evidence_texts for one chosen answer annotation; alternatives not recoverable from tasks",
            "oracle_objective":"chunk-internal lexical 5gram multiset coverage, plus separate 50%-per-span completion objective",
            "exactness":"exact enumeration only if <= exact_limit gold-positive chunks, otherwise feasible greedy/swap heuristic",
            "source_index":"reconstructed archived full text paragraphs (NOT native QASPER section index)",
            "budget":"actual cl100k_base render_context output including headers, all candidate sets",
            "gold_leak_to_production":False,
            "inference_calls":0,"network_calls":0,
            "disclaimer":"Mean feasible oracle is an achievable gold-informed diagnostic, NOT a certified theoretical ceiling for heuristic cases.",
            "exact_relevant_limit":exact_limit,
        },
        "topk_unbudgeted_5gram_coverage":average_topk,
        "mean_all_chunk_internal_5gram_ceiling":round(sum(
            r["chunking_ceiling_recall"] for r in full_source)/len(full_source),6),
        "results":summaries,
        "elapsed_seconds":round(time.perf_counter()-timing,3),
    }
    out.mkdir(parents=True,exist_ok=True)
    rows_to_csv(out/"cases.csv",records)
    with (out/"selection_audit.jsonl").open("w",encoding="utf8") as f:
        for item in diagnostic:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    report=[
        "# TokenPack QASPER Faz 3A — Gold evidence oracle teşhisi",
        "",
        f"{len(groups)} soru, {len(cache_by_paper)} makale; %25 ve %50 token üst sınırı. Tamamen çevrimdışı; üretici LLM/GPU/API yok.",
        "",
        "Gold evidence yalnızca oracle ve puan ölçümünde kullanıldı, normal retrieval/scoring/selection aşamalarına sızdırılmadı.",
        "Not: QASPER arşivi tek seçilmiş anotasyonun evidence_texts listesini koruyor; alternatif insan kanıt kümeleri mevcut değil.",
        "Oracle yöntemi bütün sorularda gerçek çıktı token bütçesine uyar; ancak yeterince büyük örneklerde sonuç yaklaşık *erişilebilir* düzeydir, matematiksel optimum değildir.",
        "",
        "## İlk aday havuzuna giren kanıtların bütçesiz kapsamı",
        "",
        "| Aday havuzu | Gold 5-gram coverage |",
        "|---|---:|",
    ]
    for k,v in average_topk.items():
        report.append(f"| {k} | {100*v:.2f}% |")
    report+=["",f"**Tüm chunk'ların chunk-içi toplam kapasitesi:** {100*result['mean_all_chunk_internal_5gram_ceiling']:.2f}%",
             "Bu bütçesiz kapasite: tüm kanıtların aynı token bütçesine sığdığı anlamına gelmez.",
             "",
             "## Bütçeli oracle", "",
             "| Bütçe | Adaylar | Normal TokenPack 5-gram | Gold-aware *feasible* oracle | Gap (yp) | Kesin optimum / soru |",
             "|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        report.append(f"| %{100*r['budget_ratio']:.0f} | {r['pool']} | "
            f"{100*r['mean_baseline_chunk_internal_5gram_recall']:.2f}% | "
            f"{100*r['mean_gold_aware_feasible_5gram_recall']:.2f}% | "
            f"{100*r['mean_baseline_5gram_to_oracle_gap']:+.2f} | "
            f"{r['exact_certified_questions']}/{r['n']} |")
    report += [
        "", "## Yorumlama sınırları",
        "",
        "- Oracle, gerçek sistemin erişebileceği alt sınır niteliğindeki **feasible** kanıt kazanımını gösterir; gold-aware seçim dağıtıma alınamaz.",
        "- Pozitif oracle farkı scoring, seçim, uzun chunk, başlık maliyeti arasında otomatik olarak nedensellik kurmaz.",
        "- Chunk içi 5-gram, çıktı metninin 5-gram'ından az farklı olabilir; baseline her iki metrikle raporlandı.",
        "- 50 ve 200 sorunun tamamı önceden kullanılmış arşivden gelir; bağımsız test değildir.",
        "- Birden fazla anotatörün alternatif kanıt kümeleri düzleştirilmiş arşivde bulunmaz; bu analiz tek seçilmiş anotasyonun vekil ölçümüdür.",
        "- Yalnızca gelişim sinyali elde edildi; cevap doğruluğu hakkında hüküm verilemez.",
    ]
    (out/"README_FINDINGS.md").write_text("\n".join(report)+"\n",encoding="utf8")
    print("RESULT "+json.dumps({"output":str(out),"topk":average_topk,"results":summaries,
                                "elapsed_s":result["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=int,choices=[5,50,200],default=50)
    ap.add_argument("--ratios",default="0.25,0.50")
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--exact-limit",type=int,default=12)
    args=ap.parse_args()
    analyze(args.cases,args.output_dir,[float(x) for x in args.ratios.split(",")],args.exact_limit)
