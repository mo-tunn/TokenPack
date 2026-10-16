"""Extended paired QASPER Phase-2 audit, strictly offline.

Recreates all compared Phase-2 arms on one local chunk pool and tokenizer.
Gold evidence is introduced ONLY after scoring, selection, and rendering.
No inference, network, Modal, or Qwen calls. Historical result files are read-only.
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
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENPACK_HF_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from submission.experiments.evidence_failure_audit import evidence_coverage
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.evidence_grouping import fit_render_to_token_budget, render_source_evidence_groups
from tokenpack.evidence_neighbors import select_neighbor_augmented, source_adjacent
from tokenpack.export import render_context
from tokenpack.models import TextBlock
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter

OUTPUT = ROOT / "submission/results/phase2_qasper_diagnostic_v2_20261011"
CACHE = ROOT / ".tokenpack/cache/phase2_qasper_evidence_offline_20261011"
ARCHIVE = ROOT / "submission/results/modal_generation_eval_qasper_full200/qasper_generation_tasks.jsonl"
METHODS = (
    "control", "grouped", "neighbor", "neighbor-grouped", "rag-budgeted",
)


def blocks_from_document(paper: str, source: str, counter: TokenCounter) -> list[TextBlock]:
    """Exact same paragraph reconstruction as the original Phase-2 QASPER audit."""
    blocks=[]
    cursor=0
    for n, text in enumerate(source.split("\n\n")):
        start=cursor
        end=start+len(text)
        cursor=end+2
        if not text.strip():
            continue
        blocks.append(TextBlock(
            text=text.strip(),source_path="qasper:"+paper,document_index=0,
            paragraph_index=n,char_start=start,char_end=end,
            metadata={"content_type":"text","token_count":counter.count(text)},
        ))
    return blocks


def prepare_cases(max_cases: int) -> list[dict]:
    methods=defaultdict(dict)
    with ARCHIVE.open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line)
                methods[row["case_id"]][row["method"]]=row
    cases=[]
    for cid,group in methods.items():
        if not all(m in group for m in ("full-document","tokenpack-50","budget-top-k-50")):
            raise ValueError("Incomplete QASPER case: "+cid)
        cases.append(group)
        if len(cases)==max_cases:
            break
    return cases


def _render_production_rag_context(items) -> str:
    ordered=sorted(items,key=lambda item:item.raw_similarity,reverse=True)
    parts=[]
    for number,item in enumerate(ordered,1):
        c=item.chunk
        parts.append(
            f"[Retrieved Chunk {number}: id={c.id}, source={c.source_path}, "
            f"tokens={c.token_count}, similarity={item.raw_similarity:.4f}]"
        )
        parts.append(c.text)
    return "\n\n".join(parts).strip()+"\n"


def source_span_audit(source: str, gold: list[str], chunks, selected_ids: set[str]) -> dict:
    """Exact, case-insensitive source offsets. Repeated spans are ambiguous.

    This reports structural provenance, not evidence sufficiency or entailment.
    """
    located = 0
    ambiguous = 0
    fragmented = 0
    joint_recovered = []
    spans = []
    lower = source.casefold()
    for span in gold:
        if not span.strip():
            continue
        needle = span.casefold()
        positions = []
        at = 0
        while (pos := lower.find(needle, at)) >= 0:
            positions.append(pos)
            at = pos + max(1, len(needle))
            if len(positions) > 2:
                break
        if not positions:
            spans.append({"status": "not_found"})
            continue
        if len(positions) > 1:
            ambiguous += 1
            spans.append({"status": "repeated", "matches_at_least": len(positions)})
            continue
        start, end = positions[0], positions[0] + len(span)
        touched = [c for c in chunks if c.char_start < end and c.char_end > start]
        if not touched:
            spans.append({"status": "no_chunk_overlap"})
            continue
        located += 1
        split = not any(c.char_start <= start and c.char_end >= end for c in touched)
        fragmented += int(split)
        all_kept = all(c.id in selected_ids for c in touched)
        joint_recovered.append(all_kept)
        spans.append({
            "status": "unique", "source_start": start, "source_end": end,
            "touching_chunk_ids": [c.id for c in touched],
            "crosses_chunk_boundary": split,
            "all_touching_chunks_selected": all_kept,
        })
    return {
        "unique_source_spans": located,
        "ambiguous_source_spans": ambiguous,
        "boundary_split_spans": fragmented,
        "all_locatable_span_chunks_selected": (
            all(joint_recovered) if joint_recovered else None
        ),
        "source_spans": spans,
    }


def adjacency_counts(chunks, selected_ids: set[str]) -> tuple[int, int]:
    ordered = sorted(chunks, key=lambda c: (c.document_index,c.source_path,c.char_start,c.char_end,c.id))
    total, kept = 0, 0
    for left,right in zip(ordered,ordered[1:]):
        if source_adjacent(left,right):
            total += 1
            kept += int(left.id in selected_ids and right.id in selected_ids)
    return kept,total


def repetition_fraction(selected) -> float | None:
    """Share of lexical 5-gram occurrences that repeat within selected raw text."""
    from submission.experiments.evidence_failure_audit import tokens
    grams = []
    for item in selected:
        words = tokens(item.chunk.text)
        grams.extend(tuple(words[i:i+5]) for i in range(max(0,len(words)-4)))
    if not grams:
        return None
    return round((len(grams)-len(set(grams)))/len(grams),6)


def paired_bootstrap(a: list[float], b: list[float], *, n: int = 6000,
                     seed: int = 20261011) -> dict:
    if len(a) != len(b) or not a:
        raise ValueError("Must supply non-empty paired observations")
    deltas = [x-y for x,y in zip(a,b)]
    rng = random.Random(seed)
    means = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas)/len(deltas)
                   for _ in range(n))
    wins = sum(d > 1e-9 for d in deltas)
    losses = sum(d < -1e-9 for d in deltas)
    ties = len(deltas) - wins - losses
    nonzero = wins + losses
    p = (min(1.0,2*sum(math.comb(nonzero,k) for k in range(min(wins,losses)+1)) /
             (2**nonzero)) if nonzero else 1.0)
    return {
        "n":len(deltas),
        "mean_difference":round(sum(deltas)/len(deltas),6),
        "bootstrap_ci95":[round(means[int(.025*(n-1))],6),
                          round(means[int(.975*(n-1))],6)],
        "wins":wins,"losses":losses,"ties":ties,
        "two_sided_sign_p_exploratory":round(p,6),
    }


def render_and_fit(selected, renderer, budget, counter, priority=None):
    t0=time.perf_counter()
    text, kept, trimmed = fit_render_to_token_budget(
        selected, render=renderer, token_counter=counter, budget=budget,
        drop_priority=priority)
    elapsed=time.perf_counter()-t0
    used=counter.count(text)
    if used > budget:
        raise AssertionError(f"Rendered budget exceeded: {used}>{budget}")
    return text,kept,trimmed,used,elapsed


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No rows")
    with path.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def do_run(cases: int, out: Path):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Analysis output already exists: {out}")
    group_cases=prepare_cases(cases)
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("cl100k_base tokenizer unavailable; cannot use fallback")
    # Both the archived tasks and regenerated contexts use cl100k_base.
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    paper_cache={}
    rows=[]
    diagnostics=[]
    total_t0=time.perf_counter()
    print(f"OFFLINE: {len(group_cases)} cases; no LLM or network",flush=True)
    for i, group in enumerate(group_cases,1):
        full=group["full-document"]
        paper, source=full["paper_id"],full["context"]
        if paper not in paper_cache:
            t0=time.perf_counter()
            blocks=blocks_from_document(paper,source,counter)
            cache=EmbeddingCache(CACHE/(paper.replace("/","_")+".embeddings.json"))
            vectors=cache.get_or_embed([b.text for b in blocks],embedder)
            embedding_sec=time.perf_counter()-t0
            t0=time.perf_counter()
            chunks=StructureAwareChunker(
                target_tokens=650,min_tokens=120,max_tokens=900,
                token_counter=counter,block_embeddings=vectors,
                semantic_threshold=.35,
            ).chunk(blocks)
            chunking_sec=time.perf_counter()-t0
            t0=time.perf_counter()
            chunk_vectors=cache.get_or_embed([c.text for c in chunks],embedder)
            embedding_sec+=time.perf_counter()-t0
            paper_cache[paper]=(chunks,chunk_vectors,
                hashlib.sha256(source.encode()).hexdigest(),
                embedding_sec,chunking_sec)
        chunks,vectors,digest,_,_=paper_cache[paper]
        if hashlib.sha256(source.encode()).hexdigest()!=digest:
            raise ValueError("Different source text for same paper: "+paper)
        budget=int(full["budget_tokens"])
        question=str(full["question"])
        t0=time.perf_counter()
        question_embedding=embedder.embed([question])[0]
        scored=score_chunks(question_embedding,chunks,vectors,
                            scoring="evidence-hybrid",query_text=question)
        scoring_seconds=time.perf_counter()-t0
        t0=time.perf_counter()
        base=select_chunks(scored,strategy="budget-top-k",budget=budget,
                           candidate_pool=300)
        base_time=time.perf_counter()-t0
        t0=time.perf_counter()
        neighbor=select_neighbor_augmented(scored,budget,candidate_pool=300,
                                           neighbor_fraction=.15)
        neigh_time=time.perf_counter()-t0
        t0=time.perf_counter()
        rag=select_chunks(scored,strategy="production-rag",budget=budget,
                          candidate_pool=300)
        rag_time=time.perf_counter()-t0

        original_renderer=lambda items:render_context([x.chunk for x in items])
        specs={
            "control":(base.selected,original_renderer,base_time,None),
            "grouped":(base.selected,render_source_evidence_groups,base_time,None),
            "neighbor":(neighbor.selected,original_renderer,neigh_time,None),
            "neighbor-grouped":(neighbor.selected,render_source_evidence_groups,neigh_time,None),
            "rag-budgeted":(rag.selected,_render_production_rag_context,
                            rag_time,lambda x:x.raw_similarity),
        }
        # All five selections are complete BEFORE the gold evidence is consulted.
        selections={}
        for method,(selected,renderer,select_sec,drop_priority) in specs.items():
            rendered,kept,trimmed,used,render_sec=render_and_fit(
                selected,renderer,budget,counter,drop_priority)
            selections[method]=(rendered,kept,trimmed,used,select_sec,render_sec)

        evidence=full.get("evidence_texts") or []
        base_ids={x.chunk.id for x in selections["control"][1]}
        neighbor_ids={x.chunk.id for x in selections["neighbor"][1]}
        for method in METHODS:
            rendered,kept,trimmed,used,select_sec,render_sec=selections[method]
            selected_ids={x.chunk.id for x in kept}
            tok,five=evidence_coverage(rendered,evidence)
            support=[]
            for span in evidence:
                _,metric=evidence_coverage(rendered,[span])
                if metric is not None:
                    support.append(metric>=.5)
            structural=source_span_audit(source,evidence,chunks,selected_ids)
            adj_selected,adj_total=adjacency_counts(chunks,selected_ids)
            data={
                "case_id":full["case_id"],"paper_id":paper,"method":method,
                "budget_tokens":budget,"rendered_tokens":used,
                "budget_used_fraction":round(used/budget,6),
                "kept_chunks":len(kept),"trimmed_for_headers":trimmed,
                "gold_token_recall":tok,"gold_fivegram_recall":five,
                "supported_gold_spans":sum(support),
                "evaluable_gold_spans":len(support),
                "all_evaluable_gold_spans_present":all(support) if support else None,
                "unique_source_spans":structural["unique_source_spans"],
                "ambiguous_source_spans":structural["ambiguous_source_spans"],
                "boundary_split_spans":structural["boundary_split_spans"],
                "all_locatable_span_chunks_selected":
                    structural["all_locatable_span_chunks_selected"],
                "adjacent_pairs_selected":adj_selected,
                "total_available_adjacent_pairs":adj_total,
                "repeated_content_fivegram_fraction":repetition_fraction(kept),
                "scoring_seconds":round(scoring_seconds,6),
                "selection_seconds":round(select_sec,6),
                "render_seconds":round(render_sec,6),
                "added_vs_control":len(selected_ids-base_ids),
                "lost_vs_control":len(base_ids-selected_ids),
            }
            rows.append(data)
            diagnostics.append({
                "case_id":full["case_id"],"paper_id":paper,"method":method,
                "gold_span_count":len(evidence),
                "source_provenance":structural,
                "added_chunk_ids":sorted(selected_ids-base_ids),
                "removed_chunk_ids":sorted(base_ids-selected_ids),
                "selected_chunk_ids":sorted(selected_ids),
            })
        if i%10==0 or i==len(group_cases):
            print(f"{i}/{len(group_cases)} complete; elapsed={time.perf_counter()-total_t0:.1f}s",flush=True)

    by_case=defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["method"]]=row
    if any(set(v)!=set(METHODS) for v in by_case.values()):
        raise AssertionError("Missing paired method rows")
    def avg(method,field):
        arr=[r[field] for r in rows if r["method"]==method and r[field] is not None]
        return round(sum(arr)/len(arr),6) if arr else None
    overview=[]
    for method in METHODS:
        r=[x for x in rows if x["method"]==method]
        overview.append({
            "method":method,"n":len(r),
            "avg_gold_token_recall":avg(method,"gold_token_recall"),
            "avg_gold_5gram_recall":avg(method,"gold_fivegram_recall"),
            "all_gold_spans_present_cases":
                sum(x["all_evaluable_gold_spans_present"] is True for x in r),
            "all_locatable_span_chunks_selected_cases":
                sum(x["all_locatable_span_chunks_selected"] is True for x in r),
            "avg_rendered_tokens":avg(method,"rendered_tokens"),
            "avg_budget_used_fraction":avg(method,"budget_used_fraction"),
            "avg_repeated_content_5gram_fraction":
                avg(method,"repeated_content_fivegram_fraction"),
            "avg_adjacent_pairs_selected":avg(method,"adjacent_pairs_selected"),
            "avg_selection_seconds":avg(method,"selection_seconds"),
            "avg_render_seconds":avg(method,"render_seconds"),
            "header_trimmed_cases":sum(x["trimmed_for_headers"]>0 for x in r),
            "budget_violations":sum(x["rendered_tokens"]>x["budget_tokens"] for x in r),
        })
    def paired(method,field):
        x=[v[method][field] for v in by_case.values()]
        y=[v["control"][field] for v in by_case.values()]
        return paired_bootstrap(x,y)
    comparisons={method:{
        "fivegram":paired(method,"gold_fivegram_recall"),
        "token":paired(method,"gold_token_recall"),
    } for method in METHODS if method!="control"}
    phase1_path=ROOT/"submission/results/phase1_evidence_diagnostics_20261011/qasper_cases.csv"
    with phase1_path.open(encoding="utf8",newline="") as f:
        losses=[x for x in csv.DictReader(f) if x["outcome"]=="tokenpack_loss"]
    loss_review=[{
        "case_id":x["case_id"],"historical_diagnostic":x["candidate_failure"],
        "control_5gram":by_case[x["case_id"]]["control"]["gold_fivegram_recall"],
        "neighbor_5gram":by_case[x["case_id"]]["neighbor"]["gold_fivegram_recall"],
        "rag_5gram":by_case[x["case_id"]]["rag-budgeted"]["gold_fivegram_recall"],
    } for x in losses if x["case_id"] in by_case]
    changed=[{
        "case_id":cid,
        "control_5gram":m["control"]["gold_fivegram_recall"],
        "neighbor_5gram":m["neighbor"]["gold_fivegram_recall"],
        "difference":round(m["neighbor"]["gold_fivegram_recall"]-
                            m["control"]["gold_fivegram_recall"],6),
        "added_chunks":m["neighbor"]["added_vs_control"],
        "lost_chunks":m["neighbor"]["lost_vs_control"],
    } for cid,m in by_case.items()
       if m["neighbor"]["gold_fivegram_recall"]!=m["control"]["gold_fivegram_recall"]]
    summary={
        "scope":"Paired, offline QASPER lexical gold-evidence audit; NOT answer accuracy",
        "cases":len(by_case),"papers":len(paper_cache),
        "embedding_model":"sentence-transformers/all-MiniLM-L6-v2 (local only)",
        "tokenizer":"cl100k_base (actual, no fallback)",
        "budget":"Same per-question archived 50% maximum; contexts not artificially padded",
        "source":"QASPER archived full-document contexts re-chunked into shared pool",
        "gold_used_in_selection":False,
        "limitations":[
            "Development set; not independent validation.",
            "Reconstructed source paragraphs/chunk boundaries differ from historical QASPER sections.",
            "Lexical 5-grams and source offsets are proxies, not semantic entailment.",
            "Source-span audit excludes ambiguous repeated occurrences and uses exact source offsets.",
            "Timing is local CPU/cache-dependent; common preprocessing is shared across methods.",
            "Grouping can influence downstream answer use but NOT tested without inference.",
        ],
        "methods":overview,"paired_vs_control":comparisons,
        "neighbor_changed_5gram_cases":changed,"phase1_historical_losses":loss_review,
        "source_spans_boundary_split_total":
            sum(x["boundary_split_spans"] for x in rows if x["method"]=="control"),
        "source_spans_uniquely_located_total":
            sum(x["unique_source_spans"] for x in rows if x["method"]=="control"),
        "source_spans_ambiguous_total":
            sum(x["ambiguous_source_spans"] for x in rows if x["method"]=="control"),
        "shared_preparation_seconds":{
            "block_and_chunk_embeddings":round(sum(x[3] for x in paper_cache.values()),3),
            "chunking":round(sum(x[4] for x in paper_cache.values()),3),
        },
        "overall_elapsed_seconds":round(time.perf_counter()-total_t0,3),
    }
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"case_metrics.csv",rows)
    with (out/"chunk_and_span_diagnostics.jsonl").open("w",encoding="utf8") as f:
        for record in diagnostics:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    lines=[
        "# QASPER Faz 2 — çevrimdışı eşleştirilmiş ek tanı",
        "",
        f"**Kapsam:** {len(by_case)} soru; {len(paper_cache)} makale. Üretici LLM, GPU ve API yok.",
        "Veri geliştirme setidir; gold yalnızca seçimden sonra değerlendirmede kullanıldı.",
        "Bu rapor cevap doğruluğu değil, sözcüksel kanıt koruma vekil ölçütüdür.",
        "",
        "| Yöntem | Token recall | 5-gram recall | Tüm gold spanlar | Ortalama çıktı tokenı | Başlık yüzünden trim |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in overview:
        lines.append(f"| {x['method']} | {100*x['avg_gold_token_recall']:.2f}% | "
            f"{100*x['avg_gold_5gram_recall']:.2f}% | {x['all_gold_spans_present_cases']}/{x['n']} | "
            f"{x['avg_rendered_tokens']:.1f} | {x['header_trimmed_cases']} |")
    lines += ["","## Kontrole karşı eşleştirilmiş farklar",
        "Birim: soru başına 5-gram recall farkı. Bootstrap %95 GA; işaret testi keşifseldir.","",
        "| Yöntem | Ortalama fark (yüzde puan) | %95 GA (yp) | Kazanım / Kayıp / Eşit | İşaret p |",
        "|---|---:|---:|---:|---:|"]
    for method,c in comparisons.items():
        z=c["fivegram"]
        ci=z["bootstrap_ci95"]
        lines.append(f"| {method} | {100*z['mean_difference']:+.2f} | "
            f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] | "
            f"{z['wins']}/{z['losses']}/{z['ties']} | {z['two_sided_sign_p_exploratory']:.3f} |")
    lines += ["","## Komşulukla değişen sorular","",
              "| Soru | Kontrol 5-gram | Komşu 5-gram | Fark (yp) | Yeni chunk | Düşen chunk |",
              "|---|---:|---:|---:|---:|---:|"]
    for x in changed:
        lines.append(f"| {x['case_id']} | {x['control_5gram']:.3f} | "
            f"{x['neighbor_5gram']:.3f} | {100*x['difference']:+.2f} | "
            f"{x['added_chunks']} | {x['lost_chunks']} |")
    lines += ["","## Faz 1'deki 5 tarihsel TokenPack kaybı","",
              "| Soru | Ön tanı | Kontrol | Komşu | Bütçeli RAG |",
              "|---|---|---:|---:|---:|"]
    for x in loss_review:
        lines.append(f"| {x['case_id']} | {x['historical_diagnostic']} | "
            f"{x['control_5gram']:.3f} | {x['neighbor_5gram']:.3f} | "
            f"{x['rag_5gram']:.3f} |")
    lines += [
        "", "## Parçalama ve maliyet",
        f"- Tekil kaynakta bulunabilen kanıt spanları: {summary['source_spans_uniquely_located_total']}",
        f"- Parça sınırında bölünme (konumsal vekil): {summary['source_spans_boundary_split_total']}",
        f"- Birden fazla konumda geçen, belirsiz span: {summary['source_spans_ambiguous_total']}",
        f"- Ön işlem (makale embedding önbelleği): {summary['shared_preparation_seconds']['block_and_chunk_embeddings']} saniye.",
        f"- Parçalama süresi: {summary['shared_preparation_seconds']['chunking']} saniye.",
        "",
        "## Metodolojik sınırlar ve karar",
        "- Hepsi aynı soru / kaynak / tokenizer / token tavanında üretilen yeni kollardır.",
        "- Arşivdeki önceki bağlamlar farklı bölümlemeden ve sıklıkla limit aşımından dolayı eşdeğer karşılaştırma değildir.",
        "- Gold spanların yerel sözcüksel bulunması doğru cevap çıkarılabileceği anlamına gelmez.",
        "- Komşuluk, semantik kanıt zinciri kurmaz. Gruplama kalite farkını ancak ileride izin verilen inference ile test edebiliriz.",
        "- Parametre değiştirilirse geliştirme verisindeki kazanım bağımsız kanıt sayılmaz.",
        "",
    ]
    (out/"README_RESULTS.md").write_text("\n".join(lines),encoding="utf8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4.4))
        labels=[x["method"] for x in overview]
        vals=[100*x["avg_gold_5gram_recall"] for x in overview]
        bars=ax.barh(labels,vals)
        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_xlabel("Mean lexical gold evidence 5-gram recall (%)")
        ax.set_title(f"QASPER Phase 2: {len(by_case)} paired offline questions")
        for rect,v in zip(bars,vals):
            ax.text(v+1,rect.get_y()+rect.get_height()/2,f"{v:.1f}%",va="center",fontsize=9)
        fig.tight_layout()
        fig.savefig(out/"evidence_recall.png",dpi=155)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; skipped PNG graph",flush=True)
    print("RESULT "+json.dumps({
        "output":str(out),"papers":len(paper_cache),"cases":len(by_case),
        "overview":overview,"neighbor_comparison":comparisons["neighbor"],
        "changed_cases":changed,
        "span_boundary":summary["source_spans_boundary_split_total"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases",type=int,choices=[50,200],default=50)
    ap.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=ap.parse_args()
    do_run(args.cases,args.output_dir)
