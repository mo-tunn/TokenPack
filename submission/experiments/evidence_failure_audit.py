"""Offline, noncausal Phase-1 evidence failure triage. No API/model requests.

Compare archived QASPER Qwen2.5-14B tasks and judged answers; also list
paired LongBench v2 wins/losses. Do not reuse annotated gold evidence in
retrieval/model inputs. Lexical coverage is NOT proof of answerability.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission" / "results"
Q_TASKS = RESULTS / "modal_generation_eval_qasper_pilot50" / "qasper_generation_tasks.jsonl"
Q_JUDGED = RESULTS / "modal_generation_eval_qasper_pilot50" / "qasper_generation_judged_14b.jsonl"
LB_RESULTS = RESULTS / "longbench_v2_modal_hybrid_greedy_83_latency" / "longbench_generation_results.jsonl"
LB_64K_RESULTS = RESULTS / "longbench_v2_modal_64k_pilot_32k58k_production_rag" / "longbench_generation_results.jsonl"
Q_ABLATION_TASKS = RESULTS / "modal_generation_eval_qasper_ablation50" / "qasper_generation_tasks.jsonl"
Q_ABLATION_JUDGED = RESULTS / "modal_generation_eval_qasper_ablation50" / "qasper_generation_judged_14b.jsonl"
OUTPUT = RESULTS / "phase1_evidence_diagnostics_20261011"

WORDS = re.compile(r"\w+", re.UNICODE)
CHUNK_IDS = re.compile(r"\[(?:Chunk|Retrieved Chunk|Priority Chunk)\s+\d+:\s*id=([^,\]\s]+)")
REFUSAL = re.compile(
    r"insufficient (?:information|context|evidence)|not enough (?:information|context|evidence)|"
    r"cannot (?:determine|answer)|can't (?:determine|answer)|"
    r"no (?:relevant|sufficient) (?:information|evidence)",
    re.IGNORECASE,
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                data = json.loads(line)
            except ValueError as e:
                raise ValueError(f"Invalid JSON: {path}:{n}") from e
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object: {path}:{n}")
            rows.append(data)
    return rows

def index_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        case, method = str(row["case_id"]), str(row["method"])
        if method in out[case]:
            raise ValueError(f"Duplicate result: {case}/{method}")
        out[case][method] = row
    return dict(out)

def tokens(s: str) -> list[str]:
    return WORDS.findall(unicodedata.normalize("NFKC", s).casefold())

def coverage(needed: list, present: list) -> float | None:
    if not needed:
        return None
    left, right = Counter(needed), Counter(present)
    return round(sum(min(n, right[t]) for t, n in left.items()) / len(needed), 6)

def evidence_coverage(context: str, evidence_texts: list[str]) -> tuple[float | None,float | None]:
    evidence = [tokens(x) for x in evidence_texts if isinstance(x, str) and x.strip()]
    ctx = tokens(context)
    words = [w for e in evidence for w in e]
    five = [tuple(e[i:i+5]) for e in evidence for i in range(max(0,len(e)-4))]
    ctx_five = [tuple(ctx[i:i+5]) for i in range(max(0,len(ctx)-4))]
    return coverage(words,ctx), coverage(five,ctx_five)

def chunk_comparison(tokenpack_context: str, baseline_context: str) -> dict[str, Any]:
    a,b = CHUNK_IDS.findall(tokenpack_context),CHUNK_IDS.findall(baseline_context)
    sa,sb = set(a),set(b)
    return {
        "identical_context":tokenpack_context==baseline_context,
        "same_chunk_set":(sa==sb) if a and b else None,
        "same_chunk_order":(a==b) if a and b else None,
        "common_chunk_ids":len(sa & sb),
        "tokenpack_only_chunk_ids":len(sa-sb),
        "baseline_only_chunk_ids":len(sb-sa),
        "chunk_jaccard":round(len(sa & sb)/len(sa | sb),6) if (sa | sb) else None,
    }

def judged_correct(r: dict[str, Any]) -> bool | None:
    if r.get("status")!="completed" or r.get("judge_parse_failed") or r.get("correctness_0_2") is None:
        return None
    return int(r["correctness_0_2"])==2

def triage_loss(tp_ans: str, t_rec: tuple, b_rec: tuple, chunks: dict,
                *, same_prompt_variant: bool=True) -> str:
    """Candidate mechanism for *manual* investigation, not proven cause."""
    if chunks["identical_context"]:
        if not same_prompt_variant:
            return "same_context_different_answer_prompt_variant"
        return "same_context_conflicting_outcome_check_judge"
    if REFUSAL.search(tp_ans):
        return "refusal_review_evidence"
    if t_rec[0] is None or b_rec[0] is None:
        return "no_gold_evidence_annotations"
    if b_rec[0]-t_rec[0]>=0.12 or (
        b_rec[1] is not None and t_rec[1] is not None and b_rec[1]-t_rec[1]>=0.20
    ):
        return "possible_evidence_loss"
    if chunks["same_chunk_set"] is True and chunks["same_chunk_order"] is False:
        return "same_chunks_different_order"
    if t_rec[0]>=0.80 and t_rec[1] is not None and t_rec[1]>=0.45:
        return "evidence_present_answer_use_candidate"
    return "ambiguous_review_required"

def analyse_qasper(tasks: list[dict], judged: list[dict], tp_method: str="tokenpack-50",
                   baseline: str="budget-top-k-50") -> tuple[list[dict],dict]:
    t,j = index_by_case(tasks),index_by_case(judged)
    if set(t)!=set(j):
        raise ValueError("QASPER task/judged case sets do not match")
    out=[]
    for case in sorted(t):
        if not all(m in t[case] and m in j[case] for m in (tp_method,baseline)):
            continue
        ta,tb,ja,jb=t[case][tp_method],t[case][baseline],j[case][tp_method],j[case][baseline]
        a,b=judged_correct(ja),judged_correct(jb)
        if a is None or b is None:outcome="unjudged"
        elif a and not b:outcome="tokenpack_win"
        elif b and not a:outcome="tokenpack_loss"
        else:outcome="tied"
        evidence=ta.get("evidence_texts") or []
        if evidence!=(tb.get("evidence_texts") or []):
            raise ValueError(f"Gold evidence differs within QASPER case: {case}")
        ac,bc=str(ta.get("context") or ""),str(tb.get("context") or "")
        arec,brec=evidence_coverage(ac,evidence),evidence_coverage(bc,evidence)
        chunks=chunk_comparison(ac,bc)
        same_prompt_variant=ta.get("answer_prompt_variant","default")==tb.get("answer_prompt_variant","default")
        category=triage_loss(
            str(ja.get("answer") or ""),arec,brec,chunks,same_prompt_variant=same_prompt_variant
        ) if outcome=="tokenpack_loss" else "not_exclusive_loss"
        out.append({
            "case_id":case,"outcome":outcome,"candidate_failure":category,
            "tokenpack_correct":a,"baseline_correct":b,
            "tokenpack_judge_score":ja.get("correctness_0_2"),"baseline_judge_score":jb.get("correctness_0_2"),
            "tokenpack_refused":bool(REFUSAL.search(str(ja.get("answer") or ""))),
            "same_answer_prompt_variant":same_prompt_variant,
            "same_context_and_prompt_variant":chunks["identical_context"] and same_prompt_variant,
            "tokenpack_gold_token_recall":arec[0],"baseline_gold_token_recall":brec[0],
            "tokenpack_gold_5gram_recall":arec[1],"baseline_gold_5gram_recall":brec[1],
            "tokenpack_context_tokens":ta.get("context_tokens"),"baseline_context_tokens":tb.get("context_tokens"),
            **chunks,
        })
    counts=Counter(x["outcome"] for x in out)
    failure=Counter(x["candidate_failure"] for x in out if x["outcome"]=="tokenpack_loss")
    def mean(field):
        vals=[float(x[field]) for x in out if x[field] is not None]
        return round(sum(vals)/len(vals),6) if vals else None
    summary={
        "dataset":"QASPER","model":"Qwen/Qwen2.5-14B-Instruct",
        "methods":[tp_method,baseline],"cases":len(out),
        "outcomes":dict(counts),"loss_triage_candidates":dict(failure),
        "identical_contexts":sum(x["identical_context"] for x in out),
        "identical_context_same_prompt_outcome_disagreements":sum(
            x["same_context_and_prompt_variant"] and x["tokenpack_correct"]!=x["baseline_correct"]
            for x in out
        ),
        "identical_context_different_prompt_variant":sum(
            x["identical_context"] and not x["same_answer_prompt_variant"] for x in out
        ),
        "identical_context_different_prompt_outcome_disagreements":sum(
            x["identical_context"] and not x["same_answer_prompt_variant"]
            and x["tokenpack_correct"]!=x["baseline_correct"] for x in out
        ),
        "same_chunk_sets":sum(x["same_chunk_set"] is True for x in out),
        "different_chunk_sets":sum(x["same_chunk_set"] is False for x in out),
        "avg_tp_gold_token_recall":mean("tokenpack_gold_token_recall"),
        "avg_baseline_gold_token_recall":mean("baseline_gold_token_recall"),
        "avg_tp_gold_5gram_recall":mean("tokenpack_gold_5gram_recall"),
        "avg_baseline_gold_5gram_recall":mean("baseline_gold_5gram_recall"),
        "limitations":"Lexical recall is an approximate overlap measure, not evidence entailment. Categories are noncausal triage. QASPER judge is the same Qwen 14B as answer model.",
    }
    return out,summary

def analyse_longbench(rows: list[dict], tp_method: str="tokenpack-50",
                      baseline: str="production-rag-50") -> tuple[list[dict],dict]:
    cases=index_by_case(rows)
    out=[]
    for case in sorted(cases):
        methods=cases[case]
        if tp_method not in methods or baseline not in methods:
            continue
        a,b=methods[tp_method],methods[baseline]
        if a.get("model_id")!=b.get("model_id") or a.get("answer")!=b.get("answer"):
            raise ValueError(f"Unmatched model/gold in LongBench: {case}")
        ca,cb=a.get("correct"),b.get("correct")
        if not isinstance(ca,bool) or not isinstance(cb,bool):outcome="unjudged"
        elif ca and not cb:outcome="tokenpack_win"
        elif cb and not ca:outcome="tokenpack_loss"
        else:outcome="tied"
        out.append({
            "case_id":case,"outcome":outcome,
            "length":a.get("length"),"domain":a.get("domain"),"difficulty":a.get("difficulty"),
            "tokenpack_correct":ca,"baseline_correct":cb,
            "tokenpack_prediction":a.get("prediction"),"baseline_prediction":b.get("prediction"),
            "source_tokens":a.get("source_tokens"),
            "tokenpack_context_tokens":a.get("context_tokens"),
            "baseline_context_tokens":b.get("context_tokens"),
            "candidate_failure":"missing_evidence_spans_manual_review" if outcome=="tokenpack_loss" else "not_exclusive_loss",
        })
    summary={
        "dataset":"LongBench v2","model":"Qwen/Qwen2.5-14B-Instruct",
        "methods":[tp_method,baseline],"cases":len(out),
        "outcomes":dict(Counter(x["outcome"] for x in out)),
        "limitations":"Gold evidence spans unavailable: accuracy differences do not identify which evidence chain failed.",
    }
    return out,summary

def save_csv(path: Path,rows: list[dict]) -> None:
    if not rows:raise ValueError("No comparable cases")
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def write_readout(qrows: list[dict],q: dict,lrows: list[dict],l: dict) -> str:
    def count(x,k):return x["outcomes"].get(k,0)
    lines=[
        "# Faz 1 – Kanıt Kaybı ve Yanıt Hataları (offline)",
        "",
        "Eski Qwen2.5-14B yanıtları yeniden değerlendirilmiştir. Yeni GPU/API isteği yok.",
        "",
        "## QASPER: TokenPack-50 vs Budget-Top-K-50",
        "",
        f"Ortak soru: {q['cases']}; TokenPack tekil kazanım: {count(q,'tokenpack_win')}; tekil kayıp: {count(q,'tokenpack_loss')}; eşit: {count(q,'tied')}.",
        f"Birebir aynı bağlam: {q['identical_contexts']}; aynı bağlam + aynı istemle farklı doğruluk: {q['identical_context_same_prompt_outcome_disagreements']}.",
        f"Aynı chunk seti: {q['same_chunk_sets']}; farklı chunk seti: {q['different_chunk_sets']}.",
        f"Ortalama yaklaşık gold token recall: TokenPack {q['avg_tp_gold_token_recall']}, baseline {q['avg_baseline_gold_token_recall']}.",
        "",
        "### TokenPack'e özel kayıplar: olası inceleme sınıfı",
        "",
        "| Sınıf | Adet |","|---|---:|",
    ]
    lines.extend("| "+k+" | "+str(v)+" |" for k,v in sorted(q["loss_triage_candidates"].items()))
    lines+=["","### QASPER hata vakaları","","| Soru ID | Ön sınıf | TP token recall | Baseline token recall | TP 5-gram | Baseline 5-gram |","|---|---|---:|---:|---:|---:|"]
    def fmt(x):return "n/a" if x is None else f"{x:.2f}"
    for a in qrows:
        if a["outcome"]=="tokenpack_loss":
            lines.append("| "+str(a["case_id"])+" | "+str(a["candidate_failure"])+" | "+fmt(a["tokenpack_gold_token_recall"])+" | "+fmt(a["baseline_gold_token_recall"])+" | "+fmt(a["tokenpack_gold_5gram_recall"])+" | "+fmt(a["baseline_gold_5gram_recall"])+" |")
    lines+=["","## LongBench v2: TokenPack-50 vs Production-RAG-50","",
            f"Ortak soru: {l['cases']}; tekil kazanım: {count(l,'tokenpack_win')}; kayıp: {count(l,'tokenpack_loss')}; eşit: {count(l,'tied')}.",
            "LongBench sonuç kayıtlarında altın kanıt aralıkları yok; bu sorular yalnızca elle kanıt incelemesine adaydır.",
            "",
            "| Soru ID | Alan | TokenPack yanıt | RAG yanıt |","|---|---|---|---|"]
    for a in lrows:
        if a["outcome"]=="tokenpack_loss":
            lines.append("| "+str(a["case_id"])+" | "+str(a["domain"])+" | "+str(a["tokenpack_prediction"])+" | "+str(a["baseline_prediction"])+" |")
    lines+=["","## Sınırlar","",
            "Bu bir **hata ayıklama / hipotez üretme** raporudur: token ve 5-gram örtüşmesi anlam doğruluğunu kanıtlamaz; LLM-hakem bağımsız değildir; gözlenen farklar nedensellik göstermiyor.",
            "Bir sonraki adım: tekil kayıp vakalarını elle doğrula; aynı aday parçalarda sıralama, aynı sıralamada seçim ablasyonu yap; ayrılmış sorularda doğrula.",""]
    return "\n".join(lines)

def qasper_loss_casebook(tasks: list[dict], judged: list[dict], qrows: list[dict]) -> str:
    """Compact, gold-aware HUMAN review sheet, never to be used for inference."""
    t,j=index_by_case(tasks),index_by_case(judged)
    lines=[
        "# QASPER – TokenPack özel kayıp vakaları",
        "",
        "Yalnızca insan hata analizi için. Gold cevap ve evidence bilgileri hiçbir model sorgusuna aktarılmamalıdır.",
        "Her vaka için sınıflandırma adaydır; model hatası ile hakem yanılgısını ayrı kontrol et.",
        "",
    ]
    for case in qrows:
        if case["outcome"]!="tokenpack_loss":
            continue
        cid=str(case["case_id"])
        a=t[cid]["tokenpack-50"]
        ja=j[cid]["tokenpack-50"]
        jb=j[cid]["budget-top-k-50"]
        lines.extend([
            "## "+cid,
            "",
            "Soru: "+str(a.get("question") or ""),
            "",
            "Gold cevap (sadece analiz): "+str(a.get("gold_answer") or ""),
            "",
            "Aday hata sınıfı: "+str(case["candidate_failure"]),
            "",
            "TP cevabı: "+str(ja.get("answer") or "")[:650],
            "",
            "Baseline cevabı: "+str(jb.get("answer") or "")[:650],
            "",
            "TP lexical token / 5-gram recall: "+str(case["tokenpack_gold_token_recall"])+" / "+str(case["tokenpack_gold_5gram_recall"]),
            "",
            "Baseline lexical token / 5-gram recall: "+str(case["baseline_gold_token_recall"])+" / "+str(case["baseline_gold_5gram_recall"]),
            "",
            "TP-only chunk sayısı: "+str(case["tokenpack_only_chunk_ids"])+", baseline-only: "+str(case["baseline_only_chunk_ids"]),
            "",
            "Gold evidence örneği (yalnızca analiz):",
            "",
        ])
        for ev in (a.get("evidence_texts") or [])[:2]:
            lines.append("- "+str(ev).replace("\n"," ")[:480])
        lines.extend([
            "",
            "Elle incelenecek:",
            "- [ ] Kritik kaynak ifadesi TokenPack bağlamında gerçekten var mı?",
            "- [ ] Gerekli komşu veya tamamlayıcı kanıt atılmış mı?",
            "- [ ] Yanlış cevap bağlam organizasyonu / distractor kaynaklı mı?",
            "- [ ] LLM-hakem kararı özgün altın cevaba göre tutarlı mı?",
            "",
        ])
    return "\n".join(lines)

def main() -> int:
    parser=argparse.ArgumentParser(description="Offline Phase-1 failure diagnosis; never sends inference")
    parser.add_argument("--qasper-tasks",type=Path,default=Q_TASKS)
    parser.add_argument("--qasper-judged",type=Path,default=Q_JUDGED)
    parser.add_argument("--longbench-results",type=Path,default=LB_RESULTS)
    parser.add_argument("--longbench-64k-results",type=Path,default=LB_64K_RESULTS)
    parser.add_argument("--qasper-ablation-tasks",type=Path,default=Q_ABLATION_TASKS)
    parser.add_argument("--qasper-ablation-judged",type=Path,default=Q_ABLATION_JUDGED)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    args=parser.parse_args()
    qtasks=read_jsonl(args.qasper_tasks)
    qjudged=read_jsonl(args.qasper_judged)
    qrows,q=analyse_qasper(qtasks,qjudged)
    lrows,l=analyse_longbench(read_jsonl(args.longbench_results))
    longer_rows,longer=analyse_longbench(read_jsonl(args.longbench_64k_results))
    ablation_rows,ablation=analyse_qasper(
        read_jsonl(args.qasper_ablation_tasks),read_jsonl(args.qasper_ablation_judged),
        tp_method="tokenpack-score-sorted-strong-50",baseline="tokenpack-score-sorted-50",
    )
    args.output_dir.mkdir(parents=True,exist_ok=True)
    save_csv(args.output_dir/"qasper_cases.csv",qrows)
    save_csv(args.output_dir/"longbench_cases.csv",lrows)
    save_csv(args.output_dir/"longbench_64k_cases.csv",longer_rows)
    save_csv(args.output_dir/"qasper_ablation_cases.csv",ablation_rows)
    (args.output_dir/"qasper_summary.json").write_text(json.dumps(q,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    (args.output_dir/"longbench_summary.json").write_text(json.dumps(l,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    (args.output_dir/"longbench_64k_summary.json").write_text(json.dumps(longer,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    (args.output_dir/"qasper_ablation_summary.json").write_text(json.dumps(ablation,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    (args.output_dir/"phase1_readout.md").write_text(write_readout(qrows,q,lrows,l),encoding="utf8")
    (args.output_dir/"qasper_manual_review.md").write_text(qasper_loss_casebook(qtasks,qjudged,qrows),encoding="utf8")
    print(json.dumps({"qasper":q,"qasper_ablation":ablation,"longbench":l,"longbench_64k":longer,"output_dir":str(args.output_dir)},ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
