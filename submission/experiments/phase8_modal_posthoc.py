"""Posthoc scientific report AFTER all 152 Qwen27B answers are frozen.

Read-only; no new Modal or model call. Compares original MS MARCO 300
versus Phase8 new atomic evidence pipeline at equal actual token upper cap.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"submission/results/phase8_qwen27b_atomic152_20261012"
REF=ROOT/"submission/results/phase8_atomic_goldblind152_20261012"
FROZEN=OUT/"answer_em_f1_report.json"


def ci_paired(x,seed=20261012,rep=10000):
    a=np.asarray(x,dtype=float)
    rng=np.random.default_rng(seed)
    vals=a[rng.integers(len(a),size=(rep,len(a)))].mean(axis=1)
    return [round(float(v),6) for v in np.quantile(vals,[.025,.975])]


def pp(x):
    return f"{x*100:+.2f}".replace(".",",")


def main():
    d=json.loads(FROZEN.read_text(encoding="utf8"))
    if d["questions"]!=152 or len(d["case_metrics"])!=456:
        raise AssertionError("Missing complete 152x3 reader results")
    groups=defaultdict(dict)
    for row in d["case_metrics"]:
        groups[row["case_id"]][row["arm"]]=row
    if len(groups)!=152 or any(len(x)!=3 for x in groups.values()):
        raise AssertionError("Reader result not question-paired")
    q=list(groups.values())
    old=np.array([x["msmarco-25"]["EM"] for x in q])
    new=np.array([x["atomic-sentence-rank"]["EM"] for x in q])
    full=np.array([x["full-source"]["EM"] for x in q])
    old_f1=np.array([x["msmarco-25"]["F1"] for x in q])
    new_f1=np.array([x["atomic-sentence-rank"]["F1"] for x in q])
    wins=int(((new==1)&(old==0)).sum())
    losses=int(((new==0)&(old==1)).sum())
    both=int(((new==1)&(old==1)).sum())
    neither=int(((new==0)&(old==0)).sum())
    p=binomtest(min(wins,losses),wins+losses,.5).pvalue
    eligible=full==1
    evidence=json.loads((REF/"PHASE8_GOLD_LOCAL_RESULTS.json").read_text(encoding="utf8"))
    stat={
        "method":"Qwen3.8 27B Modal on HotpotQA dev questions, Phase8",
        "n":152,
        "answers":d["arms"],
        "paid_calls":d["calls_paid"],
        "duplicate_input_reuses":d["reused_identical_prompts"],
        "paired_differences":{
            "new_minus_old_EM":{
                "difference":round(float(np.mean(new-old)),6),
                "question_bootstrap_ci95":ci_paired(new-old),
                "new_only_correct":wins,
                "old_only_correct":losses,
                "both_correct":both,
                "both_incorrect":neither,
                "exact_McNemar_two_sided_p":round(float(p),7),
            },
            "new_minus_old_F1":{
                "difference":round(float(np.mean(new_f1-old_f1)),6),
                "question_bootstrap_ci95":ci_paired(new_f1-old_f1),
            },
            "new_minus_full_EM":{
                "difference":round(float(np.mean(new-full)),6),
                "question_bootstrap_ci95":ci_paired(new-full),
            },
            "old_minus_full_EM":{
                "difference":round(float(np.mean(old-full)),6),
                "question_bootstrap_ci95":ci_paired(old-full),
            },
        },
        "full_source_correct_subset":{
            "n":int(eligible.sum()),
            "new_retained":int(new[eligible].sum()),
            "old_retained":int(old[eligible].sum()),
            "new_minus_old_retention":round(
                float(np.mean(new[eligible]-old[eligible])),6),
            "paired_question_bootstrap_ci95":ci_paired(
                new[eligible]-old[eligible]),
        },
        "evidence_retention_development":{
            "old_all_support_rate":evidence["arms"]["msmarco-25"]["all_support_rate"],
            "new_all_support_rate":evidence["arms"]["atomic-sentence-rank"]["all_support_rate"],
            "old_mean_realized_tokens":evidence["arms"]["msmarco-25"]["mean_used_tokens"],
            "new_mean_realized_tokens":evidence["arms"]["atomic-sentence-rank"]["mean_used_tokens"],
        },
        "limits":[
            "HotpotQA public validation, 152 selected questions from prior Phase7 frozen contexts",
            "Evidence gold analysis on these 152 was observed before reader test; not an independent confirmation of method selection",
            "True answer EM and token F1 on generated Qwen27B answers, but not hidden test",
            "Both methods equal MAX rendered cl100k cap, unequal realized tokens",
            "New method uses grouped compact source headers while control uses verbose original chunk headers",
            "Answer reader endpoint may contain pretrained knowledge about Wikipedia",
            "Reader question-bootstrap nominal CI; not necessarily document-cluster independent",
        ],
    }
    file=OUT/"PHASE8_READER_PAIRED_POSTHOC.json"
    if file.exists():raise FileExistsError("Posthoc already saved")
    file.write_text(json.dumps(stat,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    dif=stat["paired_differences"]["new_minus_old_EM"]
    dif1=stat["paired_differences"]["new_minus_old_F1"]
    sub=stat["full_source_correct_subset"]
    lines=[
        "# Faz 8 — Cümle düzeyi kanıt seçiminin Qwen 27B gerçek cevap doğruluğu",
        "",
        "**12 Ekim 2026 | 152 HotpotQA distractor validation sorusu | "
        "Kamuya açık, keşifsel geliştirme verisi**",
        "",
        "## Yöntem",
        "",
        "- Aynı Modal Qwen3.8 27B okuyucu, aynı prompt ve greedy cevap üretimi.",
        "- Üç bağlam: tüm Wikipedia kaynağı, eski MS MARCO 300-token "
        "render-aware sıkıştırması (%25), yeni MS MARCO sentence-level "
        "atomic sıkıştırması (%25).",
        "- Yeni model fine-tuning yok. Kaynak cümleler özgün biçimde, "
        "kaynak başlıklarıyla ve gerçek cl100k_base bütçe sınırı altında.",
        "- Altın cevaplar model istemlerinde yoktu, 456 yanıtın tamamı "
        "dondurulduktan sonra yerel EM/F1 hesabı yapıldı.",
        "",
        "## Gerçek cevap doğruluğu",
        "",
        "| Bağlam | Doğru cevap | Exact Match | Ortalama token F1 |",
        "|---|---:|---:|---:|",
    ]
    for arm,label in (
        ("full-source","Tam kaynak"),
        ("msmarco-25","Eski MS MARCO 300"),
        ("atomic-sentence-rank","**Yeni cümle düzeyinde MS MARCO**"),
    ):
        r=d["arms"][arm]
        lines.append(
            f"| {label} | {r['correct']}/{r['n']} "
            f"| %{100*r['EM']:.2f} | %{100*r['F1']:.2f} |")
    lines.extend([
        "",
        "## Eşleştirilmiş fark ve belirsizlik",
        "",
        f"**Yeni − eski Exact Match farkı: {pp(dif['difference'])} "
        f"yüzde puan, nominal %95 eşleştirilmiş bootstrap GA "
        f"[{pp(dif['question_bootstrap_ci95'][0])}; "
        f"{pp(dif['question_bootstrap_ci95'][1])}]**.",
        f"Yalnız yeni yöntemin doğru cevapladığı: {wins}; yalnız eski "
        f"yöntemin doğru cevapladığı: {losses}; ikisinin de doğru "
        f"cevapladığı: {both}; ikisinin de yanlış cevapladığı: {neither}. "
        f"İki yönlü exact McNemar p={p:.5f}.",
        "",
        f"Yeni − eski ortalama token F1 farkı: {pp(dif1['difference'])} "
        f"puan ve nominal %95 GA "
        f"[{pp(dif1['question_bootstrap_ci95'][0])}; "
        f"{pp(dif1['question_bootstrap_ci95'][1])}].",
        "",
        "## Tam kaynakta doğru cevaplanan sorularda korunma",
        "",
        f"Tam kaynakta doğru cevaplanan {sub['n']} sorudan eski yöntem "
        f"{sub['old_retained']}, yeni atomik yöntem "
        f"{sub['new_retained']} doğru cevabı koruyor. "
        f"Koruma farkı {pp(sub['new_minus_old_retention'])} puan "
        f"[{pp(sub['paired_question_bootstrap_ci95'][0])}; "
        f"{pp(sub['paired_question_bootstrap_ci95'][1])}].",
        "",
        "## Kanıt koruma ile neden aynı şey değil?",
        "",
        f"Bu sorularda gold cümlelerin tümünün korunması eski yöntem "
        f"%{100*stat['evidence_retention_development']['old_all_support_rate']:.2f}, "
        f"yeni yöntem "
        f"%{100*stat['evidence_retention_development']['new_all_support_rate']:.2f}. "
        "Bu büyük evidence proxy kazanımının kaç puanı gerçek Qwen EM "
        "kazancına dönüştü, yukarıdaki bağımsız cevap ölçümüyle görülebilir.",
        "",
        "## Sınırlamalar",
        "",
        "- Bu 152 sorunun kanıt gold'u **okuyucu deneyinden önce "
        "analiz edildi**; dolayısıyla yöntem seçimi açısından aynı "
        "development örnekleri bağımsız gizli test sayılamaz.",
        "- Yalnızca MS MARCO 300 vs atomik yöntem için yeni Qwen ölçümü "
        "yapıldı; manuel ve belge-çeşitlendirmeli yöntemler yeniden çağrılmadı.",
        "- Her iki yöntemin azami bütçesi aynı, kullanılan gerçek token "
        "miktarı farklı. Atomik yöntem daha kompakt kaynak başlıkları "
        "kullanıyor; bu format farkı kazancın bir bileşeni olabilir.",
        "- HotpotQA distractor 10 kaynak pasajla çalışılıyor, full Wikipedia "
        "retrieval veya bilimsel makale QA genellemesi henüz ölçülmedi.",
        "- QASPER resmî test split'i ve üretim kodu kullanılmadı.",
        "",
        "## Kayıtlar",
        "",
        "- gold_blind_tasks.jsonl — 456 soru/bağlam istemi.",
        "- remote_answers.jsonl — modelin 456 cevabı.",
        "- answer_em_f1_report.json — yerel gold cevap karşılaştırması.",
        "- PHASE8_READER_PAIRED_POSTHOC.json — ikili analiz.",
        "- ../phase8_atomic_goldblind152_20261012/PHASE8_GOLD_LOCAL_RESULTS.json — "
        "önce dondurulmuş evidence seçiminin gold-local metriği.",
    ])
    (OUT/"PHASE8_TURKCE_NIHAI_RAPOR.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    print("PHASE8_READER_POSTHOC",json.dumps({
        "arms":d["arms"],"pairs":stat["paired_differences"],
        "full_correct_retention":sub,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
