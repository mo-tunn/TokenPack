"""Paired posthoc MuSiQue Qwen27B real-answer analysis and Turkish report.

Never invokes an inference endpoint; only complete 64-question frozen
prediction file and local gold scores. 10k paired question bootstrap;
exact two-sided McNemar test, aliases recognized by reader scoring.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"submission/results/phase9_musique_qwen27b_reader64_20261012"
PARENT=ROOT/"submission/results/phase9_musique_bridge_new256_20261012"
N=64


def pct(value):
    return f"{100*value:.2f}".replace(".",",")


def pp(value):
    return f"{100*value:+.2f}".replace(".",",")


def interval(values):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261012)
    draws=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
    return [round(float(x),6) for x in np.quantile(draws,[.025,.975])]


def run():
    raw=json.loads((OUT/"QWEN_READER_EM_F1.json").read_text(encoding="utf8"))
    if raw["questions"]!=N or len(raw["case_metrics"])!=N*3:
        raise AssertionError("Need complete MuSiQue reader")
    by=defaultdict(dict)
    for x in raw["case_metrics"]:
        by[x["case_id"]][x["arm"]]=x
    if len(by)!=N or any(set(x)!={
            "full-source","atomic-rank","crossdoc-bridge"} for x in by.values()):
        raise AssertionError("Unpaired answers")
    cases=list(by.values())
    old=np.asarray([x["atomic-rank"]["EM"] for x in cases])
    new=np.asarray([x["crossdoc-bridge"]["EM"] for x in cases])
    full=np.asarray([x["full-source"]["EM"] for x in cases])
    oldf=np.asarray([x["atomic-rank"]["F1"] for x in cases])
    newf=np.asarray([x["crossdoc-bridge"]["F1"] for x in cases])
    win=int(((new==1)&(old==0)).sum())
    loss=int(((new==0)&(old==1)).sum())
    discordant=win+loss
    p=binomtest(min(win,loss),discordant,.5).pvalue if discordant else 1.
    sub=full==1
    annot=json.loads((PARENT/"EXTERNAL_MUSIQUE_RESULTS.json").read_text(encoding="utf8"))
    saved=[json.loads(s) for s in (OUT/"remote_answers.jsonl")
           .read_text(encoding="utf8").splitlines() if s.strip()]
    paid=[r for r in saved if r.get("paid_remote_call")]
    result={
        "benchmark":"MuSiQue public validation, 64 ID-only questions from the new 256",
        "model":raw["model"],"questions":N,
        "answers":raw["arms"],
        "same_real_output_maximum_budget":True,
        "same_model_and_prompt":True,
        "gold_asked_remotely":False,
        "real_modal_requests":len(paid),
        "reused_identical_prompts":len(saved)-len(paid),
        "input_tokens_reported":sum(int(x.get("input_tokens") or 0) for x in paid),
        "output_tokens_reported":sum(int(x.get("output_tokens") or 0) for x in paid),
        "finish_reasons":dict(Counter(str(x.get("finish_reason")) for x in paid)),
        "paired":{
            "bridge_minus_atomic_EM":{
                "difference":round(float(np.mean(new-old)),6),
                "question_bootstrap_ci95":interval(new-old),
                "bridge_only_correct":win,"atomic_only_correct":loss,
                "both_correct":int(((new==1)&(old==1)).sum()),
                "neither_correct":int(((new==0)&(old==0)).sum()),
                "mcnemar_exact_two_sided_p":round(float(p),7),
            },
            "bridge_minus_atomic_F1":{
                "difference":round(float(np.mean(newf-oldf)),6),
                "question_bootstrap_ci95":interval(newf-oldf),
            },
            "bridge_minus_full_EM":{
                "difference":round(float(np.mean(new-full)),6),
                "question_bootstrap_ci95":interval(new-full),
            },
            "atomic_minus_full_EM":{
                "difference":round(float(np.mean(old-full)),6),
                "question_bootstrap_ci95":interval(old-full),
            },
        },
        "full_correct_subgroup":{
            "full_correct_n":int(sub.sum()),
            "bridge_retained_n":int(new[sub].sum()),
            "atomic_retained_n":int(old[sub].sum()),
            "difference":round(float(np.mean(new[sub]-old[sub])),6)
                if sub.any() else None,
            "ci95":interval(new[sub]-old[sub]) if sub.any() else None,
        },
        "previous_prospective_256_retrieval":{
            arm:{
                "all_supporting_paragraph_rate":annot["arms"][arm][
                    "all_supporting_paragraph_rate"],
                "mean_used_tokens":annot["arms"][arm]["mean_used_tokens"],
            } for arm in ("atomic-rank","crossdoc-bridge")
        },
        "limitations":[
            "Only 64 fresh MuSiQue questions; public validation, not hidden test",
            "64-question reader sample drawn from the 256 retrieval evaluation",
            "MuSiQue gold retrieval support paragraphs hit by any one sentence is a weak proxy",
            "Reader conditioned on context; pretrained general-world knowledge remains",
            "Observed differences cannot guarantee performance on QASPER or other corpora",
            "True Modal invoice/cost not audited",
        ],
    }
    target=OUT/"MU_SIQUE_QWEN_PAIRED_POSTHOC.json"
    if target.exists():raise FileExistsError("Previous reader posthoc preserved")
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                      encoding="utf8")
    es=result["paired"]["bridge_minus_atomic_EM"]
    fs=result["paired"]["bridge_minus_atomic_F1"]
    sub=result["full_correct_subgroup"]
    lines=[
        "# TokenPack Faz 9 — Belgeler arası köprü ve MuSiQue bağımsız doğrulaması",
        "",
        "**12 Ekim 2026 | MuSiQue public validation, 2.417 soru kaynağı; "
        "256 ID-only sabit örneklem, 64 Qwen27B okuyucu sorusu.**",
        "",
        "## Amaç ve deney kontrolü",
        "",
        "Faz 8'deki 152 HotpotQA sorusunun gold bilgisine dayanarak eşik "
        "optimize etmedik. Sıfır-atış MS MARCO'nun cümle skorlarına "
        "eklenen deterministik belgeler arası başlık/cümle bağlantısı "
        "ve belge çeşitliliği seçicisini önceden sabitledik. "
        "Aynı kaynak cümlelerini, aynı başlık render'ını, "
        "aynı cl100k_base %25 maksimum bütçeyi kullandık.",
        "",
        "## Yeni benchmark'ta paragraf erişimi — 256 soru",
        "",
        "| Ölçüm | Atomik sıralama | Köprü seçicisi |",
        "|---|---:|---:|",
    ]
    a=annot["arms"]["atomic-rank"]
    b=annot["arms"]["crossdoc-bridge"]
    lines.extend([
        f"| Tüm gold destek paragraflarına erişim | "
        f"%{pct(a['all_supporting_paragraph_rate'])} "
        f"| **%{pct(b['all_supporting_paragraph_rate'])}** |",
        f"| Ortalama destek paragrafı recall | "
        f"%{pct(a['mean_supporting_paragraph_recall'])} "
        f"| %{pct(b['mean_supporting_paragraph_recall'])} |",
        f"| Ortalama gerçek token | {a['mean_used_tokens']:.2f} "
        f"| {b['mean_used_tokens']:.2f} |",
        "",
        "MuSiQue gold yalnızca hangi **paragrafların** destek olduğunu "
        "etiketler. Bunlardan birinin **en az bir özgün cümlesinin** "
        "bulunmasını paragraf erişimi sayıyoruz; "
        "bu doğruluğu destekleyecek bütün ifadeler demek değildir.",
        "",
        "## Qwen 27B ile gerçek cevap doğruluğu — 64 soru",
        "",
        "| Okuyucunun gördüğü bağlam | EM doğru/64 | Exact Match | Cevap F1 |",
        "|---|---:|---:|---:|",
    ])
    for arm,label in (
        ("full-source","Tam kaynak"),
        ("atomic-rank","Eski atomik sıralama (%25)"),
        ("crossdoc-bridge","**Yeni köprü seçici (%25)**"),
    ):
        r=raw["arms"][arm]
        lines.append(
            f"| {label} | {r['correct']}/64 | %{pct(r['EM'])} "
            f"| %{pct(r['F1'])} |")
    lines.extend([
        "",
        f"**Köprü − atomik Exact Match: {pp(es['difference'])} yüzde puan.** "
        f"Nominal soru-bootstrap %95 GA "
        f"[{pp(es['question_bootstrap_ci95'][0])}; "
        f"{pp(es['question_bootstrap_ci95'][1])}]. "
        f"Yalnız köprü doğru: {win}, yalnız eski doğru: {loss}. "
        f"McNemar exact iki yönlü p={p:.5f}.",
        "",
        f"**Cevap F1 farkı: {pp(fs['difference'])} puan**, "
        f"%95 GA [{pp(fs['question_bootstrap_ci95'][0])}; "
        f"{pp(fs['question_bootstrap_ci95'][1])}].",
        "",
        "## Tam kaynak doğru sorularda doğru cevabı koruma",
        "",
        f"Tam kaynağı gören model {sub['full_correct_n']}/64 soruyu "
        f"doğru cevapladı. Bu alt kümede atomik seçici "
        f"{sub['atomic_retained_n']}, köprü seçici "
        f"{sub['bridge_retained_n']} doğru cevabı korudu. "
        f"Eşleştirilmiş fark "
        f"{pp(sub['difference']) if sub['difference'] is not None else 'tanımsız'} "
        "yüzde puan.",
        "",
        "## Bilimsel ve operasyonel sınırlar",
        "",
        "- 256 MuSiQue değerlendirme sorusu daha önce TokenPack seçiminde "
        "kullanılmamıştır. Model/parametreler bu gold'a göre ayarlanmadı; "
        "Qwen okuyucu testindeki 64 örnek ise 256 içinde yer alıyor.",
        "- Yöntem altın gold paragrafları, dekompozisyon alt-cevapları "
        "veya gold cevabı seçim sırasında görmez. İsteklerde gold yoktur.",
        "- MuSiQue public validation gizli test değildir. "
        "Daha güçlü iddia için ayrı yeni benchmark veya korunan split gerekir.",
        "- Atomik ve köprü yöntemleri aynı maksimum gerçek token bütçesini "
        "kullanır; aynı fiilî token adedi garanti değildir.",
        f"- Modal gerçek uzaktan çağrı sayısı {result['real_modal_requests']}; "
        f"aynı istem yeniden kullanımı "
        f"{result['reused_identical_prompts']}. Gerçek fatura yok.",
        "- QASPER resmî test veya arşivleri yeniden kullanılmadı; "
        "production koduna dokunulmadı.",
        "",
        "## Artefaktlar",
        "",
        "- MuSiQue veri sürümü ve SHA256: "
        "submission/data/phase9_musique_validation_20261012/PROVENANCE.json.",
        "- 256 gold-blind bağlam ve sonuçlar: "
        "submission/results/phase9_musique_bridge_new256_20261012.",
        "- 64 Qwen cevabı: remote_answers.jsonl, yerel gold: QWEN_READER_EM_F1.json.",
        "- Eşleştirilmiş rapor: MU_SIQUE_QWEN_PAIRED_POSTHOC.json.",
    ])
    (OUT/"PHASE9_MUSIQUE_TURKCE_RAPOR.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    print("MUSIQUE_READER_POSTHOC",json.dumps({
        "arms":raw["arms"],"paired":result["paired"],
        "full_correct_subgroup":sub,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()
