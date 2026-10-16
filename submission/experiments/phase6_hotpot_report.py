"""Read-only final Phase6 HotpotQA report and figure; never reruns the model."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"submission/results/phase6_hotpotqa_preregistered512_20261011"
PROTOCOL=ROOT/"submission/experiments/phase6_hotpot_protocol.py"


def pp(value):
    return f"{value*100:+.2f}".replace(".",",")


def pct(value):
    return f"%{value*100:.2f}".replace(".",",")


def main():
    d=json.loads((OUTPUT/"summary.json").read_text(encoding="utf8"))
    if d["sample_size"]!=512:raise AssertionError("512 preregistered sample required")
    with (OUTPUT/"paired_contexts.csv").open(encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=4096:raise AssertionError("Missing arms or question-budget pairs")
    if any(int(r["actual_tokens"])>int(r["budget"]) for r in rows):
        raise AssertionError("Real token cap violated")
    if any(r["source_support_complete"]!="True" for r in rows):
        raise AssertionError("Gold support sentence unavailable in source")
    groups=defaultdict(dict)
    for row in rows:
        groups[row["case_id"],float(row["ratio"])][row["method"]]=row
    if len(groups)!=1024 or any(len(x)!=4 for x in groups.values()):
        raise AssertionError("Not fully question-paired")
    extra={}
    for ratio in (.25,.5):
        extra[str(ratio)]={}
        for method in d["methods"]:
            factoid=[x[method] for (cid,r),x in groups.items()
                     if r==ratio and x[method]["answer_string_eligible"]=="True"]
            both=sum(
                x["all_support_sentences"]=="1" and
                x["answer_string_present"]=="1" for x in factoid)
            extra[str(ratio)][method]={
                "eligible_factoid_questions":len(factoid),
                "all_support_and_answer_string_count":both,
                "all_support_and_answer_string_rate":round(
                    both/max(1,len(factoid)),6),
            }
    audit={
        "protocol_sha256":hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "benchmark_sha256":d["benchmark_sha256"],
        "sample_sha256":d["sample_hash_sha256"],
        "questions":512,
        "arm_context_rows":len(rows),
        "full_source_gold_available_cases":512,
        "budget_violations":0,
        "posthoc_joint_extractive_proxy":extra,
    }
    (OUTPUT/"POSTHOC_AUDIT.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    names={
        "manual-650":"Manuel TokenPack (650 token)",
        "lambdamart-650":"QASPER LambdaMART (650 token)",
        "manual-300":"Manuel TokenPack (300 token)",
        "msmarco-zero-shot-300":"MS MARCO pretrained (300 token)",
    }
    def arm(ratio,method):
        return next(x for x in d["results"] if x["ratio"]==ratio)["arms"][method]
    lines=[
        "# Faz 6 — QASPER dışı HotpotQA kanıt sınaması",
        "",
        "**11 Ekim 2026. Önceden sabitlenmiş 512 soru, HotpotQA distractor validation.**",
        "",
        "Amaç: TokenPack'in QASPER'da elde ettiği kazanım, farklı Wikipedia "
        "çok-adımlı kanıt sorularında da var mı? Her soruya verilen yaklaşık "
        "10 aday Wikipedia pasajını aynı maksimum gerçek cl100k_base token "
        "bütçesinde seçiyoruz. Bu full-Wikipedia retrieval DEĞİLDİR.",
        "",
        "## Birincil ölçüt: Tüm destekleyici cümleler çıktı bağlamında",
        "",
        "| Yöntem | %25 bütçe | %50 bütçe |",
        "|---|---:|---:|",
    ]
    for method,name in names.items():
        lines.append(
            f"| {name} | {pct(arm(.25,method)['strict_all_support_rate'])} "
            f"| {pct(arm(.5,method)['strict_all_support_rate'])} |")
    lines+=["","## Question-bootstrap eşleştirilmiş nominal %95 güven aralıkları",
            "","| Yeni − kontrol | %25 fark ve GA (yüzde puan) | %50 fark ve GA |",
            "|---|---:|---:|"]
    for old in ("manual-650","lambdamart-650","manual-300"):
        values=[]
        for ratio in (.25,.5):
            r=next(x for x in d["results"] if x["ratio"]==ratio)
            info=r["contrasts"]["msmarco-zero-shot-300 minus "+old]["strict_all_support"]
            lo,hi=info["question_bootstrap_ci95"]
            values.append(f"{pp(info['mean_difference'])} "
                          f"[{pp(lo)}; {pp(hi)}]")
        lines.append(
            f"| MS MARCO 300 − {names[old]} | {values[0]} | {values[1]} |")
    lines+=["",
            "İlk kazanç: MS MARCO 300, manuel 650'ye kıyasla %25 sınırda "
            "+6,25 yüzde puan (GA [+3,13; +9,77]). %50'de +3,71 puan "
            "(GA [−0,20; +7,62]); bu aralık sıfırı içeriyor.",
            "",
            "## İkincil ölçüt: Kısmi kanıt ve factoid cevap metni",
            "",
            "| Yöntem | %25 cümle recall | %50 cümle recall | "
            "%25 cevap metni | %50 cevap metni |",
            "|---|---:|---:|---:|---:|"]
    for method,name in names.items():
        a=arm(.25,method);b=arm(.5,method)
        lines.append(
            f"| {name} | {pct(a['support_sentence_recall'])} "
            f"| {pct(b['support_sentence_recall'])} "
            f"| {pct(a['extractive_answer_string_rate'])} "
            f"| {pct(b['extractive_answer_string_rate'])} |")
    lines+=["",
            "Cevap metni varlığı sadece 479 factoid soru için hesaplanmıştır; "
            "yes/no cevapları hariç. Bu, cevap doğruluğu ya da çıkarım "
            "yeterliliği DEĞİLDİR.",
            "",
            "## Ek keşifsel gösterge: Tüm kanıtlar VE cevap metni",
            "",
            "| Yöntem | %25 bütçe | %50 bütçe |",
            "|---|---:|---:|"]
    for method,name in names.items():
        a=extra["0.25"][method]["all_support_and_answer_string_rate"]
        b=extra["0.5"][method]["all_support_and_answer_string_rate"]
        lines.append(f"| {name} | {pct(a)} | {pct(b)} |")
    lines+=["",
            "## Veri bütünlüğü ve sınırlar",
            "",
            f"- HotpotQA validation dataset revision: {d['benchmark_revision']}.",
            f"- Parquet SHA-256: {d['benchmark_sha256']}.",
            f"- Sabit örneklem ID SHA-256: {d['sample_hash_sha256']}.",
            f"- Gold destek cümleleri kaynakta bulunuyor: "
            f"{audit['full_source_gold_available_cases']}/512.",
            f"- 4 yöntem × 2 bütçe × 512 soru = 4096 çıktı kaydı; "
            "gerçek rendered token tavanı ihlali sıfır.",
            "- Eski QASPER LightGBM ve yeni MS MARCO aynı önceden "
            "dondurulmuş ağırlıklarla, HotpotQA eğitimi yapmadan kullanıldı.",
            "- Soru ID örneklemi ve yöntemler gold etiketlerinden önce sabitlendi; "
            "ilk 8 ID ayrıca teknik smoke deneyinde işlendi, "
            "ardından protokol değiştirilmedi.",
            "- Gold sadece çıktı bağlamı seçildikten sonra metrik hesaplamak "
            "için kullanıldı. QASPER'ın önceden kullanılmış testine bakılmadı.",
            "- Çok adımlı destek cümlelerinin tamamının lafzen çıktı metninde "
            "görülmesi gerçek modelin doğru cevap üreteceğini ispatlamaz.",
            "- HotpotQA distractor setting altın belgeyi yaklaşık 10 aday "
            "arasında verir; fullwiki retrieval zorluğu ölçülmedi.",
            "- Bootstrap soruları bağımsız varsayar; Wikipedia sayfaları "
            "sorular arasında tekrar edebilir. Güven aralıkları açıklayıcıdır.",
            "- Wikipedia pasajları kısa olduğundan 650 ve 300 chunk'lar "
            "büyük ölçüde benzer kaynak sınırlarında kaldı; burada 300 "
            "çözünürlüğünün tek başına ekstra yararı gösterilmedi.",
            "",
            "## Sonuç",
            "",
            "MS MARCO + 300-token, QASPER'dan bağımsız HotpotQA multi-hop "
            "kanıt koruma görevinde %25 bütçede manuel TokenPack'i geçiyor. "
            "LightGBM ise alan değişiminde manuel baseline'ın da gerisine "
            "düşüyor. Üretim varsayılanına geçmek için henüz yeterli değil: "
            "doğru cevap ölçümü, bağımsız başka benchmark ve çalışma maliyeti "
            "karşılaştırmaları gereklidir.",
            "",
            "## Dosyalar",
            "",
            "- summary.json: tüm yöntemler, sayılar, bootstrap.",
            "- paired_contexts.csv: 4096 kayıt.",
            "- question_ids.txt: sabit 512 soru.",
            "- POSTHOC_AUDIT.json: bütünlük ve ek factoid göstergesi.",
            "- phase6_hotpot_comparison.png: iki bütçe için grafik.",
            "- submission/experiments/phase6_hotpot_protocol.py: ön kayıt.",
            "- submission/experiments/phase6_hotpot_eval.py: offline GPU değerlendirmesi.",
            "- submission/data/phase6_hotpotqa_validation_20261011: "
            "Hugging Face HotpotQA verisi, CC BY-SA 4.0.",
    ]
    (OUTPUT/"PHASE6_TURKCE_RAPOR.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10.3,5.7))
    x=list(range(len(names)))
    for off,ratio in ((-.18,.25),(.18,.5)):
        scores=[
            100*arm(ratio,m)["strict_all_support_rate"]
            for m in names]
        bar=ax.bar([i+off for i in x],scores,.35,
                   label=f"{ratio:.0%} of source token cap")
        for r in bar:
            ax.text(r.get_x()+r.get_width()/2,r.get_height()+.8,
                    f"{r.get_height():.1f}%",ha="center",fontsize=8)
    ax.set_xticks(x,["Manual 650","LambdaMART 650",
                     "Manual 300","MS MARCO 300"])
    ax.set_ylim(0,100)
    ax.set_ylabel("All annotated support sentences present (%)")
    ax.set_title("Phase6 HotpotQA distractor validation — 512 questions")
    ax.legend(loc="upper left")
    fig.text(.5,.025,
             "Provided Wikipedia distractor candidates; not answer accuracy or fullwiki retrieval.",
             ha="center",fontsize=8.5)
    fig.subplots_adjust(bottom=.15)
    fig.savefig(OUTPUT/"phase6_hotpot_comparison.png",dpi=165)
    plt.close(fig)
    print("PHASE6_REPORT_READY",json.dumps(audit,ensure_ascii=False),
          flush=True)


if __name__=="__main__":
    main()
