"""Faz 4C saved development result diagnostic. Does not run test or train."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parents[2]
OUTPUT=BASE/"submission/results/phase4c_marginal_full_train_validation889_20261011"


def main():
    data=json.loads((OUTPUT/"summary.json").read_text(encoding="utf8"))
    if data["validation_questions"]!=889:raise AssertionError("full archive needed")
    with (OUTPUT/"paired_contexts.csv").open(encoding="utf8") as f:
        records=list(csv.DictReader(f))
    seen={}
    for line in (OUTPUT/"selected_chunks.jsonl").read_text(
            encoding="utf8").splitlines():
        rec=json.loads(line)
        seen[rec["case_id"],float(rec["ratio"]),rec["method"]]=set(rec["chunks"])
    state_names=data["features"][13:]
    feature_importance=data["model_feature_importances"]
    state_total=sum(feature_importance[x] for x in state_names)
    initial_total=sum(feature_importance[x] for x in data["features"][:13])
    diagnostics=[]
    for ratio in (.25,.5):
        questions={r["case_id"] for r in records if float(r["ratio"])==ratio}
        difference=[]
        changed=0
        for cid in questions:
            static=seen[cid,ratio,"marginal-static"]
            conditional=seen[cid,ratio,"marginal-conditional"]
            difference.append(len(static^conditional))
            changed+=int(static!=conditional)
        diagnostics.append({
            "ratio":ratio,"n":len(questions),
            "changed_selection_set":changed,
            "mean_symmetric_difference":round(statistics.mean(difference),3),
        })
    audit={
        "context_dependent_feature_importance":state_total,
        "initial_feature_importance":initial_total,
        "context_dependent_feature_importance_fraction":
            round(state_total/max(1,state_total+initial_total),5),
        "context_features_with_no_importance":[
            k for k in state_names if feature_importance[k]==0],
        "selection_disagreement":diagnostics,
    }
    (OUTPUT/"DIAGNOSTIC_POSTHOC.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    lines=[
        "# Faz 4C — Koşullu Marjinal Kanıt Katkısı Öğrenme",
        "",
        "**QASPER train 2.102 soru, validation 889 soru. Bağımsız test kullanılmadı.**",
        "",
        "## Gold evidence lexical 5-gram recall (cevap doğruluğu değil)",
        "",
        "| Yöntem | %25 bütçe | %50 bütçe |",
        "|---|---:|---:|",
    ]
    names={
        "manual-render":"Manuel + render-aware",
        "lambdamart-render":"Önceki LambdaMART + render-aware",
        "marginal-static":"Marjinal model, statik",
        "marginal-conditional":"Marjinal model, dinamik",
    }
    for method,label in names.items():
        vals=[next(r for r in data["results"] if r["ratio"]==ratio)["arms"][method]
              for ratio in (.25,.5)]
        lines.append(f"| {label} | %{100*vals[0]['gold_5gram_mean']:.2f} "
                     f"| %{100*vals[1]['gold_5gram_mean']:.2f} |")
    lines.extend([
        "",
        "## Makale-kümeli %95 güven aralıkları, eşleştirilmiş",
        "",
        "| Bütçe | Kontrast | Ortalama fark (yp) | %95 GA (yp) | Kazanç/kayıp/eşit |",
        "|---|---|---:|---:|---:|",
    ])
    for r in data["results"]:
        for name,v in r["comparisons"].items():
            lo,hi=v["paper_cluster_bootstrap"]["cluster_bootstrap_ci95"]
            lines.append(
                f"| %{100*r['ratio']:.0f} | {name} | "
                f"{100*v['mean_difference']:+.2f} | "
                f"[{100*lo:+.2f}; {100*hi:+.2f}] | "
                f"{v['wins']}/{v['losses']}/{v['ties']} |")
    lines.extend([
        "",
        "## Eğitim / dağılım teşhisi",
        "",
        f"- Train 2.102 soru; {data['train_feature_shape'][0]:,} "
        f"aday ve durum örneği, {data['train_feature_shape'][1]} özellik.",
        f"- Pozitif marjinal kazanç etiket oranı: "
        f"%{100*data['train_positive_label_fraction']:.2f}.",
        f"- İlk relevance özelliklerinin ağaç split önem toplamı: {initial_total}; "
        f"koşullu seçili-küme özelliklerinin önemi: {state_total} "
        f"(yaklaşık %{100*audit['context_dependent_feature_importance_fraction']:.2f}).",
        f"- Sıfır öneme sahip durum özellikleri: "
        f"{', '.join(audit['context_features_with_no_importance'])}.",
    ])
    for row in diagnostics:
        lines.append(
            f"- %{100*row['ratio']:.0f} bütçede statik ve dinamik seçili "
            f"chunk kümeleri {row['changed_selection_set']}/{row['n']} soruda farklı. "
            f"Ortalama simetrik küme farkı {row['mean_symmetric_difference']:.3f}.")
    lines.extend([
        "",
        "## Belirlenen önemli etiket-hedef uyumsuzluğu",
        "",
        "İlk modelin etiket formülü, her kanıt alternatifi için ayrı katkı "
        "hesaplayıp marjinal katkıların en büyüğünü almak üzerine kuruldu: "
        "max_g[coverage_g(S+c)-coverage_g(S)]. Ancak nihai QASPER gold-recall "
        "metriği alternatif anotasyonlardan EN İYİSİNİ seçer: max_g coverage_g(S). "
        "Dolayısıyla tutarlı marjinal hedef, "
        "max_g coverage_g(S+c) - max_g coverage_g(S) olmalıdır. "
        "Bu eşit olmayan iki amaç, özellikle alternatif anotasyonlardan "
        "biri zaten büyük ölçüde korunmuşken sapabilir. Bu bilimsel hipotez "
        "yeni model ile yapılacak ayrı A/B eğitimde sınanmalı.",
        "",
        "## Sonuç ve sınırlar",
        "",
        "Marjinal model aynı 650-token chunk havuzunda, eski LightGBM'ye "
        "karşı istatistiksel olarak belirgin bir kazanım vermedi. "
        "Koşullu inference, bir aday seçildikçe kalan tüm adaylara yeniden "
        "tahmin çalıştırır ve CPU maliyeti artar. Eğitim durumları yalnızca "
        "gold-blind manual Top-0/Top-1/Top-3 prefixleridir; gerçek seçici "
        "rolloutlarından farklı olabilir.",
        "",
        "Bu veri daha önce kullanılan VALIDATION olduğu için sonuç "
        "keşifseldir. Yeni eğitimi test bölümünde yapmak yasaktır. "
        "Gold inference özelliklerine girmez. Final-render token bütçesi "
        "aşımı sıfır. GPU, network, LLM, Modal veya ücretli API yok. "
        "Production kodu ve eski sonuç/model dosyaları değiştirilmedi.",
        "",
        "Tekrar üretilebilir: summary.json, paired_contexts.csv, "
        "selected_chunks.jsonl, marginal_regressor.model.txt, "
        "DIAGNOSTIC_POSTHOC.json.",
    ])
    (OUTPUT/"PHASE4C_TURKCE_RAPOR.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
