"""Posthoc full Phase4C3 validation report. No inference or model training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

RESULT=Path(__file__).resolve().parents[2]/"submission/results/phase4c3_rollout_full_train_validation889_20261011"


def main():
    d=json.loads((RESULT/"summary.json").read_text(encoding="utf8"))
    if d["train_questions"]!=2102 or d["validation_questions"]!=889:
        raise ValueError("Full data required for report")
    with (RESULT/"paired_contexts.csv").open(encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=889*2*5:
        raise AssertionError("Missing validation records")
    breaches=sum(int(r["actual_tokens"])>int(r["budget"]) for r in rows)
    if breaches:raise AssertionError("Final-render cap violations")
    state=d["train_rollout_states_by_selected_count"]
    count_high=sum(v for k,v in state.items() if int(k)>3)
    state_total=sum(state.values())
    conditional=[
        "selected_count_fraction","selected_budget_fraction",
        "mean_selected_hybrid_score","new_query_terms_fraction",
        "candidate_new_5grams_fraction","max_text_jaccard_to_selected",
        "max_embedding_cosine_to_selected","max_source_block_containment",
    ]
    state_imp=sum(d["feature_importances"].get(k,0) for k in conditional)
    all_imp=sum(d["feature_importances"].values())
    audit={
        "paired_records":len(rows),"budget_breaches":breaches,
        "states_after_three":count_high,"total_train_states":state_total,
        "states_after_three_fraction":round(count_high/max(1,state_total),6),
        "conditional_importance_fraction":round(state_imp/max(1,all_imp),6),
        "conditional_importance_count":state_imp,
        "total_importance_count":all_imp,
    }
    (RESULT/"DIAGNOSTIC_POSTHOC.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    names={
        "manual-render":"Manuel evidence-hybrid + render-aware",
        "lambdamart-render":"Eski LightGBM LambdaMART + render-aware",
        "old-marginal-conditional":"Faz 4C ilk dinamik regresyon",
        "rollout-marginal-static":"4C3 bütçeli-rollout eğitim / statik",
        "rollout-marginal-conditional":"4C3 bütçeli-rollout eğitim / dinamik",
    }
    lines=[
        "# TokenPack Faz 4C3 — Gerçek Bütçeli Rollout Durumlarında Marjinal Öğrenme",
        "",
        "**QASPER train: 2.102 ölçülebilir soru / 888 makale; "
        "QASPER validation: 889 ölçülebilir soru / 276 makale. "
        "Resmî QASPER test bölümü kullanılmadı.**",
        "",
        "## Araştırma sorusu",
        "",
        "Faz4C'nin eski regresörü sadece 0,1,3 parça seçilmiş manuel skor "
        "prefixlerinden öğreniyordu. Yeni 4C3, aynı 650-token native aday "
        "havuzunda manuel skorlu, gerçek bütçeye uygun seçim izlerinden "
        "0,1,2,3,5,7,9,12 ve iz sonu prefixleri oluşturdu. Eylem "
        "etiketleri yalnızca kalan tahmini token kapasitesine sığan adaylarda "
        "oluşturuldu. Train hedefi, alternatif gold anotasyonlar arasından "
        "en iyi tek alternatifin toplam 5-gram coverage'ındaki koşullu artış:",
        "",
        "max_group coverage(group, selected + candidate) "
        "- max_group coverage(group, selected).",
        "",
        "Gold bilgi sadece QASPER train etiketi üretilirken kullanılır. "
        "Train rollout, validation scoring ve selection gold-blind. "
        "Özellik sayısı ve LightGBM hiperparametreleri ilk Faz4C ile aynı; "
        "ancak bu ablation'da hem **train durum dağılımı hem de etiket formülü** "
        "değişmiştir, farkın tek bir nedene ait olduğu ileri sürülemez.",
        "",
        "## Gold evidence 5-gram recall",
        "",
        "| Yöntem | %25 bütçe | %50 bütçe |",
        "|---|---:|---:|",
    ]
    for key,name in names.items():
        pair=[next(x for x in d["results"] if x["ratio"]==ratio)["arms"][key]
              for ratio in (.25,.5)]
        lines.append(f"| {name} | %{100*pair[0]['gold_5gram_mean']:.2f} "
                     f"| %{100*pair[1]['gold_5gram_mean']:.2f} |")
    lines.extend([
        "",
        "## Makale-kümeli bootstrap %95 güven aralığı, eşleştirilmiş",
        "",
        "| Bütçe | Yeni − eski | Ortalama fark (yp) | %95 GA (yp) | "
        "Kazanç/kayıp/eşit |",
        "|---|---|---:|---:|---:|",
    ])
    for r in d["results"]:
        for contrast,vals in r["contrasts"].items():
            lo,hi=vals["paper_cluster_bootstrap"]["cluster_bootstrap_ci95"]
            lines.append(
                f"| %{100*r['ratio']:.0f} | {contrast} | "
                f"{100*vals['mean_difference']:+.2f} | "
                f"[{100*lo:+.2f}; {100*hi:+.2f}] | "
                f"{vals['wins']}/{vals['losses']}/{vals['ties']} |")
    lines.extend([
        "",
        "## Eğitim kapsamı ve gerçek bütçe bütünlüğü",
        "",
        f"- Kullanılan train sorusu: {d['train_questions']}.",
        f"- Üretilen rollout: {d['train_action_audit']['rollout_count']}; "
        f"durumlar: {state_total}; etiketli feasible aday–durum satırı: "
        f"{d['train_action_audit']['training_action_rows']}.",
        f"- Pozitif marjinal etiket sayısı: "
        f"{d['train_action_audit']['positive_action_labels']}.",
        f"- Önceden 3 chunk'tan fazla seçilmiş training state: "
        f"{count_high}/{state_total} (%{100*audit['states_after_three_fraction']:.2f}).",
        f"- Yeni modelin selected-set özelliklerinin ağaç split-importance "
        f"payı: %{100*audit['conditional_importance_fraction']:.2f}. "
        "Bu split sayısı nedensel feature importance değildir.",
        f"- Validation bağlamı: {len(rows)}, gerçek cl100k_base rendered "
        f"token bütçe ihlali: {breaches}.",
        "",
        "## Deney sınırları",
        "",
        "- Ana metric lexical gold evidence 5-gram recall'dur; "
        "LLM cevap doğruluğu / semantik entailment değildir.",
        "- Validation daha önce çok kez incelenmiştir. Bu sonuç "
        "KEŞİFSELDİR, yeni bağımsız hold-out değildir. Faz3D testine dokunulmadı.",
        "- Rollout gerçekten token-bütçeli olsa da MANUEL gold-blind "
        "politikanın izidir; yeni öğrenilen politikanın on-policy dağılımı "
        "ile tam eşleşmez. DAgger uygulanmadı.",
        "- Training eylem fizibilitesi tek-chunk maliyetlerini toplayan "
        "yaklaşık filtredir; inference son render'ın gerçek tokenını "
        "ayrıca sayıp kesin sınır uygular.",
        "- Frozen Phase4C/LightGBM/manüel geçmiş sonuçları arşivden "
        "alındı ve bu aşamada yeniden hesaplanmadı. Yeni seçici gecikmesi "
        "ayrı; eski yöntemler için raporlanmayan süre 0 olarak kaydedilmiş "
        "olsa da karşılaştırılabilir gerçek CPU süresi sayılmaz.",
        "- Ağ, GPU, generative LLM, ücretli API yok; üretim kodu ve "
        "eski deney sonuçları değiştirilmedi.",
        "",
        "## Artefaktlar",
        "",
        "- summary.json: bütün parametreler, split, feature ve CI özeti.",
        "- rollout_marginal_regressor.model.txt: yeni deneysel model.",
        "- paired_contexts.csv: vaka bazında 5 kol × 2 bütçe.",
        "- selected_chunks.jsonl: seçilen chunk ID'leri.",
        "- DIAGNOSTIC_POSTHOC.json: training state ve maliyet denetimi.",
        "- phase4c3_rollout_comparison.png: sonuç grafiği.",
        "- Kod: phase4c3_rollout_core.py, phase4c3_rollout_train_eval.py.",
    ])
    (RESULT/"PHASE4C3_TURKCE_RAPOR.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=list(range(len(names)))
    fig,ax=plt.subplots(figsize=(10.5,5.7))
    for offset,ratio in ((-.19,.25),(.19,.5)):
        arm=next(v for v in d["results"] if v["ratio"]==ratio)["arms"]
        ax.bar([z+offset for z in x],
               [100*arm[n]["gold_5gram_mean"] for n in names],.36,
               label=f"{100*ratio:.0f}% max budget")
    ax.set_ylim(0,100)
    ax.set_xticks(x,[
        "Manual","LambdaMART","Old conditional","Rollout static","Rollout conditional"])
    ax.set_ylabel("Gold evidence 5-gram recall (%)")
    ax.set_title("Phase4C3, 889 QASPER validation questions (development only)")
    ax.legend()
    fig.text(.5,.035,"Gold lexical evidence coverage, not answer accuracy; final token maxima equal.",
             ha="center",fontsize=9)
    fig.subplots_adjust(bottom=.16)
    fig.savefig(RESULT/"phase4c3_rollout_comparison.png",dpi=165)
    plt.close(fig)
    print("REPORT_READY",RESULT/"PHASE4C3_TURKCE_RAPOR.md",
          json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
