"""Full validation Phase4A clustered bootstrap and Turkish report. No test read."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci

SCALES=(150,300,650)
RATIOS=(.25,.5)
PAIRS=(
    ("manual150-minus-650",150,"render",650,"render"),
    ("manual300-minus-650",300,"render",650,"render"),
    ("oracle150-minus-650",150,"oracle",650,"oracle"),
    ("oracle300-minus-650",300,"oracle",650,"oracle"),
    ("oracle-gap650",650,"oracle",650,"render"),
    ("oracle-gap300",300,"oracle",300,"render"),
    ("oracle-gap150",150,"oracle",150,"render"),
)

def metric(record,method):
    return (record["render"]["recall"] if method=="render"
            else record["best_feasible_rendered_recall"])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dir",required=True,type=Path)
    args=parser.parse_args()
    path=args.dir
    summary=json.loads((path/"summary.json").read_text(encoding="utf8"))
    if summary["n_development_questions"]!=889:
        raise ValueError("Full validation report requires 889 questions")
    records=[json.loads(line) for line in (path/"cases.jsonl").read_text(
        encoding="utf8").splitlines() if line]
    if len(records)!=889*6:raise AssertionError("Incomplete paired cases")
    index=defaultdict(dict)
    for rec in records:
        key=(rec["case_id"],rec["budget_ratio"])
        if rec["scale"] in index[key]:raise AssertionError("Duplicate scale")
        index[key][rec["scale"]]=rec
    if any(set(v)!=set(SCALES) for v in index.values()):
        raise AssertionError("Unpaired scales")
    comparisons=[]
    failures=[]
    for ratio in RATIOS:
        sample={case:v for (case,b),v in index.items() if b==ratio}
        for name,new,ni,old,oi in PAIRS:
            pairs={
                case:{
                    "control":{"paper_id":v[old]["paper_id"],"delta":0.},
                    "new":{"paper_id":v[new]["paper_id"],
                           "delta":metric(v[new],ni)-metric(v[old],oi)},
                } for case,v in sample.items()
            }
            delta=[x["new"]["delta"] for x in pairs.values()]
            comparisons.append({
                "ratio":ratio,"comparison":name,
                "mean":round(statistics.mean(delta),6),
                "wins":sum(x>1e-10 for x in delta),
                "losses":sum(x< -1e-10 for x in delta),
                "ties":sum(abs(x)<=1e-10 for x in delta),
                "ci":paper_cluster_ci(pairs,"delta","new",replicates=4000),
            })
        failed=[r for r in sample.values() if r[650]["render"]["recall"]<1e-9]
        failures.append({
            "ratio":ratio,
            "zero_650_manual":len(failed),
            "recovered_by_150_manual":sum(
                r[150]["render"]["recall"]>0 for r in failed),
            "recovered_by_300_manual":sum(
                r[300]["render"]["recall"]>0 for r in failed),
            "reachable_by_650_oracle":sum(
                r[650]["best_feasible_rendered_recall"]>0 for r in failed),
            "reachable_by_150_oracle":sum(
                r[150]["best_feasible_rendered_recall"]>0 for r in failed),
        })
    (path/"PAPER_CLUSTER_POSTHOC.json").write_text(
        json.dumps({"comparisons":comparisons,"failure_recovery":failures},
                   indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    def pc(x):return f"{100*x:.2f}%"
    def pp(x):return f"{100*x:+.2f}"
    report=[
        "# Faz 4A — Çok Çözünürlüklü Chunking ve Kanıt Oracle Analizi",
        "",
        "**Resmî QASPER validation: 889 soru, 281 makale. Test split kullanılmadı.**",
        "",
        "Bu keşifsel analizde 150, 300, 650 hedef tokenlık yapı-farkındalıklı "
        "parçalar, aynı %25 / %50 son-render token bütçelerinde incelendi.",
        "",
        "**İki ayrı kaynak:** Manuel evidence-hybrid + Top-K veya render-aware "
        "seçici gold-blind; oracle ise gold metinlerini önceden görür. "
        "Oracle bir gerçek algoritma kazanımı değildir.",
        "",
        "## Ortalama gold evidence 5-gram recall",
        "",
        "| Bütçe | Chunk hedefi | Manuel Top-K | Manuel Render | "
        "Ulaşılmış gold-aware oracle alt sınır | Bütçesiz chunk-içi üst sınır | "
        "Kesin optimal grup çözümü |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in RATIOS:
        for size in SCALES:
            s=next(x for x in summary["per_scale"]
                   if x["scale"]==size and x["ratio"]==ratio)
            report.append(
                f"| %{100*ratio:.0f} | {size} | {pc(s['manual_topk_recall'])} | "
                f"{pc(s['manual_render_recall'])} | "
                f"{pc(s['gold_feasible_oracle_lower'])} | "
                f"{pc(s['gold_pool_internal_upper_no_budget'])} | "
                f"{s['oracle_all_group_certified']}/{s['questions']} |")
    report.extend([
        "",
        "ÖNEMLİ: Bütçesiz havuz üst sınırı yalnızca chunk içindeki gold 5-gram "
        "multiset amacı içindir, son-render lexical recall'un matematiksel "
        "üst sınırı diye kullanılamaz. Oracle'ın alt sınırı gerçek bütçeye "
        "uyan ve gold görerek bulunan somut seçimlerden gelir. Birden çok "
        "anotasyon ayrı alternatif olarak ele alınmıştır.",
        "",
        "## 276 değerlendirilebilir makale kümeli eşleştirilmiş karşılaştırmalar",
        "",
        "| Bütçe | Fark | Fark (yüzde puan) | Bootstrap %95 GA (yp) | "
        "Kazanç/Kayıp/Eşit |",
        "|---|---|---:|---:|---:|",
    ])
    for c in comparisons:
        low,hi=c["ci"]["cluster_bootstrap_ci95"]
        report.append(
            f"| %{c['ratio']*100:.0f} | {c['comparison']} | {pp(c['mean'])} | "
            f"[{pp(low)}; {pp(hi)}] | {c['wins']}/{c['losses']}/{c['ties']} |")
    report.extend([
        "",
        "## Gold 5-gram recall sıfır olan 650-token manuel sorular",
        "",
        "| Bütçe | Kontrol sıfır | 150 manuel kurtardı | 300 manuel kurtardı | "
        "650 oracle kanıt bulabildi | 150 oracle kanıt bulabildi |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in failures:
        report.append(
            f"| %{r['ratio']*100:.0f} | {r['zero_650_manual']} | "
            f"{r['recovered_by_150_manual']} | {r['recovered_by_300_manual']} | "
            f"{r['reachable_by_650_oracle']} | {r['reachable_by_150_oracle']} |")
    report.extend([
        "",
        "## Deney ve iddia sınırları",
        "",
        "- Oracle >9 ilgili chunk olan durumlarda sezgisel ULAŞILABİLİR ALT SINIR "
        "olup tam optimum değildir. Küçük aday kümelerinde chunk-içi gold "
        "gramlar üzerinde eksiksiz alt-küme taraması kullanıldı.",
        "- Oracle için gold kullanımı, deploy edilebilir scoring sonucu değildir. "
        "Skorlayıcı ve seçici gold'a erişmedi.",
        "- Paragraf/bölüm sınırları korunur; yeni chunk ölçekleri sadece deneysel. "
        "Birden çok anlaşılan evidence grubu birleştirilmedi.",
        "- Üst token limiti her yöntemde aynı; gerçek kullanılan token sayısı "
        "ve kaynak içindeki 5-gram sınırları değişebilir.",
        "- Sadece lexical evidence proxy ölçüldü. LLM cevap doğruluğu değil.",
        "- QASPER'ın daha önce araştırmada görülmüş VALIDATION split'i kullanıldı. "
        "Faz3D testine tekrar bakılmadı; bu sonuç bağımsız test değildir.",
        "- Çoklu test düzeltmesi uygulanmadı; sonuçlar keşifsel.",
        "- Üretim dosyaları ve eski modeller/raporlar değiştirilmedi; "
        "GPU, LLM, Modal, ücretli API, network kullanılmadı.",
        "",
        "## Artefaktlar",
        "",
        "- cases.jsonl: 889 soru x 3 chunk hedefi x 2 bütçe = 5.334 vaka",
        "- summary.json: özet ve sert bütçe denetimi",
        "- PAPER_CLUSTER_POSTHOC.json: vaka-kümeli bootstrap",
        "- phase4a_multires_comparison.png: karşılaştırma grafiği",
        "- phase4a_oracle_core.py ve phase4a_qasper_multires_oracle.py: deney",
    ])
    (path/"PHASE4A_TURKCE_RAPOR.md").write_text(
        "\n".join(report)+"\n",encoding="utf8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.8,5.5))
    for ratio in RATIOS:
        records=[next(r for r in summary["per_scale"]
                      if r["scale"]==s and r["ratio"]==ratio)
                 for s in SCALES]
        ax.plot(SCALES,[100*r["manual_render_recall"] for r in records],
                "o-",label=f"Blind manual {ratio:.0%}")
        ax.plot(SCALES,[100*r["gold_feasible_oracle_lower"] for r in records],
                "s--",label=f"Gold-aware feasible {ratio:.0%}")
    ax.set_xticks(SCALES)
    ax.set_xlabel("Target chunk tokens")
    ax.set_ylabel("Mean gold 5gram recall (%)")
    ax.set_ylim(0,105)
    ax.legend()
    ax.set_title("Phase4A QASPER validation, 889 questions")
    fig.text(.5,.035,"Privileged oracle feasible coverage is not deployable inference.",
             ha="center",fontsize=9)
    fig.subplots_adjust(bottom=.17)
    fig.savefig(path/"phase4a_multires_comparison.png",dpi=160)
    plt.close(fig)
    print("REPORT WRITTEN",path/"PHASE4A_TURKCE_RAPOR.md")


if __name__=="__main__":
    main()
