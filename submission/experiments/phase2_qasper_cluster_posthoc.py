"""Source-paper-clustered sensitivity intervals for offline Phase 2.

Works only on saved CSV/JSONL audits. No model loading, gold-aware retrieval,
network, or modification to prior outputs.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[2]
INPUTS={
    "pilot50":ROOT/"submission/results/phase2_qasper_diagnostic_v2_20261011",
    "exploratory200":ROOT/"submission/results/phase2_qasper_diagnostic_v2_full200_20261011",
}
OUTPUT=ROOT/"submission/results/phase2_qasper_cluster_posthoc_20261011"
METHODS=("grouped","neighbor","neighbor-grouped","rag-budgeted")


def paper_cluster_ci(data:dict, field:str, method:str,
                     *, replicates:int=6000,seed:int=20261011) -> dict:
    """Resample papers, retaining all their questions; weighted by question."""
    papers=defaultdict(list)
    for group in data.values():
        papers[group["control"]["paper_id"]].append(
            group[method][field]-group["control"][field]
        )
    identities=sorted(papers)
    means=[]
    rng=random.Random(seed)
    for _ in range(replicates):
        sample=[papers[identities[rng.randrange(len(identities))]]
                for _ in identities]
        combined=[v for values in sample for v in values]
        means.append(sum(combined)/len(combined))
    means.sort()
    original=[v for values in papers.values() for v in values]
    return {
        "papers":len(papers),"questions":len(original),
        "mean_difference":round(sum(original)/len(original),6),
        "cluster_bootstrap_ci95":[round(means[int(.025*(replicates-1))],6),
                                  round(means[int(.975*(replicates-1))],6)],
    }


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Existing immutable posthoc analysis")
    output={}
    docs=["# QASPER Faz 2: makale kümeli bootstrap hassasiyet analizi","",
          "Çevrimdışı, yeni inference yok. Makale bazında yeniden örnekleme (6000 tekrar).",
          "Bir makaledeki birden fazla sorunun bağımsız gözlem kabul edilmesine karşı hassasiyet kontrolü.",
          "Sonuçlar geliştirme ve eski deney arşivinden; bağımsız test değildir.",""]
    for title,folder in INPUTS.items():
        with (folder/"case_metrics.csv").open(encoding="utf8",newline="") as f:
            rows=list(csv.DictReader(f))
        data=defaultdict(dict)
        for r in rows:
            data[r["case_id"]][r["method"]]={
                **r,
                "gold_fivegram_recall":float(r["gold_fivegram_recall"]),
                "gold_token_recall":float(r["gold_token_recall"]),
                "trimmed_for_headers":int(r["trimmed_for_headers"]),
                "rendered_tokens":int(r["rendered_tokens"]),
            }
        if any(set(group)!={"control",*METHODS} for group in data.values()):
            raise ValueError("Unpaired case in "+title)
        details={}
        for method in METHODS:
            details[method]={
                "fivegram":paper_cluster_ci(data,"gold_fivegram_recall",method),
                "token":paper_cluster_ci(data,"gold_token_recall",method),
            }
        by_case=defaultdict(dict)
        with (folder/"chunk_and_span_diagnostics.jsonl").open(encoding="utf8") as f:
            for line in f:
                row=json.loads(line)
                by_case[row["case_id"]][row["method"]]=row
        grouped_drops=[]
        for cid, methods in data.items():
            a,b=methods["control"],methods["grouped"]
            if b["gold_fivegram_recall"]<a["gold_fivegram_recall"]:
                orig=set(by_case[cid]["control"]["selected_chunk_ids"])
                altered=set(by_case[cid]["grouped"]["selected_chunk_ids"])
                grouped_drops.append({
                    "case_id":cid,
                    "control_fivegram":a["gold_fivegram_recall"],
                    "grouped_fivegram":b["gold_fivegram_recall"],
                    "control_trimmed":a["trimmed_for_headers"],
                    "grouped_trimmed":b["trimmed_for_headers"],
                    "removed_ids":sorted(orig-altered),
                    "added_ids":sorted(altered-orig),
                })
        output[title]={
            "cases":len(data),
            "papers":len({r["paper_id"] for r in rows}),
            "method_comparisons":details,
            "grouped_recall_losses":grouped_drops,
            "mean_scoring_seconds":round(sum(r["scoring_seconds"] if isinstance(r["scoring_seconds"],float)
                                             else float(r["scoring_seconds"])
                                             for r in rows if r["method"]=="control")/len(data),6),
        }
        docs.extend([f"## {title}: {len(data)} soru", "",
              "| Kol | 5-gram farkı (yp) | Makale kümeli %95 GA (yp) |",
              "|---|---:|---:|"])
        for method in METHODS:
            z=details[method]["fivegram"]
            lo,hi=z["cluster_bootstrap_ci95"]
            docs.append(f"| {method} | {100*z['mean_difference']:+.2f} | "
                        f"[{100*lo:+.2f}, {100*hi:+.2f}] |")
        docs.extend(["",f"Gruplamada 5-gram kaybı: {len(grouped_drops)} soru.", ""])
        for case in grouped_drops:
            docs.append(f"- {case['case_id']}: {case['control_fivegram']:.3f} → "
                        f"{case['grouped_fivegram']:.3f}, "
                        f"çıkarılan chunk {len(case['removed_ids'])}, "
                        f"eklenen chunk {len(case['added_ids'])}.")
        docs.append("")
    OUTPUT.mkdir(parents=True)
    (OUTPUT/"cluster_sensitivity.json").write_text(
        json.dumps(output,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    (OUTPUT/"README.md").write_text("\n".join(docs)+"\n",encoding="utf8")
    print(json.dumps({
        name:{
            "cases":z["cases"],"papers":z["papers"],
            "neighbor_5gram":z["method_comparisons"]["neighbor"]["fivegram"],
            "grouped_5gram":z["method_comparisons"]["grouped"]["fivegram"],
            "grouped_losses":len(z["grouped_recall_losses"]),
            "avg_scoring_seconds":z["mean_scoring_seconds"],
        } for name,z in output.items()},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
