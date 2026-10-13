"""Phase 30A: offline mechanism and heterogeneity analysis.

This script reads only the already frozen, post-gold result files from Phase 28
and the Phase 29 pilot.  It makes no remote model calls and does not alter any
experiment archive.  The pooled figures are descriptive; confirmatory claims
remain tied to the preregistered Phase 28 validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import binomtest, spearmanr


ROOT = Path(__file__).resolve().parents[2]
PHASE28 = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013"
PHASE29 = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot"
OUT = ROOT / "submission/results/phase30a_frozen_mechanism_analysis_20261013"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_mcnemar(rows: list[dict], metric: str) -> dict:
    wins = sum(r["treatment"][metric] > r["control"][metric] for r in rows)
    losses = sum(r["treatment"][metric] < r["control"][metric] for r in rows)
    discordant = wins + losses
    p = float(binomtest(min(wins, losses), discordant, 0.5).pvalue) if discordant else 1.0
    return {"win": wins, "loss": losses, "net": wins - losses,
            "two_sided_exact_mcnemar_p": p}


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def hop_family(identifier: str) -> str:
    prefix = identifier.split("__", 1)[0]
    if prefix.startswith("2hop"):
        return "2-hop"
    if prefix.startswith("3hop"):
        return "3-hop"
    if prefix.startswith("4hop"):
        return "4-hop"
    return "unknown-hop"


def make_rows(dataset: str, result_path: Path, audit_path: Path) -> tuple[list[dict], dict]:
    result = read_json(result_path)
    audit = read_json(audit_path)
    if audit.get("status") != "PASS_INDEPENDENT_RECOMPUTATION":
        raise RuntimeError(f"Frozen input did not pass independent audit: {audit_path}")
    rows: list[dict] = []
    for case in result["per_case"]:
        ident = case["id"]
        control = case["atomic-sentence-rank"]
        treatment = case["semantic-joint-citations"]
        if dataset == "HotpotQA":
            stratum = case.get("type", "unknown")
            hop = "multi-hop"
        else:
            stratum = hop_family(ident)
            hop = stratum
        row = {
            "dataset": dataset,
            "id": ident,
            "stratum": stratum,
            "hop_family": hop,
            "source_sentence_set_changed": bool(case.get("source_sentence_set_changed", False)),
            "control": {m: float(control[m]) for m in ("EM", "F1", "full_support", "support_recall")},
            "treatment": {m: float(treatment[m]) for m in ("EM", "F1", "full_support", "support_recall")},
            "control_tokens": float(control.get("actual_tokens", control.get("tokens", 0))),
            "treatment_tokens": float(treatment.get("actual_tokens", treatment.get("tokens", 0))),
        }
        row["delta"] = {m: row["treatment"][m] - row["control"][m]
                         for m in ("EM", "F1", "full_support", "support_recall")}
        row["token_delta"] = row["treatment_tokens"] - row["control_tokens"]
        row["support_gain_without_EM_gain"] = row["delta"]["support_recall"] > 0 and row["delta"]["EM"] <= 0
        row["EM_gain_without_support_gain"] = row["delta"]["EM"] > 0 and row["delta"]["support_recall"] <= 0
        row["both_support_and_EM_gain"] = row["delta"]["support_recall"] > 0 and row["delta"]["EM"] > 0
        if not row["source_sentence_set_changed"]:
            # Unchanged contexts should have identical reader prompts and outputs.
            for m in row["delta"]:
                if row["delta"][m] != 0:
                    raise AssertionError(f"Unchanged context has metric delta: {dataset} {ident} {m}")
        rows.append(row)
    return rows, {"result_sha256": sha256(result_path), "audit_sha256": sha256(audit_path),
                  "n": len(rows), "audit_status": audit["status"]}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {
        "n": n,
        "control_EM_correct": int(sum(r["control"]["EM"] for r in rows)),
        "treatment_EM_correct": int(sum(r["treatment"]["EM"] for r in rows)),
        "control_full_support": int(sum(r["control"]["full_support"] for r in rows)),
        "treatment_full_support": int(sum(r["treatment"]["full_support"] for r in rows)),
        "control_mean_F1": mean_or_zero([r["control"]["F1"] for r in rows]),
        "treatment_mean_F1": mean_or_zero([r["treatment"]["F1"] for r in rows]),
        "control_mean_support_recall": mean_or_zero([r["control"]["support_recall"] for r in rows]),
        "treatment_mean_support_recall": mean_or_zero([r["treatment"]["support_recall"] for r in rows]),
        "mean_F1_delta": mean_or_zero([r["delta"]["F1"] for r in rows]),
        "mean_support_recall_delta": mean_or_zero([r["delta"]["support_recall"] for r in rows]),
        "mean_token_delta": mean_or_zero([r["token_delta"] for r in rows]),
        "changed_contexts": int(sum(r["source_sentence_set_changed"] for r in rows)),
        "unchanged_contexts": int(sum(not r["source_sentence_set_changed"] for r in rows)),
    }
    out["control_EM_fraction"] = out["control_EM_correct"] / n
    out["treatment_EM_fraction"] = out["treatment_EM_correct"] / n
    out["control_full_support_fraction"] = out["control_full_support"] / n
    out["treatment_full_support_fraction"] = out["treatment_full_support"] / n
    out["paired_EM"] = exact_mcnemar(rows, "EM")
    out["paired_full_support"] = exact_mcnemar(rows, "full_support")
    out["transitions"] = {
        "both_EM_correct": int(sum(r["control"]["EM"] == 1 and r["treatment"]["EM"] == 1 for r in rows)),
        "EM_gain": int(sum(r["delta"]["EM"] > 0 for r in rows)),
        "EM_loss": int(sum(r["delta"]["EM"] < 0 for r in rows)),
        "both_EM_wrong": int(sum(r["control"]["EM"] == 0 and r["treatment"]["EM"] == 0 for r in rows)),
        "support_gain": int(sum(r["delta"]["support_recall"] > 0 for r in rows)),
        "support_loss": int(sum(r["delta"]["support_recall"] < 0 for r in rows)),
        "support_gain_without_EM_gain": int(sum(r["support_gain_without_EM_gain"] for r in rows)),
        "EM_gain_without_support_gain": int(sum(r["EM_gain_without_support_gain"] for r in rows)),
        "both_support_and_EM_gain": int(sum(r["both_support_and_EM_gain"] for r in rows)),
    }
    support_gains = [r for r in rows if r["delta"]["support_recall"] > 0]
    out["support_gain_to_EM_gain_rate"] = (
        sum(r["delta"]["EM"] > 0 for r in support_gains) / len(support_gains)
        if support_gains else None
    )
    x = [r["delta"]["support_recall"] for r in rows]
    y = [r["delta"]["EM"] for r in rows]
    if len(set(x)) > 1 and len(set(y)) > 1:
        corr = spearmanr(x, y)
        out["support_recall_delta_vs_EM_delta_spearman"] = {
            "rho": float(corr.statistic), "p_descriptive": float(corr.pvalue)
        }
    else:
        out["support_recall_delta_vs_EM_delta_spearman"] = None
    return out


def grouped(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def md_table(grouped_data: dict) -> str:
    lines = ["| Grup | n | EM kontrol → yeni | Tam destek kontrol → yeni | Δ destek recall | Δ F1 | Δ token |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, s in grouped_data.items():
        lines.append(
            f"| {key} | {s['n']} | {s['control_EM_correct']}/{s['n']} → {s['treatment_EM_correct']}/{s['n']} "
            f"| {s['control_full_support']}/{s['n']} → {s['treatment_full_support']}/{s['n']} "
            f"| {pct(s['mean_support_recall_delta'])} | {pct(s['mean_F1_delta'])} | {s['mean_token_delta']:.2f} |"
        )
    return "\n".join(lines)


def build_report(analysis: dict) -> str:
    h = analysis["datasets"]["HotpotQA"]
    m = analysis["datasets"]["MuSiQue"]
    pooled = analysis["pooled_descriptive"]
    lines = [
        "# Faz 30A — Donmuş kayıtların mekanizma ve heterojenlik analizi",
        "",
        "**Tarih:** 13 Ekim 2026  ",
        "**Kapsam:** Faz 28 HotpotQA doğrulaması + Faz 29 MuSiQue pilotu  ",
        "**Uzak model çağrısı:** 0",
        "",
        "## Amaç",
        "",
        "Bu analiz, önceden tamamlanmış ve bağımsız denetimden geçmiş sonuç kayıtlarını yeniden kullanır. Yeni soru seçimi, yeni gold açılması veya yeni model çağrısı yapılmamıştır. Amaç, semantic-joint-citations yönteminin destek seçimi kazanımının hangi koşullarda cevap doğruluğuna dönüştüğünü ölçmektir.",
        "",
        "Faz 28 doğrulaması önceden kayıtlı confirmatory sonuçtur. Faz 29 pilotu çapraz-benchmark yön göstergesidir. Birleştirilmiş sonuçlar yalnızca açıklayıcıdır ve yeni bir doğrulama p değeri olarak yorumlanmaz.",
        "",
        "## Veri bütünlüğü",
        "",
        f"- HotpotQA kayıtları: {h['n']} soru; bağımsız denetim `{h['audit_status']}`.",
        f"- MuSiQue kayıtları: {m['n']} soru; bağımsız denetim `{m['audit_status']}`.",
        f"- HotpotQA sonuç SHA-256: `{h['result_sha256']}`.",
        f"- MuSiQue sonuç SHA-256: `{m['result_sha256']}`.",
        "- Analiz yalnızca mevcut `per_case` kayıtlarını okur; `remote_calls=0`.",
        "",
        "## Veri kümesi bazında özet",
        "",
        "### Faz 28 — HotpotQA",
        "",
        f"- EM: {h['control_EM_correct']}/{h['n']} → {h['treatment_EM_correct']}/{h['n']}; net {h['paired_EM']['net']}, exact McNemar p = {h['paired_EM']['two_sided_exact_mcnemar_p']:.12g}.",
        f"- Tam destek kapsaması: {h['control_full_support']}/{h['n']} → {h['treatment_full_support']}/{h['n']}; net {h['paired_full_support']['net']}, exact McNemar p = {h['paired_full_support']['two_sided_exact_mcnemar_p']:.12g}.",
        f"- Ortalama destek recall farkı: {pct(h['mean_support_recall_delta'])}; ortalama F1 farkı: {pct(h['mean_F1_delta'])}.",
        f"- Kaynak cümlesi kümesi {h['changed_contexts']}/{h['n']} soruda değişti.",
        "",
        md_table(analysis["groups"]["HotpotQA_type"]),
        "",
        "### Faz 29 — MuSiQue",
        "",
        f"- EM: {m['control_EM_correct']}/{m['n']} → {m['treatment_EM_correct']}/{m['n']}; net {m['paired_EM']['net']}, exact McNemar p = {m['paired_EM']['two_sided_exact_mcnemar_p']:.12g}.",
        f"- Tam destek paragrafı kapsamı: {m['control_full_support']}/{m['n']} → {m['treatment_full_support']}/{m['n']}; net {m['paired_full_support']['net']}, exact McNemar p = {m['paired_full_support']['two_sided_exact_mcnemar_p']:.12g}.",
        f"- Ortalama destek recall farkı: {pct(m['mean_support_recall_delta'])}; ortalama F1 farkı: {pct(m['mean_F1_delta'])}.",
        f"- Kaynak cümlesi kümesi {m['changed_contexts']}/{m['n']} soruda değişti.",
        "",
        md_table(analysis["groups"]["MuSiQue_hop"]),
        "",
        "## Mekanizma: destek kazanımı neden her zaman EM kazanımı olmadı?",
        "",
        "| Veri kümesi | Destek recall artan | Bu artışta EM de artan | Destek artıp EM artmayan | EM artıp destek artmayan |",
        "|---|---:|---:|---:|---:|",
        f"| HotpotQA | {h['transitions']['support_gain']} | {h['transitions']['both_support_and_EM_gain']} | {h['transitions']['support_gain_without_EM_gain']} | {h['transitions']['EM_gain_without_support_gain']} |",
        f"| MuSiQue | {m['transitions']['support_gain']} | {m['transitions']['both_support_and_EM_gain']} | {m['transitions']['support_gain_without_EM_gain']} | {m['transitions']['EM_gain_without_support_gain']} |",
        "",
        f"HotpotQA’da destek recall artışı olan vakaların %{100*h['support_gain_to_EM_gain_rate']:.1f}’inde EM de arttı. MuSiQue’de bu oran %{100*m['support_gain_to_EM_gain_rate']:.1f} oldu. Bu, seçicinin kanıt kapsamını iyileştirmesinin cevap okuyucusu için yeterli veya doğru kullanılmış kanıt garantisi vermediğini gösteriyor.",
        "",
        "Başka bir deyişle MuSiQue sonucu yöntem etkisinin yok olduğunu göstermiyor; destek seçimi ile cevap üretimi arasındaki aktarımın veri kümesine ve soru yapısına bağlı olduğunu gösteriyor.",
        "",
        "## Heterojenlik bulguları",
        "",
        f"- HotpotQA’da değişen bağlamlar: {analysis['groups']['HotpotQA_changed']['True']['n']} soru; değişmeyen bağlamlar: {analysis['groups']['HotpotQA_changed']['False']['n']} soru.",
        f"- MuSiQue’de değişen bağlamlar: {analysis['groups']['MuSiQue_changed']['True']['n']} soru; değişmeyen bağlamlar: {analysis['groups']['MuSiQue_changed']['False']['n']} soru.",
        "- Değişmeyen bağlamlarda delta sıfırdır; bu beklenen bir kontrol ve aynı istem yanıtlarının yeniden kullanımının doğrulanmasıdır.",
        "- MuSiQue hop katmanları arasında etki büyüklüğü farklıdır; küçük alt gruplardaki p değerleri keşif amaçlıdır.",
        "",
        "### HotpotQA: kaynak cümlesi kümesi değişimine göre",
        "",
        md_table(analysis["groups"]["HotpotQA_changed"]),
        "",
        "### MuSiQue: kaynak cümlesi kümesi değişimine göre",
        "",
        md_table(analysis["groups"]["MuSiQue_changed"]),
        "",
        "## Açıklayıcı havuz özeti",
        "",
        f"İki arşiv birlikte {pooled['n']} eşleştirilmiş soru içerir. EM {pooled['control_EM_correct']}/{pooled['n']} → {pooled['treatment_EM_correct']}/{pooled['n']}; tam destek {pooled['control_full_support']}/{pooled['n']} → {pooled['treatment_full_support']}/{pooled['n']}. Ortalama destek recall farkı {pct(pooled['mean_support_recall_delta'])}, ortalama F1 farkı {pct(pooled['mean_F1_delta'])} ve ortalama token farkı {pooled['mean_token_delta']:.2f}.",
        "",
        "Bu havuz özeti iki farklı benchmark ailesini tek bir confirmatory test gibi birleştirmez. Yalnızca yönün ve mekanizmanın genel görünümünü verir.",
        "",
        "## Karar ve sonraki deney için sonuç",
        "",
        "1. Semantic-joint-citations’ın destek seçimi üzerindeki etkisi iki arşivde de pozitiftir.",
        "2. Cevap doğruluğu etkisi HotpotQA’da güçlü, MuSiQue’de küçük ve belirsizdir.",
        "3. Bir sonraki deney selector’ı rastgele değiştirmemeli; önce okuyucunun iyileştirilmiş kanıtı kullanamama vakalarını ayrı bir ön kayıtlı ablation olarak test etmelidir.",
        "4. Yeni bir benchmark doğrulaması yapılacaksa yöntem ve okuyucu sürümü önceden dondurulmalı, ayrı bir pilot ve doğrulama seti kullanılmalı, Faz 29’un ayrılmış 96 kimliği açılmamalıdır.",
        "",
        "**Faz 30A sonucu:** Kanıt seçimi kazanımı tekrarlanıyor; EM kazanımı heterojendir. En güçlü sonraki aday, semantic selector sabitken cevap okuyucusunun kanıtı kullanma davranışını ölçen ayrı bir deneydir.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    phase28_rows, phase28_input = make_rows(
        "HotpotQA", PHASE28 / "VALIDATION_POST_GOLD_RESULTS.json",
        PHASE28 / "VALIDATION_INTEGRITY_INDEPENDENT_AUDIT.json")
    phase29_rows, phase29_input = make_rows(
        "MuSiQue", PHASE29 / "PILOT_POST_GOLD_RESULTS.json",
        PHASE29 / "INTEGRITY_INDEPENDENT_AUDIT.json")
    all_rows = phase28_rows + phase29_rows
    hotpot_summary = summarize(phase28_rows)
    hotpot_summary.update({"audit_status": phase28_input["audit_status"],
                           "result_sha256": phase28_input["result_sha256"],
                           "audit_sha256": phase28_input["audit_sha256"]})
    musique_summary = summarize(phase29_rows)
    musique_summary.update({"audit_status": phase29_input["audit_status"],
                            "result_sha256": phase29_input["result_sha256"],
                            "audit_sha256": phase29_input["audit_sha256"]})
    analysis = {
        "phase": "30A",
        "created": "2026-10-13",
        "remote_calls": 0,
        "frozen_inputs": {"HotpotQA": phase28_input, "MuSiQue": phase29_input},
        "datasets": {"HotpotQA": hotpot_summary, "MuSiQue": musique_summary},
        "groups": {
            "HotpotQA_type": grouped(phase28_rows, "stratum"),
            "MuSiQue_hop": grouped(phase29_rows, "stratum"),
            "HotpotQA_changed": grouped(phase28_rows, "source_sentence_set_changed"),
            "MuSiQue_changed": grouped(phase29_rows, "source_sentence_set_changed"),
        },
        "pooled_descriptive": summarize(all_rows),
        "interpretation": {
            "support_gain_repeats": True,
            "EM_gain_is_heterogeneous": True,
            "pooled_results_are_descriptive": True,
            "support_overlap_is_not_semantic_entailment": True,
        },
    }
    # The first failed run may have left the newly-created empty directory;
    # overwrite only this analysis' own two output files on a rerun.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PHASE30A_FROZEN_MECHANISM_ANALYSIS.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "PHASE30A_TURKCE_RAPOR.md").write_text(build_report(analysis), encoding="utf-8")
    print(json.dumps({"status": "PHASE30A_COMPLETE", "out": str(OUT),
                      "remote_calls": 0, "n": len(all_rows),
                      "hotpot": analysis["datasets"]["HotpotQA"],
                      "musique": analysis["datasets"]["MuSiQue"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
