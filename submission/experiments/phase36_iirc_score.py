"""Open IIRC gold only after every paired Qwen response is hash-frozen; score official metrics."""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import string
import sys
import tarfile
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase36_iirc_core import ARMS, answer_parts
from submission.experiments.phase36_iirc_register import IIRC_ARCHIVE, OUT, SCRATCH
from submission.experiments.phase7_modal27b_matched8 import atomic_write, read_rows

CONTEXTS = OUT / "gold_blind_paired_contexts.jsonl"
READER_TASKS = OUT / "reader_tasks.jsonl"
READER_ANSWERS = OUT / "reader_responses.jsonl"
READER_FROZEN = OUT / "READER_RESPONSES_FROZEN.json"
SCORE_OUTPUT = OUT / "IIRC_OFFICIAL_EM_DROP_F1_RESULTS.json"
CASES_OUTPUT = OUT / "paired_case_results.jsonl"
GOLD_ACCESS = OUT / "GOLD_ACCESS_AUTHORIZED.json"
REPORT = OUT / "PHASE36_TURKCE_RAPOR.md"
BOOTSTRAP_SEED = 360014
BOOTSTRAP_REPLICATES = 10000
EXCLUDE = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    return str(float(text)) if _is_number(text) else text


def normalize_answer(text: str) -> str:
    """Port of AllenNLP's official DROP answer normalizer."""
    normalized_parts = []
    for token in re.split(" |-", str(text).lower()):
        if not _is_number(token):
            token = "".join(char for char in token if char not in EXCLUDE)
        token = _normalize_number(token)
        token = re.sub(r"\b(a|an|the)\b", " ", token, flags=re.UNICODE)
        token = " ".join(token.split())
        if token.strip():
            normalized_parts.append(token)
    return " ".join(normalized_parts).strip()


def _answer_to_bags(answer):
    raw_spans = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized = [normalize_answer(span) for span in raw_spans]
    bags = [set(span.split()) for span in normalized]
    return normalized, bags


def _compute_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(gold.intersection(predicted))
    precision = 1.0 if not predicted else intersection / float(len(predicted))
    recall = 1.0 if not gold else intersection / float(len(gold))
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _match_numbers_if_present(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {word for word in gold if _is_number(word)}
    predicted_numbers = {word for word in predicted if _is_number(word)}
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))


def official_drop_metrics(prediction, gold) -> tuple[int, float]:
    """Official DROP scorer: normalized exact match + number-aware aligned F1."""
    predicted_norm, predicted_bags = _answer_to_bags(prediction)
    gold_norm, gold_bags = _answer_to_bags(gold)
    exact = int(set(predicted_norm) == set(gold_norm) and len(predicted_norm) == len(gold_norm))
    scores = np.zeros((len(gold_bags), len(predicted_bags)), dtype=np.float64)
    for gold_index, gold_bag in enumerate(gold_bags):
        for pred_index, pred_bag in enumerate(predicted_bags):
            if _match_numbers_if_present(gold_bag, pred_bag):
                scores[gold_index, pred_index] = _compute_f1(pred_bag, gold_bag)
    row_indexes, col_indexes = linear_sum_assignment(-scores)
    aligned = np.zeros(max(len(gold_bags), len(predicted_bags)), dtype=np.float64)
    for row, column in zip(row_indexes, col_indexes):
        aligned[row] = max(aligned[row], scores[row, column])
    return exact, round(float(np.mean(aligned)), 2)


def gold_strings(answer: dict) -> tuple[str, ...]:
    answer_type = answer["type"]
    if answer_type == "span":
        return tuple(str(span["text"]) for span in answer["answer_spans"])
    if answer_type == "value":
        return (str(answer["answer_value"]),)
    if answer_type == "binary":
        return (str(answer["answer_value"]),)
    if answer_type == "none":
        return ("NONE",)
    raise AssertionError(f"Unexpected IIRC answer type: {answer_type}")


def normalized_span(text: str) -> str:
    text = html.unescape(str(text)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def normalize_title(text: str) -> str:
    return " ".join(str(text).replace("_", " ").casefold().split())


def load_gold() -> tuple[dict[str, dict], dict[str, dict]]:
    with tarfile.open(IIRC_ARCHIVE, "r:gz") as archive:
        passages = json.load(archive.extractfile("iirc/dev.json"))
    gold = {}
    meta = {}
    for passage in passages:
        pid = str(passage["pid"])
        for question in passage["questions"]:
            qid = str(question["qid"])
            gold[qid] = {
                "answer": question["answer"],
                "context": question["context"],
                "question_links": question["question_links"],
            }
            meta[qid] = {
                "pid": pid,
                "main_title": str(passage["title"]),
            }
    return gold, meta


def cluster_bootstrap_ci(rows: list[dict], field: str) -> dict:
    cluster_diffs = defaultdict(float)
    cluster_sizes = Counter()
    for row in rows:
        cluster_diffs[row["pid"]] += row["treatment"][field] - row["baseline"][field]
        cluster_sizes[row["pid"]] += 1
    clusters = sorted(cluster_sizes)
    sums = np.asarray([cluster_diffs[key] for key in clusters], dtype=np.float64)
    counts = np.asarray([cluster_sizes[key] for key in clusters], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(clusters), size=(BOOTSTRAP_REPLICATES, len(clusters)))
    sampled_sums = sums[draws].sum(axis=1)
    sampled_counts = counts[draws].sum(axis=1)
    estimates = sampled_sums / sampled_counts
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "unit": "source passage pid cluster",
        "clusters": len(clusters),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "difference": float(np.mean([row["treatment"][field] - row["baseline"][field] for row in rows])),
        "ci_95_percentile": [float(low), float(high)],
    }


def selected_titles(context_record: dict, arm: str) -> set[str]:
    return {
        normalize_title(item["title"])
        for item in context_record["contexts"][arm]["selected_original_pairs"]
    }


def support_diagnostics(context_record: dict, gold_record: dict, meta: dict, arm: str) -> dict:
    docs = []
    for span in gold_record["context"]:
        passage_name = str(span.get("passage", ""))
        title = meta["main_title"] if passage_name.casefold() in {"main", "main passage"} else passage_name
        docs.append({
            "title": title,
            "text": str(span.get("text", "")),
            "passage_key": normalize_title(title),
        })
    selected = selected_titles(context_record, arm)
    gold_doc_keys = {doc["passage_key"] for doc in docs if doc["passage_key"]}
    covered_doc_keys = gold_doc_keys & selected
    selected_context = normalized_span(context_record["contexts"][arm]["text"])
    spans = [normalized_span(doc["text"]) for doc in docs if normalized_span(doc["text"])]
    span_hits = sum(int(span in selected_context) for span in spans)

    links = gold_record["question_links"] or []
    link_titles = set()
    for link in links:
        if isinstance(link, dict):
            value = link.get("target") or link.get("passage") or link.get("title") or ""
        else:
            value = link
        if str(value).strip():
            link_titles.add(normalize_title(value))
    selected_link_titles = selected - {normalize_title(meta["main_title"])}
    return {
        "gold_context_document_n": len(gold_doc_keys),
        "gold_context_documents_covered_n": len(covered_doc_keys),
        "full_gold_context_document_coverage": int(bool(gold_doc_keys) and gold_doc_keys <= selected),
        "gold_context_span_n": len(spans),
        "exact_gold_context_span_hits": span_hits,
        "gold_context_span_coverage": span_hits / len(spans) if spans else 0.0,
        "gold_question_link_n": len(link_titles),
        "gold_question_links_covered_n": len(link_titles & selected_link_titles),
        "gold_question_link_recall": len(link_titles & selected_link_titles) / len(link_titles) if link_titles else 0.0,
    }


def main() -> None:
    if SCORE_OUTPUT.exists():
        raise FileExistsError("Post-gold scoring is immutable once frozen")
    prereg = json.loads((OUT / "PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(Path(__file__)) == prereg["code_sha256"]["phase36_iirc_score.py"]
    contexts_freeze = json.loads((OUT / "CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    reader_freeze = json.loads(READER_FROZEN.read_text(encoding="utf8"))
    assert reader_freeze["gold_seen"] is False
    assert reader_freeze["n"] == 2 * prereg["sample"]["questions"]
    assert sha_file(READER_ANSWERS) == reader_freeze["responses_sha256"]
    assert sha_file(READER_TASKS) == reader_freeze["task_sha256"]
    assert sha_file(CONTEXTS) == contexts_freeze["sha256"]

    gold_access_spec = {
        "gold_access_authorized_by_protocol": True,
        "protocol_sha256": sha_file(OUT / "PRE_GOLD_PROTOCOL.json"),
        "contexts_sha256": contexts_freeze["sha256"],
        "reader_responses_sha256": reader_freeze["responses_sha256"],
        "all_1_301_questions_have_both_reader_outputs_frozen": True,
    }
    if GOLD_ACCESS.exists():
        existing_gate = json.loads(GOLD_ACCESS.read_text(encoding="utf8"))
        assert all(existing_gate.get(key) == value for key, value in gold_access_spec.items())
    else:
        atomic_write(GOLD_ACCESS, {
            **gold_access_spec,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        })

    contexts = read_rows(CONTEXTS)
    reader_rows = read_rows(READER_ANSWERS)
    tasks = read_rows(READER_TASKS)
    gold, metadata = load_gold()
    assert len(contexts) == len(gold) == prereg["sample"]["questions"]
    response_by_task = {row["task_id"]: row for row in reader_rows}
    task_ids = {task["task_id"] for task in tasks}
    assert len(response_by_task) == len(reader_rows) == len(task_ids)
    assert set(response_by_task) == task_ids
    context_by_id = {row["id"]: row for row in contexts}
    paired = []
    type_results = defaultdict(lambda: {arm: {"n": 0, "em_sum": 0, "f1_sum": 0.0} for arm in ARMS})
    answer_type_counts = Counter()

    cases_partial = CASES_OUTPUT.with_suffix(CASES_OUTPUT.suffix + ".partial")
    cases_partial.unlink(missing_ok=True)
    with cases_partial.open("x", encoding="utf8") as stream:
        for context_record in contexts:
            qid = context_record["id"]
            gold_record = gold[qid]
            answer_type = gold_record["answer"]["type"]
            answer_type_counts[answer_type] += 1
            target = gold_strings(gold_record["answer"])
            pair = {"id": qid, "pid": metadata[qid]["pid"], "answer_type": answer_type}
            for arm in ARMS:
                response = response_by_task[qid + "|" + arm]
                prediction = answer_parts(response["answer"])
                em, f1 = official_drop_metrics(prediction, target)
                diagnostics = support_diagnostics(context_record, gold_record, metadata[qid], arm)
                pair[arm] = {
                    "EM": em,
                    "F1": f1,
                    "answer": response["answer"],
                    "predicted_parts": prediction,
                    "context_tokens": context_record["contexts"][arm]["tokens"],
                    "support_diagnostics": diagnostics,
                }
                type_results[answer_type][arm]["n"] += 1
                type_results[answer_type][arm]["em_sum"] += em
                type_results[answer_type][arm]["f1_sum"] += f1
            paired.append(pair)
            stream.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    if CASES_OUTPUT.exists():
        if sha_file(CASES_OUTPUT) != sha_file(cases_partial):
            raise AssertionError("Existing completed paired cases differ from deterministic rescoring")
        cases_partial.unlink()
    else:
        cases_partial.replace(CASES_OUTPUT)

    base_em = [row[ARMS[0]]["EM"] for row in paired]
    treat_em = [row[ARMS[1]]["EM"] for row in paired]
    base_f1 = [row[ARMS[0]]["F1"] for row in paired]
    treat_f1 = [row[ARMS[1]]["F1"] for row in paired]
    wins = sum(t > b for t, b in zip(treat_em, base_em))
    losses = sum(t < b for t, b in zip(treat_em, base_em))
    ties = len(paired) - wins - losses
    mcnemar_p = float(binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue) if wins + losses else 1.0

    metric_rows = [
        {"pid": row["pid"], "baseline": {"EM": row[ARMS[0]]["EM"], "F1": row[ARMS[0]]["F1"]},
         "treatment": {"EM": row[ARMS[1]]["EM"], "F1": row[ARMS[1]]["F1"]}}
        for row in paired
    ]
    bootstrap_em = cluster_bootstrap_ci(metric_rows, "EM")
    bootstrap_f1 = cluster_bootstrap_ci(metric_rows, "F1")

    arm_scores = {}
    for arm, ems, f1s in ((ARMS[0], base_em, base_f1), (ARMS[1], treat_em, treat_f1)):
        arm_scores[arm] = {
            "n": len(paired),
            "official_EM": float(np.mean(ems)),
            "official_DROP_F1": float(np.mean(f1s)),
            "support_diagnostics_mean": {
                key: float(np.mean([row[arm]["support_diagnostics"][key] for row in paired]))
                for key in (
                    "full_gold_context_document_coverage",
                    "gold_context_span_coverage",
                    "gold_question_link_recall",
                )
            },
        }
    delta_em = float(np.mean(np.asarray(treat_em) - np.asarray(base_em)))
    delta_f1 = float(np.mean(np.asarray(treat_f1) - np.asarray(base_f1)))
    claim_gate = (
        delta_em > 0 and delta_f1 > 0
        and bootstrap_em["ci_95_percentile"][0] > 0
        and bootstrap_f1["ci_95_percentile"][0] > 0
    )
    results = {
        "phase": 36,
        "dataset": "IIRC public dev; all 1,301 questions; not an external hidden test",
        "protocol_sha256": sha_file(OUT / "PRE_GOLD_PROTOCOL.json"),
        "reader_response_sha256": reader_freeze["responses_sha256"],
        "paired_cases_sha256": sha_file(CASES_OUTPUT),
        "official_scoring": "IIRC EMNLP 2020 EM and DROP macro-F1; AllenNLP official DROP normalization, numeric match, and bipartite span alignment.",
        "answer_type_counts": dict(answer_type_counts),
        "arms": arm_scores,
        "paired_delta_treatment_minus_baseline": {
            "EM": delta_em,
            "EM_net_correct_questions": int(sum(treat_em) - sum(base_em)),
            "EM_wins": wins,
            "EM_losses": losses,
            "EM_ties": ties,
            "EM_exact_McNemar_two_sided_p": mcnemar_p,
            "EM_cluster_bootstrap_95_CI": bootstrap_em,
            "DROP_F1": delta_f1,
            "DROP_F1_cluster_bootstrap_95_CI": bootstrap_f1,
        },
        "by_answer_type": {
            answer_type: {
                arm: {
                    "n": score[arm]["n"],
                    "EM": score[arm]["em_sum"] / score[arm]["n"],
                    "DROP_F1": score[arm]["f1_sum"] / score[arm]["n"],
                }
                for arm in ARMS
            }
            for answer_type, score in type_results.items()
        },
        "predeclared_improvement_claim_gate_met": bool(claim_gate),
        "calls_and_token_usage": {
            "verifier_paid_calls": json.loads((OUT / "VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))["actual_paid_calls"],
            "reader_paid_calls": reader_freeze["actual_paid_reader_calls"],
            "reader_duplicate_prompt_reuse": reader_freeze["identical_prompt_reuse_n"],
            "verifier_input_tokens_reported": sum(row.get("input_tokens") or 0 for row in read_rows(OUT / "semantic_verifier_responses.jsonl")),
            "verifier_output_tokens_reported": sum(row.get("output_tokens") or 0 for row in read_rows(OUT / "semantic_verifier_responses.jsonl")),
            "reader_input_tokens_reported": sum(row.get("input_tokens") or 0 for row in reader_rows if row.get("paid_remote_call")),
            "reader_output_tokens_reported": sum(row.get("output_tokens") or 0 for row in reader_rows if row.get("paid_remote_call")),
            "price_or_credit_invoice_available": False,
        },
        "support_metric_caveat": "Gold context/link coverage measures annotated source-page/span retrieval, not independent proof of entailment.",
        "public_dev_caveat": "All IIRC dev answers are public; this is a frozen full-dev cross-benchmark experiment, not a sealed generalization test.",
    }
    em_ci = bootstrap_em["ci_95_percentile"]
    f1_ci = bootstrap_f1["ci_95_percentile"]
    report = f"""# Faz 36 — IIRC tam geliştirme kümesi

## Tasarım

IIRC’nin herkese açık geliştirme bölümündeki {len(paired):,} sorunun tamamında, aynı soru ve aynı %25 kaynak-token bütçesi altında iki kol karşılaştırıldı. Kontrol kolu MiniLM cümle sıralaması; yöntem kolu Faz 28’de dondurulan anlamsal ortak atıf seçicisi ve geçersiz seçimde aynı kontrole dönüş kuralıdır. Her iki kol aynı Qwen 27B okuyucusunu kullandı.

Model yanıtları tamamlanıp SHA256 ile dondurulduktan sonra cevap, bağlam ve `question_links` anotasyonları açıldı. EM ve DROP F1, IIRC makalesindeki/AllenNLP DROP değerlendiricisindeki sayı eşleştirmesi ve çoklu cevap hizalamasıyla hesaplandı.

## Sonuçlar

| Ölçüt | Cümle sıralama | Anlamsal atıf | Fark |
|---|---:|---:|---:|
| Resmî EM | {arm_scores[ARMS[0]]['official_EM']*100:.2f}% | {arm_scores[ARMS[1]]['official_EM']*100:.2f}% | {delta_em*100:+.2f} yüzde puan ({int(sum(treat_em)-sum(base_em)):+d} doğru cevap) |
| Resmî DROP F1 | {arm_scores[ARMS[0]]['official_DROP_F1']*100:.2f}% | {arm_scores[ARMS[1]]['official_DROP_F1']*100:.2f}% | {delta_f1*100:+.2f} yüzde puan |

- EM farkı için pasaj-kümeli %95 bootstrap aralığı: [{em_ci[0]*100:+.2f}, {em_ci[1]*100:+.2f}] yüzde puan; eşlenik tam McNemar p={mcnemar_p:.5g}.
- DROP F1 farkı için pasaj-kümeli %95 bootstrap aralığı: [{f1_ci[0]*100:+.2f}, {f1_ci[1]*100:+.2f}] yüzde puan.
- EM eşlenik sonuçları: {wins} kazanım, {losses} kayıp, {ties} eşit.
- Altın bağlam pasajlarının tamamını kapsama: kontrol {arm_scores[ARMS[0]]['support_diagnostics_mean']['full_gold_context_document_coverage']*100:.1f}%, yöntem {arm_scores[ARMS[1]]['support_diagnostics_mean']['full_gold_context_document_coverage']*100:.1f}%.
- Altın bağlam metni tam-alt-dize kapsaması: kontrol {arm_scores[ARMS[0]]['support_diagnostics_mean']['gold_context_span_coverage']*100:.1f}%, yöntem {arm_scores[ARMS[1]]['support_diagnostics_mean']['gold_context_span_coverage']*100:.1f}%.

## Yorum sınırı

Önceden belirlenen çift-metrik iddia eşiği **{'geçildi' if claim_gate else 'geçilmedi'}**: EM ve DROP F1 farklarının ikisi de pozitif olmalı ve ikisinin pasaj-kümeli %95 aralığı da sıfırın üzerinde kalmalı. IIRC dev etiketleri açık olduğundan, bu sonuç gizli test veya bağımsız kapalı benchmark kanıtı değildir. Destek bağlamı kapsamı, metinsel anotasyonların seçilen kaynaklarda bulunmasını ölçer; tek başına anlamca destek kanıtı sayılmaz.

## Çağrılar

- Verifier uzak çağrısı: {results['calls_and_token_usage']['verifier_paid_calls']:,}
- Okuyucu uzak çağrısı: {results['calls_and_token_usage']['reader_paid_calls']:,}; aynı prompt tekrar kullanımı: {results['calls_and_token_usage']['reader_duplicate_prompt_reuse']:,}
- Sunucunun raporladığı giriş/çıkış token toplamı: verifier {results['calls_and_token_usage']['verifier_input_tokens_reported']:,}/{results['calls_and_token_usage']['verifier_output_tokens_reported']:,}, okuyucu {results['calls_and_token_usage']['reader_input_tokens_reported']:,}/{results['calls_and_token_usage']['reader_output_tokens_reported']:,}.
- Modal kredi faturası API yanıtlarında verilmediği için dolar/kredi toplamı bu kayıtlardan doğrulanamıyor.

Ön kayıt: `PRE_GOLD_PROTOCOL.json`; ham eşlenik sonuçlar: `paired_case_results.jsonl`; ölçüt ayrıntıları: `IIRC_OFFICIAL_EM_DROP_F1_RESULTS.json`.
"""
    report_partial = REPORT.with_suffix(REPORT.suffix + ".partial")
    report_partial.write_text(report, encoding="utf8")
    if REPORT.exists():
        if REPORT.read_text(encoding="utf8") != report:
            raise AssertionError("Existing report differs from deterministic rescoring")
        report_partial.unlink()
    else:
        report_partial.replace(REPORT)
    atomic_write(SCORE_OUTPUT, results)
    print("PHASE36_IIRC_SCORE_COMPLETE", json.dumps({
        "n": len(paired),
        "baseline_em": arm_scores[ARMS[0]]["official_EM"],
        "treatment_em": arm_scores[ARMS[1]]["official_EM"],
        "delta_em": delta_em,
        "baseline_f1": arm_scores[ARMS[0]]["official_DROP_F1"],
        "treatment_f1": arm_scores[ARMS[1]]["official_DROP_F1"],
        "delta_f1": delta_f1,
        "em_wins_losses_ties": [wins, losses, ties],
        "claim_gate": claim_gate,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
