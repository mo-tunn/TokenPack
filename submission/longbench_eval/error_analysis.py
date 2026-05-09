from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "longbench_v2_modal_error_analysis"
DEFAULT_RUNS = {
    "evidence-hybrid": ROOT / "submission" / "results" / "longbench_v2_modal_pilot30",
    "query-support": ROOT / "submission" / "results" / "longbench_v2_modal_pilot30_query_support",
    "decision-aware": ROOT / "submission" / "results" / "longbench_v2_modal_pilot30_decision_aware",
    "coverage": ROOT / "submission" / "results" / "longbench_v2_modal_pilot30_coverage",
}
METHODS = [
    "full-context",
    "tokenpack-50",
    "only-longllmlingua-rate050",
    "tokenpack-50+longllmlingua-rate050",
]
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "both",
    "can",
    "does",
    "from",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "most",
    "not",
    "only",
    "our",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze LongBench v2 TokenPack error patterns.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Optional label=path override. Can be passed multiple times.",
    )
    args = parser.parse_args()

    runs = dict(DEFAULT_RUNS)
    for item in args.run:
        label, _, path = item.partition("=")
        if not label or not path:
            raise ValueError(f"--run must be label=path, got: {item}")
        runs[label] = Path(path)

    loaded = {label: _load_run(path) for label, path in runs.items() if path.exists()}
    if "evidence-hybrid" not in loaded:
        raise FileNotFoundError("The evidence-hybrid baseline run is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_rows = _build_case_rows(loaded)
    swing_rows = _build_swing_rows(loaded)
    profile_rows = _build_profile_rows(loaded)

    _write_csv(case_rows, output_dir / "longbench_error_case_matrix.csv")
    _write_csv(swing_rows, output_dir / "longbench_profile_swings.csv")
    _write_csv(profile_rows, output_dir / "longbench_profile_summary.csv")
    _write_report(profile_rows, case_rows, swing_rows, output_dir / "longbench_error_analysis.md")
    print(f"Wrote LongBench error analysis to {output_dir}")
    return 0


def _load_run(path: Path) -> dict[str, Any]:
    results = _read_jsonl(path / "longbench_generation_results.jsonl")
    tasks = _read_jsonl(path / "longbench_generation_tasks.jsonl")
    return {
        "path": path,
        "results": {
            (str(row.get("case_id")), str(row.get("method"))): row
            for row in results
        },
        "tasks": {
            (str(row.get("case_id")), str(row.get("method"))): row
            for row in tasks
        },
    }


def _build_profile_rows(loaded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run in loaded.items():
        by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
        for (_case_id, method), row in run["results"].items():
            if method in by_method:
                by_method[method].append(row)
        for method in METHODS:
            method_rows = by_method[method]
            rows.append(
                {
                    "profile": label,
                    "method": method,
                    "runs": len(method_rows),
                    "accuracy": _mean(1.0 if row.get("correct") is True else 0.0 for row in method_rows),
                    "avg_context_tokens": _mean(row.get("context_tokens") for row in method_rows),
                    "avg_saving": _mean(row.get("token_saving_vs_full") for row in method_rows),
                }
            )
    return rows


def _build_case_rows(loaded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = loaded["evidence-hybrid"]
    case_ids = sorted({case_id for case_id, _method in baseline["results"]})
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        full = baseline["results"].get((case_id, "full-context"), {})
        tokenpack = baseline["results"].get((case_id, "tokenpack-50"), {})
        lll = baseline["results"].get((case_id, "only-longllmlingua-rate050"), {})
        cascade = baseline["results"].get((case_id, "tokenpack-50+longllmlingua-rate050"), {})
        task = baseline["tasks"].get((case_id, "tokenpack-50"), {})
        full_task = baseline["tasks"].get((case_id, "full-context"), {})
        gold = str(full.get("answer") or tokenpack.get("answer") or "").strip()
        tokenpack_prediction = str(tokenpack.get("prediction") or "")
        gold_choice = _choice_text(full, gold)
        predicted_choice = _choice_text(full, tokenpack_prediction)
        tp_context = str(task.get("context") or "")
        full_context = str(full_task.get("context") or "")
        gold_recall = _choice_recall(tp_context, gold_choice)
        full_gold_recall = _choice_recall(full_context, gold_choice)
        predicted_recall = _choice_recall(tp_context, predicted_choice)
        row = {
            "case_id": case_id,
            "domain": full.get("domain", ""),
            "sub_domain": full.get("sub_domain", ""),
            "difficulty": full.get("difficulty", ""),
            "question": _shorten(str(full.get("question") or ""), 180),
            "gold": gold,
            "full_prediction": full.get("prediction", ""),
            "tokenpack_prediction": tokenpack_prediction,
            "longllmlingua_prediction": lll.get("prediction", ""),
            "cascade_prediction": cascade.get("prediction", ""),
            "full_correct": _bool_int(full.get("correct") is True),
            "tokenpack_correct": _bool_int(tokenpack.get("correct") is True),
            "longllmlingua_correct": _bool_int(lll.get("correct") is True),
            "cascade_correct": _bool_int(cascade.get("correct") is True),
            "outcome": _outcome(full, tokenpack, lll, cascade),
            "diagnosis": _diagnose(full, tokenpack, lll, cascade, gold_recall, full_gold_recall, predicted_recall),
            "tokenpack_gold_choice_recall": f"{gold_recall:.3f}",
            "full_gold_choice_recall": f"{full_gold_recall:.3f}",
            "tokenpack_pred_choice_recall": f"{predicted_recall:.3f}",
            "tokenpack_context_tokens": tokenpack.get("context_tokens", ""),
            "tokenpack_saving": tokenpack.get("token_saving_vs_full", ""),
        }
        for profile, run in loaded.items():
            prof_row = run["results"].get((case_id, "tokenpack-50"), {})
            row[f"{profile}_tp_prediction"] = prof_row.get("prediction", "")
            row[f"{profile}_tp_correct"] = _bool_int(prof_row.get("correct") is True)
        rows.append(row)
    return rows


def _build_swing_rows(loaded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = loaded["evidence-hybrid"]
    case_ids = sorted({case_id for case_id, method in baseline["results"] if method == "tokenpack-50"})
    rows: list[dict[str, Any]] = []
    for profile, run in loaded.items():
        if profile == "evidence-hybrid":
            continue
        for case_id in case_ids:
            base = baseline["results"].get((case_id, "tokenpack-50"), {})
            other = run["results"].get((case_id, "tokenpack-50"), {})
            if not other:
                continue
            base_correct = base.get("correct") is True
            other_correct = other.get("correct") is True
            if base_correct and not other_correct:
                swing = "broken_by_ablation"
            elif not base_correct and other_correct:
                swing = "fixed_by_ablation"
            else:
                swing = "unchanged"
            rows.append(
                {
                    "profile": profile,
                    "case_id": case_id,
                    "swing": swing,
                    "gold": base.get("answer", ""),
                    "evidence_hybrid_prediction": base.get("prediction", ""),
                    f"{profile}_prediction": other.get("prediction", ""),
                    "question": _shorten(str(base.get("question") or ""), 180),
                }
            )
    return rows


def _write_report(
    profile_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    swing_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# LongBench v2 Error Analysis",
        "",
        "## Profile Summary",
        "",
        "| Profile | Method | Runs | Accuracy | Avg ctx toks | Saving |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in profile_rows:
        lines.append(
            f"| {row['profile']} | {row['method']} | {int(row['runs'])} | "
            f"{float(row['accuracy']):.3f} | {float(row['avg_context_tokens']):.0f} | {float(row['avg_saving']):.3f} |"
        )

    outcome_counts = Counter(str(row["outcome"]) for row in case_rows)
    diagnosis_counts = Counter(str(row["diagnosis"]) for row in case_rows)
    lines.extend(["", "## Evidence-Hybrid Outcome Counts", "", "| Outcome | Count |", "|---|---:|"])
    for outcome, count in outcome_counts.most_common():
        lines.append(f"| {outcome} | {count} |")

    lines.extend(["", "## TokenPack Diagnosis Counts", "", "| Diagnosis | Count |", "|---|---:|"])
    for diagnosis, count in diagnosis_counts.most_common():
        lines.append(f"| {diagnosis} | {count} |")

    swing_counts = Counter((row["profile"], row["swing"]) for row in swing_rows)
    lines.extend(["", "## Ablation Swings vs Evidence-Hybrid TokenPack", "", "| Profile | Swing | Count |", "|---|---|---:|"])
    for (profile, swing), count in sorted(swing_counts.items()):
        lines.append(f"| {profile} | {swing} | {count} |")

    lines.extend(
        [
            "",
            "## Important Case Lists",
            "",
            "### Evidence-Hybrid TokenPack Wins over LongLLMLingua",
            "",
            "| Case | Gold | TokenPack | LongLLMLingua | Question |",
            "|---|---|---|---|---|",
        ]
    )
    for row in case_rows:
        if row["tokenpack_correct"] == 1 and row["longllmlingua_correct"] == 0:
            lines.append(
                f"| {row['case_id']} | {row['gold']} | {row['tokenpack_prediction']} | "
                f"{row['longllmlingua_prediction']} | {row['question']} |"
            )

    lines.extend(["", "### LongLLMLingua Wins over Evidence-Hybrid TokenPack", "", "| Case | Gold | TokenPack | LongLLMLingua | Diagnosis | Question |", "|---|---|---|---|---|---|"])
    for row in case_rows:
        if row["tokenpack_correct"] == 0 and row["longllmlingua_correct"] == 1:
            lines.append(
                f"| {row['case_id']} | {row['gold']} | {row['tokenpack_prediction']} | "
                f"{row['longllmlingua_prediction']} | {row['diagnosis']} | {row['question']} |"
            )

    lines.extend(["", "### Ablations That Broke Evidence-Hybrid TokenPack", "", "| Profile | Case | Gold | Evidence-Hybrid | Ablation | Question |", "|---|---|---|---|---|---|"])
    for row in swing_rows:
        if row["swing"] == "broken_by_ablation":
            profile = str(row["profile"])
            lines.append(
                f"| {profile} | {row['case_id']} | {row['gold']} | {row['evidence_hybrid_prediction']} | "
                f"{row.get(profile + '_prediction', '')} | {row['question']} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The main evidence-hybrid TokenPack setting remains the strongest TokenPack-50 profile in this 30-case window.",
            "- Query-support and decision-aware fix a small number of cases, but they also break cases that evidence-hybrid gets right.",
            "- Coverage slightly increases saving but hurts both TokenPack-50 and the cascade, so it should stay experimental.",
            "- The next useful improvement is likely not another lexical heuristic; it is either a stronger learned reranker/value model or a second-stage compressor/reranker after TokenPack selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _diagnose(
    full: dict[str, Any],
    tokenpack: dict[str, Any],
    lll: dict[str, Any],
    cascade: dict[str, Any],
    gold_recall: float,
    full_gold_recall: float,
    predicted_recall: float,
) -> str:
    if tokenpack.get("correct") is True:
        if full.get("correct") is not True:
            return "tokenpack_correct_full_wrong_possible_distractor_removal"
        if lll.get("correct") is not True:
            return "tokenpack_correct_llmlingua_wrong"
        return "tokenpack_correct"
    if full.get("correct") is True:
        if gold_recall + 0.20 < full_gold_recall:
            return "tokenpack_likely_missing_gold_choice_terms"
        if predicted_recall > gold_recall + 0.20:
            return "tokenpack_distractor_terms_dominate"
        return "tokenpack_loss_vs_full_generation_or_paraphrase"
    if lll.get("correct") is True:
        return "llmlingua_wins_possible_compression_filtering"
    if cascade.get("correct") is True:
        return "cascade_recovers_after_selection"
    return "all_wrong_or_gold_not_surface_form"


def _outcome(full: dict[str, Any], tokenpack: dict[str, Any], lll: dict[str, Any], cascade: dict[str, Any]) -> str:
    flags = {
        "full": full.get("correct") is True,
        "tokenpack": tokenpack.get("correct") is True,
        "llmlingua": lll.get("correct") is True,
        "cascade": cascade.get("correct") is True,
    }
    if all(flags.values()):
        return "all_correct"
    if not any(flags.values()):
        return "all_wrong"
    return "+".join(name for name, correct in flags.items() if correct)


def _choice_text(row: dict[str, Any], letter: str) -> str:
    letter = letter.strip().upper()
    return str(row.get(f"choice_{letter}") or "")


def _choice_recall(context: str, choice: str) -> float:
    terms = _content_terms(choice)
    if not terms:
        return 0.0
    context_terms = _content_terms(context)
    return len(terms & context_terms) / len(terms)


def _content_terms(text: str) -> set[str]:
    return {token for token in (match.group(0).lower() for match in TOKEN_RE.finditer(text)) if token not in STOPWORDS}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[Any]) -> float:
    numeric: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(numeric) / len(numeric) if numeric else 0.0


def _bool_int(value: bool) -> int:
    return 1 if value else 0


def _shorten(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
