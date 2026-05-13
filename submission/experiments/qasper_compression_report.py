from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qasper_selector_eval import _load_qasper_rows, _questions_from_qasper_row  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "qasper_compression_report"
DEFAULT_CURRENT_SUMMARY = (
    ROOT / "submission" / "results" / "qasper_compression_eval_200q_rate050_full" / "qasper_compression_eval_summary.csv"
)
DEFAULT_CURRENT_RAW = (
    ROOT / "submission" / "results" / "qasper_compression_eval_200q_rate050_full" / "qasper_compression_eval_raw.csv"
)
DEFAULT_CURRENT_FRONTIER_SUMMARY = (
    ROOT
    / "submission"
    / "results"
    / "qasper_compression_eval_200q_rates075_085_current"
    / "qasper_compression_eval_summary.csv"
)
DEFAULT_LEGACY_SUMMARIES = [
    ROOT / "submission" / "results" / "qasper_compression_eval_200q_rate075" / "qasper_compression_eval_summary.csv",
    ROOT / "submission" / "results" / "qasper_compression_eval_200q_rate085" / "qasper_compression_eval_summary.csv",
]
DEFAULT_DATA_FILE = Path(r"C:\tmp\qasper-validation.parquet")


@dataclass(slots=True)
class ReportRow:
    method: str
    pipeline: str
    sample_questions: int
    selection_strategy: str
    compression_rate: str
    token_saving: float
    evidence_recall: float
    complete_evidence_rate: float | None
    answer_token_f1: float
    avg_seconds: float
    source_tag: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready compression comparison tables and error analysis.")
    parser.add_argument("--current-summary", default=str(DEFAULT_CURRENT_SUMMARY))
    parser.add_argument("--current-raw", default=str(DEFAULT_CURRENT_RAW))
    parser.add_argument("--current-frontier-summary", default=str(DEFAULT_CURRENT_FRONTIER_SUMMARY))
    parser.add_argument("--legacy-summaries", default=",".join(str(path) for path in DEFAULT_LEGACY_SUMMARIES))
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-cases", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_summary_rows = _read_csv(Path(args.current_summary))
    current_raw_rows = _read_csv(Path(args.current_raw))
    current_frontier_rows = _read_csv(Path(args.current_frontier_summary))
    legacy_summary_paths = [Path(item) for item in _csv_list(args.legacy_summaries)]
    legacy_summary_rows = [
        {"_source_path": str(path), **row}
        for path in legacy_summary_paths
        if path.exists()
        for row in _read_csv(path)
    ]

    report_rows = _build_report_rows(current_summary_rows, current_frontier_rows, legacy_summary_rows)
    comparison_rows = [
        row
        for row in report_rows
        if row.method in {
            "Only TokenPack",
            "Only LLMLingua-2 rate=0.50",
            "TokenPack + LLMLingua-2 rate=0.50",
        }
    ]
    frontier_rows = sorted(report_rows, key=lambda row: (row.token_saving, row.evidence_recall))
    question_lookup = _load_question_lookup(Path(args.data_file)) if Path(args.data_file).exists() else {}
    wins = _pairwise_wins(current_raw_rows)
    top_cases = _largest_recall_gaps(current_raw_rows, question_lookup, limit=args.top_cases)

    combined_csv = output_dir / "qasper_compression_methods.csv"
    comparison_tex = ROOT / "submission" / "paper" / "tables" / "qasper_compression_comparison_table.tex"
    frontier_tex = ROOT / "submission" / "paper" / "tables" / "qasper_compression_frontier_table.tex"
    notes_md = output_dir / "qasper_compression_error_analysis.md"

    _write_csv_report(report_rows, combined_csv)
    comparison_tex.write_text(_comparison_table_tex(comparison_rows), encoding="utf-8")
    frontier_tex.write_text(_frontier_table_tex(frontier_rows), encoding="utf-8")
    notes_md.write_text(_error_analysis_markdown(wins, top_cases), encoding="utf-8")

    print(f"Wrote {combined_csv}")
    print(f"Wrote {comparison_tex}")
    print(f"Wrote {frontier_tex}")
    print(f"Wrote {notes_md}")
    return 0


def _csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_report_rows(
    current_summary_rows: list[dict[str, str]],
    current_frontier_rows: list[dict[str, str]],
    legacy_summary_rows: list[dict[str, str]],
) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for row in current_summary_rows:
        report_row = _current_report_row(row)
        if report_row is None:
            continue
        rows.append(report_row)
    for row in current_frontier_rows:
        report_row = _current_report_row(row)
        if report_row is None or any(existing.method == report_row.method for existing in rows):
            continue
        rows.append(report_row)
    for row in legacy_summary_rows:
        if row.get("strategy") != "budget-top-k":
            continue
        rate = _legacy_rate(row)
        method = f"TokenPack + LLMLingua-2 rate={rate:.2f}"
        if any(existing.method == method for existing in rows):
            continue
        rows.append(
            ReportRow(
                method=method,
                pipeline="tokenpack+llmlingua2",
                sample_questions=int(float(row["runs"])),
                selection_strategy=row["strategy"],
                compression_rate=f"{rate:.2f}",
                token_saving=float(row["total_saving_vs_full"]),
                evidence_recall=float(row["compressed_evidence_recall"]),
                complete_evidence_rate=None,
                answer_token_f1=float(row["compressed_answer_token_f1"]),
                avg_seconds=float(row["avg_compression_seconds"]),
                source_tag="earlier-200q rerun",
            )
        )
    return sorted(rows, key=lambda item: (item.token_saving, item.method))


def _current_report_row(row: dict[str, str]) -> ReportRow | None:
    pipeline = row["pipeline"]
    if pipeline == "only-tokenpack":
        method = "Only TokenPack"
    elif pipeline == "only-llmlingua2":
        method = f"Only LLMLingua-2 rate={float(row['compression_rate']):.2f}"
    elif pipeline == "tokenpack+llmlingua2":
        method = f"TokenPack + LLMLingua-2 rate={float(row['compression_rate']):.2f}"
    else:
        return None
    return ReportRow(
        method=method,
        pipeline=pipeline,
        sample_questions=int(float(row["processed_questions"])),
        selection_strategy=row["selection_strategy"],
        compression_rate=row["compression_rate"],
        token_saving=float(row["total_saving_vs_full"]),
        evidence_recall=float(row["final_evidence_recall"]),
        complete_evidence_rate=float(row["final_complete_evidence_rate"]),
        answer_token_f1=float(row["final_answer_token_f1"]),
        avg_seconds=float(row["avg_total_seconds"]),
        source_tag="current-200q",
    )


def _write_csv_report(rows: list[ReportRow], path: Path) -> None:
    fieldnames = [
        "method",
        "pipeline",
        "sample_questions",
        "selection_strategy",
        "compression_rate",
        "token_saving",
        "evidence_recall",
        "complete_evidence_rate",
        "answer_token_f1",
        "avg_seconds",
        "source_tag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "pipeline": row.pipeline,
                    "sample_questions": row.sample_questions,
                    "selection_strategy": row.selection_strategy,
                    "compression_rate": row.compression_rate,
                    "token_saving": row.token_saving,
                    "evidence_recall": row.evidence_recall,
                    "complete_evidence_rate": row.complete_evidence_rate if row.complete_evidence_rate is not None else "",
                    "answer_token_f1": row.answer_token_f1,
                    "avg_seconds": row.avg_seconds,
                    "source_tag": row.source_tag,
                }
            )


def _legacy_rate(row: dict[str, str]) -> float:
    path_text = row.get("_source_path", "")
    for token in path_text.replace("\\", "/").split("/"):
        if token.startswith("qasper_compression_eval_200q_rate"):
            suffix = token.removeprefix("qasper_compression_eval_200q_rate")
            if suffix.isdigit():
                return float(suffix) / 100.0
    raise ValueError(f"Unable to infer compression rate from legacy path: {path_text}")


def _comparison_table_tex(rows: list[ReportRow]) -> str:
    ordered = sorted(rows, key=lambda item: item.method)
    lines = [
        r"\begin{table}[!t]",
        r"\caption{QASPER Compression Comparison at Approximately Matched 50\% Token Saving. LLM2 denotes LLMLingua-2.}",
        r"\label{tab:qasper-compression-comparison}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Method & Saving & Evidence Recall & Complete & Answer Overlap F1 & Sec. \\",
        r"\hline",
    ]
    for row in ordered:
        lines.append(
            " & ".join(
                [
                    _latex_text(_display_method(row.method)),
                    f"{100.0 * row.token_saving:.1f}\\%",
                    f"{row.evidence_recall:.3f}",
                    f"{row.complete_evidence_rate:.3f}" if row.complete_evidence_rate is not None else "--",
                    f"{row.answer_token_f1:.3f}",
                    f"{row.avg_seconds:.2f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def _frontier_table_tex(rows: list[ReportRow]) -> str:
    has_legacy = any(row.source_tag != "current-200q" for row in rows)
    note = (
        r"{\footnotesize Run notes: \textit{current-200q} denotes the current full 200-question rerun. "
        r"\textit{earlier-200q rerun} denotes rows evaluated on the same 200-question split before the final report-script and manifest cleanup.}"
        if has_legacy
        else r"{\footnotesize Run notes: all rows are refreshed \textit{current-200q} runs on the same 200-question split.}"
    )
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Compression Frontier on QASPER (All Rows Use 200 Questions). LLM2 denotes LLMLingua-2.}",
        r"\label{tab:qasper-compression-frontier}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\hline",
        r"Method & Saving & Evidence Recall & Answer Overlap F1 & Run \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_text(_display_method(row.method)),
                    f"{100.0 * row.token_saving:.1f}\\%",
                    f"{row.evidence_recall:.3f}",
                    f"{row.answer_token_f1:.3f}",
                    _latex_text(row.source_tag),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            note,
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _pairwise_wins(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["paper_id"], row["question_id"])
        grouped.setdefault(key, {})[row["pipeline"]] = row
    same_saving_total = 0
    same_saving_tokenpack_better = 0
    same_saving_llm_better = 0
    higher_saving_total = 0
    higher_saving_combo_better = 0
    higher_saving_llm_better = 0
    for bundle in grouped.values():
        tokenpack = bundle.get("only-tokenpack")
        only_llm = bundle.get("only-llmlingua2")
        combo = bundle.get("tokenpack+llmlingua2")
        if tokenpack and only_llm:
            same_saving_total += 1
            tokenpack_recall = float(tokenpack["final_evidence_recall"])
            llm_recall = float(only_llm["final_evidence_recall"])
            if tokenpack_recall > llm_recall:
                same_saving_tokenpack_better += 1
            elif llm_recall > tokenpack_recall:
                same_saving_llm_better += 1
        if combo and only_llm:
            higher_saving_total += 1
            combo_recall = float(combo["final_evidence_recall"])
            llm_recall = float(only_llm["final_evidence_recall"])
            if combo_recall > llm_recall:
                higher_saving_combo_better += 1
            elif llm_recall > combo_recall:
                higher_saving_llm_better += 1
    return {
        "same_saving_total": same_saving_total,
        "same_saving_tokenpack_better_rate": same_saving_tokenpack_better / max(1, same_saving_total),
        "same_saving_llm_better_rate": same_saving_llm_better / max(1, same_saving_total),
        "higher_saving_total": higher_saving_total,
        "higher_saving_combo_better_rate": higher_saving_combo_better / max(1, higher_saving_total),
        "higher_saving_llm_better_rate": higher_saving_llm_better / max(1, higher_saving_total),
    }


def _load_question_lookup(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _load_qasper_rows(str(path), split="validation"):
        paper_id = str(row.get("id") or row.get("paper_id") or "")
        for question in _questions_from_qasper_row(row):
            lookup[(paper_id, question.question_id)] = {
                "title": question.title,
                "question": question.question,
                "answer": question.answer,
                "evidence_texts": question.evidence_texts,
            }
    return lookup


def _largest_recall_gaps(
    rows: list[dict[str, str]],
    question_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["paper_id"], row["question_id"]), {})[row["pipeline"]] = row
    ranked: list[dict[str, Any]] = []
    for key, bundle in grouped.items():
        tokenpack = bundle.get("only-tokenpack")
        only_llm = bundle.get("only-llmlingua2")
        combo = bundle.get("tokenpack+llmlingua2")
        if not tokenpack or not only_llm or not combo:
            continue
        tokenpack_recall = float(tokenpack["final_evidence_recall"])
        llm_recall = float(only_llm["final_evidence_recall"])
        combo_recall = float(combo["final_evidence_recall"])
        lookup = question_lookup.get(key, {})
        ranked.append(
            {
                "paper_id": key[0],
                "question_id": key[1],
                "title": lookup.get("title", ""),
                "question": lookup.get("question", ""),
                "answer": lookup.get("answer", ""),
                "evidence_texts": lookup.get("evidence_texts", []),
                "tokenpack_recall": tokenpack_recall,
                "llm_recall": llm_recall,
                "combo_recall": combo_recall,
                "tokenpack_minus_llm": tokenpack_recall - llm_recall,
                "combo_minus_llm": combo_recall - llm_recall,
                "tokenpack_saving": float(tokenpack["total_saving_vs_full"]),
                "llm_saving": float(only_llm["total_saving_vs_full"]),
                "combo_saving": float(combo["total_saving_vs_full"]),
            }
        )
    ranked.sort(key=lambda item: (item["tokenpack_minus_llm"], item["combo_minus_llm"]), reverse=True)
    return ranked[:limit]


def _error_analysis_markdown(wins: dict[str, float], top_cases: list[dict[str, Any]]) -> str:
    lines = [
        "# QASPER Compression Error Analysis",
        "",
        "## Pairwise Win Rates",
        "",
        (
            f"- Same-saving comparison (`Only TokenPack` vs `Only LLMLingua-2`, {int(wins['same_saving_total'])} questions): "
            f"TokenPack higher evidence recall on {100.0 * wins['same_saving_tokenpack_better_rate']:.1f}% of questions; "
            f"LLMLingua-2 higher on {100.0 * wins['same_saving_llm_better_rate']:.1f}%."
        ),
        (
            f"- Higher-saving comparison (`TokenPack + LLMLingua-2` vs `Only LLMLingua-2`, {int(wins['higher_saving_total'])} questions): "
            f"selection-first pipeline higher evidence recall on {100.0 * wins['higher_saving_combo_better_rate']:.1f}% of questions; "
            f"LLMLingua-2 higher on {100.0 * wins['higher_saving_llm_better_rate']:.1f}%."
        ),
        "",
        "## Largest Same-Saving Gaps",
        "",
    ]
    for index, case in enumerate(top_cases, start=1):
        lines.extend(
            [
                f"### Case {index}",
                "",
                f"- Paper: `{case['paper_id']}`",
                f"- Question ID: `{case['question_id']}`",
                f"- Title: {case['title'] or '(not loaded)'}",
                f"- Question: {case['question'] or '(not loaded)'}",
                f"- Gold answer: {case['answer'] or '(not loaded)'}",
                (
                    f"- Evidence recall: TokenPack `{case['tokenpack_recall']:.3f}`, "
                    f"Only LLMLingua-2 `{case['llm_recall']:.3f}`, "
                    f"TokenPack + LLMLingua-2 `{case['combo_recall']:.3f}`"
                ),
                (
                    f"- Token saving: TokenPack `{100.0 * case['tokenpack_saving']:.1f}%`, "
                    f"Only LLMLingua-2 `{100.0 * case['llm_saving']:.1f}%`, "
                    f"TokenPack + LLMLingua-2 `{100.0 * case['combo_saving']:.1f}%`"
                ),
                "- Evidence snippets:",
            ]
        )
        for evidence in case.get("evidence_texts", [])[:3]:
            lines.append(f"  - {evidence}")
        lines.append("")
    return "\n".join(lines)


def _display_method(method: str) -> str:
    if method.startswith("Only LLMLingua-2 rate="):
        return "Only LLM2-" + method.rsplit("=", 1)[-1]
    if method.startswith("TokenPack + LLMLingua-2 rate="):
        return "TokenPack + LLM2-" + method.rsplit("=", 1)[-1]
    return method


def _latex_text(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


if __name__ == "__main__":
    raise SystemExit(main())
