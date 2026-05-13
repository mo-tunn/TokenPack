from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qasper_selector_eval import (  # noqa: E402
    DEFAULT_PARQUET_URLS,
    BudgetSpec,
    _as_list,
    _blocks_from_qasper_row,
    _build_index,
    _clean_text,
    _load_qasper_rows,
    _parse_budgets,
    _questions_from_qasper_row,
    _tokens,
)
from tokenpack.chunk_profiles import resolve_chunk_size_config  # noqa: E402
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.selectors import select_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "qasper_cost_quality"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "qasper_cost_quality"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure QASPER answerability retained per token budget."
    )
    parser.add_argument("--data-file", help="Local QASPER parquet/json/jsonl file.")
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--chunkers", default="structure-aware")
    parser.add_argument("--scorings", default="evidence-hybrid")
    parser.add_argument("--strategies", default="production-rag,budget-top-k,greedy-density,knapsack,knapsack-redundancy")
    parser.add_argument("--budget-ratios", default="0.05,0.10,0.20,0.40")
    parser.add_argument("--budgets")
    parser.add_argument("--max-papers", type=int, default=10_000)
    parser.add_argument("--max-questions", type=int, default=100_000)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--chunk-size-preset",
        choices=["manual", "default", "low-budget"],
        default="manual",
        help="Override target/min/max token limits with a named chunking preset.",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_load_qasper_rows(args.data_file, args.split))
    token_counter = TokenCounter()
    embedder = make_embedder(
        model_name=args.model,
        local_files_only=not args.allow_model_download,
    )
    chunkers = _csv_list(args.chunkers)
    scorings = _csv_list(args.scorings)
    unsupported_scorings = sorted(set(scorings) - set(SCORING_PROFILES))
    if unsupported_scorings:
        raise SystemExit(f"Unsupported production scoring profile(s): {', '.join(unsupported_scorings)}")
    strategies = _csv_list(args.strategies)
    chunk_size = resolve_chunk_size_config(
        args.chunk_size_preset,
        args.target_tokens,
        args.min_tokens,
        args.max_tokens,
    )
    raw_rows: list[dict[str, Any]] = []
    processed_papers = 0
    processed_questions = 0
    for row in rows:
        if processed_papers >= args.max_papers or processed_questions >= args.max_questions:
            break
        paper_id = str(row.get("id") or row.get("paper_id") or processed_papers)
        blocks = _blocks_from_qasper_row(row, document_index=processed_papers, token_counter=token_counter)
        questions = _questions_from_qasper_row(row)
        if not blocks or not questions:
            continue
        source_text = " ".join(block.text for block in blocks)
        source_tokens = sum(block.metadata.get("token_count", token_counter.count(block.text)) for block in blocks)
        budget_specs = _parse_budgets(args.budgets, args.budget_ratios, source_tokens)
        full_doc_answer_cache: dict[str, float] = {}
        full_doc_evidence_cache: dict[str, float] = {}
        indexes = {
            chunker_name: _build_index(
                paper_id=paper_id,
                blocks=blocks,
                embedder=embedder,
                work_dir=work_dir,
                chunker_name=chunker_name,
                target_tokens=chunk_size.target_tokens,
                min_tokens=chunk_size.min_tokens,
                max_tokens=chunk_size.max_tokens,
                semantic_threshold=args.semantic_threshold,
                token_counter=token_counter,
            )
            for chunker_name in chunkers
        }
        processed_papers += 1

        for question in questions:
            if processed_questions >= args.max_questions:
                break
            processed_questions += 1
            full_answer_recall = _answer_token_recall(source_text, question.answer)
            full_evidence_recall = _evidence_recall(question.evidence_texts, source_text)
            full_doc_answer_cache[question.question_id] = full_answer_recall
            full_doc_evidence_cache[question.question_id] = full_evidence_recall
            for chunker_name, index in indexes.items():
                query_embedding = embedder.embed([question.question])[0]
                for scoring in scorings:
                    scored = score_chunks(
                        query_embedding,
                        index.chunks,
                        index.embeddings,
                        scoring=scoring,
                        query_text=question.question,
                        redundancy_candidate_pool=args.candidate_pool,
                    )
                    for budget_spec in budget_specs:
                        for strategy in strategies:
                            selection = select_chunks(
                                scored,
                                strategy=strategy,
                                budget=budget_spec.tokens,
                                candidate_pool=args.candidate_pool,
                            )
                            selected_text = " ".join(item.chunk.text for item in selection.selected)
                            answer_recall = _answer_token_recall(selected_text, question.answer)
                            evidence_recall = _evidence_recall(question.evidence_texts, selected_text)
                            raw_rows.append(
                                {
                                    "paper_id": paper_id,
                                    "question_id": question.question_id,
                                    "chunker": chunker_name,
                                    "scoring": scoring,
                                    "strategy": strategy,
                                    "budget": budget_spec.label,
                                    "budget_sort": budget_spec.sort_key,
                                    "source_tokens": source_tokens,
                                    "budget_tokens": budget_spec.tokens,
                                    "used_tokens": selection.used_tokens,
                                    "token_ratio_vs_full": selection.used_tokens / max(1, source_tokens),
                                    "cost_saving_vs_full": 1.0 - selection.used_tokens / max(1, source_tokens),
                                    "budget_utilization": selection.used_tokens / max(1, budget_spec.tokens),
                                    "avg_total_value": selection.total_value,
                                    "full_doc_answer_recall": full_answer_recall,
                                    "full_doc_evidence_recall": full_evidence_recall,
                                    "answer_token_recall": answer_recall,
                                    "answer_recall_retained": _safe_ratio(answer_recall, full_answer_recall),
                                    "evidence_recall": evidence_recall,
                                    "evidence_recall_retained": _safe_ratio(evidence_recall, full_evidence_recall),
                                    "answer_90_retained": 1.0
                                    if _safe_ratio(answer_recall, full_answer_recall) >= 0.90
                                    else 0.0,
                                    "evidence_complete": 1.0 if evidence_recall >= 0.80 else 0.0,
                                    "answer_and_evidence_preserved": 1.0
                                    if _safe_ratio(answer_recall, full_answer_recall) >= 0.90
                                    and evidence_recall >= 0.80
                                    else 0.0,
                                }
                            )

    summary_rows = _summarize(raw_rows, processed_papers, processed_questions)
    raw_path = output_dir / "qasper_cost_quality_raw.csv"
    summary_path = output_dir / "qasper_cost_quality_summary.csv"
    table_path = output_dir / "qasper_cost_quality_table.tex"
    paper_table_path = ROOT / "submission" / "paper" / "tables" / "qasper_cost_quality_table.tex"
    _write_csv(raw_rows, raw_path)
    _write_csv(summary_rows, summary_path)
    _write_latex(summary_rows, table_path)
    _write_latex(summary_rows, paper_table_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Processed papers={processed_papers}, questions={processed_questions}")
    return 0


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _answer_token_recall(context: str, answer: str) -> float:
    answer_terms = set(_content_tokens(answer))
    if not answer_terms:
        return 0.0
    context_terms = set(_content_tokens(context))
    return len(answer_terms & context_terms) / len(answer_terms)


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(str(text).lower()) if token not in STOPWORDS]


def _evidence_recall(evidence_texts: list[str], selected_text: str) -> float:
    selected_norm = _normalize(selected_text)
    if not evidence_texts:
        return 0.0
    hits = 0.0
    for evidence in evidence_texts:
        evidence_norm = _normalize(evidence)
        if not evidence_norm:
            continue
        if evidence_norm in selected_norm:
            hits += 1.0
            continue
        evidence_terms = set(_content_tokens(evidence_norm))
        selected_terms = set(_content_tokens(selected_norm))
        hits += len(evidence_terms & selected_terms) / max(1, len(evidence_terms))
    return hits / max(1, len(evidence_texts))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return min(1.0, numerator / denominator)


def _summarize(
    rows: list[dict[str, Any]],
    processed_papers: int,
    processed_questions: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["chunker"]),
            str(row["scoring"]),
            str(row["strategy"]),
            str(row["budget"]),
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (chunker, scoring, strategy, budget), group in sorted(
        grouped.items(), key=lambda item: (float(item[1][0]["budget_sort"]), item[0])
    ):
        summary.append(
            {
                "chunker": chunker,
                "scoring": scoring,
                "strategy": strategy,
                "budget": budget,
                "processed_papers": processed_papers,
                "processed_questions": processed_questions,
                "runs": len(group),
                "avg_source_tokens": _mean(row["source_tokens"] for row in group),
                "avg_budget_tokens": _mean(row["budget_tokens"] for row in group),
                "avg_used_tokens": _mean(row["used_tokens"] for row in group),
                "token_ratio_vs_full": _mean(row["token_ratio_vs_full"] for row in group),
                "cost_saving_vs_full": _mean(row["cost_saving_vs_full"] for row in group),
                "budget_utilization": _mean(row["budget_utilization"] for row in group),
                "answer_token_recall": _mean(row["answer_token_recall"] for row in group),
                "answer_recall_retained": _mean(row["answer_recall_retained"] for row in group),
                "answer_90_retained_rate": _mean(row["answer_90_retained"] for row in group),
                "evidence_recall": _mean(row["evidence_recall"] for row in group),
                "evidence_complete_rate": _mean(row["evidence_complete"] for row in group),
                "answer_and_evidence_preserved_rate": _mean(
                    row["answer_and_evidence_preserved"] for row in group
                ),
                "avg_total_value": _mean(row["avg_total_value"] for row in group),
            }
        )
    _add_relative_to_100_budget(summary)
    return summary


def _add_relative_to_100_budget(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["budget"] == "100%":
            baselines[(str(row["chunker"]), str(row["scoring"]), str(row["strategy"]))] = row
    for row in rows:
        baseline = baselines.get((str(row["chunker"]), str(row["scoring"]), str(row["strategy"])))
        if not baseline:
            row["token_saving_vs_100_budget"] = ""
            row["answer_retention_vs_100_budget"] = ""
            row["evidence_retention_vs_100_budget"] = ""
            row["preserved_rate_vs_100_budget"] = ""
            continue
        row["token_saving_vs_100_budget"] = 1.0 - float(row["avg_used_tokens"]) / max(
            1.0, float(baseline["avg_used_tokens"])
        )
        row["answer_retention_vs_100_budget"] = _safe_ratio(
            float(row["answer_recall_retained"]),
            float(baseline["answer_recall_retained"]),
        )
        row["evidence_retention_vs_100_budget"] = _safe_ratio(
            float(row["evidence_recall"]),
            float(baseline["evidence_recall"]),
        )
        row["preserved_rate_vs_100_budget"] = _safe_ratio(
            float(row["answer_and_evidence_preserved_rate"]),
            float(baseline["answer_and_evidence_preserved_rate"]),
        )


def _mean(values: Any) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(1, len(values))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["chunker"] == "structure-aware"
        and row["scoring"] == "evidence-hybrid"
        and row["strategy"] in {"production-rag", "budget-top-k", "knapsack", "knapsack-redundancy"}
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{QASPER Cost-Quality Proxy Under Reduced Context Budgets}",
        r"\label{tab:qasper-cost-quality}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Selector & Budget & Used Tok. & Saving & Ans. vs 100 & Ev. vs 100 & Preserved \\",
        r"\hline",
    ]
    for row in selected:
        strategy = str(row["strategy"])
        prefix = r"\rowcolor{tokenpackhighlight}" if strategy == "budget-top-k" else ""
        strategy_text = r"\textbf{TokenPack hybrid-greedy}" if strategy == "budget-top-k" else strategy
        lines.append(
            f"{prefix}{strategy_text} & {_latex_text(str(row['budget']))} & "
            f"{float(row['avg_used_tokens']):.0f} & "
            f"{100.0 * float(row['cost_saving_vs_full']):.1f}\\% & "
            f"{_fmt_optional(row['answer_retention_vs_100_budget'])} & "
            f"{_fmt_optional(row['evidence_retention_vs_100_budget'])} & "
            f"{float(row['answer_and_evidence_preserved_rate']):.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_text(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def _fmt_optional(value: Any) -> str:
    if value == "":
        return "--"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
