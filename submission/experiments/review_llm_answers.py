from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively review TokenPack LLM answer-quality rows.")
    parser.add_argument("--input", default="submission/results/llm_quality/llm_answer_quality.jsonl")
    parser.add_argument("--output", default="submission/results/llm_quality/llm_answer_quality_reviewed.jsonl")
    parser.add_argument("--summary", default="submission/results/llm_quality/llm_answer_quality_reviewed_summary.csv")
    parser.add_argument("--paper-table", default="submission/paper/tables/llm_answer_quality_table.tex")
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    reviewed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("status") != "completed" and not args.include_invalid:
            reviewed.append(row)
            continue
        print_record(index, len(rows), row)
        score = prompt_score()
        row["human_score"] = score
        row["human_notes"] = input("Notes (optional): ").strip()
        reviewed.append(row)

    write_jsonl(reviewed, Path(args.output))
    summary_rows = build_summary(reviewed)
    write_summary(summary_rows, Path(args.summary))
    write_paper_table(summary_rows, Path(args.paper_table))
    print(f"Reviewed rows written to: {args.output}")
    print(f"Summary written to: {args.summary}")
    return 0


def print_record(index: int, total: int, row: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f"Record {index}/{total}")
    print("-" * 88)
    print(f"Model: {row.get('model_label')} / {row.get('model')}")
    print(f"Strategy: {row.get('strategy')}")
    print(f"Status: {row.get('status')}")
    print(f"Context tokens: {row.get('context_tokens')} / budget {row.get('effective_budget')}")
    print(f"Evidence recall: {float(row.get('evidence_recall') or 0):.2f}")
    print(f"Query: {row.get('query')}")
    print(f"Gold answer: {row.get('gold_answer')}")
    print("\nAnswer:")
    print(str(row.get("answer") or "").strip() or "[empty answer]")
    print("-" * 88)


def prompt_score() -> int:
    while True:
        value = input("Score 0=wrong, 1=partial, 2=correct grounded: ").strip()
        if value in {"0", "1", "2"}:
            return int(value)
        print("Please enter 0, 1, or 2.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("model_label")), str(row.get("strategy"))), []).append(row)

    summary: list[dict[str, Any]] = []
    for (model_label, strategy), group in sorted(groups.items()):
        completed = [row for row in group if row.get("status") == "completed"]
        scored = [row for row in completed if row.get("human_score") is not None]
        scores = [float(row["human_score"]) for row in scored]
        summary.append(
            {
                "model_label": model_label,
                "strategy": strategy,
                "completed": len(completed),
                "human_scored": len(scored),
                "invalid_over_budget": sum(1 for row in group if row.get("status") == "invalid-over-budget"),
                "avg_context_tokens": mean([float(row.get("context_tokens") or 0) for row in group]),
                "avg_evidence_recall": mean([float(row.get("evidence_recall") or 0) for row in group]),
                "avg_human_score": "" if not scores else mean(scores),
                "correct_rate": "" if not scores else mean([1.0 if score >= 2 else 0.0 for score in scores]),
            }
        )

    return summary


def write_summary(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_paper_table(summary: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in summary if row["strategy"] in {"document-prefix", "top-k", "knapsack"}]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{LLM Answer Quality After Human Review}",
        r"\label{tab:llm-answer-quality}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Model & Strategy & Scored & Invalid & Avg. Tokens & Avg. Score \\",
        r"\hline",
    ]
    for row in selected:
        strategy = str(row["strategy"])
        prefix = r"\rowcolor{tokenpackhighlight}" if strategy == "knapsack" else ""
        strategy_text = r"\textbf{TokenPack knapsack}" if strategy == "knapsack" else strategy
        score = "--" if row["avg_human_score"] == "" else f"{float(row['avg_human_score']):.2f}"
        lines.append(
            f"{prefix}{row['model_label']} & {strategy_text} & {row['human_scored']} & "
            f"{row['invalid_over_budget']} & {float(row['avg_context_tokens']):.1f} & {score} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


if __name__ == "__main__":
    raise SystemExit(main())
