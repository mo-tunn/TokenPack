from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hotpotqa_selector_eval import (  # noqa: E402
    _evaluate_question,
    _load_hotpot_rows,
    _question_from_row,
)
from qasper_selector_eval import _build_index, _csv_list, _parse_budgets  # noqa: E402
from tokenpack.chunk_profiles import resolve_chunk_size_config  # noqa: E402
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "heterogeneity_advantage"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "heterogeneity_advantage"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether knapsack's advantage over budget-top-k grows when "
            "documents have more heterogeneous chunk token costs."
        )
    )
    parser.add_argument("--dataset", choices=["hotpotqa"], default="hotpotqa")
    parser.add_argument("--data-file", help="Optional local HotpotQA JSON/JSONL/Arrow file.")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["semantic-threshold", "structure-aware"], default="structure-aware")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="evidence-hybrid",
    )
    parser.add_argument("--baseline", default="budget-top-k")
    parser.add_argument("--challenger", default="knapsack-redundancy")
    parser.add_argument("--budget-ratios", default="0.30,0.40,0.50")
    parser.add_argument("--budgets")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--block-level", choices=["title", "sentence"], default="title")
    parser.add_argument("--levels", default="medium,hard")
    parser.add_argument("--target-tokens", type=int, default=140)
    parser.add_argument("--min-tokens", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument(
        "--chunk-size-preset",
        choices=["manual", "default", "low-budget"],
        default="manual",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter()
    embedder = make_embedder(model_name=args.model, local_files_only=True)
    chunk_size = resolve_chunk_size_config(
        args.chunk_size_preset,
        args.target_tokens,
        args.min_tokens,
        args.max_tokens,
    )
    levels = {item.strip() for item in args.levels.split(",") if item.strip()}
    rows = list(_load_hotpot_rows(args.data_file, args.split))

    per_example: list[dict[str, Any]] = []
    processed = 0
    for row in rows:
        if processed >= args.max_examples:
            break
        if levels and str(row.get("level", "")) not in levels:
            continue
        question = _question_from_row(row, processed, args.block_level, token_counter)
        if not question.evidence_texts or not question.blocks:
            continue
        index = _build_index(
            paper_id=question.example_id,
            blocks=question.blocks,
            embedder=embedder,
            work_dir=work_dir,
            chunker_name=args.chunker,
            target_tokens=chunk_size.target_tokens,
            min_tokens=chunk_size.min_tokens,
            max_tokens=chunk_size.max_tokens,
            semantic_threshold=args.semantic_threshold,
            token_counter=token_counter,
        )
        if len(index.chunks) < 2:
            continue
        query_embedding = embedder.embed([question.question])[0]
        scored = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            scoring=args.scoring,
            query_text=question.question,
            redundancy_candidate_pool=args.candidate_pool,
        )
        stats = _chunk_stats([chunk.token_count for chunk in index.chunks])
        budgets = _parse_budgets(args.budgets, args.budget_ratios, question.source_tokens)
        processed += 1
        for budget_spec in budgets:
            baseline = _evaluate_question(question, scored, args.baseline, budget_spec, args.candidate_pool)
            challenger = _evaluate_question(question, scored, args.challenger, budget_spec, args.candidate_pool)
            per_example.append(
                {
                    "dataset": args.dataset,
                    "example_id": question.example_id,
                    "budget": budget_spec.label,
                    "budget_tokens": budget_spec.tokens,
                    "source_tokens": question.source_tokens,
                    "chunk_count": len(index.chunks),
                    **stats,
                    "baseline": args.baseline,
                    "challenger": args.challenger,
                    "baseline_supporting_fact_recall": baseline["supporting_fact_recall"],
                    "challenger_supporting_fact_recall": challenger["supporting_fact_recall"],
                    "delta_supporting_fact_recall": challenger["supporting_fact_recall"]
                    - baseline["supporting_fact_recall"],
                    "baseline_complete_support_rate": baseline["complete_support_rate"],
                    "challenger_complete_support_rate": challenger["complete_support_rate"],
                    "delta_complete_support_rate": challenger["complete_support_rate"]
                    - baseline["complete_support_rate"],
                    "baseline_supporting_title_recall": baseline["supporting_title_recall"],
                    "challenger_supporting_title_recall": challenger["supporting_title_recall"],
                    "delta_supporting_title_recall": challenger["supporting_title_recall"]
                    - baseline["supporting_title_recall"],
                    "baseline_answer_token_recall": baseline["answer_token_recall"],
                    "challenger_answer_token_recall": challenger["answer_token_recall"],
                    "delta_answer_token_recall": challenger["answer_token_recall"] - baseline["answer_token_recall"],
                    "baseline_used_tokens": baseline["avg_used_tokens"],
                    "challenger_used_tokens": challenger["avg_used_tokens"],
                    "baseline_budget_utilization": baseline["budget_utilization"],
                    "challenger_budget_utilization": challenger["budget_utilization"],
                }
            )

    if not per_example:
        raise SystemExit("No examples were evaluated.")

    _assign_quantile_buckets(per_example, "chunk_token_cv", "heterogeneity_bucket")
    _assign_quantile_buckets(per_example, "source_tokens", "length_bucket")
    summary = _summarize(per_example)
    correlations = _correlations(per_example)

    per_example_path = output_dir / "heterogeneity_advantage_raw.csv"
    summary_path = output_dir / "heterogeneity_advantage_summary.csv"
    correlation_path = output_dir / "heterogeneity_advantage_correlations.csv"
    _write_csv(per_example, per_example_path)
    _write_csv(summary, summary_path)
    _write_csv(correlations, correlation_path)
    print(f"Wrote {per_example_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {correlation_path}")
    print(f"Processed examples={processed}, evaluated rows={len(per_example)}")
    return 0


def _chunk_stats(weights: list[int]) -> dict[str, float]:
    clean = [max(1, int(weight)) for weight in weights]
    mean = statistics.fmean(clean)
    stdev = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "chunk_token_mean": mean,
        "chunk_token_stdev": stdev,
        "chunk_token_cv": stdev / mean if mean else 0.0,
        "chunk_token_min": float(min(clean)),
        "chunk_token_max": float(max(clean)),
        "chunk_token_max_min_ratio": max(clean) / max(1, min(clean)),
        "chunk_token_gini": _gini(clean),
    }


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    total = sum(ordered)
    if total <= 0.0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted) / (len(ordered) * total) - (len(ordered) + 1.0) / len(ordered)


def _assign_quantile_buckets(rows: list[dict[str, Any]], value_key: str, bucket_key: str) -> None:
    values = sorted(float(row[value_key]) for row in rows)
    low_cut = values[max(0, math.floor((len(values) - 1) / 3))]
    high_cut = values[max(0, math.floor(2 * (len(values) - 1) / 3))]
    for row in rows:
        value = float(row[value_key])
        if value <= low_cut:
            bucket = "low"
        elif value <= high_cut:
            bucket = "medium"
        else:
            bucket = "high"
        row[bucket_key] = bucket


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for view, bucket_key in [
            ("heterogeneity", "heterogeneity_bucket"),
            ("length", "length_bucket"),
            ("heterogeneity_x_length", "joint_bucket"),
        ]:
            if view == "heterogeneity_x_length":
                row["joint_bucket"] = f"{row['heterogeneity_bucket']}_hetero/{row['length_bucket']}_length"
            groups.setdefault((str(row["budget"]), view, str(row[bucket_key])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (budget, view, bucket), group_rows in sorted(groups.items(), key=_summary_sort_key):
        summary.append(
            {
                "budget": budget,
                "view": view,
                "bucket": bucket,
                "runs": len(group_rows),
                "avg_source_tokens": _avg(group_rows, "source_tokens"),
                "avg_chunk_count": _avg(group_rows, "chunk_count"),
                "avg_chunk_token_cv": _avg(group_rows, "chunk_token_cv"),
                "avg_chunk_token_gini": _avg(group_rows, "chunk_token_gini"),
                "baseline_supporting_fact_recall": _avg(group_rows, "baseline_supporting_fact_recall"),
                "challenger_supporting_fact_recall": _avg(group_rows, "challenger_supporting_fact_recall"),
                "delta_supporting_fact_recall": _avg(group_rows, "delta_supporting_fact_recall"),
                "baseline_complete_support_rate": _avg(group_rows, "baseline_complete_support_rate"),
                "challenger_complete_support_rate": _avg(group_rows, "challenger_complete_support_rate"),
                "delta_complete_support_rate": _avg(group_rows, "delta_complete_support_rate"),
                "baseline_supporting_title_recall": _avg(group_rows, "baseline_supporting_title_recall"),
                "challenger_supporting_title_recall": _avg(group_rows, "challenger_supporting_title_recall"),
                "delta_supporting_title_recall": _avg(group_rows, "delta_supporting_title_recall"),
                "win_rate": _rate(group_rows, "delta_supporting_fact_recall", lambda value: value > 1e-9),
                "tie_rate": _rate(group_rows, "delta_supporting_fact_recall", lambda value: abs(value) <= 1e-9),
                "loss_rate": _rate(group_rows, "delta_supporting_fact_recall", lambda value: value < -1e-9),
            }
        )
    return summary


def _summary_sort_key(item: tuple[tuple[str, str, str], list[dict[str, Any]]]) -> tuple[float, str, int, str]:
    budget, view, bucket = item[0]
    budget_sort = float(budget.strip("%")) if budget.endswith("%") else float(budget)
    bucket_order = {"low": 0, "medium": 1, "high": 2}
    first = bucket.split("_", maxsplit=1)[0]
    return (budget_sort, view, bucket_order.get(first, 9), bucket)


def _correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    budgets = sorted({str(row["budget"]) for row in rows}, key=lambda value: float(value.strip("%")) if value.endswith("%") else float(value))
    for budget in budgets:
        budget_rows = [row for row in rows if row["budget"] == budget]
        for metric in ["chunk_token_cv", "chunk_token_gini", "source_tokens", "chunk_count"]:
            out.append(
                {
                    "budget": budget,
                    "x_metric": metric,
                    "y_metric": "delta_supporting_fact_recall",
                    "pearson_r": _pearson(
                        [float(row[metric]) for row in budget_rows],
                        [float(row["delta_supporting_fact_recall"]) for row in budget_rows],
                    ),
                    "runs": len(budget_rows),
                }
            )
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0.0 or y_den == 0.0:
        return 0.0
    return numerator / (x_den * y_den)


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def _rate(rows: list[dict[str, Any]], key: str, predicate) -> float:
    return sum(1 for row in rows if predicate(float(row[key]))) / max(1, len(rows))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
