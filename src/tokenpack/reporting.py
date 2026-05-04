from __future__ import annotations

import csv
from pathlib import Path


def save_markdown_report(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# TokenPack Benchmark Report", ""]
    for run in _budget_runs(payload):
        lines.append(f"## Budget {run['budget']} (effective {run.get('effective_budget', run['budget'])})")
        lines.append("")
        lines.append(
            "| Strategy | Evidence Recall | Evidence Precision | Coverage | Avg Tokens | Budget Util. | Over Budget | Value Density | Redundancy | Latency (s) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for strategy, metrics in run["summary"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        strategy,
                        _fmt(metrics.get("evidence_recall_at_budget")),
                        _fmt(metrics.get("evidence_precision")),
                        _fmt(metrics.get("coverage_ratio")),
                        _fmt(metrics.get("avg_used_tokens")),
                        _fmt(metrics.get("budget_utilization")),
                        _fmt(metrics.get("over_budget_rate")),
                        _fmt(metrics.get("value_density") or metrics.get("avg_value_density")),
                        _fmt(metrics.get("redundancy_score")),
                        _fmt(metrics.get("latency_seconds") or metrics.get("avg_latency_seconds")),
                    ]
                )
                + " |"
            )
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")


def save_csv_report(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget",
        "effective_budget",
        "strategy",
        "evidence_recall_at_budget",
        "evidence_precision",
        "coverage_ratio",
        "avg_used_tokens",
        "budget_utilization",
        "over_budget_rate",
        "avg_over_budget_tokens",
        "value_density",
        "redundancy_score",
        "latency_seconds",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in _budget_runs(payload):
            for strategy, metrics in run["summary"].items():
                writer.writerow(
                    {
                        "budget": run["budget"],
                        "effective_budget": run.get("effective_budget", run["budget"]),
                        "strategy": strategy,
                        "evidence_recall_at_budget": metrics.get("evidence_recall_at_budget"),
                        "evidence_precision": metrics.get("evidence_precision"),
                        "coverage_ratio": metrics.get("coverage_ratio"),
                        "avg_used_tokens": metrics.get("avg_used_tokens"),
                        "budget_utilization": metrics.get("budget_utilization"),
                        "over_budget_rate": metrics.get("over_budget_rate"),
                        "avg_over_budget_tokens": metrics.get("avg_over_budget_tokens"),
                        "value_density": metrics.get("value_density") or metrics.get("avg_value_density"),
                        "redundancy_score": metrics.get("redundancy_score"),
                        "latency_seconds": metrics.get("latency_seconds") or metrics.get("avg_latency_seconds"),
                    }
                )


def _budget_runs(payload: dict) -> list[dict]:
    if "budgets" in payload:
        return list(payload["budgets"])
    return [payload]


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
