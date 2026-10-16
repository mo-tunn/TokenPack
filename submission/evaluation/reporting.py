from __future__ import annotations

import statistics
from pathlib import Path

from .core import compare, correct, judge_known, prepare, ratio, summarize, supported
from .timing import STAGES, stage_total


def mean(values):
    values = [float(value) for value in values if value not in (None, "")]
    return statistics.fmean(values) if values else None


def percentile(values, q):
    values = sorted(float(x) for x in values if x not in (None, ""))
    if not values:
        return None
    p = (len(values) - 1) * q
    low, high = int(p), min(int(p) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (p - low)


def recorded_latency(row):
    # Legacy totals covered selection + compression + generation only.
    value = row.get("recorded_stage_total_seconds", row.get("total_latency_seconds"))
    if value is not None:
        return float(value)
    values = [row.get(key) for key in ("selection_seconds", "compression_seconds", "answer_latency_seconds")]
    return sum(values) if all(x is not None for x in values) else None


def summarize_longbench(rows, expected=None, methods=None):
    result = summarize(rows, expected, methods)
    base = mean(recorded_latency(row) for row in rows if row["method"] == "full-context" and row.get("status") == "completed")
    for summary in result:
        group = [row for row in rows if row["method"] == summary["method"]]
        complete = [row for row in group if row.get("status") == "completed"]
        judged = [row for row in group if judge_known(row)]
        totals = [recorded_latency(row) for row in complete]
        average = mean(totals)
        summary.update({
            "avg_source_tokens": mean(row.get("source_tokens") for row in group),
            "avg_context_tokens": mean(row.get("context_tokens") for row in group),
            "avg_prompt_tokens": mean(row.get("prompt_tokens") for row in group),
            "avg_token_saving_vs_full": mean(row.get("token_saving_vs_full") for row in group),
            "avg_recorded_preprocessing_seconds": mean(row.get("preprocessing_seconds") for row in complete),
            "avg_answer_latency_seconds": mean(row.get("answer_latency_seconds") for row in complete),
            "avg_recorded_stage_total_seconds": average,
            "p50_recorded_stage_total_seconds": percentile(totals, .5),
            "p90_recorded_stage_total_seconds": percentile(totals, .9),
            "p95_recorded_stage_total_seconds": percentile(totals, .95),
            "recorded_stage_speedup_vs_full": base / average if base is not None and average else None,
            "latency_scope": "selection_compression_generation_only_not_end_to_end",
            "unsupported_claim_rate": ratio(sum(row["unsupported_claims"] for row in judged), len(judged)),
            "strict_grounding_failure_rate": ratio(sum(not supported(row) for row in judged), len(judged)),
            "answer_supported_rate": ratio(sum(row["supported_answer"] for row in judged), len(judged)),
            "rationale_supported_rate": ratio(sum(row["supported_rationale"] for row in judged), len(judged)),
            "quote_supports_answer_rate": ratio(sum(row["evidence_quote_supports_answer"] for row in judged), len(judged)),
            "quote_missing_rate": ratio(sum(row.get("quote_found") is not True for row in judged), len(judged)),
            "correct_but_unsupported_rate": ratio(sum(correct(row) and not supported(row) for row in judged), sum(correct(row) for row in judged)),
        })
        timings = [row["timing"] for row in complete if row.get("timing")]
        for stage in STAGES:
            values = [item["stages"][stage]["seconds"] for item in timings
                      if item["stages"][stage]["status"] in {"measured", "skipped"}]
            summary[f"avg_{stage}_seconds"] = mean(values)
            summary[f"{stage}_timing_available"] = len(values)
        summary["avg_new_document_stage_sum_seconds"] = mean(stage_total(item) for item in timings)
        summary["avg_indexed_query_stage_sum_seconds"] = mean(stage_total(item, indexed=True) for item in timings)
    return result


def pairwise_longbench(rows, baseline, expected=None, grounded=False):
    _, planned = prepare(rows, expected)
    names = sorted({method for _, method in planned})
    if baseline not in names:
        return []
    return [compare(rows, name, baseline, expected, grounded=grounded) for name in names if name != baseline]


def write_readout(summary, pairwise, path: Path):
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
    lines = ["# LongBench metric-v2 readout", "",
             "Exploratory. Missing planned results count as incorrect. Unknown judges are not hallucinations.",
             "Latency covers recorded selection + compression + generation only; it is not end-to-end.", "",
             "| Method | Status | Correct / denominator | All-planned accuracy | Observed-only accuracy | Judge coverage | Recorded stage seconds |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['evaluation_status']} | {row['correct']}/{row['evaluated_denominator']} | {fmt(row['accuracy_all_planned'])} | {fmt(row['accuracy_all_observed'])} | {fmt(row['judge_coverage'])} | {fmt(row['avg_recorded_stage_total_seconds'])} |")
    if any(row["judge_evaluable"] for row in summary):
        lines += ["", "| Method | Verified correct-supported / planned | Judge evaluable | Judge unknown | Grounding (evaluable only) | Unsupported claims (evaluable only) |", "|---|---:|---:|---:|---:|---:|"]
        for row in summary:
            lines.append(f"| {row['method']} | {fmt(row['verified_correct_supported_all_planned'])} | {row['judge_evaluable']} | {row['judge_unknown']} | {fmt(row['grounding_rate'])} | {fmt(row['hallucination_rate'])} |")
    lines += ["", "| Method vs baseline | N | Win / tie / loss | Difference (pp) | 95% paired CI | Exact McNemar p |", "|---|---:|---:|---:|---|---:|"]
    for row in pairwise:
        lines.append(f"| {row['method']} vs {row['baseline']} | {row['compared']} | {row['wins']}/{row['ties']}/{row['losses']} | {fmt(row['difference_percentage_points'])} | {row['bootstrap_95']} | {fmt(row['exact_mcnemar_p'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
