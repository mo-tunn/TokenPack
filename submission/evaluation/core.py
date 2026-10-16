from __future__ import annotations

import math
import random
from collections import Counter

from . import METRIC_VERSION, SCHEMA_VERSION


def indexed(rows):
    output = {}
    for row in rows:
        key = (row.get("case_id"), row.get("method"))
        if not all(isinstance(value, str) and value.strip() for value in key):
            raise ValueError(f"Invalid case/method identifier: {key!r}")
        if key in output:
            raise ValueError(f"Duplicate case/method: {key!r}")
        output[key] = row
    return output


def prepare(rows, expected=None):
    observed = indexed(rows)
    planned = indexed(expected) if expected is not None else dict(observed)
    unexpected = sorted(observed.keys() - planned.keys())
    if unexpected:
        raise ValueError(f"Results outside task manifest: {unexpected!r}")
    return observed, planned


def outcome(row):
    if row is None:
        return "missing_result"
    if row.get("status") != "completed":
        return "execution_failure"
    if row.get("prediction") not in {"A", "B", "C", "D"}:
        return "parse_failure"
    return "completed_answered"


def correct(row):
    return outcome(row) == "completed_answered" and row.get("correct") is True


def judge_known(row):
    return bool(row and outcome(row) == "completed_answered"
                and row.get("judge_parse_failure") is not True
                and all(isinstance(row.get(key), bool) for key in (
                    "supported_answer", "supported_rationale", "unsupported_claims",
                    "evidence_quote_supports_answer")))


def supported(row):
    return bool(judge_known(row) and row["supported_answer"]
                and row["supported_rationale"] and not row["unsupported_claims"]
                and row["evidence_quote_supports_answer"] and row.get("quote_found") is True)


def ratio(n, d):
    return n / d if d else None


def summarize(rows, expected=None, methods=None):
    observed, planned = prepare(rows, expected)
    names = sorted(set(methods or []) | {method for _, method in planned})
    summaries = []
    for method in names:
        keys = sorted(key for key in planned if key[1] == method)
        group = [observed.get(key) for key in keys]
        counts = Counter(outcome(row) for row in group)
        n = len(keys)
        successes = sum(correct(row) for row in group)
        judged = [row for row in group if judge_known(row)]
        answered = counts["completed_answered"]
        summaries.append({
            "schema_version": SCHEMA_VERSION, "metric_version": METRIC_VERSION,
            "method": method,
            "evaluation_status": "not_run" if not n else (
                "partial" if counts["missing_result"] or counts["execution_failure"] else "complete"),
            "denominator_source": "task_manifest" if expected is not None else "observed_only_exploratory",
            "planned": n if expected is not None else None,
            "evaluated_denominator": n, "runs": sum(row is not None for row in group),
            "completed": answered + counts["parse_failure"], "answered": answered,
            **{name: counts[name] for name in ("completed_answered", "parse_failure", "execution_failure", "missing_result")},
            "correct": successes,
            "accuracy_all_planned": ratio(successes, n) if expected is not None else None,
            "accuracy_all_observed": ratio(successes, sum(row is not None for row in group)),
            "accuracy_answered_only": ratio(successes, answered),
            "parse_failure_rate": ratio(counts["parse_failure"], answered + counts["parse_failure"]),
            "judge_evaluable": len(judged), "judge_unknown": answered - len(judged),
            "judge_coverage": ratio(len(judged), answered),
            "grounding_rate": ratio(sum(supported(row) for row in judged), len(judged)),
            "hallucination_rate": ratio(sum(row["unsupported_claims"] for row in judged), len(judged)),
            "verified_correct_supported_all_planned": ratio(sum(correct(row) and supported(row) for row in group), n) if expected is not None else None,
        })
    return summaries


def paired_statistics(differences, clusters=None, *, seed=20260916, replicates=20000, binary=True):
    if not differences or replicates < 1:
        raise ValueError("Nonempty differences and positive replicates required")
    if clusters is not None and (len(clusters) != len(differences) or any(not x for x in clusters)):
        raise ValueError("Complete cluster identifiers required")
    groups = {}
    for index, value in enumerate(differences):
        groups.setdefault(clusters[index] if clusters is not None else index, []).append(value)
    units = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    samples = []
    if len(units) > 1:
        for _ in range(replicates):
            selected = rng.choices(units, k=len(units))
            samples.append(sum(sum(unit) for unit in selected) / sum(len(unit) for unit in selected))
        samples.sort()
    wins, losses = sum(x > 0 for x in differences), sum(x < 0 for x in differences)
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    return {
        "difference": sum(differences) / len(differences),
        "wins": wins, "losses": losses, "ties": len(differences) - wins - losses,
        "bootstrap_95": [samples[int(replicates * .025)], samples[max(0, math.ceil(replicates * .975) - 1)]] if samples else None,
        "bootstrap_unit": "cluster" if clusters is not None else "case",
        "bootstrap_status": "available" if samples else "unavailable_single_unit",
        "clusters": len(units) if clusters is not None else None,
        "exact_mcnemar_p": p if binary and clusters is None else None,
        "seed": seed, "replicates": replicates, "interpretation": "exploratory_unadjusted",
    }


def compare(rows, method, baseline, expected=None, *, grounded=False, seed=20260916, replicates=20000):
    observed, planned = prepare(rows, expected)
    left = {case for case, name in planned if name == method}
    right = {case for case, name in planned if name == baseline}
    if left != right:
        raise ValueError(f"Different case sets: {method} vs {baseline}")
    if not left:
        return {"method": method, "baseline": baseline, "status": "not_run", "compared": 0}
    ids = sorted(left)
    def value(row):
        return int(correct(row) and (not grounded or supported(row)))
    differences = [value(observed.get((case, method))) - value(observed.get((case, baseline))) for case in ids]
    clusters = []
    for case in ids:
        a, b = planned[(case, method)].get("cluster_id"), planned[(case, baseline)].get("cluster_id")
        if a != b:
            raise ValueError("Cluster mismatch between methods")
        clusters.append(a)
    if any(clusters) and not all(clusters):
        raise ValueError("Partial cluster metadata")
    result = paired_statistics(differences, clusters if all(clusters) else None, seed=seed, replicates=replicates)
    base_accuracy = sum(value(observed.get((case, baseline))) for case in ids) / len(ids)
    return {"method": method, "baseline": baseline, "compared": len(ids),
            "status": "partial" if any(outcome(observed.get((case, name))) in {"missing_result", "execution_failure"} for case in ids for name in (method, baseline)) else "complete",
            "denominator_source": "task_manifest" if expected is not None else "observed_only_exploratory",
            "metric": "verified_correct_supported" if grounded else "correctness",
            **result, "difference_percentage_points": 100 * result["difference"],
            "relative_difference": ratio(result["difference"], base_accuracy),
            "win_rate": result["wins"] / len(ids)}
