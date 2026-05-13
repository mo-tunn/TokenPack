from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


METHODS = [
    "full-context",
    "production-rag-50",
    "similarity-knapsack-50",
    "hybrid-greedy-50",
    "hybrid-knapsack-50",
    "tokenpack-50",
    "only-longllmlingua-rate050",
    "tokenpack-50+longllmlingua-rate050",
    "tokenpack-60+longllmlingua-rate050",
    "tokenpack-50+longllmlingua-rate065",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # Use literal LF splitting instead of splitlines(); LongBench contexts can
    # contain Unicode line-separator characters that are valid inside JSON strings.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_mc_prompt(task: dict[str, Any]) -> str:
    return (
        "Use only the provided context to answer the multiple-choice question.\n"
        "Choose the single best option. Return only the letter A, B, C, or D.\n\n"
        f"Context:\n{str(task['context']).strip()}\n\n"
        f"Question:\n{str(task['question']).strip()}\n\n"
        f"A. {str(task['choice_A']).strip()}\n"
        f"B. {str(task['choice_B']).strip()}\n"
        f"C. {str(task['choice_C']).strip()}\n"
        f"D. {str(task['choice_D']).strip()}\n\n"
        "Answer:"
    )


def render_grounded_prompt(task: dict[str, Any]) -> str:
    return (
        "Use only the provided context to answer the multiple-choice question.\n"
        "Choose the single best option only when it is directly supported by the context; do not use outside knowledge.\n"
        "Then provide one short rationale and one exact contiguous supporting quote copied from the context.\n"
        "If the context is ambiguous, choose the best-supported option and keep the rationale limited to what the quote proves.\n"
        "Return valid JSON with exactly these keys: answer, rationale, evidence_quote.\n"
        "The answer value must be one of A, B, C, or D.\n\n"
        f"Context:\n{str(task['context']).strip()}\n\n"
        f"Question:\n{str(task['question']).strip()}\n\n"
        f"A. {str(task['choice_A']).strip()}\n"
        f"B. {str(task['choice_B']).strip()}\n"
        f"C. {str(task['choice_C']).strip()}\n"
        f"D. {str(task['choice_D']).strip()}\n\n"
        "JSON:"
    )


def render_grounding_judge_prompt(task: dict[str, Any], answer_row: dict[str, Any]) -> str:
    return (
        "You are checking whether a model answer is grounded in the provided context.\n"
        "Use only the context. Do not use outside knowledge.\n"
        "Return valid JSON with exactly these keys:\n"
        "- supported_answer: true if the selected option is supported by the context.\n"
        "- supported_rationale: true if the rationale is supported by the context.\n"
        "- unsupported_claims: true only if the rationale or answer adds factual claims not supported by the context.\n"
        "- evidence_quote_supports_answer: true if the quoted text directly supports the selected option.\n"
        "- explanation: one short sentence.\n\n"
        f"Context:\n{str(task['context']).strip()}\n\n"
        f"Question:\n{str(task['question']).strip()}\n\n"
        f"A. {str(task['choice_A']).strip()}\n"
        f"B. {str(task['choice_B']).strip()}\n"
        f"C. {str(task['choice_C']).strip()}\n"
        f"D. {str(task['choice_D']).strip()}\n\n"
        f"Model answer: {str(answer_row.get('prediction') or '').strip()}\n"
        f"Model rationale: {str(answer_row.get('rationale') or '').strip()}\n"
        f"Model evidence quote: {str(answer_row.get('evidence_quote') or '').strip()}\n\n"
        "JSON:"
    )


def parse_choice(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"\b([ABCD])\b", cleaned, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def parse_grounded_answer(text: str) -> dict[str, str]:
    payload = _parse_json_object(text)
    if payload:
        answer = parse_choice(str(payload.get("answer") or ""))
        return {
            "prediction": answer,
            "rationale": str(payload.get("rationale") or "").strip(),
            "evidence_quote": str(payload.get("evidence_quote") or "").strip(),
        }
    salvaged = _salvage_json_fields(text, ("answer", "rationale", "evidence_quote"))
    if salvaged:
        return {
            "prediction": parse_choice(salvaged.get("answer", "")),
            "rationale": salvaged.get("rationale", "").strip(),
            "evidence_quote": salvaged.get("evidence_quote", "").strip(),
        }
    return {
        "prediction": parse_choice(text),
        "rationale": "",
        "evidence_quote": "",
    }


def parse_grounding_judge(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    if not payload:
        salvaged = _salvage_json_fields(
            text,
            ("supported_answer", "supported_rationale", "unsupported_claims", "evidence_quote_supports_answer", "explanation"),
        )
        if salvaged:
            return {
                "judge_parse_failure": False,
                "supported_answer": _as_bool(salvaged.get("supported_answer")),
                "supported_rationale": _as_bool(salvaged.get("supported_rationale")),
                "unsupported_claims": _as_bool(salvaged.get("unsupported_claims")),
                "evidence_quote_supports_answer": _as_bool(salvaged.get("evidence_quote_supports_answer")),
                "judge_explanation": salvaged.get("explanation", "").strip(),
            }
        return {
            "judge_parse_failure": True,
            "supported_answer": False,
            "supported_rationale": False,
            "unsupported_claims": True,
            "evidence_quote_supports_answer": False,
            "judge_explanation": "",
        }
    return {
        "judge_parse_failure": False,
        "supported_answer": _as_bool(payload.get("supported_answer")),
        "supported_rationale": _as_bool(payload.get("supported_rationale")),
        "unsupported_claims": _as_bool(payload.get("unsupported_claims")),
        "evidence_quote_supports_answer": _as_bool(payload.get("evidence_quote_supports_answer")),
        "judge_explanation": str(payload.get("explanation") or "").strip(),
    }


def quote_found_in_context(quote: str, context: str) -> bool:
    cleaned_quote = _normalize_space(quote).strip(" \"'")
    if len(cleaned_quote) < 12:
        return False
    return cleaned_quote.lower() in _normalize_space(context).lower()


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method")), []).append(row)
    full_total_latency = _mean(_row_total_latency(row) for row in grouped.get("full-context", []))
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        group = grouped.get(method, [])
        completed = [row for row in group if row.get("status") == "completed"]
        answered = [row for row in completed if row.get("prediction")]
        total_latencies = [_row_total_latency(row) for row in completed]
        avg_total_latency = _mean(total_latencies)
        summary.append(
            {
                "method": method,
                "runs": len(group),
                "completed": len(completed),
                "answered": len(answered),
                "accuracy": _mean(1.0 if row.get("correct") is True else 0.0 for row in answered),
                "avg_source_tokens": _mean(row.get("source_tokens") for row in group),
                "avg_context_tokens": _mean(row.get("context_tokens") for row in group),
                "avg_token_saving_vs_full": _mean(row.get("token_saving_vs_full") for row in group),
                "avg_preprocessing_seconds": _mean(_row_preprocessing_latency(row) for row in completed),
                "avg_answer_latency_seconds": _mean(row.get("answer_latency_seconds") for row in completed),
                "avg_total_latency_seconds": avg_total_latency,
                "p50_total_latency_seconds": _percentile(total_latencies, 0.50),
                "p90_total_latency_seconds": _percentile(total_latencies, 0.90),
                "speedup_vs_full": full_total_latency / avg_total_latency if avg_total_latency > 0 else 0.0,
                "parse_failure_rate": 1.0 - (len(answered) / len(completed)) if completed else 0.0,
            }
        )
    return summary


def summarize_grounded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method")), []).append(row)
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        group = grouped.get(method, [])
        completed = [row for row in group if row.get("status") == "completed"]
        answered = [row for row in completed if row.get("prediction")]
        correct = [row for row in answered if row.get("correct") is True]
        grounded = [row for row in answered if row.get("grounded") is True]
        hallucinated = [row for row in answered if _is_hallucinated(row)]
        unsupported_claims = [row for row in answered if row.get("unsupported_claims") is True]
        strict_ungrounded = [row for row in answered if _has_strict_grounding_failure(row)]
        summary.append(
            {
                "method": method,
                "runs": len(group),
                "completed": len(completed),
                "answered": len(answered),
                "accuracy": _mean(1.0 if row.get("correct") is True else 0.0 for row in answered),
                "grounded_accuracy": _mean(
                    1.0 if row.get("correct") is True and row.get("grounded") is True else 0.0 for row in answered
                ),
                "grounding_rate": len(grounded) / len(answered) if answered else 0.0,
                "hallucination_rate": len(hallucinated) / len(answered) if answered else 0.0,
                "unsupported_claim_rate": len(unsupported_claims) / len(answered) if answered else 0.0,
                "strict_grounding_failure_rate": len(strict_ungrounded) / len(answered) if answered else 0.0,
                "answer_supported_rate": _mean(1.0 if row.get("supported_answer") is True else 0.0 for row in answered),
                "rationale_supported_rate": _mean(1.0 if row.get("supported_rationale") is True else 0.0 for row in answered),
                "quote_supports_answer_rate": _mean(
                    1.0 if row.get("evidence_quote_supports_answer") is True else 0.0 for row in answered
                ),
                "quote_found_rate": _mean(1.0 if row.get("quote_found") is True else 0.0 for row in answered),
                "quote_missing_rate": _mean(1.0 if row.get("quote_found") is not True else 0.0 for row in answered),
                "correct_but_unsupported_rate": (
                    sum(1 for row in correct if row.get("grounded") is not True) / len(correct) if correct else 0.0
                ),
                "judge_parse_failure_rate": _mean(1.0 if row.get("judge_parse_failure") is True else 0.0 for row in completed),
                "avg_source_tokens": _mean(row.get("source_tokens") for row in group),
                "avg_context_tokens": _mean(row.get("context_tokens") for row in group),
                "avg_token_saving_vs_full": _mean(row.get("token_saving_vs_full") for row in group),
                "parse_failure_rate": 1.0 - (len(answered) / len(completed)) if completed else 0.0,
            }
        )
    return summary


def pairwise_rows(rows: list[dict[str, Any]], baseline: str = "only-longllmlingua-rate050") -> list[dict[str, Any]]:
    by_case_method = {
        (str(row.get("case_id")), str(row.get("method"))): row
        for row in rows
        if row.get("status") == "completed" and row.get("prediction")
    }
    compared_methods = [method for method in METHODS if method != baseline]
    output: list[dict[str, Any]] = []
    for method in compared_methods:
        wins = ties = losses = compared = 0
        for (case_id, row_method), row in by_case_method.items():
            if row_method != method:
                continue
            baseline_row = by_case_method.get((case_id, baseline))
            if baseline_row is None:
                continue
            compared += 1
            left = 1 if row.get("correct") is True else 0
            right = 1 if baseline_row.get("correct") is True else 0
            if left > right:
                wins += 1
            elif left < right:
                losses += 1
            else:
                ties += 1
        output.append(
            {
                "method": method,
                "baseline": baseline,
                "compared": compared,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / compared if compared else 0.0,
            }
        )
    return output


def pairwise_grounded_rows(rows: list[dict[str, Any]], baseline: str = "only-longllmlingua-rate050") -> list[dict[str, Any]]:
    by_case_method = {
        (str(row.get("case_id")), str(row.get("method"))): row
        for row in rows
        if row.get("status") == "completed" and row.get("prediction")
    }
    compared_methods = [method for method in METHODS if method != baseline]
    output: list[dict[str, Any]] = []
    for method in compared_methods:
        wins = ties = losses = compared = 0
        for (case_id, row_method), row in by_case_method.items():
            if row_method != method:
                continue
            baseline_row = by_case_method.get((case_id, baseline))
            if baseline_row is None:
                continue
            compared += 1
            left = 1 if row.get("correct") is True and row.get("grounded") is True else 0
            right = 1 if baseline_row.get("correct") is True and baseline_row.get("grounded") is True else 0
            if left > right:
                wins += 1
            elif left < right:
                losses += 1
            else:
                ties += 1
        output.append(
            {
                "method": method,
                "baseline": baseline,
                "compared": compared,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / compared if compared else 0.0,
            }
        )
    return output


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


def _percentile(values: Iterable[Any], q: float) -> float:
    numeric: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    numeric.sort()
    if not numeric:
        return 0.0
    if len(numeric) == 1:
        return numeric[0]
    position = (len(numeric) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(numeric) - 1)
    fraction = position - lower
    return numeric[lower] * (1.0 - fraction) + numeric[upper] * fraction


def _row_preprocessing_latency(row: dict[str, Any]) -> float:
    explicit = row.get("preprocessing_seconds")
    if explicit not in ("", None):
        return float(explicit)
    return float(row.get("selection_seconds") or 0.0) + float(row.get("compression_seconds") or 0.0)


def _row_total_latency(row: dict[str, Any]) -> float:
    explicit = row.get("total_latency_seconds")
    if explicit not in ("", None):
        return float(explicit)
    return _row_preprocessing_latency(row) + float(row.get("answer_latency_seconds") or 0.0)


def _is_hallucinated(row: dict[str, Any]) -> bool:
    if "unsupported_claims" in row:
        return row.get("unsupported_claims") is True
    return row.get("hallucinated") is True


def _has_strict_grounding_failure(row: dict[str, Any]) -> bool:
    if "strict_grounding_failure" in row:
        return row.get("strict_grounding_failure") is True
    return bool(
        row.get("grounded") is not True
        or row.get("quote_found") is not True
        or row.get("supported_answer") is not True
        or row.get("supported_rationale") is not True
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}
    for candidate in _json_candidates(cleaned):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    return candidates


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "supported"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _salvage_json_fields(text: str, keys: tuple[str, ...]) -> dict[str, str]:
    output: dict[str, str] = {}
    for index, key in enumerate(keys):
        next_keys = keys[index + 1 :]
        next_pattern = "|".join(re.escape(next_key) for next_key in next_keys)
        if next_pattern:
            pattern = rf'"{re.escape(key)}"\s*:\s*(.*?)(?=,\s*"(?:{next_pattern})"\s*:|,?\s*\}}|$)'
        else:
            pattern = rf'"{re.escape(key)}"\s*:\s*(.*?)(?=,?\s*\}}|$)'
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        value = match.group(1).strip().rstrip(",").strip()
        output[key] = _decode_partial_json_value(value)
    return output


def _decode_partial_json_value(value: str) -> str:
    if not value:
        return ""
    lowered = value.lower()
    if lowered.startswith("true"):
        return "true"
    if lowered.startswith("false"):
        return "false"
    if value.startswith('"'):
        value = value[1:]
    if value.endswith('"'):
        value = value[:-1]
    value = value.replace('\\"', '"').replace("\\n", "\n")
    return value.strip()
