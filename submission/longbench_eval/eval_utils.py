from __future__ import annotations

import csv
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from submission.evaluation.reporting import summarize_longbench, pairwise_longbench


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
    keys = ("supported_answer", "supported_rationale", "unsupported_claims", "evidence_quote_supports_answer")
    payload = _parse_json_object(text) or _salvage_json_fields(text, (*keys, "explanation"))
    def boolean(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return None
    values = {key: boolean(payload.get(key)) for key in keys}
    failure = any(value is None for value in values.values())
    return {"judge_parse_failure": failure,
            **({key: None for key in keys} if failure else values),
            "judge_explanation": str(payload.get("explanation") or "")}


def quote_found_in_context(quote: str, context: str) -> bool:
    cleaned_quote = _normalize_space(quote).strip(" \"'")
    if len(cleaned_quote) < 12:
        return False
    return cleaned_quote.lower() in _normalize_space(context).lower()


def summarize_rows(rows, expected=None):
    return summarize_longbench(rows, expected, METHODS)


def summarize_grounded_rows(rows, expected=None):
    return summarize_longbench(rows, expected, METHODS)


def pairwise_rows(rows, baseline="only-longllmlingua-rate050", expected=None):
    return pairwise_longbench(rows, baseline, expected)


def pairwise_grounded_rows(rows, baseline="only-longllmlingua-rate050", expected=None):
    return pairwise_longbench(rows, baseline, expected, grounded=True)


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
