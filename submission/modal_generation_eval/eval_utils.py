from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METHODS = [
    "full-document",
    "budget-top-k-50",
    "tokenpack-50",
    "tokenpack-score-sorted-50",
    "tokenpack-density-sorted-50",
    "tokenpack-score-sorted-strong-50",
    "tokenpack-score-sorted-grounded-50",
    "tokenpack-score-sorted-extractive-50",
    "only-llmlingua2-rate050",
]
TOKENPACK_METHOD = "tokenpack-50"
JUDGE_SCORE_FIELDS = ("correctness_0_2", "groundedness_0_2", "completeness_0_2")
JUDGE_BOOL_FIELDS = ("hallucination", "insufficient_when_answerable")


@dataclass(frozen=True, slots=True)
class PromptTemplates:
    answer: str
    judge: str
    answer_strict: str
    answer_grounded: str
    answer_extractive: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def load_prompt_templates(prompt_dir: Path) -> PromptTemplates:
    answer = (prompt_dir / "answer_prompt.txt").read_text(encoding="utf-8")
    strict_path = prompt_dir / "answer_prompt_strict.txt"
    grounded_path = prompt_dir / "answer_prompt_grounded.txt"
    extractive_path = prompt_dir / "answer_prompt_extractive.txt"
    return PromptTemplates(
        answer=answer,
        judge=(prompt_dir / "judge_prompt.txt").read_text(encoding="utf-8"),
        answer_strict=strict_path.read_text(encoding="utf-8") if strict_path.exists() else answer,
        answer_grounded=grounded_path.read_text(encoding="utf-8") if grounded_path.exists() else answer,
        answer_extractive=extractive_path.read_text(encoding="utf-8") if extractive_path.exists() else answer,
    )


def answer_template_for_variant(templates: PromptTemplates, variant: str) -> str:
    if variant == "strict":
        return templates.answer_strict
    if variant == "grounded":
        return templates.answer_grounded
    if variant == "extractive":
        return templates.answer_extractive
    return templates.answer


def render_answer_prompt(template: str, *, question: str, context: str) -> str:
    return template.format(question=question.strip(), context=context.strip())


def render_judge_prompt(
    template: str,
    *,
    question: str,
    gold_answer: str,
    evidence_texts: list[str],
    model_answer: str,
) -> str:
    evidence = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(evidence_texts, start=1))
    return template.format(
        question=question.strip(),
        gold_answer=gold_answer.strip(),
        evidence=evidence.strip(),
        model_answer=model_answer.strip(),
    )


def stable_case_id(paper_id: str, question_id: str) -> str:
    return f"{paper_id}::{question_id}"


def prompt_token_estimate(text: str) -> int:
    # Cheap, tokenizer-free estimate used only for dry-run sizing.
    return max(1, len(re.findall(r"\S+", text)))


def parse_judge_json(raw: str) -> dict[str, Any]:
    payload_text = _extract_json_object(raw)
    if payload_text is None:
        recovered = _parse_judge_fields_from_text(raw)
        if recovered is not None:
            return recovered
        return _parse_failure(raw, "no_json_object")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        recovered = _parse_judge_fields_from_text(raw)
        if recovered is not None:
            recovered["judge_parse_error"] = f"recovered_from_json_decode_error:{exc.msg}"
            return recovered
        return _parse_failure(raw, f"json_decode_error:{exc.msg}")
    parsed: dict[str, Any] = {
        "judge_parse_failed": False,
        "judge_raw": raw,
    }
    for field in JUDGE_SCORE_FIELDS:
        parsed[field] = _clamp_score(payload.get(field))
    for field in JUDGE_BOOL_FIELDS:
        parsed[field] = _to_bool(payload.get(field))
    parsed["short_rationale"] = str(payload.get("short_rationale") or "").strip()[:600]
    missing = [field for field in (*JUDGE_SCORE_FIELDS, *JUDGE_BOOL_FIELDS) if parsed[field] is None]
    if missing:
        parsed["judge_parse_failed"] = True
        parsed["judge_parse_error"] = "missing_or_invalid:" + ",".join(missing)
    return parsed


def _parse_judge_fields_from_text(raw: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] = {
        "judge_parse_failed": False,
        "judge_parse_recovered": True,
        "judge_raw": raw,
    }
    for field in JUDGE_SCORE_FIELDS:
        parsed[field] = _recover_score_field(raw, field)
    for field in JUDGE_BOOL_FIELDS:
        parsed[field] = _recover_bool_field(raw, field)
    parsed["short_rationale"] = _recover_rationale(raw)
    missing = [field for field in (*JUDGE_SCORE_FIELDS, *JUDGE_BOOL_FIELDS) if parsed[field] is None]
    if missing:
        return None
    return parsed


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in [_refresh_judge_parse(row) for row in rows]:
        grouped.setdefault(str(row.get("method")), []).append(row)
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        group = grouped.get(method, [])
        completed = [row for row in group if row.get("status") == "completed"]
        judged = [row for row in completed if not row.get("judge_parse_failed")]
        summary.append(
            {
                "method": method,
                "runs": len(group),
                "completed": len(completed),
                "judged": len(judged),
                "avg_context_tokens": _mean(row.get("context_tokens") for row in group),
                "avg_answer_tokens": _mean(row.get("answer_tokens") for row in completed),
                "avg_judge_score": _mean(_overall_score(row) for row in judged),
                "correct_rate": _mean(1.0 if _score(row, "correctness_0_2") >= 2 else 0.0 for row in judged),
                "grounded_rate": _mean(1.0 if _score(row, "groundedness_0_2") >= 2 else 0.0 for row in judged),
                "hallucination_rate": _mean(1.0 if row.get("hallucination") is True else 0.0 for row in judged),
                "insufficient_rate": _mean(
                    1.0 if row.get("insufficient_when_answerable") is True else 0.0 for row in judged
                ),
                "parse_failure_rate": 1.0 - (len(judged) / len(completed)) if completed else 0.0,
            }
        )
    return summary


def pairwise_tokenpack_wins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case_method: dict[tuple[str, str], dict[str, Any]] = {}
    for row in [_refresh_judge_parse(row) for row in rows]:
        if row.get("status") != "completed" or row.get("judge_parse_failed"):
            continue
        case_id = str(row.get("case_id") or stable_case_id(str(row.get("paper_id")), str(row.get("question_id"))))
        by_case_method[(case_id, str(row.get("method")))] = row
    results: list[dict[str, Any]] = []
    baselines = [method for method in METHODS if method != TOKENPACK_METHOD]
    for baseline in baselines:
        wins = ties = losses = compared = 0
        for (case_id, method), tokenpack_row in by_case_method.items():
            if method != TOKENPACK_METHOD:
                continue
            baseline_row = by_case_method.get((case_id, baseline))
            if baseline_row is None:
                continue
            compared += 1
            delta = _overall_score(tokenpack_row) - _overall_score(baseline_row)
            if delta > 0.05:
                wins += 1
            elif delta < -0.05:
                losses += 1
            else:
                ties += 1
        results.append(
            {
                "baseline": baseline,
                "compared": compared,
                "tokenpack_wins": wins,
                "ties": ties,
                "tokenpack_losses": losses,
                "tokenpack_win_rate": wins / compared if compared else 0.0,
            }
        )
    return results


def _refresh_judge_parse(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("judge_parse_failed") or not row.get("judge_raw"):
        return row
    refreshed = dict(row)
    refreshed.update(parse_judge_json(str(row.get("judge_raw") or "")))
    return refreshed


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(summary_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{LLM-as-Judge Generation Quality on 200 QASPER Questions}",
        r"\label{tab:qasper-generation-quality}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Method & Completed & Tokens & Judge & Correct & Grounded & Halluc. \\",
        r"\hline",
    ]
    for row in summary_rows:
        method = _latex_method(str(row["method"]))
        lines.append(
            f"{method} & {int(row['completed'])} & {float(row['avg_context_tokens']):.0f} & "
            f"{float(row['avg_judge_score']):.2f} & {float(row['correct_rate']):.3f} & "
            f"{float(row['grounded_rate']):.3f} & {float(row['hallucination_rate']):.3f} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"{\footnotesize Local open-source LLM-as-a-judge scores; not a substitute for human evaluation.}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_json_object(raw: str) -> str | None:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_failure(raw: str, reason: str) -> dict[str, Any]:
    return {
        "correctness_0_2": None,
        "groundedness_0_2": None,
        "completeness_0_2": None,
        "hallucination": None,
        "insufficient_when_answerable": None,
        "short_rationale": "",
        "judge_parse_failed": True,
        "judge_parse_error": reason,
        "judge_raw": raw,
    }


def _clamp_score(value: Any) -> int | None:
    try:
        return max(0, min(2, int(value)))
    except (TypeError, ValueError):
        return None


def _recover_score_field(raw: str, field: str) -> int | None:
    label = re.escape(field)
    fuzzy_label = field.replace("_0_2", "").replace("_", r"[\s_-]*")
    patterns = [
        rf'"{label}"\s*:\s*([0-2])',
        rf"{label}\s*[:=]\s*([0-2])",
        rf"{fuzzy_label}\s*(?:score\s*)?(?:is|=|:)\s*([0-2])",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _clamp_score(match.group(1))
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _recover_bool_field(raw: str, field: str) -> bool | None:
    label = re.escape(field)
    fuzzy_label = field.replace("_", r"[\s_-]*")
    patterns = [
        rf'"{label}"\s*:\s*(true|false)',
        rf"{label}\s*[:=]\s*(true|false|yes|no|0|1)",
        rf"{fuzzy_label}\s*(?:is|=|:)\s*(true|false|yes|no|0|1)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return _to_bool(match.group(1))
    return None


def _recover_rationale(raw: str) -> str:
    match = re.search(r'"short_rationale"\s*:\s*"([^"]*)"', raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()[:600]
    return raw.strip().splitlines()[0][:600] if raw.strip() else ""


def _score(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    return float(value) if value is not None else 0.0


def _overall_score(row: dict[str, Any]) -> float:
    return sum(_score(row, field) for field in JUDGE_SCORE_FIELDS) / 3.0


def _mean(values: Iterable[Any]) -> float:
    numeric: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(numeric) / len(numeric) if numeric else 0.0


def _latex_method(method: str) -> str:
    labels = {
        "full-document": "Full document",
        "budget-top-k-50": "Budget-top-k 50\\%",
        "tokenpack-50": "TokenPack 50\\%",
        "tokenpack-score-sorted-50": "TokenPack score-sorted 50\\%",
        "tokenpack-density-sorted-50": "TokenPack density-sorted 50\\%",
        "tokenpack-score-sorted-strong-50": "TokenPack score-sorted strict 50\\%",
        "tokenpack-score-sorted-grounded-50": "TokenPack score-sorted grounded 50\\%",
        "tokenpack-score-sorted-extractive-50": "TokenPack score-sorted extractive 50\\%",
        "only-llmlingua2-rate050": "Only LLMLingua-2 50\\%",
    }
    return labels.get(method, method.replace("_", r"\_"))
