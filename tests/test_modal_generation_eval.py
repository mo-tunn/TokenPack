from __future__ import annotations

from submission.modal_generation_eval.eval_utils import (
    PromptTemplates,
    answer_template_for_variant,
    pairwise_tokenpack_wins,
    parse_judge_json,
    stable_case_id,
    summarize_rows,
)


def test_parse_judge_json_from_fenced_response() -> None:
    parsed = parse_judge_json(
        """```json
        {
          "correctness_0_2": 2,
          "groundedness_0_2": 1,
          "completeness_0_2": 2,
          "hallucination": false,
          "insufficient_when_answerable": "false",
          "short_rationale": "Supported but a little terse."
        }
        ```"""
    )

    assert parsed["judge_parse_failed"] is False
    assert parsed["correctness_0_2"] == 2
    assert parsed["groundedness_0_2"] == 1
    assert parsed["hallucination"] is False
    assert parsed["insufficient_when_answerable"] is False


def test_parse_judge_json_marks_malformed_response() -> None:
    parsed = parse_judge_json("not json at all")

    assert parsed["judge_parse_failed"] is True
    assert parsed["judge_parse_error"] == "no_json_object"
    assert parsed["correctness_0_2"] is None


def test_parse_judge_json_recovers_fields_from_text() -> None:
    parsed = parse_judge_json(
        "The correctness score is 2, groundedness score is 1, completeness score is 0. "
        '"hallucination": false, "insufficient_when_answerable": true'
    )

    assert parsed["judge_parse_failed"] is False
    assert parsed["judge_parse_recovered"] is True
    assert parsed["correctness_0_2"] == 2
    assert parsed["groundedness_0_2"] == 1
    assert parsed["completeness_0_2"] == 0
    assert parsed["hallucination"] is False
    assert parsed["insufficient_when_answerable"] is True


def test_answer_template_for_variant_selects_prompt_family() -> None:
    templates = PromptTemplates(
        answer="default",
        judge="judge",
        answer_strict="strict",
        answer_grounded="grounded",
        answer_extractive="extractive",
    )

    assert answer_template_for_variant(templates, "grounded") == "grounded"
    assert answer_template_for_variant(templates, "extractive") == "extractive"
    assert answer_template_for_variant(templates, "strict") == "strict"
    assert answer_template_for_variant(templates, "unknown") == "default"


def test_summary_and_pairwise_tokenpack_wins() -> None:
    rows = [
        _row("paper-a", "q1", "tokenpack-50", 2, 2, 2, 1000),
        _row("paper-a", "q1", "budget-top-k-50", 1, 2, 1, 980),
        _row("paper-a", "q1", "full-document", 2, 2, 2, 2000),
        _row("paper-a", "q2", "tokenpack-50", 1, 1, 1, 900),
        _row("paper-a", "q2", "budget-top-k-50", 2, 2, 2, 970),
    ]

    summary = summarize_rows(rows)
    tokenpack_summary = next(row for row in summary if row["method"] == "tokenpack-50")
    pairwise = pairwise_tokenpack_wins(rows)
    topk_pair = next(row for row in pairwise if row["baseline"] == "budget-top-k-50")

    assert tokenpack_summary["completed"] == 2
    assert tokenpack_summary["judged"] == 2
    assert tokenpack_summary["avg_context_tokens"] == 950
    assert topk_pair["compared"] == 2
    assert topk_pair["tokenpack_wins"] == 1
    assert topk_pair["tokenpack_losses"] == 1


def _row(
    paper_id: str,
    question_id: str,
    method: str,
    correctness: int,
    groundedness: int,
    completeness: int,
    context_tokens: int,
) -> dict:
    return {
        "paper_id": paper_id,
        "question_id": question_id,
        "case_id": stable_case_id(paper_id, question_id),
        "method": method,
        "status": "completed",
        "judge_parse_failed": False,
        "correctness_0_2": correctness,
        "groundedness_0_2": groundedness,
        "completeness_0_2": completeness,
        "hallucination": False,
        "insufficient_when_answerable": False,
        "context_tokens": context_tokens,
        "answer_tokens": 24,
    }
