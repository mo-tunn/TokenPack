from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LONGBENCH = ROOT / "submission" / "longbench_eval"
if str(LONGBENCH) not in sys.path:
    sys.path.insert(0, str(LONGBENCH))

from client import _render_selected_context, build_tasks  # type: ignore[import-not-found]
from eval_utils import (  # type: ignore[import-not-found]
    parse_choice,
    parse_grounded_answer,
    parse_grounding_judge,
    quote_found_in_context,
    summarize_grounded_rows,
    summarize_rows,
)
from tokenpack.models import Chunk, ScoredChunk
import client as longbench_client  # type: ignore[import-not-found]


def test_parse_choice_extracts_first_letter() -> None:
    assert parse_choice("The answer is C.") == "C"
    assert parse_choice("b") == "B"
    assert parse_choice("not sure") == ""


def test_parse_grounded_answer_json() -> None:
    parsed = parse_grounded_answer(
        '{"answer": "C", "rationale": "The context supports gamma.", "evidence_quote": "Gamma is explicitly named."}'
    )

    assert parsed["prediction"] == "C"
    assert parsed["rationale"] == "The context supports gamma."
    assert parsed["evidence_quote"] == "Gamma is explicitly named."


def test_parse_grounded_answer_truncated_json() -> None:
    parsed = parse_grounded_answer(
        '{"answer": "D", "rationale": "The context supports delta.", "evidence_quote": "Delta appears in the context'
    )

    assert parsed["prediction"] == "D"
    assert parsed["rationale"] == "The context supports delta."
    assert parsed["evidence_quote"] == "Delta appears in the context"


def test_parse_grounding_judge_json_and_quote_lookup() -> None:
    parsed = parse_grounding_judge(
        '{"supported_answer": true, "supported_rationale": false, "unsupported_claims": true, '
        '"evidence_quote_supports_answer": true, "explanation": "Rationale adds detail."}'
    )

    assert parsed["judge_parse_failure"] is False
    assert parsed["supported_answer"] is True
    assert parsed["supported_rationale"] is False
    assert parsed["unsupported_claims"] is True
    assert parsed["evidence_quote_supports_answer"] is True
    assert quote_found_in_context("Beta evidence appears here", "Alpha. Beta evidence appears here. Gamma.") is True
    assert quote_found_in_context("missing evidence", "Alpha. Beta evidence appears here. Gamma.") is False


def test_summarize_rows_accuracy() -> None:
    rows = [
        {"method": "tokenpack-50", "status": "completed", "prediction": "A", "correct": True, "source_tokens": 10, "context_tokens": 5, "token_saving_vs_full": 0.5},
        {"method": "tokenpack-50", "status": "completed", "prediction": "B", "correct": False, "source_tokens": 10, "context_tokens": 5, "token_saving_vs_full": 0.5},
    ]

    summary = summarize_rows(rows)
    tokenpack = next(row for row in summary if row["method"] == "tokenpack-50")

    assert tokenpack["answered"] == 2
    assert tokenpack["accuracy"] == 0.5


def test_summarize_grounded_rows_metrics() -> None:
    rows = [
        {
            "method": "tokenpack-50",
            "status": "completed",
            "prediction": "A",
            "correct": True,
            "grounded": True,
            "hallucinated": False,
            "quote_found": True,
            "supported_answer": True,
            "supported_rationale": True,
            "unsupported_claims": False,
            "evidence_quote_supports_answer": True,
            "strict_grounding_failure": False,
            "judge_parse_failure": False,
            "source_tokens": 10,
            "context_tokens": 5,
            "token_saving_vs_full": 0.5,
        },
        {
            "method": "tokenpack-50",
            "status": "completed",
            "prediction": "B",
            "correct": True,
            "grounded": False,
            "hallucinated": True,
            "quote_found": False,
            "supported_answer": True,
            "supported_rationale": False,
            "unsupported_claims": False,
            "evidence_quote_supports_answer": False,
            "strict_grounding_failure": True,
            "judge_parse_failure": False,
            "source_tokens": 10,
            "context_tokens": 5,
            "token_saving_vs_full": 0.5,
        },
    ]

    summary = summarize_grounded_rows(rows)
    tokenpack = next(row for row in summary if row["method"] == "tokenpack-50")

    assert tokenpack["accuracy"] == 1.0
    assert tokenpack["grounded_accuracy"] == 0.5
    assert tokenpack["hallucination_rate"] == 0.0
    assert tokenpack["unsupported_claim_rate"] == 0.0
    assert tokenpack["strict_grounding_failure_rate"] == 0.5
    assert tokenpack["answer_supported_rate"] == 1.0
    assert tokenpack["rationale_supported_rate"] == 0.5
    assert tokenpack["quote_supports_answer_rate"] == 0.5
    assert tokenpack["quote_missing_rate"] == 0.5
    assert tokenpack["correct_but_unsupported_rate"] == 0.5


def test_build_tasks_from_local_json_without_compression(monkeypatch) -> None:
    monkeypatch.setattr(longbench_client, "make_embedder", lambda **_: _TinyEmbedder())
    data_path = ROOT / ".tokenpack" / "test_longbench_eval_fixture.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    context = " ".join(f"word{i}" for i in range(140))
    try:
        data_path.write_text(
            json.dumps(
                [
                    {
                        "_id": "case-1",
                        "domain": "QA",
                        "sub_domain": "Toy",
                        "difficulty": "easy",
                        "length": "short",
                        "question": "Which option is supported?",
                        "choice_A": "Alpha",
                        "choice_B": "Beta",
                        "choice_C": "Gamma",
                        "choice_D": "Delta",
                        "answer": "A",
                        "context": context,
                    }
                ]
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            data_file=str(data_path),
            limit=1,
            source_min_tokens=10,
            source_max_tokens=500,
            max_scanned=10,
            embedding_model="unused",
            chunker="structure-aware",
            scoring="evidence-hybrid",
            selection_strategy="knapsack-redundancy",
            budget_ratio=0.5,
            context_order="source",
            candidate_pool=20,
            target_tokens=40,
            min_tokens=10,
            max_tokens=60,
            reranker="none",
            reranker_model="unused",
            reranker_candidate_pool=10,
            reranker_weight=0.35,
            reranker_allow_download=False,
            cascade_frontier=False,
            skip_compression=True,
        )

        tasks, report = build_tasks(args)

        assert report["cases"] == 1
        assert {task["method"] for task in tasks} == {"full-context", "production-rag-50", "tokenpack-50"}
        assert all(task["answer"] == "A" for task in tasks)
        assert report["production_rag_baseline"] is True

        args.selection_strategy = "budget-top-k"
        greedy_tasks, greedy_report = build_tasks(args)
        assert {task["method"] for task in greedy_tasks} == {"full-context", "production-rag-50", "tokenpack-50"}
        assert greedy_report["selection_strategy"] == "budget-top-k"

        args.diagnostic_selectors = True
        diagnostic_tasks, diagnostic_report = build_tasks(args)
        assert {task["method"] for task in diagnostic_tasks} == {
            "full-context",
            "production-rag-50",
            "similarity-knapsack-50",
            "hybrid-greedy-50",
            "hybrid-knapsack-50",
        }
        assert diagnostic_report["diagnostic_selectors"] is True
    finally:
        if data_path.exists():
            data_path.unlink()


def test_render_context_order_score_and_source() -> None:
    items = [
        _scored("early-low", value=0.10, paragraph=1),
        _scored("late-high", value=0.90, paragraph=3),
        _scored("middle", value=0.50, paragraph=2),
    ]

    score_rendered = _render_selected_context(items, order="score")
    source_rendered = _render_selected_context(items, order="source")

    assert score_rendered.index("Text late-high") < score_rendered.index("Text middle")
    assert source_rendered.index("Text early-low") < source_rendered.index("Text middle")
    assert source_rendered.index("Text middle") < source_rendered.index("Text late-high")


def test_render_context_order_score_then_source_has_priority_without_duplicates() -> None:
    items = [_scored(f"chunk-{index}", value=float(index), paragraph=index) for index in range(8)]

    rendered = _render_selected_context(items, order="score-then-source")

    assert "[Priority Evidence]" in rendered
    assert "[Context Evidence]" in rendered
    assert rendered.index("Text chunk-7") < rendered.index("[Context Evidence]")
    assert rendered.index("Text chunk-0") > rendered.index("[Context Evidence]")
    for index in range(8):
        assert rendered.count(f"Text chunk-{index}") == 1


def _scored(chunk_id: str, *, value: float, paragraph: int) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        text=f"Text {chunk_id}",
        source_path="doc.txt",
        document_index=0,
        start_page=None,
        end_page=None,
        start_paragraph=paragraph,
        end_paragraph=paragraph,
        char_start=paragraph * 10,
        char_end=paragraph * 10 + 5,
        token_count=5,
    )
    return ScoredChunk(chunk=chunk, value=value, raw_similarity=value, weight=5)


class _TinyEmbedder:
    model_name = "test-tiny-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lowered else 0.0,
                    1.0 if "beta" in lowered else 0.0,
                    1.0 if "gamma" in lowered else 0.0,
                    1.0 if "delta" in lowered else 0.0,
                ]
            )
        return vectors
