from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "submission" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from qasper_compression_eval import (  # type: ignore[import-not-found]
    CompressionSetting,
    _build_result_row,
    _parse_compression_settings,
    _parse_pipeline_aliases,
    _setting_label,
    _summarize,
)
from qasper_selector_eval import BudgetSpec, QasperQuestion  # type: ignore[import-not-found]


def test_parse_pipeline_aliases_normalizes_and_deduplicates():
    pipelines = _parse_pipeline_aliases("TokenPack,only-llmlingua2,tokenpack+llmlingua2,tokenpack")

    assert pipelines == ["only-tokenpack", "only-llmlingua2", "tokenpack+llmlingua2"]


def test_parse_compression_settings_builds_variant_specific_grid():
    settings = _parse_compression_settings(
        raw_rates="0.50,0.85",
        raw_targets="128",
        pipelines=["only-llmlingua2", "tokenpack+longllmlingua"],
    )

    assert settings == [
        CompressionSetting(variant="llmlingua2", mode="rate", rate=0.5),
        CompressionSetting(variant="llmlingua2", mode="rate", rate=0.85),
        CompressionSetting(variant="llmlingua2", mode="target", target_tokens=128),
        CompressionSetting(variant="longllmlingua", mode="rate", rate=0.5),
        CompressionSetting(variant="longllmlingua", mode="rate", rate=0.85),
        CompressionSetting(variant="longllmlingua", mode="target", target_tokens=128),
    ]


def test_setting_label_omits_full_document_strategy_name():
    label = _setting_label(
        pipeline="only-llmlingua2",
        selection_strategy="full-document",
        budget_spec=None,
        compression_setting=CompressionSetting(variant="llmlingua2", mode="rate", rate=0.5),
    )

    assert label == "only-llmlingua2 / rate=0.50"


def test_build_result_row_and_summary_capture_compression_deltas():
    question = QasperQuestion(
        paper_id="paper-1",
        title="Paper",
        question_id="q1",
        question="What matters?",
        answer="alpha gamma",
        evidence_texts=["alpha beta gamma"],
    )
    budget = BudgetSpec(label="50%", tokens=8, sort_key=0.5)
    compression = CompressionSetting(variant="llmlingua2", mode="rate", rate=0.5)

    row_a = _build_result_row(
        paper_id="paper-1",
        question=question,
        pipeline="tokenpack+llmlingua2",
        selection_strategy="knapsack-redundancy",
        budget_spec=budget,
        compression_setting=compression,
        source_tokens=10,
        selected_tokens=8,
        final_tokens=4,
        selected_text="alpha beta gamma delta",
        final_text="alpha gamma",
        selection_seconds=0.1,
        compression_seconds=0.2,
    )
    row_b = _build_result_row(
        paper_id="paper-2",
        question=question,
        pipeline="tokenpack+llmlingua2",
        selection_strategy="knapsack-redundancy",
        budget_spec=budget,
        compression_setting=compression,
        source_tokens=10,
        selected_tokens=8,
        final_tokens=6,
        selected_text="alpha beta",
        final_text="alpha",
        selection_seconds=0.2,
        compression_seconds=0.3,
    )

    assert row_a["compression_saving_vs_selected"] == 0.5
    assert row_a["selected_evidence_recall"] == 1.0
    assert row_a["final_answer_token_f1"] == 1.0
    assert row_b["compression_hurt_evidence"] == 1.0

    summary = _summarize([row_a, row_b], processed_papers=2, processed_questions=2)

    assert len(summary) == 1
    assert summary[0]["pipeline"] == "tokenpack+llmlingua2"
    assert summary[0]["runs"] == 2
    assert summary[0]["selection_saving_vs_full"] == pytest.approx(0.2)
    assert summary[0]["compression_saving_vs_selected"] == pytest.approx(0.375)
    assert summary[0]["avg_final_tokens"] == 5.0
    assert summary[0]["compression_helped_evidence_rate"] == 0.0
    assert summary[0]["compression_hurt_evidence_rate"] == pytest.approx(1.0)
