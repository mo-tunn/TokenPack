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
    assert row_a["selected_lexical_evidence_coverage"] == 1.0
    assert row_a["final_answer_context_token_overlap_f1"] == 1.0
    assert row_b["compression_hurt_lexical_coverage"] == 1.0

    summary = _summarize([row_a, row_b], processed_papers=2, processed_questions=2)

    assert len(summary) == 1
    assert summary[0]["pipeline"] == "tokenpack+llmlingua2"
    assert summary[0]["runs"] == 2
    assert summary[0]["selection_saving_vs_full"] == pytest.approx(0.2)
    assert summary[0]["compression_saving_vs_selected"] == pytest.approx(0.375)
    assert summary[0]["avg_final_tokens"] == 5.0
    assert summary[0]["compression_helped_lexical_coverage_rate"] == 0.0
    assert summary[0]["compression_hurt_lexical_coverage_rate"] == pytest.approx(1.0)


def test_offline_qasper_records_preserve_boundaries_and_config(monkeypatch, tmp_path):
    import json
    import qasper_compression_eval as module
    class Embedder:
        model_name = "fixture"
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(module, "make_embedder", lambda **kwargs: Embedder())
    monkeypatch.setattr(sys, "argv", ["qasper", "--data-file", str(ROOT / "tests/fixtures/qasper_tiny.json"),
                        "--pipelines", "only-tokenpack", "--budget-ratios", "1.0", "--output-dir", str(tmp_path / "out"),
                        "--work-dir", str(tmp_path / "work")])
    assert module.main() == 0
    records = [json.loads(x) for x in (tmp_path / "out/qasper_evaluation_records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["selected_segments"]
    assert records[0]["config_hash"]
    assert records[0]["annotation_groups"]
    assert records[0]["selected_metrics"]["source_span_coverage"] == 1
    assert records[0]["timing"]["stages"]["generation"]["status"] == "skipped"


def test_qasper_alternative_annotations_and_exclusions():
    from qasper_selector_eval import _questions_from_qasper_row
    counts = {}
    data = {"id": "p", "qas": {"question": ["q", "unknown"], "question_id": ["a", "b"],
            "answers": [{"answer": [{"evidence": ["first"], "free_form_answer": "yes"},
                                    {"evidence": ["alternative"], "free_form_answer": "yes"}]},
                        {"answer": [{"unanswerable": True}]}]}}
    result = _questions_from_qasper_row(data, counts)
    assert result[0].evidence_groups == [["first"], ["alternative"]]
    assert counts == {"unanswerable": 1}


def test_report_reads_legacy_and_v2_scalar_names():
    import qasper_compression_report as report
    from submission.evaluation.evidence import rename_legacy_metrics
    rows = report._read_csv(report.DEFAULT_CURRENT_SUMMARY)
    only_tp = next(row for row in rows if row["pipeline"] == "only-tokenpack")
    converted = report._current_report_row(only_tp)
    assert converted.lexical_evidence_coverage == pytest.approx(.9340091954753705)
    assert report._current_report_row(rename_legacy_metrics(only_tp)) == converted
    assert "Lexical Coverage" in report._comparison_table_tex([converted])
