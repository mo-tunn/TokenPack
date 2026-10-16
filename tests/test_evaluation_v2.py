from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submission.evaluation.core import compare, paired_statistics, summarize
from submission.evaluation.evidence import evidence_metrics, span_coverage, rename_legacy_metrics
from submission.evaluation.timing import StageTimer, stage_total, timing_record


def row(case, method="tp", **changes):
    return {"case_id": case, "method": method, "status": "completed", "prediction": "A", "correct": True, **changes}


def test_all_planned_denominator_and_unknown_judge():
    expected = [row(str(i)) for i in range(4)]
    rows = [row("0"), row("1", prediction=""), row("2", status="failed")]
    result = summarize(rows, expected, ["absent"])
    tp = next(x for x in result if x["method"] == "tp")
    assert tp["accuracy_all_planned"] == .25
    assert tp["accuracy_answered_only"] == 1
    assert [tp[k] for k in ("completed_answered", "parse_failure", "execution_failure", "missing_result")] == [1, 1, 1, 1]
    assert tp["evaluation_status"] == "partial"
    assert tp["hallucination_rate"] is None
    assert tp["judge_unknown"] == 1
    assert result[0]["evaluation_status"] == "not_run"
    assert result[0]["accuracy_all_planned"] is None
    assert summarize(rows)[0]["accuracy_all_planned"] is None


def test_invalid_inputs_not_silently_intersected():
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row("a"), row("a")])
    with pytest.raises(ValueError, match="Invalid"):
        summarize([row("")])
    with pytest.raises(ValueError, match="outside"):
        summarize([row("b")], [row("a")])
    with pytest.raises(ValueError, match="Different case sets"):
        compare([row("a"), row("b", "base")], "tp", "base")


def test_judge_failure_cannot_become_hallucination():
    bad = row("a", judge_parse_failure=True, unsupported_claims=True,
              supported_answer=False, supported_rationale=False, evidence_quote_supports_answer=False)
    summary = summarize([bad], [bad])[0]
    assert summary["hallucination_rate"] is None
    assert summary["judge_coverage"] == 0
    assert summary["verified_correct_supported_all_planned"] == 0


def test_evidence_order_numbers_negation_and_boundaries():
    groups = [["Dose is not 10 mg", "Second evidence"], ["Alternative evidence"]]
    for text in ("Dose is 10 mg", "Dose is not 20 mg", "mg 10 not is Dose"):
        assert evidence_metrics(groups, [text])["verbatim_evidence_recall"] == 0
    partial = evidence_metrics(groups, ["Dose is not 10 mg"])
    assert partial["verbatim_evidence_recall"] == .5
    assert partial["complete_evidence_set_retained"] is False
    assert evidence_metrics(groups, ["Alternative evidence"])["complete_evidence_set_retained"] is True
    assert evidence_metrics([["alpha beta"]], ["alpha", "beta"])["verbatim_evidence_recall"] == 0
    assert evidence_metrics(groups, None)["verbatim_evidence_recall"] is None
    assert evidence_metrics([], ["anything"])["evidence_status"] == "excluded_no_valid_evidence"
    assert evidence_metrics([["café  x"]], ["cafe\u0301\nx"])["complete_evidence_set_retained"] is True
    assert evidence_metrics([["value 10"]], ["value 100"])["verbatim_evidence_recall"] == 0


def test_metric_migration_is_idempotent_and_preserves_new_metrics():
    migrated = rename_legacy_metrics({"final_evidence_recall": .9, "complete_evidence_rate": .8,
                                     "final_verbatim_evidence_recall": .5, "complete_evidence_set_retained": False,
                                     "total_seconds": 2.0})
    assert rename_legacy_metrics(migrated) == migrated
    assert migrated["final_verbatim_evidence_recall"] == .5
    assert migrated["recorded_selection_compression_seconds"] == 2


def test_source_span_union_and_alternatives():
    gold = [[["p", 0, 10], ["p", 5, 15]], [["p", 20, 30]]]
    assert span_coverage(gold, [["p", 0, 10], ["p", 5, 15]]) == 1
    assert span_coverage(gold, [["p", 0, 5]]) == pytest.approx(1 / 3)
    assert span_coverage(gold, []) == 0
    assert span_coverage(gold, None) is None
    with pytest.raises(ValueError):
        span_coverage(gold, [["p", 10, 2]])


def test_statistics_deterministic_cluster_resampling():
    values = [1, 1, -1, -1]
    result = paired_statistics(values, ["p1", "p1", "p2", "p2"], replicates=100)
    assert result == paired_statistics(values, ["p1", "p1", "p2", "p2"], replicates=100)
    assert result["bootstrap_95"] == [-1, 1]
    assert result["exact_mcnemar_p"] is None
    assert paired_statistics([1, 1], ["p", "p"], replicates=10)["bootstrap_95"] is None
    assert paired_statistics([1, 1, 1, 1, 1], replicates=10)["exact_mcnemar_p"] == .0625


def test_stage_timing_never_double_counts_or_fills_missing():
    ticks = iter([10, 12, 20, 23])
    timer = StageTimer(clock=lambda: next(ticks))
    with timer.measure("selection"):
        with pytest.raises(ValueError):
            with timer.measure("scoring"):
                pass
    assert timer.stages["selection"]["seconds"] == 2
    with pytest.raises(ValueError):
        with timer.measure("selection"):
            pass
    record = timing_record(selection=2, compression=0, generation=3)
    assert stage_total(record) is None
    assert record["request_wall_seconds"] is None
    record = timing_record(document_preparation=10, query_embedding=1, scoring=2,
                           selection=3, render=4, compression=0, generation=5, skipped=("compression",))
    assert stage_total(record) == 25
    assert stage_total(record, indexed=True) == 15


def test_canonical_pilot_counts_and_pairs():
    folder = ROOT / "submission/results/longbench_v2_modal_hybrid_greedy_83_latency"
    def read(name):
        return [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").split("\n") if line.strip()]
    rows = read("longbench_generation_results.jsonl")
    tasks = read("longbench_generation_tasks.jsonl")
    counts = {item["method"]: item["correct"] for item in summarize(rows, tasks)}
    assert counts == {"full-context": 32, "production-rag-50": 34, "tokenpack-50": 37,
                      "only-longllmlingua-rate050": 30, "tokenpack-50+longllmlingua-rate050": 37}
    comparison = compare(rows, "tokenpack-50", "full-context", tasks, replicates=100)
    assert comparison["exact_mcnemar_p"] == pytest.approx(.3017578125)
    assert compare(rows, "tokenpack-50", "production-rag-50", tasks, replicates=100)["exact_mcnemar_p"] == pytest.approx(.5810546875)
    cascade = compare(rows, "tokenpack-50+longllmlingua-rate050", "tokenpack-50", tasks, replicates=100)
    assert (cascade["wins"], cascade["losses"]) == (8, 8)


def test_pairwise_failures_keep_manifest_denominator():
    tasks = [row(case, method) for case in ("a", "b") for method in ("tp", "base")]
    rows = [row("a"), row("a", "base", prediction=""), row("b", "base")]
    result = compare(rows, "tp", "base", tasks, replicates=100)
    assert result["compared"] == 2
    assert (result["wins"], result["losses"]) == (1, 1)
    assert result["status"] == "partial"


def test_render_in_compression_not_double_counted():
    record = timing_record(document_preparation=1, query_embedding=1, scoring=1,
                           selection=1, compression=3, generation=2)
    record["stages"]["render"] = {"status": "included_in_compression", "seconds": None}
    assert stage_total(record) == 9
    record["stages"]["compression"] = {"status": "unmeasured", "seconds": None}
    assert stage_total(record) is None
