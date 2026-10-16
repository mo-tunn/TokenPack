"""Freeze-policy regression tests; do NOT open/download official test split."""
from __future__ import annotations

import pytest

from submission.experiments.phase3d_frozen_qasper_test import (
    ARMS,PAIRS,RATIOS,fixed_budget,paired_stats,verify_frozen_sources,
)


def test_frozen_dependencies_and_model_hashes_still_match():
    verify_frozen_sources()


def test_frozen_four_arm_policy_and_budgets():
    assert ARMS==(
        "manual-budget-top-k",
        "manual-render-value",
        "lightgbm-budget-top-k",
        "lightgbm-render-value",
    )
    assert RATIOS==(.25,.5)
    assert fixed_budget(1001,.25)==250
    assert fixed_budget(1001,.5)==500
    with pytest.raises(ValueError):
        fixed_budget(1000,.75)


def test_paired_metrics_reject_unmatched_result():
    rows=[{"case_id":"paper:q","paper_id":"paper","budget_ratio":.25,
           "method":"manual-budget-top-k","gold_5gram_recall":.2}]
    with pytest.raises(AssertionError,match="paired"):
        paired_stats(rows)


def test_primary_comparison_is_predeclared():
    assert PAIRS[0]==("manual-render-value","manual-budget-top-k")
    assert ("lightgbm-render-value","manual-render-value") in PAIRS
