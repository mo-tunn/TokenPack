"""Focused matched-realized-cap unit tests; no model training or network."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from submission.experiments.phase3c_matched_realized_cap import (
    METHODS,empty_choice,load_old_rows
)


def test_zero_cap_context_is_empty_and_safe():
    empty=empty_choice()
    assert empty["used_tokens"]==0
    assert empty["context"]==""
    assert not empty["selected"]
    assert empty["latency_s"]==0.


def test_previous_contexts_loaded_with_unique_paper_question_budget(tmp_path):
    file=tmp_path/"validation_paired_contexts.csv"
    rows=[
        {"case_id":"p:q1","paper_id":"p","budget_ratio":"0.25",
         "method":"manual-budget-top-k","rendered_tokens":"0"},
        {"case_id":"p:q1","paper_id":"p","budget_ratio":"0.5",
         "method":"manual-budget-top-k","rendered_tokens":"500"},
    ]
    with file.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    loaded=load_old_rows(tmp_path)
    assert len(loaded)==2
    assert loaded[("p:q1",.25,"manual-budget-top-k")]["rendered_tokens"]=="0"
    assert loaded[("p:q1",.5,"manual-budget-top-k")]["rendered_tokens"]=="500"


def test_duplicate_historical_control_is_rejected(tmp_path):
    file=tmp_path/"validation_paired_contexts.csv"
    fields=("case_id","budget_ratio","method")
    with file.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        duplicate={"case_id":"dup","budget_ratio":"0.25","method":"manual-budget-top-k"}
        writer.writerow(duplicate)
        writer.writerow(duplicate)
    with pytest.raises(ValueError,match="Duplicate"):
        load_old_rows(tmp_path)


def test_full_design_contains_frozen_control_and_both_frozen_scores():
    assert "manual-original" in METHODS
    assert "manual-render-value-capped" in METHODS
    assert "lightgbm-render-value-capped" in METHODS
