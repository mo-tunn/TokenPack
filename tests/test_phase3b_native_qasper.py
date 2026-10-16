"""Synthetic native QASPER schema regression; no download, ML or inference."""
from __future__ import annotations

import json
import pytest

from submission.experiments.phase3b_native_qasper import (
    audit_native_annotations,load_local_qasper_rows,
    native_blocks_with_section_hints,validate_native_splits,
)
from submission.experiments.qasper_selector_eval import _questions_from_qasper_row
from tokenpack.tokenization import TokenCounter


def row(paper="A"):
    return {
        "id":paper,
        "title":"A research paper",
        "full_text":{
            "section_name":["Methods","Results"],
            "paragraphs":[["Models are trained with data."],
                          ["Accuracy improves on this dataset."]],
        },
        "qas":{
            "question":["How do the results compare?"],
            "question_id":["q1"],
            "answers":[{"answer":[
                {"unanswerable":False,"evidence":["Accuracy improves on this dataset."],
                 "highlighted_evidence":[],"free_form_answer":"Improves."},
                {"unanswerable":False,"evidence":["Models are trained with data."],
                 "highlighted_evidence":[],"free_form_answer":"Training."},
            ]}],
        },
    }


def test_preserves_alternative_annotations_not_union():
    questions=_questions_from_qasper_row(row())
    assert len(questions)==1
    assert len(questions[0].evidence_groups)==2
    audit=audit_native_annotations([row()])
    assert audit["questions_with_alternatives"]==1
    assert audit["evidence_groups"]==2
    assert audit["evidence_spans"]==2


def test_native_section_name_maps_to_scoring_section_hint_without_changing_source():
    blocks=native_blocks_with_section_hints(row(),TokenCounter())
    assert blocks[0].metadata["section_name"]=="Methods"
    assert blocks[0].metadata["section_hint"]=="Methods"
    assert blocks[1].metadata["section_hint"]=="Results"
    assert len(blocks)==2


def test_paper_level_splits_require_disjoint_paper_ids():
    assert validate_native_splits([row("A"),row("B")],[row("C")])=={
        "train":2,"validation":1,
    }
    with pytest.raises(ValueError,match="leakage"):
        validate_native_splits([row("A")],[row("A")])
    with pytest.raises(ValueError,match="Duplicate"):
        validate_native_splits([row("A"),row("A")],[row("C")])


def test_native_loader_is_local_only(tmp_path):
    p=tmp_path/"qasper.jsonl"
    p.write_text(json.dumps(row())+"\n",encoding="utf8")
    assert load_local_qasper_rows(p)[0]["id"]=="A"
    with pytest.raises(FileNotFoundError):
        load_local_qasper_rows(tmp_path/"remote.parquet")
