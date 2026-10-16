"""STRICTLY local QASPER train/validation preflight with annotation provenance.

This does not download datasets, train, run embedding inference, or change
production.  This helper prepares the data-integrity checks required before
Phase 3B can be evaluated on a genuinely untouched native QASPER split.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from submission.experiments.qasper_selector_eval import (
    _blocks_from_qasper_row,
    QasperQuestion,
    _as_list,
    _clean_text,
    _answer_text,
)


def load_local_qasper_rows(path: str | Path) -> list[dict]:
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError("Native QASPER local file unavailable: "+str(path))
    if path.suffix.lower()==".parquet":
        import pandas as pd
        rows=pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix.lower()==".jsonl":
        with path.open(encoding="utf8") as f:
            rows=[json.loads(line) for line in f if line.strip()]
    elif path.suffix.lower()==".json":
        obj=json.loads(path.read_text(encoding="utf8"))
        rows=list(obj.values()) if isinstance(obj,dict) else obj
    else:
        raise ValueError("Only local parquet/json/jsonl files are permitted")
    if not isinstance(rows,list) or not all(isinstance(r,dict) for r in rows):
        raise ValueError("Invalid QASPER paper records")
    return rows


def validate_native_splits(train:list[dict],validation:list[dict],
                           test:list[dict]|None=None)->dict:
    """Hard paper-level isolation and unique IDs, never question-level splitting."""
    inputs={"train":train,"validation":validation}
    if test is not None:inputs["test"]=test
    paper_sets={}
    for name,records in inputs.items():
        ids=[str(x.get("id") or x.get("paper_id") or "") for x in records]
        if not ids or "" in ids:
            raise ValueError("Missing paper IDs in "+name)
        if len(ids)!=len(set(ids)):
            raise ValueError("Duplicate paper IDs in "+name)
        paper_sets[name]=set(ids)
    for left in paper_sets:
        for right in paper_sets:
            if left<right and (paper_sets[left]&paper_sets[right]):
                raise ValueError(f"Paper leakage: {left} overlaps {right}")
    return {name:len(ids) for name,ids in paper_sets.items()}


def audit_native_annotations(rows:list[dict])->dict:
    """Count alternate evidence sets without flattening them into a union."""
    count=Counter()
    for row in rows:
        questions,_=native_questions_from_qasper_row(row)
        count["papers"]+=1
        count["usable_questions"]+=len(questions)
        count["questions_with_alternatives"]+=sum(
            len(q.evidence_groups)>1 for q in questions)
        count["evidence_groups"]+=sum(len(q.evidence_groups) for q in questions)
        count["evidence_spans"]+=sum(
            len(group) for q in questions for group in q.evidence_groups)
    return dict(count)


def native_questions_from_qasper_row(row:dict)->tuple[list[QasperQuestion],dict]:
    """Handle original QASPER parquet's NumPy arrays without boolean coercion.

    Keeps a single *alternative* group per annotator. Gold evidence is not
    concatenated across annotators; annotation with no valid evidence is not
    silently converted into an all-negative retrieval query.
    """
    qas=row.get("qas") or {}
    questions=_as_list(qas.get("question"))
    ids=_as_list(qas.get("question_id"))
    all_answers=_as_list(qas.get("answers"))
    output=[]
    excluded=Counter()
    for index,q in enumerate(questions):
        payload=all_answers[index] if index<len(all_answers) else {}
        alternatives=_as_list(payload.get("answer")) if isinstance(payload,dict) else []
        groups=[]
        chosen=None
        for item in alternatives:
            if not isinstance(item,dict):
                continue
            if item.get("unanswerable") is True:
                continue
            ev=[_clean_text(x) for x in _as_list(item.get("evidence"))
                if _clean_text(x)]
            if not ev:
                ev=[_clean_text(x) for x in _as_list(item.get("highlighted_evidence"))
                    if _clean_text(x)]
            if not ev:
                continue
            answer=_answer_text(item)
            if not answer:continue
            groups.append(ev)
            if chosen is None:chosen=(item,ev,answer)
        if chosen is None:
            excluded["unanswerable_or_absent_gold_evidence"]+=1
            continue
        _item,evidence,answer=chosen
        output.append(QasperQuestion(
            paper_id=str(row.get("id")),
            title=str(row.get("title") or ""),
            question_id=str(ids[index] if index<len(ids) else index),
            question=str(q),answer=answer,
            evidence_texts=evidence,
            evidence_groups=groups,
        ))
    return output,dict(excluded)


def native_blocks_with_section_hints(row,token_counter):
    """Map native section_name to scorer-readable section_hint; do not guess."""
    blocks=_blocks_from_qasper_row(row,document_index=0,
                                    token_counter=token_counter)
    for block in blocks:
        section=block.metadata.get("section_name")
        if section:
            block.metadata["section_hint"]=str(section)
    return blocks
