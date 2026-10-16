"""Local full QASPER schema and annotation safeguards, no network/model run."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from submission.experiments.phase3b_native_qasper import (
    native_questions_from_qasper_row,validate_native_splits,
)
from submission.experiments.phase3b_full_qasper_train import (
    groups_with_ngrams,alt_relevance_labels,
    best_annotator_coverage,batch_vectors,
)
from tokenpack.models import Chunk,ScoredChunk


def scored(name,text):
    c=Chunk(id=name,text=text,source_path="x",document_index=0,
        start_page=None,end_page=None,start_paragraph=0,end_paragraph=0,
        char_start=0,char_end=len(text),token_count=6)
    return ScoredChunk(chunk=c,value=.8,raw_similarity=.5,weight=6)


def native_numpy_row():
    anns=[
      {"unanswerable":False,
       "evidence":np.asarray(["word one two three four five"],dtype=object),
       "highlighted_evidence":np.asarray([],dtype=object),
       "extractive_spans":np.asarray([],dtype=object),
       "free_form_answer":"Answer 1","yes_no":None},
      {"unanswerable":False,
       "evidence":np.asarray(["another different proof five six seven"],dtype=object),
       "highlighted_evidence":np.asarray([],dtype=object),
       "extractive_spans":np.asarray([],dtype=object),
       "free_form_answer":"Answer 2","yes_no":None},
    ]
    return {"id":"paper","title":"Title","qas":{
        "question":np.asarray(["What is shown?"],dtype=object),
        "question_id":np.asarray(["q1"],dtype=object),
        "answers":np.asarray([{"answer":np.asarray(anns,dtype=object)}],dtype=object),
    }}


def test_numpy_native_alternatives_do_not_boolean_cast_array():
    qs,excluded=native_questions_from_qasper_row(native_numpy_row())
    assert len(qs)==1
    assert len(qs[0].evidence_groups)==2
    assert qs[0].evidence_texts==["word one two three four five"]
    assert excluded=={}


def test_best_of_separate_alternatives_does_not_require_union():
    groups=[
        ["word one two three four five"],
        ["another different proof five six seven"],
    ]
    q1=scored("one","word one two three four five")
    q2=scored("two","different unrelated filler unrelated none")
    assert groups_with_ngrams(groups)==groups
    labels=alt_relevance_labels([q1,q2],groups)
    assert labels.tolist()==[3,0]
    token,grams,full=best_annotator_coverage(q1.chunk.text,groups)
    assert grams==1.0 and token==1.0 and full
    _,grams2,full2=best_annotator_coverage(q2.chunk.text,groups)
    assert grams2<1.0 and not full2


def test_group_isolation_requires_disjoint_papers():
    with pytest.raises(ValueError,match="leakage"):
        validate_native_splits([{"id":"same"}],[{"id":"same"}])


def test_empty_batch_has_no_effect():
    class E:
        model_name="offline"
        def embed(self,texts):
            raise AssertionError("Unexpected model call")
    assert batch_vectors([],E())==[]
