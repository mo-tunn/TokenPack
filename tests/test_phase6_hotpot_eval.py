"""Frozen Phase6 out-of-domain protocol: labels never enter inference."""
from __future__ import annotations

from submission.experiments.phase6_hotpot_protocol import (
    N_CASES,SAMPLE_SEED,BUDGET_RATIOS,METHODS,
)
from submission.experiments.phase6_hotpot_eval import (
    hashed_sample,measure_evidence,build_blocks,aggregate,
)
from tokenpack.tokenization import TokenCounter


def sample():
    return {
        "id":"case-1","question":"Where did they meet?",
        "answer":"New York",
        "context":{
            "title":["Museum","City"],
            "sentences":[
                ["The museum was established in 2001.",
                 "It hosted the meeting in New York."],
                ["New York is in the United States.",
                 "It has several museums."],
            ],
        },
        "supporting_facts":{"title":["Museum","City"],"sent_id":[1,0]},
    }


def test_id_only_selection_never_uses_gold_or_answer():
    rows=[]
    for j in range(8):
        z=sample();z["id"]=f"case-{j}";rows.append(z)
    selection=[x["id"] for x in hashed_sample(rows,3)]
    for z in rows:
        z["answer"]="DIFFERENT ANSWER"
        z["supporting_facts"]["sent_id"]=[99,99]
    assert selection==[x["id"] for x in hashed_sample(rows,3)]
    assert len(set(selection))==3


def test_both_support_sentences_required():
    row=sample()
    one="It hosted the meeting in New York."
    both=one+"\n\nNew York is in the United States."
    a=measure_evidence(row,one)
    b=measure_evidence(row,both)
    assert a["all_support_sentences"]==0
    assert a["support_sentence_recall"]==.5
    assert b["all_support_sentences"]==1
    assert b["all_support_titles"]==1
    assert b["answer_string_present"]==1


def test_yes_no_answer_not_scored_as_extractive_sufficiency():
    row=sample();row["answer"]="yes"
    result=measure_evidence(row,"yes and all evidence")
    assert not result["answer_string_eligible"]
    assert result["answer_string_present"] is None


def test_title_provenance_and_context_only_blocks():
    row=sample()
    blocks,source=build_blocks(row,TokenCounter())
    assert len(blocks)==2
    assert blocks[0].source_path!=blocks[1].source_path
    assert blocks[0].document_index!=blocks[1].document_index
    assert "Museum" in source
    assert "City" in source
    assert "Where did they meet?" not in source


def test_predeclared_no_hotpot_tuning():
    assert N_CASES==512
    assert len(SAMPLE_SEED)>12
    assert BUDGET_RATIOS==(.25,.5)
    assert METHODS==(
        "manual-650","lambdamart-650","manual-300",
        "msmarco-zero-shot-300",
    )
