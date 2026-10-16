"""Phase5A gold-blind inference input and question/paper train sampling tests.

No model download, GPU requirement, or generation in these unit tests.
"""
from types import SimpleNamespace
import numpy as np

from submission.experiments.phase5a_gpu_crossencoder import (
    BACKBONE, RATIOS, METHODS, training_pairs,
)


def fake_question():
    chunks=[
        SimpleNamespace(chunk=SimpleNamespace(text=f"chunk-{i} source"),
                        value=(1-i*.1))
        for i in range(7)
    ]
    return SimpleNamespace(
        text="Which paragraph gives the evidence?",
        scored=chunks,
        labels=np.asarray([2,0,0,1,0,0,0]),
    )


def test_train_positive_and_hard_negative_labels_balanced():
    pairs,audit=training_pairs([fake_question()],seed=42,
                               negative_per_question=3)
    assert audit["questions_with_contrast"]==1
    assert audit["positives"]==2
    assert audit["negatives"]==4
    assert sum(x[2]==1 for x in pairs)==2
    assert sum(x[2]==0 for x in pairs)==4
    assert all(x[0].startswith("Which paragraph") for x in pairs)


def test_train_sample_is_seed_reproducible_and_invalid_questions_skipped():
    p=fake_question()
    empty=SimpleNamespace(text="question",scored=p.scored,
                          labels=np.zeros(7,dtype=int))
    a,audit=training_pairs([p,empty],seed=123)
    b,_=training_pairs([p,empty],seed=123)
    assert a==b
    assert audit["questions_without_contrast"]==1


def test_frozen_crossencoder_evaluation_uses_existing_budgets():
    assert RATIOS==(.25,.50)
    assert METHODS==("manual-render","lambdamart-render",
                     "crossencoder-render")
    assert BACKBONE=="sentence-transformers/all-MiniLM-L6-v2"
