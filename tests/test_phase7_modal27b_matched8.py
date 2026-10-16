"""Small local checks: Modal pilot never includes gold answers in payload."""
from submission.experiments.phase7_modal27b_matched8 import (
    ARMS,MAX_REMOTE_CALLS,answer_metrics,make_payload,
)


def test_no_gold_answer_or_question_id_in_remote_payload():
    task={
        "task_id":"private-id|manual-25",
        "case_id":"private-id",
        "arm":"manual-25",
        "question":"Who founded the group?",
        "context":"The group was founded by Alice.",
        "gold_answer":"Alice",
    }
    content=str(make_payload(task))
    assert "gold_answer" not in content
    assert "private-id" not in content
    assert "Alice" in content  # context can independently include answer words
    assert "reasoning_effort" in content


def test_hotpot_normalized_exact_match_and_yes_no():
    assert answer_metrics("The New York.", "new york")== (1,1.)
    assert answer_metrics("yes","no")== (0,0.)
    exact,f1=answer_metrics("John Brown","Brown")
    assert exact==0 and round(f1,6)==round(2/3,6)


def test_experiment_is_small_and_methods_fixed():
    assert ARMS==("full-source","manual-25","msmarco-25")
    assert MAX_REMOTE_CALLS==24
