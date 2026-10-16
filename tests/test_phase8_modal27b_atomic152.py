"""Phase8 bounded Modal Qwen experiment sample isolation and gold-free inputs."""
from submission.experiments.phase8_modal27b_atomic152 import (
    EVAL_COUNT,MODAL_ARMS,prepare_tasks,make_payload,PLAN,
)


def test_new_reader_question_subset_is_disjoint_from_earlier_104():
    tasks=prepare_tasks()
    assert len(tasks)==EVAL_COUNT*3==456
    assert len({row["case_id"] for row in tasks})==152
    assert all(row["arm"] in MODAL_ARMS for row in tasks)


def test_gold_not_in_reader_remote_payload():
    sample=prepare_tasks()[0]
    payload=make_payload(sample)
    assert "gold_answer" not in str(payload)
    assert "supporting_facts" not in str(payload)
    assert sample["case_id"] not in str(payload)
    assert payload["temperature"]==0
    assert payload["model"]=="qwen3.8-27b-orcarouter-q4km"


def test_predeclared_control_and_new_methods_equal_limits():
    tasks=prepare_tasks()
    for j in range(0,len(tasks),3):
        assert [x["arm"] for x in tasks[j:j+3]]==list(MODAL_ARMS)
    assert PLAN.exists()
