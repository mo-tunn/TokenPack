"""MuSiQue reader input integrity; selected cases and gold excluded."""
from submission.experiments.phase9_musique_qwen64 import (
    prepare,make_payload,METHODS,READER_SAMPLES,
)
from submission.experiments.phase9_musique_protocol import DATA_SHA256
from submission.experiments.phase9_musique_contexts import candidates


def test_reader_64_question_ids_and_three_arms():
    tasks=prepare()
    assert len(tasks)==READER_SAMPLES*3==192
    assert len({x["case_id"] for x in tasks})==64
    for i in range(0,len(tasks),3):
        assert tuple(x["arm"] for x in tasks[i:i+3])==METHODS
        assert len({x["case_id"] for x in tasks[i:i+3]})==1


def test_gold_not_sent_with_modal_prompt():
    task=prepare()[0]
    payload=make_payload(task)
    assert "gold_answer" not in str(payload)
    assert "is_supporting" not in str(payload)
    assert "question_decomposition" not in str(payload)
    assert task["case_id"] not in str(payload)
    assert payload["temperature"]==0


def test_physical_projection_does_not_need_label():
    row={
        "id":"demo","question":"Why?",
        "paragraphs":[{"idx":1,"title":"Person",
                       "paragraph_text":"A was born in 1980. B is related."}],
    }
    atom,source=candidates(row)
    assert len(atom)==2
    assert atom[0]["original_paragraph_idx"]==1
    assert "is_supporting" not in source
    assert len(DATA_SHA256)==64
