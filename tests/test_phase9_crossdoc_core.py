"""Gold-free cross-document bridge correctness and true token budget."""
from submission.experiments.phase9_crossdoc_core import (
    bridge_link,choose,render_atomic,title_in_sentence,
)
from submission.experiments.phase9_musique_contexts import (
    candidates,sample_ids,split_sentences,
)
from submission.experiments.phase9_musique_protocol import (
    N_QUESTIONS,BUDGET_RATIO,METHODS,
)
from tokenpack.tokenization import TokenCounter


def source():
    return {
        "id":"fictional",
        "question":"Where was the scientist born?",
        "paragraphs":[
            {"idx":12,"title":"Space Academy",
             "paragraph_text":"The academy employed Professor Hale. Hale discovered the asteroid."},
            {"idx":42,"title":"Professor Hale",
             "paragraph_text":"Professor Hale was born in Copenhagen. She studied astronomy."},
            {"idx":77,"title":"Volcanic Mountain",
             "paragraph_text":"The mountain was first observed in winter. It has a northern ridge."},
        ],
    }


def test_candidates_do_not_need_support_gold():
    atoms,full=candidates(source())
    assert len(atoms)==6
    assert "Copenhagen" in full
    assert all(set(x)=={
        "article","original_paragraph_idx","sent_id",
        "title","sentence","input_text",
    } for x in atoms)
    assert len(split_sentences("Alice walked. Then Bob ran."))==2


def test_title_link_and_provenance():
    atoms,_=candidates(source())
    assert bridge_link(atoms[0],atoms[2])==1
    assert bridge_link(atoms[0],atoms[1])==0.
    assert title_in_sentence("Professor Hale","The academy employed Professor Hale.")


def test_real_budget_same_arms_with_no_new_text():
    atoms,_=candidates(source())
    counter=TokenCounter()
    scores=[3.0,2.0,1.9,1.1,.1,.05]
    budget=counter.count(render_atomic([0,2],atoms))
    a=choose(atoms,scores,budget,counter,graph=False)
    b=choose(atoms,scores,budget,counter,graph=True)
    for result in (a,b):
        assert result["tokens"]==counter.count(result["text"])
        assert result["tokens"]<=budget
        assert set(result["selected_source_idxs"])<={12,42,77}
        assert "gold_answer" not in result["text"]


def test_id_hash_deterministic_and_primary_protocol():
    arr=["a","b","c","d","e"]
    assert sample_ids(arr,3)==sample_ids(list(reversed(arr)),3)
    assert N_QUESTIONS==256
    assert BUDGET_RATIO==.25
    assert METHODS==("atomic-rank","crossdoc-bridge")
