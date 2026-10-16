"""Research-only atomic evidence packing: provenance, cost and gold blindness."""
from submission.experiments.phase8_atomic_contexts import (
    sentence_candidates,render,choose_atoms,
)
from submission.experiments.phase8_atomic_protocol import (
    EVAL_START,EVAL_STOP,EVAL_COUNT,ARMS,
)
from tokenpack.tokenization import TokenCounter


def fixture():
    return {
        "context":{
            "title":["First Wiki","Second Wiki","Decoy"],
            "sentences":[
                ["One sentence about the question.","The linked person is Ann."],
                ["Ann worked for the space agency.","She was born in March."],
                ["Unrelated sentence.","Irrelevant sentence."],
            ],
        },
        # These should NEVER be used inside candidate extraction:
        "answer":"gold secret answer",
        "supporting_facts":{"title":["First Wiki"],"sent_id":[1]},
    }


def test_candidate_generation_reads_source_sentences_not_labels():
    original=fixture()
    arts,atoms=sentence_candidates(original)
    original["answer"]="another gold secret"
    original["supporting_facts"]["sent_id"]=[99]
    aa,bb=sentence_candidates(original)
    assert arts==aa and atoms==bb
    assert len(atoms)==6
    assert len(arts)==3
    assert "gold secret" not in str(atoms)


def test_source_order_and_literal_evidence_unmodified():
    _,atoms=sentence_candidates(fixture())
    context=render([3,1,0,2],atoms)
    assert context.index("One sentence")<context.index("linked person")
    assert context.index("First Wiki")<context.index("Second Wiki")
    assert "Ann worked for the space agency." in context


def test_true_rendered_budget_and_distinct_source_bridge():
    _,atoms=sentence_candidates(fixture())
    counter=TokenCounter()
    scores=[2.,1.9,1.8,1.7,.3,.2]
    docs={0:2.,1:1.9,2:.01}
    cap=counter.count(render([0,2],atoms))
    out=choose_atoms(atoms,scores,docs,cap,counter,bridge=True)
    assert out["source_articles"]>=2
    assert counter.count(out["text"])<=cap
    assert out["tokens"]<=cap
    pure=choose_atoms(atoms,scores,docs,cap,counter,bridge=False)
    assert pure["tokens"]<=cap


def test_phase8_does_not_reuse_previous_reader_cases():
    assert (EVAL_START,EVAL_STOP,EVAL_COUNT)==(104,256,152)
    assert ARMS==("msmarco-25","atomic-sentence-rank",
                  "atomic-bridge-first")
