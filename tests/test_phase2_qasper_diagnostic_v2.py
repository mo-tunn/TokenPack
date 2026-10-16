"""Small deterministic unit tests; never load embeddings or run inference."""
from submission.experiments.phase2_qasper_diagnostic_v2 import (
    adjacency_counts, paired_bootstrap, repetition_fraction, source_span_audit,
)
from tokenpack.models import Chunk, ScoredChunk


def chunk(name, text, start, end):
    return Chunk(
        id=name,text=text,source_path="paper",document_index=0,
        start_page=None,end_page=None,start_paragraph=start,
        end_paragraph=start,char_start=start,char_end=end,
        token_count=10,
    )


def test_unique_span_that_crosses_chunk_boundary_and_needs_both():
    source="alpha beta gamma delta epsilon zeta eta"
    cut=source.index("epsilon")
    a=chunk("a",source[:cut],0,cut)
    b=chunk("b",source[cut:],cut,len(source))
    gold=["gamma delta epsilon zeta"]
    all_kept=source_span_audit(source,gold,[a,b],{"a","b"})
    assert all_kept["unique_source_spans"]==1
    assert all_kept["boundary_split_spans"]==1
    assert all_kept["all_locatable_span_chunks_selected"] is True
    one_kept=source_span_audit(source,gold,[a,b],{"a"})
    assert one_kept["all_locatable_span_chunks_selected"] is False
    assert one_kept["source_spans"][0]["touching_chunk_ids"]==["a","b"]


def test_ambiguous_span_is_not_claimed_to_be_precisely_mapped():
    source="intro science middle science close"
    a=chunk("a",source,0,len(source))
    audit=source_span_audit(source,["science"],[a],{"a"})
    assert audit["ambiguous_source_spans"]==1
    assert audit["unique_source_spans"]==0
    assert audit["all_locatable_span_chunks_selected"] is None


def test_paired_bootstrap_and_exact_sign_test():
    neutral=paired_bootstrap([1.,.5],[1.,.5],n=100)
    assert neutral["mean_difference"]==0
    assert neutral["bootstrap_ci95"]==[0,0]
    assert neutral["two_sided_sign_p_exploratory"]==1.0
    diff=paired_bootstrap([1.,0.,0.,0.],[0.,1.,1.,1.],n=300)
    assert diff["wins"]==1 and diff["losses"]==3
    assert diff["two_sided_sign_p_exploratory"]==.625


def test_adjacent_pairs_and_nonadjacent_gap():
    chunks=[
        chunk("a","text",0,10),chunk("b","text",12,20),
        chunk("c","text",150,170),
    ]
    assert adjacency_counts(chunks,{"a","b"})==(1,1)
    assert adjacency_counts(chunks,{"b","c"})==(0,1)


def test_duplicate_content_measure_does_not_use_headers():
    a=ScoredChunk(chunk("a","one two three four five six",0,6),.5,.5,10)
    b=ScoredChunk(chunk("b","one two three four five six",10,16),.5,.5,10)
    assert repetition_fraction([a])==0
    assert repetition_fraction([a,b])==.5
