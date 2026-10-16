from __future__ import annotations

import pytest

from submission.experiments.phase3c_set_aware import (
    select_render_aware,relevance_values,terms,
)
from tokenpack.export import render_context
from tokenpack.models import Chunk,ScoredChunk
from tokenpack.tokenization import TokenCounter

def item(name,text,value=.5,para=0):
    chunk=Chunk(
        id=name,text=text,source_path="source.txt",document_index=0,
        start_page=None,end_page=None,start_paragraph=para,end_paragraph=para,
        char_start=para*1000,char_end=para*1000+len(text),
        token_count=TokenCounter().count(text),
    )
    return ScoredChunk(chunk=chunk,value=value,raw_similarity=value,
                       weight=chunk.token_count)

def render(items):
    return render_context([x.chunk for x in items])

def test_query_terms_and_relevance_are_gold_blind():
    assert "paper" not in terms("Paper example for relevance")
    assert "relevance" in terms("Paper example for relevance")
    a=item("a","one two three four five",.1)
    b=item("b","a different section lorem ipsum",.9)
    scores=relevance_values([a,b])
    assert scores["b"]>scores["a"]>0

def test_exact_real_budget_and_source_order():
    counter=TokenCounter()
    a=item("a","Evidence discussion on downstream datasets.",.8,3)
    b=item("b","Another section focused on comparison experiments.",.7,1)
    budget=counter.count(render([a,b]))
    x=select_render_aware([a,b],query="comparison experiments",budget=budget,
                          counter=counter,render=render)
    assert [z.chunk.id for z in x.selected]==["b","a"]
    assert counter.count(x.context)==x.real_tokens<=budget

def test_real_render_headers_reject_content_that_looks_affordable():
    counter=TokenCounter()
    a=item("a","the five words are stored here",.8,0)
    budget=counter.count(render([a]))-1
    assert budget>=counter.count(render([]))
    for policy in ("render-value","set-aware"):
        x=select_render_aware([a],query="five words",budget=budget,
                              counter=counter,render=render,policy=policy)
        assert x.selected==[]
        assert x.real_tokens<=budget

def test_set_aware_prefers_new_query_concept_after_initial_choice():
    counter=TokenCounter()
    a=item("a","model alpha improved accuracy in classification",.9,0)
    b=item("b","model alpha improved accuracy in classification again",.8,1)
    c=item("c","model beta improved robustness on new datasets",.78,2)
    cap=counter.count(render([a,c]))
    # Candidate b and c are both feasible only until one is chosen.
    result=select_render_aware([a,b,c],
        query="model alpha accuracy beta robustness datasets",
        budget=cap,counter=counter,render=render,policy="set-aware")
    assert result.real_tokens<=cap
    assert len({x.chunk.id for x in result.selected})==len(result.selected)

def test_duplicate_ids_rejected_and_budget_positive():
    a=item("a","text")
    with pytest.raises(ValueError,match="Duplicate"):
        select_render_aware([a,a],query="text",budget=1000,counter=TokenCounter(),render=render)
    with pytest.raises(ValueError):
        select_render_aware([a],query="text",budget=0,counter=TokenCounter(),render=render)

def test_must_not_truncate_any_gold_blind_chunk():
    a=item("a","important evidence in a paragraph with source provenance",.9)
    counter=TokenCounter()
    x=select_render_aware([a],query="important evidence",budget=400,
                          counter=counter,render=render)
    assert x.selected and a.chunk.text in x.context
