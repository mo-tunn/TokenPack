"""Phase 2b offline add-only selection tests."""
from __future__ import annotations

import pytest

from tokenpack.evidence_slack import fill_render_slack
from tokenpack.models import Chunk, ScoredChunk


class Words:
    @staticmethod
    def count(s): return len(s.split())


def mk(name,text,start,end,score=.6):
    c=Chunk(
        id=name,text=text,source_path="paper",document_index=0,
        start_page=None,end_page=None,start_paragraph=start,end_paragraph=start,
        char_start=start,char_end=end,token_count=len(text.split()),
    )
    return ScoredChunk(chunk=c,value=score,raw_similarity=score,weight=c.token_count)


def renderer(items):
    return "\n".join(item.chunk.text for item in sorted(items,key=lambda x:x.chunk.char_start))


def test_slack_add_neighbor_preserves_base_and_budget():
    base=mk("seed","Neural algorithms optimize performance",0,38,.9)
    near=mk("near","Neural performance improves",38,65,.6)
    far=mk("far","Unrelated highly scored performance",200,236,.95)
    text, kept, meta=fill_render_slack(
        [base,near,far],[base],question="Neural performance",
        budget=8,render=renderer,counter=Words(),only_neighbors=True)
    assert {x.chunk.id for x in kept}=={"seed","near"}
    assert "performance improves" in text
    assert meta["displaced_baseline_chunks"]==0
    assert meta["rendered_tokens"]<=8


def test_evergreen_base_not_displaced_even_when_neighbor_more_relevant():
    base=mk("a","neural science basics",0,21,.2)
    near=mk("b","neural experimental evidence strong",21,56,.99)
    txt,kept,info=fill_render_slack(
        [base,near],[base],question="neural experimental",
        budget=3,render=renderer,counter=Words(),only_neighbors=True)
    assert [x.chunk.id for x in kept]==["a"]
    assert txt==renderer([base])
    assert info["added_chunks"]==0


def test_non_neighbor_slack_baseline_can_fill_global_candidate():
    seed=mk("a","Neural algorithms",0,20,.8)
    far=mk("b","Neural approaches",300,323,.7)
    text,items,_=fill_render_slack(
        [seed,far],[seed],question="Neural algorithms",
        budget=10,render=renderer,counter=Words(),only_neighbors=False)
    assert {x.chunk.id for x in items}=={"a","b"}


def test_gold_not_exposed_as_parameter_and_duplicates_rejected():
    a=mk("a","neural science basics",0,21,.8)
    with pytest.raises(ValueError):
        fill_render_slack([a,a],[a],question="neural",budget=9,
                          render=renderer,counter=Words(),only_neighbors=True)


def test_neighbor_requires_relevance_and_same_source():
    seed=mk("a","neural science basics",0,21,.8)
    adjacent_wrong=mk("b","weather forecasting today",21,46,.8)
    adjacent_weak=mk("c","neural",46,52,.02)
    txt,items,_=fill_render_slack(
        [seed,adjacent_wrong,adjacent_weak],[seed],question="neural science",
        budget=16,render=renderer,counter=Words(),only_neighbors=True)
    assert [x.chunk.id for x in items]==["a"]
