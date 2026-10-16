"""Gold-aware oracle is a separate diagnostic, not an inference selector."""
from __future__ import annotations

from submission.experiments.phase3a_oracle_core import (
    gold_counts,gram_counts,matched_count,oracle_search,pool_coverage,
)
from tokenpack.export import render_context
from tokenpack.models import Chunk,ScoredChunk
from tokenpack.tokenization import TokenCounter


def sample(name,text,start,value=.9):
    c=Chunk(
        id=name,text=text,source_path="paper",document_index=0,
        start_page=None,end_page=None,start_paragraph=start,end_paragraph=start,
        char_start=start,char_end=start+len(text),token_count=10)
    return ScoredChunk(chunk=c,value=value,raw_similarity=value,weight=10)


def renderer(items):
    return render_context([x.chunk for x in items])


def test_oracle_exact_best_can_pick_lower_scored_true_gold_chunk():
    a=sample("distractor","distracting text about unrelated things",0,value=.99)
    b=sample("evidence","the actual gold evidence five words",30,value=.1)
    gold,spans=gold_counts(["the actual gold evidence five words"])
    out=oracle_search([a,b],gold=gold,span_grams=spans,
                      budget=100,render=renderer,counter=TokenCounter())
    assert out["oracle_5gram"]["gold_5gram_recall"]==1.
    assert out["oracle_5gram"]["selected_ids"]==["evidence"]
    assert out["oracle_5gram"]["mode"].startswith("exact")


def test_oracle_prevents_budget_violation_even_technical_header_cost():
    a=sample("a","the actual gold evidence five words",0)
    gold,spans=gold_counts([a.chunk.text])
    counter=TokenCounter()
    cap=counter.count(renderer([a]))-1
    assert cap>0
    out=oracle_search([a],gold=gold,span_grams=spans,
                      budget=cap,render=renderer,counter=counter)
    assert out["oracle_5gram"]["gold_5gram_recall"]==0.
    assert out["oracle_5gram"]["rendered_tokens"]<=cap


def test_multi_span_joint_oracle_optimizes_both_evidence_pieces():
    a=sample("one","paper reports a dataset named green",0)
    b=sample("two","measured metric increased by twenty percent",150)
    gold,spans=gold_counts([a.chunk.text,b.chunk.text])
    out=oracle_search([a,b],gold=gold,span_grams=spans,
                      budget=150,render=renderer,counter=TokenCounter())
    assert out["oracle_joint"]["completed_spans"]==2
    assert out["oracle_joint"]["all_spans_present"]
    assert set(out["oracle_joint"]["selected_ids"])=={"one","two"}


def test_unbudgeted_candidate_ceiling_tracks_absent_evidence():
    a=sample("a","the actual gold evidence five words",0)
    b=sample("b","a completely different irrelevant selection",500)
    gold,spans=gold_counts([a.chunk.text])
    assert pool_coverage([b],gold,spans)["gold_5gram_recall"]==0.
    assert pool_coverage([a,b],gold,spans)["gold_5gram_recall"]==1.


def test_forced_heuristic_cannot_claim_exactness():
    a=sample("a","the actual gold evidence five words",0)
    b=sample("b","another factual result from this paper",100)
    gold,spans=gold_counts([a.chunk.text,b.chunk.text])
    out=oracle_search([a,b],gold=gold,span_grams=spans,budget=100,
                      render=renderer,counter=TokenCounter(),
                      exact_max_relevant=1)
    assert out["oracle_5gram"]["mode"]=="heuristic_feasible_lower_bound"
    assert out["oracle_5gram"]["rendered_tokens"]<=100


def test_repeated_gold_ngram_is_not_overcounted():
    s="gold evidence found in first paragraph"
    a=sample("a",s,0);b=sample("b",s,100)
    gold,spans=gold_counts([s])
    result=pool_coverage([a,b],gold,spans)
    assert result["gold_5gram_recall"]==1.
    assert matched_count(gram_counts(s)+gram_counts(s),gold)==sum(gold.values())
