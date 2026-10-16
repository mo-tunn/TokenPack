"""Small exhaustive objective, confidence bounds, and no-gold-in-inference tests."""
from __future__ import annotations

from tokenpack.models import Chunk,ScoredChunk
from tokenpack.tokenization import TokenCounter
from tokenpack.export import render_context
from submission.experiments.phase4a_oracle_core import (
    solve_group,as_grams,
)
from submission.experiments.phase4a_qasper_multires_oracle import (
    oracle_for_alternatives,SCALES,RATIOS,
)


def wrap(text:str,index:int):
    c=Chunk(id=f"unit-{index}",text=text,source_path="paper",
            document_index=0,start_page=None,end_page=None,
            start_paragraph=index,end_paragraph=index,
            char_start=index*100,char_end=index*100+len(text),
            token_count=TokenCounter().count(text))
    return ScoredChunk(chunk=c,value=.5,raw_similarity=.5,weight=c.token_count)


def render(items):
    return render_context([x.chunk for x in items])


def test_exact_when_only_one_of_two_evidence_chunks_fits():
    a=wrap("important finding red green blue yellow purple",0)
    b=wrap("important finding orange black brown grey white",1)
    counter=TokenCounter()
    budget=max(counter.count(render([a])),counter.count(render([b])))
    gold=as_grams([a.chunk.text,b.chunk.text])
    x=solve_group([a,b],gold,budget,counter,render)
    assert x.mode=="exact"
    assert x.achieved_hits< x.pool_hits
    assert counter.count(render([([a,b])[i] for i in x.chosen_indices]))<=budget


def test_pool_ceiling_certified_if_full_selection_fits():
    a=wrap("blue green orange purple white gold",0)
    b=wrap("green violet blue white cream yellow",1)
    gold=as_grams([a.chunk.text])
    cap=TokenCounter().count(render([a,b]))
    result=solve_group([a,b],gold,cap,TokenCounter(),render)
    assert result.certified_optimal
    assert result.recall==1
    assert result.pool_upper_recall==1


def test_heuristic_never_claims_optimum_without_proof():
    scored=[wrap(f"unique_{i} alpha beta gamma delta epsilon",i) for i in range(8)]
    gold=as_grams([x.chunk.text for x in scored])
    cap=TokenCounter().count(render(scored[:2]))
    result=solve_group(scored,gold,cap,TokenCounter(),render,max_exact=2)
    assert result.mode in ("feasible_heuristic_lower_bound","pool_ceiling_attained")
    assert result.recall<=result.pool_upper_recall
    assert result.recall<1


def test_non_relevant_candidate_never_improves_internal_gold_coverage():
    a=wrap("blue green orange purple white gold",0)
    b=wrap("no matching phrase a b c d e",1)
    result=solve_group([a,b],as_grams([a.chunk.text]),
                       TokenCounter().count(render([a,b])),
                       TokenCounter(),render)
    assert result.recall==1
    assert result.relevant_chunks==1


def test_annotation_groups_are_alternatives_not_pooled_union():
    a=wrap("evidence first purple green blue yellow red",0)
    b=wrap("evidence second orange brown white cream black",1)
    counter=TokenCounter()
    budget=max(counter.count(render([a])),counter.count(render([b])))
    out=oracle_for_alternatives(
        [a,b],[[a.chunk.text],[b.chunk.text]],budget,counter)
    assert out["num_alternatives"]==2
    assert out["oracle_rendered_5gram"]==1.0
    assert out["oracle_used_tokens"]<=budget


def test_resolution_priors_and_main_budget_ratios_remain_fixed():
    assert tuple(SCALES)==(150,300,650)
    assert SCALES[650]=={
        "target_tokens":650,"min_tokens":120,"max_tokens":900}
    assert RATIOS==(.25,.50)
