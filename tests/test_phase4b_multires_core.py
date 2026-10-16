"""Gold-blind multires candidate merging, overlap, and source provenance."""
from __future__ import annotations

from tokenpack.models import Chunk,ScoredChunk
from submission.experiments.phase4b_multires_core import (
    construct_multiresolution,covered_fraction,filter_heavy_overlap,
    count_selected_scale,
)


def chunk(name,start,end,text="evidence statement"):
    return Chunk(
        id=name,text=text,source_path="paper123",
        document_index=0,start_page=None,end_page=None,
        start_paragraph=start//100,end_paragraph=end//100,
        char_start=start,char_end=end,token_count=max(1,end-start),
    )


def scored(c,val):
    return ScoredChunk(chunk=c,value=val,raw_similarity=val,weight=c.token_count)


def test_unique_global_ids_no_duplicate_spans():
    same=chunk("same",0,50)
    short=chunk("short",50,100,"different short text")
    groups={150:[same,short],300:[same],650:[same]}
    vectors={150:[[.1],[.2]],300:[[.1]],650:[[.1]]}
    merged,vectors,scales=construct_multiresolution(groups,vectors)
    assert len(merged)==2
    assert len({x.id for x in merged})==2
    assert scales[merged[0].id]==150
    assert merged[0].char_start==0
    assert merged[0].char_end==50
    assert merged[0].metadata["research_resolution"]==150


def test_contained_child_is_source_overlap_conflict():
    inner=chunk("inner",100,200)
    outer=chunk("outer",0,400)
    assert covered_fraction(inner,outer)==1.
    kept,stats=filter_heavy_overlap([scored(inner,.95),scored(outer,.7)])
    assert len(kept)==1 and stats["duplicates_removed"]==1


def test_unrelated_span_survives():
    a=chunk("a",0,100)
    b=chunk("b",150,250)
    kept,stats=filter_heavy_overlap([scored(a,.8),scored(b,.7)])
    assert len(kept)==2
    assert stats["duplicates_removed"]==0


def test_qasper_per_block_character_offsets_must_not_merge_distant_chunks():
    a=chunk("a",0,110)
    b=chunk("b",0,110)
    a.block_ids=[0,1]
    b.block_ids=[11,12]
    # QASPER's per-block character offsets are identical but source
    # block IDs prove that the evidence comes from different paragraphs.
    assert covered_fraction(a,b)==0
    kept,_=filter_heavy_overlap([scored(a,.8),scored(b,.7)])
    assert len(kept)==2


def test_same_source_block_splits_use_local_char_intervals():
    a=chunk("a",0,80)
    b=chunk("b",85,155)
    a.block_ids=[7]
    b.block_ids=[7]
    assert covered_fraction(a,b)==0


def test_overlap_fraction_of_shorter_and_cross_source_not_overlap():
    a=chunk("a",10,110)
    b=chunk("b",90,200)
    a.start_paragraph=a.end_paragraph=5
    b.start_paragraph=b.end_paragraph=5
    assert abs(covered_fraction(a,b)-.2)<1e-9
    c=chunk("c",10,110)
    c.source_path="another_paper"
    assert covered_fraction(a,c)==0.


def test_redundant_scale_counting_and_zero_inputs():
    x=chunk("a",0,100)
    x.metadata["research_resolution"]=300
    counts=count_selected_scale([scored(x,.5)])
    assert counts=={"150":0,"300":1,"650":0}
    assert filter_heavy_overlap([])[0]==[]
