"""No network/model load in Phase 3B algorithmic unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from submission.experiments.phase3b_learned_scoring import (
    FEATURES,Question,check_group_isolation,data_for_ranker,
    features_for_question,make_learned_candidates,ndcg_at,positive_rescale,
    relevance_label,relevance_labels,
)
from tokenpack.models import Chunk,ScoredChunk


def make_chunk(name,text,value=.4):
    c=Chunk(
        id=name,text=text,source_path="paper",document_index=0,
        start_page=None,end_page=None,start_paragraph=1,end_paragraph=1,
        char_start=0,char_end=len(text),token_count=len(text.split()),
    )
    return ScoredChunk(
        chunk=c,value=value,raw_similarity=.1,weight=c.token_count,
        score_components={
            "cosine":.1,"bm25":.3,"query_coverage":.4,
            "structural_prior":0.,"neighbor_coherence":.2,
        },
    )


def test_feature_extractor_gold_blind_and_matches_names():
    scored=[make_chunk("a","the exact gold evidence right here",.8),
            make_chunk("b","unrelated wording",.1)]
    a=features_for_question(scored,"exact evidence")
    b=features_for_question(scored,"unrelated")
    assert a.shape==(2,len(FEATURES))
    assert np.all(np.isfinite(a))
    assert not np.array_equal(a,b)
    assert not any("gold" in field for field in FEATURES)


def test_labeling_matches_gold_5gram_and_is_not_a_feature():
    s=[make_chunk("a","one two three four five six"),
       make_chunk("b","unrelated meaningless text")]
    assert list(relevance_labels(s,["one two three four five six"]))==[3,0]
    assert [relevance_label(k) for k in (0,.01,.15,.6)]==[0,1,2,3]


def test_positive_scaling_preserves_rank_and_keeps_all_candidates():
    assert positive_rescale([-40,-5,3])==[.01,.01+.99*(35/43),1.0]
    assert positive_rescale([5,5])==[.5,.5]
    s=[make_chunk("a","chunk one"),make_chunk("b","chunk two")]
    out=make_learned_candidates(s,[-100,20])
    assert out[0].value>0 and out[1].value>out[0].value
    assert s[0].value==.4
    with pytest.raises(ValueError):
        make_learned_candidates(s,[float("nan"),.4])


def test_paper_isolation_guard_rejects_cross_fold_leakage():
    s=[make_chunk("a","one two three four five")]
    q=lambda paper,cid:Question(cid,paper,1000,500,["one two three four five"],
                                s,np.ones((1,len(FEATURES))),np.array([3]))
    records=[q("paper-a","a1"),q("paper-a","a2"),q("paper-b","b1")]
    with pytest.raises(ValueError,match="overlapping"):
        check_group_isolation(records,[0,2],[1])
    assert check_group_isolation(records,[0,1],[2])==(1,1)
    X,y,groups=data_for_ranker(records,[0,1])
    assert X.shape==(2,len(FEATURES)) and groups==[1,1]


def test_ndcg_prefers_relevant_chunk():
    gold=np.asarray([3,0],dtype=np.int32)
    assert ndcg_at(gold,[10,0],k=2)==1.0
    assert ndcg_at(gold,[0,10],k=2)<1.0
