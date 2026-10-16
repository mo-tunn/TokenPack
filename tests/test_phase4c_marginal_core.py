from types import SimpleNamespace
import numpy as np

from submission.experiments.phase4c_marginal_core import (
    feature_rows,marginal_gain_targets,training_states,NAMES,
)
from tokenpack.models import Chunk,ScoredChunk


def case():
    texts=[
        "evidence alpha beta gamma delta epsilon zeta",
        "evidence alpha beta gamma delta epsilon zeta",
        "another distinct paragraph with orange blue green white",
    ]
    chunks=[]
    for i,t in enumerate(texts):
        c=Chunk(id=f"chunk{i}",text=t,source_path="paper",
                document_index=0,start_page=None,end_page=None,
                start_paragraph=i,end_paragraph=i,char_start=0,char_end=len(t),
                token_count=15,block_ids=[i])
        chunks.append(ScoredChunk(chunk=c,value=1-i*.1,raw_similarity=.7,
                                  weight=15,embedding=[1.,0.]))
    return SimpleNamespace(
        scored=chunks,feature_matrix=np.zeros((3,13)),
        text="evidence alpha beta",
        evidence_groups=[[texts[0]],[texts[2]]],
    )


def test_marginal_duplicate_evidence_gain_goes_to_zero():
    q=case()
    y_empty=marginal_gain_targets(q,[0,1,2],[])
    y_after=marginal_gain_targets(q,[0,1,2],[0])
    assert y_empty[0]>0
    assert y_empty[1]>0
    assert y_after[1]==0
    assert y_after[2]>0


def test_features_depend_on_selected_state_without_gold():
    q=case()
    a=feature_rows(q,[1],(),500,.25)
    b=feature_rows(q,[1],(0,),500,.25)
    assert a.shape==b.shape==(1,len(NAMES))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert b[0,-4]<a[0,-4]  # NEW 5gram fraction shrinks after same text
    assert b[0,-1]==0  # no shared source block IDs between these chunks
    assert not np.array_equal(a,b)


def test_training_states_are_gold_blind_manual_score_prefixes():
    q=case()
    assert training_states(q)==((),(0,),(0,1,2))
