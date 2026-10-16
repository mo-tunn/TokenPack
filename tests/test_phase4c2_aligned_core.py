"""Demonstrate mismatch between old max-per-alternative-gain and aligned gain."""
from types import SimpleNamespace
import numpy as np

from submission.experiments.phase4c_marginal_core import marginal_gain_targets
from submission.experiments.phase4c2_aligned_core import aligned_marginal_gain_targets
from tokenpack.models import Chunk,ScoredChunk


def qasper_style_alternatives():
    text1="one bright red orange yellow green blue purple"
    text2="two birds fly toward snowy mountains every day"
    items=[]
    for i,text in enumerate((text1,text2)):
        c=Chunk(id=f"c{i}",text=text,source_path="p",document_index=0,
                start_page=None,end_page=None,start_paragraph=i,end_paragraph=i,
                char_start=0,char_end=len(text),token_count=20)
        items.append(ScoredChunk(chunk=c,value=.6,raw_similarity=.7,weight=20))
    return SimpleNamespace(scored=items,evidence_groups=[[text1],[text2]])


def test_aligned_objective_does_not_reward_switching_to_worse_annotation():
    q=qasper_style_alternatives()
    # With annotation A fully covered, B adds no value to the best-of-OR score.
    previous=marginal_gain_targets(q,[1],[0])[0]
    aligned=aligned_marginal_gain_targets(q,[1],[0])[0]
    assert previous==1.
    assert aligned==0.


def test_aligned_and_previous_match_from_empty_set():
    q=qasper_style_alternatives()
    left=marginal_gain_targets(q,[0,1],[])
    right=aligned_marginal_gain_targets(q,[0,1],[])
    np.testing.assert_allclose(left,right)


def test_single_annotation_always_matches_previous_label():
    q=qasper_style_alternatives()
    q.evidence_groups=q.evidence_groups[:1]
    np.testing.assert_allclose(
        aligned_marginal_gain_targets(q,[0,1],[0]),
        marginal_gain_targets(q,[0,1],[0]),
    )
