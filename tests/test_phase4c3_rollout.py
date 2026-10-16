"""Phase4C3 actually-bounded train rollout, no gold feature leakage."""
from types import SimpleNamespace
import numpy as np

from submission.experiments.phase4c3_rollout_core import (
    manual_budget_rollout,candidates_with_remaining_budget,STATE_COUNTS,
)
from submission.experiments.phase4c_marginal_core import feature_rows
from tokenpack.models import Chunk,ScoredChunk
from tokenpack.tokenization import TokenCounter
from tokenpack.export import render_context


def sample():
    arr=[]
    for i in range(16):
        text=" ".join([f"paper evidence passage number {i}",
                       "separate context from different paragraphs"]*7)
        c=Chunk(id=f"chunk{i}",text=text,source_path="paper",
                document_index=0,start_page=None,end_page=None,
                start_paragraph=i,end_paragraph=i,char_start=0,char_end=len(text),
                token_count=TokenCounter().count(text),block_ids=[i])
        arr.append(ScoredChunk(chunk=c,value=1.0-.04*i,
                               raw_similarity=.7,weight=c.token_count))
    return SimpleNamespace(
        scored=arr,feature_matrix=np.zeros((len(arr),13),dtype=np.float32),
        text="paper evidence passage",
    )


def test_rollout_states_include_beyond_top3_and_fit_real_render():
    q=sample()
    counter=TokenCounter()
    cap=counter.count(render_context([x.chunk for x in q.scored[:9]]))
    x=manual_budget_rollout(q,counter,cap)
    assert any(len(s)>=5 for s in x.states)
    assert x.states[0]==()
    assert len(set(x.ordered))==len(x.ordered)
    for s in x.states:
        source_order=sorted(s,key=lambda i:q.scored[i].chunk.order_key)
        actual=counter.count(render_context([q.scored[i].chunk for i in source_order]))
        assert actual<=cap
        ids=candidates_with_remaining_budget(q,x,s,cap,counter)
        assert not set(ids)&set(s)
        f=feature_rows(q,ids,s,cap,.25)
        assert f.shape==(len(ids),24)


def test_small_budget_has_no_infeasible_rollout_chunks():
    q=sample()
    counter=TokenCounter()
    cap=counter.count(render_context([]))+1
    x=manual_budget_rollout(q,counter,cap)
    assert x.states==((),)
    assert x.ordered==()
    assert candidates_with_remaining_budget(q,x,(),cap,counter)==[]


def test_determinism_and_no_gold_dependency():
    q=sample()
    counter=TokenCounter()
    cap=counter.count(render_context([v.chunk for v in q.scored[:5]]))
    a=manual_budget_rollout(q,counter,cap)
    b=manual_budget_rollout(q,counter,cap)
    assert a.states==b.states and a.ordered==b.ordered
    assert STATE_COUNTS[:4]==(0,1,2,3)


def test_bounded_booster_matches_unbounded_predictions():
    import lightgbm as lgb
    from submission.experiments.phase4c3_rollout_train_eval import BoundedBooster
    rng=np.random.default_rng(42)
    X=rng.normal(size=(35,24)).astype(np.float32)
    y=(X[:,0]+.1*X[:,1])
    model=lgb.LGBMRegressor(n_estimators=7,num_leaves=4,
                            min_child_samples=2,verbosity=-1,n_jobs=2)
    model.fit(X,y)
    raw=model.booster_.predict(X,num_threads=2)
    bounded=BoundedBooster(model.booster_).predict(X)
    np.testing.assert_allclose(raw,bounded,rtol=0,atol=0)
