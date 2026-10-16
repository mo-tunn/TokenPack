"""Phase 2: deterministic structural-neighbor & provenance-grouping ablation tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from tokenpack.evidence_grouping import fit_render_to_token_budget, render_source_evidence_groups
from tokenpack.evidence_neighbors import select_neighbor_augmented, source_adjacent
from tokenpack.models import Chunk, ScoredChunk
from tokenpack.selectors import select_chunks

ROOT = Path(__file__).resolve().parents[1]


def item(name: str, *, start: int, end: int, value: float, weight: int = 10,
         doc: int = 0, source: str = "source-a", section: str | None = None):
    chunk = Chunk(
        id=name, text="TEXT_"+name, source_path=source, document_index=doc,
        start_page=None, end_page=None, start_paragraph=start//10,
        end_paragraph=start//10, char_start=start, char_end=end,
        token_count=weight, metadata={"section_hint":section} if section else {},
    )
    return ScoredChunk(chunk=chunk, value=value, raw_similarity=value, weight=weight)


class WhitespaceCounter:
    def count(self, s: str) -> int:
        return len(s.split())


def test_neighbour_fills_real_adjacent_chunk_not_unrelated_top_chunk():
    pool=[
        item("seed-a",start=0,end=10,value=1.0),
        item("neighbor-b",start=10,end=20,value=0.03),
        item("seed-c",start=100,end=110,value=0.95),
        item("unrelated-d",start=200,end=210,value=0.80),
    ]
    existing=select_chunks(pool,strategy="budget-top-k",budget=30,candidate_pool=4)
    assert {x.chunk.id for x in existing.selected}=={"seed-a","seed-c","unrelated-d"}
    revised=select_neighbor_augmented(pool,30,candidate_pool=2,neighbor_fraction=1/3)
    assert {x.chunk.id for x in revised.selected}=={"seed-a","seed-c","neighbor-b"}
    assert revised.used_tokens==30
    assert revised.strategy=="neighbor-augment"
    assert existing.strategy=="budget-top-k"


def test_only_physical_neighbors_and_same_source_are_linked():
    left=item("a",start=0,end=10,value=.9)
    right=item("b",start=11,end=20,value=.1)
    assert source_adjacent(left.chunk,right.chunk)
    assert not source_adjacent(left.chunk,item("c",start=11,end=20,value=.2,doc=1).chunk)
    assert not source_adjacent(left.chunk,item("d",start=11,end=20,value=.2,source="other").chunk)
    assert not source_adjacent(left.chunk,item("e",start=100,end=110,value=.2).chunk)
    assert source_adjacent(left.chunk,item("f",start=11,end=20,value=.2,section="other").chunk)
    sectioned=item("sectioned",start=0,end=10,value=.9,section="intro")
    assert not source_adjacent(sectioned.chunk,item("other-section",start=10,end=20,value=.2,section="methods").chunk)


def test_neighbor_uses_existing_selector_when_fraction_zero():
    pool=[item(str(i),start=i*30,end=i*30+10,value=.1*(8-i)) for i in range(6)]
    a=select_chunks(pool,strategy="budget-top-k",budget=30,candidate_pool=6)
    b=select_neighbor_augmented(pool,budget=30,candidate_pool=6,neighbor_fraction=0)
    assert [x.chunk.id for x in a.selected]==[x.chunk.id for x in b.selected]


def test_no_boundary_hopping_to_other_document_and_not_exceed_budget():
    pool=[
        item("a",start=0,end=10,value=1.0),
        item("wrong-document",start=10,end=20,value=.05,doc=1),
        item("far-c",start=100,end=110,value=.9),
        item("far-d",start=200,end=210,value=.8),
    ]
    result=select_neighbor_augmented(pool,30,neighbor_fraction=.34)
    assert "wrong-document" not in {x.chunk.id for x in result.selected}
    assert result.used_tokens<=result.budget


def test_grouping_source_order_exact_content_no_inferred_causality():
    items=[
        item("late",start=10,end=20,value=.9),
        item("early",start=0,end=10,value=.1),
        item("other",start=0,end=10,value=.8,source="source-b",doc=1),
    ]
    txt=render_source_evidence_groups(items)
    assert txt.count("[Evidence Group")==2
    assert txt.count("TEXT_early")==1
    assert txt.index("TEXT_early") < txt.index("TEXT_late")
    assert txt.index("TEXT_late") < txt.index("TEXT_other")
    assert "relation=source-contiguous" in txt
    assert "relation=cause" not in txt


def test_budget_includes_grouping_headers_and_removes_whole_chunks_only():
    items=[item("a",start=0,end=10,value=1.0),item("b",start=10,end=20,value=.05)]
    counter=WhitespaceCounter()
    full=render_source_evidence_groups(items)
    one=render_source_evidence_groups(items[:1])
    assert counter.count(full)>counter.count(one)
    rendered,kept,dropped=fit_render_to_token_budget(
        items,render=render_source_evidence_groups,token_counter=counter,budget=counter.count(one)
    )
    assert kept==items[:1]
    assert dropped==1
    assert rendered==one
    assert counter.count(rendered)<=counter.count(one)
    assert "TEXT_b" not in rendered


def test_duplicate_chunks_and_invalid_fraction_rejected():
    a=item("same",start=0,end=10,value=.8)
    with pytest.raises(ValueError):
        select_neighbor_augmented([a,a],30)
    with pytest.raises(ValueError):
        select_neighbor_augmented([a],30,neighbor_fraction=1)
    with pytest.raises(ValueError):
        fit_render_to_token_budget([a,a],render=render_source_evidence_groups,
                                   token_counter=WhitespaceCounter(),budget=50)

def test_budgeted_rag_drops_lowest_raw_similarity_not_hybrid_score():
    items=[
        item("relevant-a",start=0,end=10,value=0.10),
        item("relevant-b",start=10,end=20,value=0.95),
    ]
    items[0].raw_similarity=.95
    items[1].raw_similarity=.10
    counter=WhitespaceCounter()
    one=render_source_evidence_groups(items[:1])
    result,kept,dropped=fit_render_to_token_budget(
        items,render=render_source_evidence_groups,
        token_counter=counter,budget=counter.count(one),
        drop_priority=lambda x:x.raw_similarity,
    )
    assert [x.chunk.id for x in kept]==["relevant-a"]
    assert dropped==1
    assert "TEXT_relevant-b" not in result


def test_disabled_ablation_keeps_legacy_checkpoint_signature():
    from submission.longbench_eval.client import _checkpoint_config_signature
    ns=argparse.Namespace(
        model="qwen",budget_ratio=.50,selection_strategy="budget-top-k",
        phase2_ablation=False,phase2_neighbor_fraction=.15,
    )
    original=argparse.Namespace(
        model="qwen",budget_ratio=.50,selection_strategy="budget-top-k",
    )
    assert _checkpoint_config_signature(ns)==_checkpoint_config_signature(original)
    ns.phase2_ablation=True
    assert _checkpoint_config_signature(ns)!=_checkpoint_config_signature(original)


def test_phase2_longbench_opt_in_builds_four_extra_methods(tmp_path,monkeypatch):
    from submission.longbench_eval import client as lb
    from tokenpack.tokenization import TokenCounter

    class TinyEmbedder:
        def embed(self,texts):
            return [[float((len(t)%23)+1), 1.0] for t in texts]

    monkeypatch.setattr(lb,"make_embedder",lambda **_:TinyEmbedder())
    raw=" ".join("word%d"%(j%70) for j in range(500))
    context=raw+"\n\n"+raw.replace("word1","someimportantanswer")+"\n\n"+raw
    case={
        "_id":"phase2-case-1","domain":"QA","sub_domain":"Toy","difficulty":"easy",
        "length":"short","question":"What does the evidence say?","choice_A":"A",
        "choice_B":"B","choice_C":"C","choice_D":"D","answer":"A","context":context,
    }
    data=tmp_path/"one.json";data.write_text(json.dumps([case]),encoding="utf8")
    args=argparse.Namespace(
        data_file=str(data),limit=1,source_min_tokens=10,source_max_tokens=50000,
        max_scanned=1,embedding_model="unused",chunker="structure-aware",
        scoring="evidence-hybrid",selection_strategy="budget-top-k",
        budget_ratio=.5,context_order="score",candidate_pool=300,
        target_tokens=150,min_tokens=40,max_tokens=200,
        reranker="none",reranker_model="unused",reranker_candidate_pool=80,
        reranker_weight=.35,reranker_allow_download=False,cascade_frontier=False,
        skip_compression=True,phase2_ablation=True,phase2_neighbor_fraction=.15,
    )
    tasks,report=lb.build_tasks(args)
    assert len(tasks)==8
    assert set(t["method"] for t in tasks)=={
        "full-context","production-rag-50","tokenpack-50",
        "production-rag-50-p2-budgeted",
        "tokenpack-50-p2-control","tokenpack-50-p2-grouped",
        "tokenpack-50-p2-neighbor","tokenpack-50-p2-neighbor-grouped",
    }
    for task in tasks:
        if "-p2-" in task["method"]:
            assert task["context_tokens"]<=task["budget_tokens"]
            assert task["phase2_ablation"] is True
            assert task["phase2_trimmed_header_overhead"]>=0
    assert report["phase2_ablation"] is True

    args.phase2_ablation=False
    baseline, _=lb.build_tasks(args)
    assert len(baseline)==3
    for method in ("full-context","production-rag-50","tokenpack-50"):
        a=next(t for t in tasks if t["method"]==method)
        b=next(t for t in baseline if t["method"]==method)
        assert a["context"]==b["context"]
