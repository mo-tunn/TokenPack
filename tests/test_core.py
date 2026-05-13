from __future__ import annotations

import uuid
import math
import re
from pathlib import Path

import pytest

from tokenpack import cli as cli_module
from tokenpack import mcp_server
from tokenpack.benchmark import redundancy_score, run_gold_benchmark
from tokenpack.chunk_profiles import resolve_chunk_size_config
from tokenpack.chunking import StructureAwareChunker
from tokenpack.compression import CompressionConfig, compress_chunks
from tokenpack.dataset import GoldRecord, load_gold_records, save_gold_records, validate_gold_records
from tokenpack.embeddings import EmbeddingCache
from tokenpack.export import render_context
from tokenpack.generation import _default_ollama_model
from tokenpack.index import ChunkIndex, load_index, save_index
from tokenpack.loaders import iter_supported_files, load_blocks, load_text_blocks
from tokenpack.pipeline import ingest_path
from tokenpack.models import Chunk, ScoredChunk, TextBlock
from tokenpack.reranking import blend_reranker_scores
from tokenpack.scoring import SCORING_PROFILES, score_chunks
from tokenpack.scoring_experimental import (
    SCORING_PROFILES as EXPERIMENTAL_SCORING_PROFILES,
    score_experimental_chunks,
)
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter


def test_text_loader_preserves_metadata():
    tmp_path = _workspace_tmp()
    source = tmp_path / "doc.md"
    source.write_text("First paragraph about alpha.\n\nSecond paragraph about beta.", encoding="utf-8")

    blocks = load_text_blocks(source)

    assert len(blocks) == 2
    assert blocks[0].source_path == str(source)
    assert blocks[0].paragraph_index == 0
    assert blocks[1].char_start > blocks[0].char_start


def test_supported_files_include_common_document_data_and_code_formats():
    tmp_path = _workspace_tmp()
    for name in ["notes.md", "page.html", "records.jsonl", "table.csv", "deck.pptx", "sheet.xlsx", "component.tsx"]:
        (tmp_path / name).write_text("alpha", encoding="utf-8")
    (tmp_path / "image.png").write_text("ignored", encoding="utf-8")

    files = {path.name for path in iter_supported_files(tmp_path)}

    assert {"notes.md", "page.html", "records.jsonl", "table.csv", "deck.pptx", "sheet.xlsx", "component.tsx"} <= files
    assert "image.png" not in files


def test_html_loader_extracts_visible_text_and_ignores_scripts():
    tmp_path = _workspace_tmp()
    source = tmp_path / "page.html"
    source.write_text(
        "<html><head><script>secret()</script></head><body><h1>Alpha Title</h1><p>Beta evidence.</p></body></html>",
        encoding="utf-8",
    )

    blocks = load_blocks(source)
    text = "\n".join(block.text for block in blocks)

    assert "Alpha Title" in text
    assert "Beta evidence" in text
    assert "secret" not in text
    assert blocks[0].metadata["source_format"] == "html"


def test_jsonl_loader_creates_structured_blocks():
    tmp_path = _workspace_tmp()
    source = tmp_path / "records.jsonl"
    source.write_text('{"title": "Alpha", "score": 3}\n{"title": "Beta"}\n', encoding="utf-8")

    blocks = load_blocks(source)

    assert len(blocks) == 2
    assert "title: Alpha" in blocks[0].text
    assert blocks[0].metadata["content_type"] == "structured"
    assert blocks[0].metadata["line"] == 1


def test_csv_loader_creates_row_blocks():
    tmp_path = _workspace_tmp()
    source = tmp_path / "table.csv"
    source.write_text("name,role\nAlpha,Researcher\nBeta,Engineer\n", encoding="utf-8")

    blocks = load_blocks(source)

    assert len(blocks) == 2
    assert "name: Alpha" in blocks[0].text
    assert "role: Researcher" in blocks[0].text
    assert blocks[0].metadata["source_format"] == "csv"


def test_yaml_loader_marks_config_as_structured_text():
    tmp_path = _workspace_tmp()
    source = tmp_path / "config.yaml"
    source.write_text("name: Alpha\nrole: Researcher\n", encoding="utf-8")

    blocks = load_blocks(source)

    assert blocks
    assert "name: Alpha" in blocks[0].text
    assert blocks[0].metadata["content_type"] == "structured"
    assert blocks[0].metadata["source_format"] == "yaml"


def test_chunker_respects_max_tokens():
    tmp_path = _workspace_tmp()
    source = tmp_path / "doc.txt"
    source.write_text("\n\n".join(f"paragraph {idx} " + "word " * 20 for idx in range(8)), encoding="utf-8")
    blocks = load_text_blocks(source)
    chunker = StructureAwareChunker(target_tokens=40, min_tokens=20, max_tokens=55, token_counter=TokenCounter())

    chunks = chunker.chunk(blocks)

    assert chunks
    assert all(chunk.token_count <= 55 for chunk in chunks)


def test_low_budget_chunk_size_preset_uses_smaller_evidence_chunks():
    config = resolve_chunk_size_config(
        "low-budget",
        target_tokens=650,
        min_tokens=120,
        max_tokens=900,
    )

    assert config.target_tokens == 250
    assert config.min_tokens == 40
    assert config.max_tokens == 320


def test_token_counter_treats_special_token_strings_as_text():
    counter = TokenCounter()

    assert counter.count("literal <|endoftext|> marker") > 0


def test_chunk_id_hash_tolerates_pdf_surrogate_text():
    source = TextBlock(
        text="PDF extractor emitted \ud835 here.",
        source_path="paper.pdf",
        document_index=0,
        page=1,
        paragraph_index=0,
        char_start=0,
        char_end=30,
    )
    chunker = StructureAwareChunker(target_tokens=20, min_tokens=1, max_tokens=50, token_counter=TokenCounter())

    chunks = chunker.chunk([source])

    assert chunks[0].id.startswith("chunk-")


def test_embedding_cache_reuses_existing_vectors():
    tmp_path = _workspace_tmp()
    cache = EmbeddingCache(tmp_path / "embeddings.json")
    embedder = _ToyEmbedder(dimensions=16)

    first = cache.get_or_embed(["alpha beta"], embedder)
    second = EmbeddingCache(tmp_path / "embeddings.json").get_or_embed(["alpha beta"], embedder)

    assert first == second


def test_embedding_cache_tolerates_pdf_surrogate_text():
    tmp_path = _workspace_tmp()
    cache = EmbeddingCache(tmp_path / "embeddings.json")
    embedder = _ToyEmbedder(dimensions=16)

    vectors = cache.get_or_embed(["bad \ud835 text"], embedder)

    assert len(vectors[0]) == 16


def test_index_save_tolerates_pdf_surrogate_text():
    tmp_path = _workspace_tmp()
    index = ChunkIndex(chunks=[_chunk("bad \ud835 text")], embeddings=[[1.0]], model_name="test")

    save_index(index, tmp_path / "index.json")
    loaded = load_index(tmp_path / "index.json")

    assert loaded.chunks[0].text


def test_gold_jsonl_validation_catches_missing_chunk():
    index = ChunkIndex(chunks=[_chunk("known")], embeddings=[[1.0, 0.0]], model_name="test")
    records = [GoldRecord(query="alpha", answer="answer", evidence_chunk_ids=["missing"])]

    errors = validate_gold_records(records, index)

    assert errors
    assert "missing" in errors[0]


def test_gold_jsonl_roundtrip():
    tmp_path = _workspace_tmp()
    records = [GoldRecord(query="alpha", answer="beta", evidence_chunk_ids=["chunk-1"], notes="reviewed")]

    save_gold_records(records, tmp_path / "gold.jsonl")
    loaded = load_gold_records(tmp_path / "gold.jsonl")

    assert loaded[0].query == "alpha"
    assert loaded[0].evidence_chunk_ids == ["chunk-1"]


def test_semantic_threshold_chunker_splits_topic_shift():
    tmp_path = _workspace_tmp()
    source = tmp_path / "doc.txt"
    source.write_text(
        "alpha alpha alpha alpha.\n\n"
        "alpha alpha alpha beta.\n\n"
        "zebra zebra zebra zebra.\n\n"
        "zebra zebra zebra yak.",
        encoding="utf-8",
    )
    index = ingest_path(
        source,
        tmp_path / "index.json",
        embedder=_ToyEmbedder(dimensions=64),
        target_tokens=80,
        min_tokens=1,
        max_tokens=80,
        chunker_name="semantic-threshold",
        semantic_threshold=0.2,
    )

    assert len(index.chunks) >= 2
    assert all(chunk.token_count <= 80 for chunk in index.chunks)


def test_structure_aware_chunker_uses_semantic_boundaries_inside_sections():
    blocks = [
        TextBlock("alpha alpha retrieval", "doc.txt", 0, paragraph_index=0, metadata={"content_type": "document"}),
        TextBlock("alpha beta evidence", "doc.txt", 0, paragraph_index=1, metadata={"content_type": "document"}),
        TextBlock("zebra yak unrelated", "doc.txt", 0, paragraph_index=2, metadata={"content_type": "document"}),
    ]
    chunker = StructureAwareChunker(
        target_tokens=100,
        min_tokens=1,
        max_tokens=120,
        token_counter=TokenCounter(),
        block_embeddings=[[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]],
        semantic_threshold=0.4,
    )

    chunks = chunker.chunk(blocks)

    assert len(chunks) == 2
    assert "alpha beta evidence" in chunks[0].text
    assert chunks[1].text == "zebra yak unrelated"
    assert chunks[0].metadata["chunker"] == "structure-aware"
    assert chunks[0].metadata["semantic_threshold"] == 0.4


def test_code_loader_extracts_python_symbols():
    tmp_path = _workspace_tmp()
    source = tmp_path / "module.py"
    source.write_text(
        "import os\n\n"
        "class Config:\n"
        "    def __init__(self, lr):\n"
        "        self.lr = lr\n\n"
        "def train_model(model, data):\n"
        "    return model.fit(data)\n",
        encoding="utf-8",
    )

    blocks = load_blocks(source, source_type="code")

    symbol_names = {block.metadata.get("symbol_name") for block in blocks}
    assert "Config" in symbol_names
    assert "train_model" in symbol_names
    assert all(block.metadata.get("content_type") == "code" for block in blocks)


def test_pdf_loader_splits_longcodezip_into_fine_blocks():
    source = Path("resources/2510.00446v1.pdf")
    if not source.exists():
        return

    blocks = load_blocks(source, source_type="document")

    assert len(blocks) > 40
    assert any(block.metadata.get("content_type") == "code" for block in blocks)


def test_structure_aware_ingest_preserves_python_function_metadata():
    tmp_path = _workspace_tmp()
    source = tmp_path / "module.py"
    source.write_text(
        "def helper(value):\n"
        "    return value + 1\n\n"
        "def train_model(model, data):\n"
        "    optimizer = AdamW(model.parameters())\n"
        "    return model.fit(data, optimizer=optimizer)\n",
        encoding="utf-8",
    )

    index = ingest_path(
        source,
        tmp_path / "index.json",
        embedder=_ToyEmbedder(dimensions=32),
        chunker_name="structure-aware",
        source_type="code",
        target_tokens=40,
        min_tokens=1,
        max_tokens=80,
    )

    train_chunks = [chunk for chunk in index.chunks if chunk.metadata.get("symbol_name") == "train_model"]
    assert train_chunks
    assert train_chunks[0].metadata.get("content_type") == "code"
    assert train_chunks[0].metadata.get("start_line") == 4


def test_knapsack_never_exceeds_budget():
    scored = [
        _scored("a", value=8.0, weight=7),
        _scored("b", value=7.0, weight=6),
        _scored("c", value=4.0, weight=4),
    ]

    result = select_chunks(scored, strategy="knapsack", budget=10)

    assert result.used_tokens <= 10
    assert {item.chunk.id for item in result.selected} == {"b", "c"}


def test_cosine_scoring_preserves_existing_value_behavior():
    chunks = [_chunk("match"), _chunk("miss", paragraph=1)]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    scored = score_experimental_chunks([1.0, 0.0], chunks, embeddings, scoring="cosine")

    assert scored[0].value == 1.0
    assert scored[1].value == 0.0


def test_hybrid_bm25_component_favors_query_terms():
    chunks = [
        _chunk("alpha", text="alpha alpha retrieval evidence"),
        _chunk("beta", paragraph=1, text="beta gamma unrelated text"),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks([1.0, 0.0], chunks, embeddings, scoring="hybrid", query_text="alpha")

    assert scored[0].score_components["bm25"] > scored[1].score_components["bm25"]


def test_hybrid_position_bias_favors_document_edges():
    chunks = [
        _chunk("first", paragraph=0),
        _chunk("middle", paragraph=1),
        _chunk("last", paragraph=2),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks([1.0, 0.0], chunks, embeddings, scoring="hybrid", query_text="")

    assert scored[0].score_components["position"] > scored[1].score_components["position"]
    assert scored[2].score_components["position"] > scored[1].score_components["position"]


def test_hybrid_scoring_exposes_all_components():
    chunks = [_chunk("alpha", text="alpha evidence"), _chunk("beta", paragraph=1, text="beta evidence")]
    embeddings = [[1.0, 0.0], [0.8, 0.2]]

    scored = score_experimental_chunks([1.0, 0.0], chunks, embeddings, scoring="hybrid", query_text="alpha")

    assert set(scored[0].score_components) == {"cosine", "bm25", "position", "neighbor_coherence"}


def test_evidence_hybrid_scoring_uses_query_and_structure_components():
    chunks = [
        _chunk(
            "code",
            text="def train_model(model, data):\n    optimizer = AdamW(model.parameters())",
            metadata={
                "content_type": "code",
                "language": "python",
                "symbol_name": "train_model",
                "symbol_kind": "function",
            },
        ),
        _chunk("text", paragraph=1, text="unrelated prose about retrieval", metadata={"content_type": "document"}),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="evidence-hybrid",
        query_text="train model optimizer",
    )

    assert scored[0].value > scored[1].value
    assert scored[0].score_components["query_coverage"] > scored[1].score_components["query_coverage"]
    assert scored[0].score_components["structural_prior"] > scored[1].score_components["structural_prior"]


def test_production_scoring_rejects_experimental_profiles():
    chunks = [_chunk("alpha", text="alpha evidence")]
    embeddings = [[1.0, 0.0]]

    with pytest.raises(ValueError, match="Unsupported production scoring profile"):
        score_chunks([1.0, 0.0], chunks, embeddings, scoring="cosine", query_text="alpha")


def test_knapsack_aware_scoring_exposes_density_and_length_components():
    chunks = [
        _chunk("short", weight=40, text="alpha optimizer"),
        _chunk("long", paragraph=1, weight=400, text="alpha optimizer " + "filler " * 100),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="knapsack-aware",
        query_text="alpha optimizer",
    )

    assert "evidence_base" in scored[0].score_components
    assert "value_density_prior" in scored[0].score_components
    assert "length_utility" in scored[0].score_components
    assert scored[0].value >= 0.0


def test_query_support_scoring_exposes_general_support_components():
    chunks = [
        _chunk("support", weight=60, text="The retrieval optimizer uses budget allocation to select evidence chunks."),
        _chunk("distractor", paragraph=1, weight=60, text="A recipe describes sugar butter flour and baking time."),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="query-support",
        query_text="How does retrieval budget allocation select evidence?",
    )

    assert scored[0].value > scored[1].value
    assert scored[0].score_components["support_likelihood"] > scored[1].score_components["support_likelihood"]
    assert "phrase_overlap" in scored[0].score_components
    assert "term_proximity" in scored[0].score_components


def test_query_support_scoring_is_not_longbench_specific():
    chunks = [
        _chunk("policy", text="Urban renewal projects declined after the legal change."),
        _chunk("coffee", paragraph=1, text="Coffee shops adjusted their menu and decor."),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="query-support",
        query_text="urban renewal legal change",
    )

    assert scored[0].value > scored[1].value
    assert all("choice_A" not in item.score_components for item in scored)


def test_decision_aware_scoring_favors_discriminative_candidate_evidence():
    chunks = [
        _chunk(
            "support",
            text="The policy change happened in Lisbon after the 2019 election.",
        ),
        _chunk(
            "distractor",
            paragraph=1,
            text="The policy change was discussed alongside several unrelated reforms.",
        ),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="decision-aware",
        query_text=(
            "Where did the policy change happen?\n"
            "A. Lisbon\n"
            "B. Madrid\n"
            "C. Rome\n"
            "D. Berlin"
        ),
    )

    assert scored[0].value > scored[1].value
    assert scored[0].score_components["candidate_support"] > scored[1].score_components["candidate_support"]
    assert scored[0].score_components["candidate_contrast"] > scored[1].score_components["candidate_contrast"]


def test_decision_aware_scoring_falls_back_without_candidates():
    chunks = [_chunk("alpha", text="alpha evidence"), _chunk("beta", paragraph=1, text="unrelated")]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="decision-aware",
        query_text="alpha evidence",
    )

    assert scored[0].value > scored[1].value
    assert scored[0].score_components["candidate_support"] == 0.0


def test_budgetmem_style_profile_exposes_feature_proxy_components():
    chunks = [
        _chunk(
            "feature-rich",
            text=(
                "NASA reports 42 benchmark results. However, our results indicate "
                "that retrieval memory policy improves accuracy."
            ),
        ),
        _chunk("plain", paragraph=1, text="unrelated lowercase prose about cooking and music"),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    scored = score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="budgetmem-style",
        query_text="retrieval memory policy benchmark accuracy",
    )

    assert SCORING_PROFILES == ("evidence-hybrid",)
    assert "budgetmem-style" in EXPERIMENTAL_SCORING_PROFILES
    assert scored[0].value > scored[1].value
    assert scored[0].score_components["bm25"] > scored[1].score_components["bm25"]
    assert scored[0].score_components["entity_density"] > scored[1].score_components["entity_density"]
    assert scored[0].score_components["numerical_density"] > scored[1].score_components["numerical_density"]
    assert scored[0].score_components["discourse_marker"] > scored[1].score_components["discourse_marker"]


def test_redundancy_penalty_tracks_lexical_and_novelty_components():
    chunks = [
        _chunk("first", text="alpha beta gamma unique source"),
        _chunk("repeat", paragraph=1, text="alpha beta gamma repeated source"),
        _chunk("novel", paragraph=2, text="delta epsilon zeta independent"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]

    scored = score_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        redundancy_penalty=0.5,
        scoring="evidence-hybrid",
        query_text="alpha beta gamma",
    )

    repeated = scored[1]
    assert repeated.redundancy_penalty > 0.0
    assert repeated.score_components["lexical_overlap"] > 0.0
    assert repeated.score_components["novelty"] < 1.0


def test_redundancy_penalty_tracks_structural_overlap_for_code_chunks():
    chunks = [
        _chunk(
            "full",
            text="def train_model(model):\n    optimizer = AdamW(model.parameters())",
            metadata={"content_type": "code", "language": "python", "symbol_name": "train_model"},
        ),
        _chunk(
            "partial",
            paragraph=1,
            text="def train_model(model):\n    return model",
            metadata={"content_type": "code", "language": "python", "symbol_name": "train_model"},
        ),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    scored = score_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        redundancy_penalty=0.5,
        scoring="evidence-hybrid",
        query_text="train model",
    )

    assert scored[1].score_components["structural_overlap"] == 1.0
    assert scored[1].redundancy_penalty > 0.0


def test_redundancy_penalty_can_be_limited_to_candidate_pool():
    chunks = [_chunk("one", text="alpha one"), _chunk("two", text="alpha two"), _chunk("three", text="alpha three")]
    embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

    scored = score_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        redundancy_penalty=0.5,
        scoring="evidence-hybrid",
        query_text="alpha",
        redundancy_candidate_pool=2,
    )

    assert "novelty" in scored[1].score_components
    assert "novelty" not in scored[2].score_components


def test_knapsack_respects_budget_with_hybrid_scoring():
    chunks = [
        _chunk("alpha", weight=7, text="alpha evidence"),
        _chunk("beta", paragraph=1, weight=6, text="beta alpha support"),
        _chunk("gamma", paragraph=2, weight=4, text="gamma alpha short"),
    ]
    embeddings = [[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]]
    scored = score_experimental_chunks([1.0, 0.0], chunks, embeddings, scoring="hybrid", query_text="alpha")

    result = select_chunks(scored, strategy="knapsack", budget=10)

    assert result.used_tokens <= 10


def test_knapsack_candidate_pool_includes_dense_tradeoff_items():
    scored = [
        _scored("large", value=10.0, weight=100),
        _scored("dense", value=3.0, weight=3),
    ]

    topk_result = select_chunks(scored, strategy="budget-top-k", budget=3, candidate_pool=1)
    knapsack_result = select_chunks(scored, strategy="knapsack", budget=3, candidate_pool=1)

    assert topk_result.selected == []
    assert [item.chunk.id for item in knapsack_result.selected] == ["dense"]


def test_production_rag_uses_dense_similarity_ranking_under_budget():
    scored = [
        ScoredChunk(_chunk("evidence-scored", weight=4), value=0.95, raw_similarity=0.30, weight=4),
        ScoredChunk(_chunk("dense-rag-hit", paragraph=1, weight=4), value=0.40, raw_similarity=0.90, weight=4),
    ]

    production_rag = select_chunks(scored, strategy="production-rag", budget=4)
    budget_top_k = select_chunks(scored, strategy="budget-top-k", budget=4)

    assert [item.chunk.id for item in production_rag.selected] == ["dense-rag-hit"]
    assert [item.chunk.id for item in budget_top_k.selected] == ["evidence-scored"]
    assert production_rag.used_tokens <= 4


def test_knapsack_redundancy_penalizes_repeated_embeddings_without_mutating_input():
    scored = [
        ScoredChunk(_chunk("first", weight=2), value=1.0, raw_similarity=1.0, weight=2, embedding=[1.0, 0.0]),
        ScoredChunk(_chunk("repeat", paragraph=1, weight=2), value=0.9, raw_similarity=0.9, weight=2, embedding=[1.0, 0.0]),
        ScoredChunk(_chunk("novel", paragraph=2, weight=2), value=0.8, raw_similarity=0.8, weight=2, embedding=[0.0, 1.0]),
    ]

    result = select_chunks(scored, strategy="knapsack-redundancy", budget=4, mmr_lambda=0.5)

    assert {item.chunk.id for item in result.selected} == {"first", "novel"}
    assert scored[1].redundancy_penalty == 0.0


def test_knapsack_augment_keeps_topk_seed_and_fills_remaining_budget():
    scored = [
        _scored("seed", value=10.0, weight=5),
        _scored("too-large", value=9.0, weight=5),
        _scored("filler", value=7.0, weight=2),
    ]

    topk_result = select_chunks(scored, strategy="budget-top-k", budget=7, candidate_pool=2)
    augmented = select_chunks(scored, strategy="knapsack-augment", budget=7, candidate_pool=2)

    assert [item.chunk.id for item in topk_result.selected] == ["seed"]
    assert {item.chunk.id for item in augmented.selected} == {"seed", "filler"}
    assert augmented.used_tokens == 7


def test_greedy_density_prefers_value_per_token():
    scored = [
        _scored("large", value=10.0, weight=10),
        _scored("dense-a", value=6.0, weight=3),
        _scored("dense-b", value=5.0, weight=3),
    ]

    result = select_chunks(scored, strategy="greedy-density", budget=6)

    assert {item.chunk.id for item in result.selected} == {"dense-a", "dense-b"}
    assert result.used_tokens == 6


def test_knapsack_coverage_favors_complementary_query_terms():
    scored = [
        ScoredChunk(
            _chunk("alpha", text="alpha alpha alpha", weight=4),
            value=1.0,
            raw_similarity=1.0,
            weight=4,
        ),
        ScoredChunk(
            _chunk("beta", paragraph=1, text="beta gamma", weight=4),
            value=0.96,
            raw_similarity=0.96,
            weight=4,
        ),
        ScoredChunk(
            _chunk("repeat", paragraph=2, text="alpha repeated", weight=4),
            value=0.95,
            raw_similarity=0.95,
            weight=4,
        ),
    ]

    result = select_chunks(scored, strategy="knapsack-coverage", budget=8, coverage_query="alpha beta gamma")

    assert {item.chunk.id for item in result.selected} == {"alpha", "beta"}
    assert all("selector_query_coverage_bonus" in item.score_components for item in result.selected)


def test_reranker_blend_increases_high_reranked_chunk_value():
    low_base = ScoredChunk(_chunk("low-base"), value=0.2, raw_similarity=0.2, weight=1)
    high_reranked = ScoredChunk(_chunk("high-reranked"), value=0.1, raw_similarity=0.1, weight=1)

    blended = blend_reranker_scores(
        [low_base, high_reranked],
        [low_base, high_reranked],
        [0.0, 10.0],
        weight=0.5,
    )
    by_id = {item.chunk.id: item for item in blended}

    assert by_id["high-reranked"].value > high_reranked.value
    assert by_id["high-reranked"].value > by_id["low-base"].value
    assert by_id["high-reranked"].score_components["reranker_norm"] == 1.0


def test_gold_benchmark_metrics_on_controlled_index():
    chunks = [_chunk("evidence", weight=2), _chunk("distractor", paragraph=1, weight=2)]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    index = ChunkIndex(chunks=chunks, embeddings=embeddings, model_name="test")
    records = [GoldRecord(query="evidence", answer="answer", evidence_chunk_ids=["evidence"])]

    payload = run_gold_benchmark(
        index=index,
        embedder=_StaticEmbedder({"evidence": [1.0, 0.0]}),
        records=records,
        budgets=[10],
        reserve_output=0,
        candidate_pool=2,
    )

    summary = payload["budgets"][0]["summary"]["budget-top-k"]
    assert summary["evidence_recall_at_budget"] == 1.0
    assert summary["over_budget_rate"] == 0.0
    assert payload["budgets"][0]["effective_budget"] == 10


def test_redundancy_score_increases_for_similar_chunks():
    similar = [
        ScoredChunk(_chunk("a"), value=1.0, raw_similarity=1.0, weight=1, embedding=[1.0, 0.0]),
        ScoredChunk(_chunk("b"), value=1.0, raw_similarity=1.0, weight=1, embedding=[1.0, 0.0]),
    ]
    dissimilar = [
        ScoredChunk(_chunk("a"), value=1.0, raw_similarity=1.0, weight=1, embedding=[1.0, 0.0]),
        ScoredChunk(_chunk("c"), value=1.0, raw_similarity=1.0, weight=1, embedding=[0.0, 1.0]),
    ]

    assert redundancy_score(similar) > redundancy_score(dissimilar)


def test_default_ollama_model_uses_local_default_for_cloud_placeholder():
    assert _default_ollama_model("gpt-4o-mini") == "llama3.2:1b"
    assert _default_ollama_model("qwen2.5:3b") == "qwen2.5:3b"


def test_pack_infers_default_output_paths():
    tmp_path = _workspace_tmp()
    source_file = tmp_path / "paper.pdf"
    source_file.write_text("placeholder", encoding="utf-8")
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    assert cli_module._infer_pack_output_path(source_file) == tmp_path / "paper-tp.md"
    assert cli_module._infer_pack_output_path(source_dir) == tmp_path / "docs-tp.md"
    assert cli_module._infer_pack_output_path(source_file, str(tmp_path / "custom.md")) == tmp_path / "custom.md"


def test_pack_auto_budget_defaults_and_clamps():
    small = cli_module._resolve_pack_budget(
        source_tokens=2_000,
        budget=None,
        budget_ratio=0.50,
        min_budget=1_200,
        max_budget=64_000,
        reserve_output=None,
    )
    medium = cli_module._resolve_pack_budget(
        source_tokens=40_000,
        budget=None,
        budget_ratio=0.50,
        min_budget=1_200,
        max_budget=64_000,
        reserve_output=None,
    )
    large = cli_module._resolve_pack_budget(
        source_tokens=200_000,
        budget=None,
        budget_ratio=0.50,
        min_budget=1_200,
        max_budget=64_000,
        reserve_output=None,
    )
    manual = cli_module._resolve_pack_budget(
        source_tokens=200_000,
        budget=9_000,
        budget_ratio=0.50,
        min_budget=1_200,
        max_budget=64_000,
        reserve_output=None,
    )

    assert small.budget == 1_200
    assert small.cap_reason == "min-budget"
    assert medium.budget == 20_000
    assert medium.reserve_output == 2_000
    assert large.budget == 64_000
    assert large.reserve_output == 4_000
    assert large.cap_reason == "max-budget"
    assert manual.mode == "manual"
    assert manual.budget == 9_000
    assert manual.reserve_output == 900


def test_pack_command_writes_markdown_with_auto_budget(monkeypatch):
    tmp_path = _workspace_tmp()
    source = tmp_path / "mini_context.txt"
    source.write_text(
        "Alpha evidence explains the TokenPack budget selector.\n\n"
        "Beta notes describe unrelated implementation details.\n\n"
        "Alpha budget packing keeps useful evidence under a context limit.",
        encoding="utf-8",
    )
    output = tmp_path / "mini_context-tp.md"
    monkeypatch.setattr(cli_module, "_make_cli_embedder", lambda args, model_name: _ToyEmbedder(dimensions=16))

    exit_code = cli_module.main(["pack", str(source), "--query", "alpha budget evidence", "--out", str(output)])

    text = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "# TokenPack Packed Context" in text
    assert "**Source:**" in text
    assert "**Selected:**" in text
    assert "[Source:" in text
    assert "Index artifact" not in text
    assert "id=chunk" not in text
    assert "Alpha evidence" in text


def test_pack_command_debug_output_keeps_technical_metadata(monkeypatch):
    tmp_path = _workspace_tmp()
    source = tmp_path / "mini_context.txt"
    source.write_text("Alpha evidence explains the TokenPack budget selector.", encoding="utf-8")
    output = tmp_path / "mini_context-debug.md"
    monkeypatch.setattr(cli_module, "_make_cli_embedder", lambda args, model_name: _ToyEmbedder(dimensions=16))

    exit_code = cli_module.main(
        [
            "pack",
            str(source),
            "--query",
            "alpha budget evidence",
            "--out",
            str(output),
            "--output-detail",
            "debug",
        ]
    )

    text = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "| Budget mode | auto |" in text
    assert "| Selector | budget-top-k |" in text
    assert "Index artifact" in text
    assert "[Chunk 1:" in text


def test_pack_command_refuses_existing_output():
    tmp_path = _workspace_tmp()
    source = tmp_path / "mini_context.txt"
    source.write_text("Alpha evidence.", encoding="utf-8")
    output = tmp_path / "mini_context-tp.md"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli_module.main(["pack", str(source), "--query", "alpha", "--out", str(output)])

    assert "Output already exists" in str(exc.value)


def test_mcp_pack_context_returns_inline_markdown_and_artifacts(monkeypatch):
    tmp_path = _workspace_tmp()
    source = tmp_path / "mini_context.txt"
    source.write_text(
        "Alpha evidence explains the TokenPack budget selector.\n\n"
        "Beta notes describe unrelated implementation details.\n\n"
        "Alpha budget packing keeps useful evidence under a context limit.",
        encoding="utf-8",
    )
    config = mcp_server.McpServerConfig(workspace=tmp_path)
    monkeypatch.setattr(mcp_server, "_make_mcp_embedder", lambda config: _ToyEmbedder(dimensions=16))

    payload = mcp_server.pack_context_tool(
        source="mini_context.txt",
        query="alpha budget evidence",
        config=config,
    )

    assert payload["output_path"] == str((tmp_path / "mini_context-tp.md").resolve())
    assert payload["selection_path"].startswith(str((tmp_path / ".tokenpack" / "runs").resolve()))
    assert payload["budget_mode"] == "auto"
    assert payload["selector"] == "budget-top-k"
    assert "# TokenPack Packed Context" in payload["markdown"]
    assert (tmp_path / "mini_context-tp.md").exists()


def test_mcp_workspace_rejects_outside_paths():
    workspace = _workspace_tmp()
    outside = _workspace_tmp() / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    config = mcp_server.McpServerConfig(workspace=workspace)

    with pytest.raises(ValueError) as exc:
        mcp_server.read_packed_context_tool(path=str(outside.resolve()), config=config)

    assert "outside the MCP workspace" in str(exc.value)


def test_mcp_read_packed_context_slices_workspace_file():
    workspace = _workspace_tmp()
    packed = workspace / "packed.md"
    packed.write_text("abcdef", encoding="utf-8")
    config = mcp_server.McpServerConfig(workspace=workspace)

    payload = mcp_server.read_packed_context_tool(path="packed.md", offset=2, max_chars=3, config=config)

    assert payload["text"] == "cde"
    assert payload["next_offset"] == 5
    assert payload["truncated"] is True


def test_export_orders_chunks_by_original_position():
    later = _chunk("later", document_index=0, paragraph=3, weight=4)
    earlier = _chunk("earlier", document_index=0, paragraph=1, weight=4)

    rendered = render_context([later, earlier], include_headers=False)

    assert rendered.index("Text earlier") < rendered.index("Text later")


def test_llmlingua_compression_adapter_uses_selected_chunks_as_context():
    chunks = [_chunk("a", text="alpha evidence one", weight=3), _chunk("b", text="beta evidence two", weight=3)]
    backend = _FakePromptCompressor()

    result = compress_chunks(
        chunks,
        CompressionConfig(
            compressor="llmlingua",
            rate=0.5,
            target_tokens=5,
            question="What evidence matters?",
            longllmlingua=True,
        ),
        backend=backend,
    )

    assert backend.context == ["alpha evidence one", "beta evidence two"]
    assert backend.kwargs["rank_method"] == "longllmlingua"
    assert result.compressed_prompt == "alpha beta"
    assert result.origin_tokens == 6
    assert result.compressed_tokens == 2
    assert result.saving_rate > 0.0


def test_longllmlingua_requires_question_before_loading_backend():
    try:
        compress_chunks([_chunk("a")], CompressionConfig(compressor="llmlingua", longllmlingua=True))
    except ValueError as exc:
        assert "requires a non-empty question" in str(exc)
    else:
        raise AssertionError("LongLLMLingua should require a question.")


def _chunk(
    chunk_id: str,
    document_index: int = 0,
    paragraph: int = 0,
    weight: int = 1,
    text: str | None = None,
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text or f"Text {chunk_id}",
        source_path="doc.txt",
        document_index=document_index,
        start_page=None,
        end_page=None,
        start_paragraph=paragraph,
        end_paragraph=paragraph,
        char_start=paragraph * 10,
        char_end=paragraph * 10 + 5,
        token_count=weight,
        metadata=metadata or {},
    )


def _scored(chunk_id: str, value: float, weight: int) -> ScoredChunk:
    chunk = _chunk(chunk_id, weight=weight)
    return ScoredChunk(chunk=chunk, value=value, raw_similarity=value, weight=weight)


def _workspace_tmp() -> Path:
    root = Path(".test-tmp")
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


class _StaticEmbedder:
    model_name = "static"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(text, [0.0, 1.0]) for text in texts]


class _ToyEmbedder:
    model_name = "test-toy-embedder"

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.lower()):
            index = sum(ord(char) for char in token) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class _FakePromptCompressor:
    def __init__(self) -> None:
        self.context: list[str] = []
        self.kwargs: dict = {}

    def compress_prompt(self, context, **kwargs):
        self.context = list(context)
        self.kwargs = kwargs
        return {
            "compressed_prompt": "alpha beta",
            "origin_tokens": 6,
            "compressed_tokens": 2,
            "ratio": "3.0x",
        }

