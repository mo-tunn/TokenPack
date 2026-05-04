from __future__ import annotations

import uuid
from pathlib import Path

from tokenpack.benchmark import redundancy_score, run_gold_benchmark
from tokenpack.chunking import ParagraphGroupChunker
from tokenpack.dataset import GoldRecord, load_gold_records, save_gold_records, validate_gold_records
from tokenpack.embeddings import EmbeddingCache, HashingEmbedder
from tokenpack.export import render_context
from tokenpack.generation import _default_ollama_model
from tokenpack.index import ChunkIndex
from tokenpack.loaders import load_text_blocks
from tokenpack.pipeline import ingest_path
from tokenpack.models import Chunk, ScoredChunk
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


def test_chunker_respects_max_tokens():
    tmp_path = _workspace_tmp()
    source = tmp_path / "doc.txt"
    source.write_text("\n\n".join(f"paragraph {idx} " + "word " * 20 for idx in range(8)), encoding="utf-8")
    blocks = load_text_blocks(source)
    chunker = ParagraphGroupChunker(target_tokens=40, min_tokens=20, max_tokens=55, token_counter=TokenCounter())

    chunks = chunker.chunk(blocks)

    assert chunks
    assert all(chunk.token_count <= 55 for chunk in chunks)


def test_embedding_cache_reuses_existing_vectors():
    tmp_path = _workspace_tmp()
    cache = EmbeddingCache(tmp_path / "embeddings.json")
    embedder = HashingEmbedder(dimensions=16)

    first = cache.get_or_embed(["alpha beta"], embedder)
    second = EmbeddingCache(tmp_path / "embeddings.json").get_or_embed(["alpha beta"], embedder)

    assert first == second


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
        embedder=HashingEmbedder(dimensions=64),
        target_tokens=80,
        min_tokens=1,
        max_tokens=80,
        chunker_name="semantic-threshold",
        semantic_threshold=0.2,
    )

    assert len(index.chunks) >= 2
    assert all(chunk.token_count <= 80 for chunk in index.chunks)


def test_knapsack_never_exceeds_budget():
    scored = [
        _scored("a", value=8.0, weight=7),
        _scored("b", value=7.0, weight=6),
        _scored("c", value=4.0, weight=4),
    ]

    result = select_chunks(scored, strategy="knapsack", budget=10)

    assert result.used_tokens <= 10
    assert {item.chunk.id for item in result.selected} == {"b", "c"}


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


def test_default_ollama_model_uses_local_default_for_openai_placeholder():
    assert _default_ollama_model("gpt-4o-mini") == "llama3.2:1b"
    assert _default_ollama_model("qwen2.5:3b") == "qwen2.5:3b"


def test_export_orders_chunks_by_original_position():
    later = _chunk("later", document_index=0, paragraph=3, weight=4)
    earlier = _chunk("earlier", document_index=0, paragraph=1, weight=4)

    rendered = render_context([later, earlier], include_headers=False)

    assert rendered.index("Text earlier") < rendered.index("Text later")


def _chunk(chunk_id: str, document_index: int = 0, paragraph: int = 0, weight: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"Text {chunk_id}",
        source_path="doc.txt",
        document_index=document_index,
        start_page=None,
        end_page=None,
        start_paragraph=paragraph,
        end_paragraph=paragraph,
        char_start=paragraph * 10,
        char_end=paragraph * 10 + 5,
        token_count=weight,
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

