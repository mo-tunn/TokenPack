from __future__ import annotations

import builtins
import json
import os
import sys
import types
import uuid
from pathlib import Path

import pytest

from tokenpack import (
    benchmark as benchmark_module,
    cli,
    generation as generation_module,
    loaders as loaders_module,
    mcp_server,
    packing as packing_module,
    pipeline as pipeline_module,
    reporting as reporting_module,
    reranking as reranking_module,
    scoring as prod_scoring,
    selectors as selectors_module,
)
from tokenpack import scoring_experimental as exp_scoring
from tokenpack.chunk_profiles import resolve_chunk_size_config
from tokenpack.chunking import SemanticThresholdChunker, StructureAwareChunker
from tokenpack.compression import (
    CompressionConfig,
    CompressionResult,
    _llmlingua_kwargs,
    _make_llmlingua_backend,
    _ratio_from_payload,
    _resolve_local_model_path,
    compress_chunks,
)
from tokenpack.dataset import (
    GoldRecord,
    _spaced_indices,
    load_gold_records,
    propose_gold_records,
    save_gold_records,
    validate_gold_records,
)
from tokenpack.doctor import _ollama_status, collect_diagnostics
from tokenpack.embeddings import (
    EmbeddingCache,
    SentenceTransformerEmbedder,
    cosine,
    make_embedder,
    normalize,
)
from tokenpack.export import export_selection, render_context, render_compressed_context
from tokenpack.generation import (
    _cerebras_answer,
    _default_ollama_model,
    _groq_answer,
    _ollama_answer,
    answer_from_selection,
    save_answer,
)
from tokenpack.index import ChunkIndex
from tokenpack.loaders import (
    _HTMLTextExtractor,
    _json_to_text,
    _split_pdf_text,
    iter_supported_files,
    load_blocks,
    load_code_blocks,
    load_csv_blocks,
    load_docx_blocks,
    load_json_blocks,
    load_office_blocks,
    load_pdf_blocks,
    load_pptx_blocks,
    load_xlsx_blocks,
)
from tokenpack.models import Chunk, ScoredChunk, SelectionResult, TextBlock
from tokenpack.reranking import CrossEncoderReranker, apply_reranker, blend_reranker_scores
from tokenpack.reporting import save_csv_report, save_markdown_report
from tokenpack.tokenization import TokenCounter


def test_reporting_writes_markdown_and_csv_for_budget_payload():
    tmp_path = _workspace_tmp()
    payload = {
        "budgets": [
            {
                "budget": 100,
                "effective_budget": 80,
                "summary": {
                    "budget-top-k": {
                        "evidence_recall_at_budget": 0.75,
                        "evidence_precision": 1.0,
                        "coverage_ratio": 0.5,
                        "avg_used_tokens": 70,
                        "budget_utilization": 0.875,
                        "over_budget_rate": 0.0,
                        "avg_over_budget_tokens": 0,
                        "avg_value_density": 0.12,
                        "redundancy_score": None,
                        "avg_latency_seconds": 0.002,
                    }
                },
            }
        ]
    }

    markdown = tmp_path / "reports" / "bench.md"
    csv_path = tmp_path / "reports" / "bench.csv"
    save_markdown_report(payload, markdown)
    save_csv_report(payload, csv_path)

    assert "Budget 100" in markdown.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "budget-top-k" in csv_text
    assert "0.12" in csv_text


def test_generation_provider_branches_and_serialization(monkeypatch):
    tmp_path = _workspace_tmp()
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selected": [{"chunk": _chunk("a", text="alpha evidence", weight=3).to_dict()}]}),
        encoding="utf-8",
    )

    none_payload = answer_from_selection("question?", selection, provider="none")
    assert none_payload["model"] is None
    assert none_payload["context_tokens"] == 3

    monkeypatch.setattr("tokenpack.generation._ollama_answer", lambda prompt, model, base_url: f"{model}:{base_url}")
    local_payload = answer_from_selection("q", selection, provider="local", model="gpt-4o-mini", ollama_url="http://ollama")
    assert local_payload["answer"] == "llama3.2:1b:http://ollama"

    out = tmp_path / "answer.json"
    save_answer(local_payload, out)
    assert json.loads(out.read_text(encoding="utf-8"))["provider"] == "local"

    with pytest.raises(ValueError, match="Unknown answer provider"):
        answer_from_selection("q", selection, provider="mystery")


def test_generation_http_helpers_use_env_and_parse_payload(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "  answer  "}}], "response": " local "}).encode()

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    assert _cerebras_answer("prompt", "model") == "answer"
    assert _groq_answer("prompt", "model") == "answer"
    assert _ollama_answer("prompt", "llama", "http://localhost:1") == "local"
    assert any("cerebras" in url for url, _ in calls)
    assert any("groq" in url for url, _ in calls)
    assert any(url.endswith("/api/generate") for url, _ in calls)

    monkeypatch.delenv("CEREBRAS_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="CEREBRAS_API_KEY"):
        _cerebras_answer("prompt", "model")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _groq_answer("prompt", "model")


def test_ollama_error_and_default_model(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(RuntimeError, match="Ollama generation failed"):
        _ollama_answer("prompt", "llama", "http://localhost:1")

    monkeypatch.setenv("TOKENPACK_OLLAMA_MODEL", "custom")
    assert _default_ollama_model("gpt-4o-mini") == "custom"
    assert _default_ollama_model("chosen") == "chosen"


def test_export_context_variants_and_compression(monkeypatch):
    tmp_path = _workspace_tmp()
    earlier = _chunk("early", paragraph=1, text="early text", weight=2)
    later = _chunk("late", paragraph=2, text="late text", weight=2)

    assert "[Source:" in render_context([later, earlier], header_style="source")
    assert render_context([later], header_style="none").strip() == "late text"
    with pytest.raises(ValueError, match="Unknown context header style"):
        render_context([later], header_style="weird")

    class FakeResult:
        compressed_prompt = "small prompt"
        origin_tokens = 10
        compressed_tokens = 4
        ratio = 2.5
        metadata = {"compressor": "fake"}

        @property
        def saving_rate(self):
            return 0.6

    fake_result = FakeResult()
    monkeypatch.setattr("tokenpack.export.compress_chunks", lambda chunks, config: fake_result)
    rendered, result = render_compressed_context([earlier], CompressionConfig(compressor="llmlingua"))
    assert "small prompt" in rendered
    assert result.saving_rate == 0.6

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": [{"chunk": earlier.to_dict()}]}), encoding="utf-8")
    plain_out = tmp_path / "plain.md"
    assert export_selection(selection, plain_out, include_headers=False) is None
    assert plain_out.read_text(encoding="utf-8").strip() == "early text"

    compressed_out = tmp_path / "compressed.md"
    assert export_selection(selection, compressed_out, compression_config=CompressionConfig(compressor="llmlingua")) is fake_result
    assert "small prompt" in compressed_out.read_text(encoding="utf-8")


def test_doctor_diagnostics_success_and_failure(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"models":[{"name":"llama"},{"missing":"name"}]}'

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TOKENPACK_HF_OFFLINE", "1")
    monkeypatch.setenv("TOKENPACK_OLLAMA_MODEL", "llama")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse())
    diagnostics = collect_diagnostics("http://ollama")
    assert diagnostics["environment"]["hf_hub_offline"] is True
    assert diagnostics["ollama"]["models"] == ["llama"]

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("nope")))
    assert _ollama_status("http://ollama")["available"] is False


def test_dataset_errors_and_proposals():
    tmp_path = _workspace_tmp()
    invalid = tmp_path / "gold.jsonl"
    invalid.write_text('{"answer":"missing query"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_gold_records(invalid)
    with pytest.raises(ValueError, match="query"):
        GoldRecord.from_dict({"evidence_chunk_ids": ["x"]})
    with pytest.raises(ValueError, match="evidence_chunk_ids"):
        GoldRecord.from_dict({"query": "alpha"})

    chunks = [
        _chunk("one", text="Alpha alpha beta evidence sentence. More text."),
        _chunk("two", text="The and with about because."),
        _chunk("three", text="Gamma delta findings."),
    ]
    records = propose_gold_records(ChunkIndex(chunks=chunks, embeddings=[[1.0]] * 3, model_name="toy"), sample_size=5)
    assert records[0].metadata["proposal"] == "keyword"
    assert records[0].answer.startswith("Alpha")


def test_embeddings_fake_sentence_transformer_paths(monkeypatch):
    created = []

    class FakeSentenceTransformer:
        def __init__(self, model_name, local_files_only=False):
            created.append((model_name, local_files_only))
            if model_name == "model" and local_files_only is True and len(created) == 1:
                raise RuntimeError("cache miss")

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return [[1, 2], [3, 4]][: len(texts)]

    module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    embedder = SentenceTransformerEmbedder("model", local_files_only=None)
    assert created == [("model", True), ("model", False)]
    assert embedder.embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]

    created.clear()
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    embedder = make_embedder("offline-model")
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert created == [("offline-model", True)]


def test_embedding_cache_loads_existing_records():
    tmp_path = _workspace_tmp()
    key = EmbeddingCache._key("alpha", "toy")
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps({key: ["1", 2]}), encoding="utf-8")
    cache = EmbeddingCache(path)

    assert cache.get_or_embed(["alpha"], _Embedder()) == [[1.0, 2.0]]


def test_compression_helpers_and_local_model_resolution(monkeypatch):
    empty = compress_chunks([], CompressionConfig(compressor="none"))
    assert empty.saving_rate == 0.0
    assert empty.compressed_prompt == ""

    with pytest.raises(ValueError, match="Unknown compressor"):
        compress_chunks([_chunk("a")], CompressionConfig(compressor="zip"))

    result = compress_chunks(
        [_chunk("a", text="alpha beta gamma")],
        CompressionConfig(compressor="llmlingua", rate=0.2),
        backend=_WeirdPromptCompressor(),
    )
    assert result.origin_tokens >= result.compressed_tokens
    assert result.ratio >= 1.0

    tmp_path = _workspace_tmp()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert _resolve_local_model_path(str(model_dir)) == str(model_dir)

    hf_module = types.SimpleNamespace(snapshot_download=lambda repo_id, local_files_only: f"/cache/{repo_id}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
    assert _resolve_local_model_path("repo") == "/cache/repo"

    hf_module.snapshot_download = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("miss"))
    with pytest.raises(RuntimeError, match="not cached locally"):
        _resolve_local_model_path("repo")


def test_llmlingua_backend_instantiation_with_fake_module(monkeypatch):
    created = {}

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(sys.modules, "llmlingua", types.SimpleNamespace(PromptCompressor=FakePromptCompressor))
    backend = _make_llmlingua_backend(CompressionConfig(compressor="llmlingua", local_files_only=False, llmlingua2=True))

    assert isinstance(backend, FakePromptCompressor)
    assert created["use_llmlingua2"] is True


def test_loader_dispatch_and_structured_edge_cases():
    tmp_path = _workspace_tmp()
    (tmp_path / "notes.md").write_text("Alpha paragraph.\n\nBeta paragraph.", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"title":{"nested":"Alpha"},"count":2}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")
    (tmp_path / "data.jsonl").write_text('{"a": 1}\nnot-json\n\n', encoding="utf-8")
    (tmp_path / "empty.csv").write_text("", encoding="utf-8")
    (tmp_path / "row.tsv").write_text("a\tb\n1\t2\n", encoding="utf-8")
    (tmp_path / "style.css").write_text(".card { color: red; }\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text("name='alpha'\n", encoding="utf-8")
    (tmp_path / "image.png").write_text("ignored", encoding="utf-8")

    assert (tmp_path / "notes.md") in iter_supported_files(tmp_path, source_type="document")
    assert (tmp_path / "style.css") not in iter_supported_files(tmp_path, source_type="document")
    assert (tmp_path / "style.css") in iter_supported_files(tmp_path, source_type="code")
    with pytest.raises(ValueError, match="Unknown source_type"):
        iter_supported_files(tmp_path, source_type="bad")

    blocks = load_blocks(tmp_path)
    assert any(block.metadata.get("source_format") == "toml" for block in blocks)
    assert any(block.metadata.get("parse_error") == "json_decode_error" for block in load_json_blocks(tmp_path / "bad.json"))
    assert any(block.metadata.get("parse_error") == "json_decode_error" for block in load_json_blocks(tmp_path / "data.jsonl"))
    assert load_csv_blocks(tmp_path / "empty.csv") == []
    assert load_csv_blocks(tmp_path / "row.tsv")[0].metadata["source_format"] == "tsv"
    assert _json_to_text({"a": [1, {"b": None}, {"c": 3}]}) == "a[0]: 1\na[2].c: 3"


def test_html_parser_ignores_nested_script_and_adds_breaks():
    parser = _HTMLTextExtractor()
    parser.feed("<div>Visible<script>secret<style>nested</style></script><p>Again</p></div>")
    text = parser.text()

    assert "Visible" in text
    assert "Again" in text
    assert "secret" not in text


def test_office_loaders_with_fake_optional_modules(monkeypatch):
    tmp_path = _workspace_tmp()
    docx = tmp_path / "doc.docx"
    pptx = tmp_path / "deck.pptx"
    xlsx = tmp_path / "sheet.xlsx"
    for path in (docx, pptx, xlsx):
        path.write_text("placeholder", encoding="utf-8")

    fake_doc = types.SimpleNamespace(
        paragraphs=[types.SimpleNamespace(text="Intro"), types.SimpleNamespace(text="")],
        tables=[types.SimpleNamespace(rows=[types.SimpleNamespace(cells=[types.SimpleNamespace(text="A"), types.SimpleNamespace(text="B")])])],
    )
    monkeypatch.setitem(sys.modules, "docx", types.SimpleNamespace(Document=lambda path: fake_doc))
    doc_blocks = load_docx_blocks(docx)
    assert [block.text for block in doc_blocks] == ["Intro", "A | B"]

    fake_slide = types.SimpleNamespace(shapes=[types.SimpleNamespace(text="Title"), types.SimpleNamespace(text="")])
    monkeypatch.setitem(sys.modules, "pptx", types.SimpleNamespace(Presentation=lambda path: types.SimpleNamespace(slides=[fake_slide])))
    assert load_pptx_blocks(pptx)[0].metadata["slide"] == 1

    fake_sheet = types.SimpleNamespace(title="Sheet1", iter_rows=lambda values_only=True: [(None, ""), ("A", 3)])
    fake_workbook = types.SimpleNamespace(worksheets=[fake_sheet])
    monkeypatch.setitem(sys.modules, "openpyxl", types.SimpleNamespace(load_workbook=lambda *args, **kwargs: fake_workbook))
    xlsx_blocks = load_xlsx_blocks(xlsx)
    assert xlsx_blocks[0].text == "Sheet: Sheet1\nRow 2: A | 3"

    assert load_office_blocks(docx)[0].text == "Intro"
    with pytest.raises(ValueError, match="Unknown office"):
        load_office_blocks(tmp_path / "file.odt")


def test_pdf_loaders_with_fake_backends(monkeypatch):
    tmp_path = _workspace_tmp()
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("placeholder", encoding="utf-8")

    class FakePage:
        def get_text(self, mode):
            return [(1, 2, 3, 4, "METHODS\n\nprint(x); y = z < 3\n\nThis is one. This is two. This is three. This is four.")]

    class FakeDoc:
        def __enter__(self):
            return [FakePage()]

        def __exit__(self, *args):
            return None

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda path: FakeDoc()))
    blocks = load_pdf_blocks(pdf)
    assert any(block.bbox == (1.0, 2.0, 3.0, 4.0) for block in blocks)
    assert any(block.metadata["content_type"] == "code" for block in blocks)

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda path: (_ for _ in ()).throw(RuntimeError("boom"))))

    class FakePdfPage:
        def extract_text(self):
            return "Plain sentence. Second sentence."

    monkeypatch.setitem(sys.modules, "pypdf", types.SimpleNamespace(PdfReader=lambda path: types.SimpleNamespace(pages=[FakePdfPage()])))
    fallback_blocks = load_pdf_blocks(pdf)
    assert fallback_blocks[0].page == 1
    assert fallback_blocks[0].metadata["source_format"] == "pdf"

    assert _split_pdf_text("   \n") == []


def test_code_loaders_cover_python_and_regex_paths():
    tmp_path = _workspace_tmp()
    py = tmp_path / "module.py"
    py.write_text("import os\n\nclass Alpha:\n    pass\n\ndef beta():\n    return 1\n\nx = 3\n", encoding="utf-8")
    py_blocks = load_code_blocks(py)
    assert any(block.metadata.get("symbol_name") == "Alpha" for block in py_blocks)
    assert any(block.metadata.get("symbol_name") == "beta" for block in py_blocks)

    bad_py = tmp_path / "bad.py"
    bad_py.write_text("def broken(:\n    pass\n", encoding="utf-8")
    assert load_code_blocks(bad_py)[0].metadata["language"] == "python"

    js = tmp_path / "app.js"
    js.write_text("const helper = () => 1;\nfunction run() { return helper(); }\n", encoding="utf-8")
    assert any(block.metadata.get("symbol_name") == "helper" for block in load_code_blocks(js))

    for name, text, expected in [
        ("main.go", "func Run() {}\ntype Thing struct{}\n", "Run"),
        ("lib.rs", "pub fn run() {}\nstruct Thing {}\n", "run"),
        ("main.cpp", "int run() {\nreturn 1;\n}\n", "run"),
        ("Main.java", "public class Main {}\nprivate int run() {\nreturn 1;\n}\n", "Main"),
    ]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        assert any(block.metadata.get("symbol_name") == expected for block in load_code_blocks(path))


def test_chunking_edge_paths():
    with pytest.raises(ValueError, match="min_tokens"):
        StructureAwareChunker(target_tokens=5, min_tokens=10, max_tokens=20)

    blocks = [
        _block("a", "one " * 20, metadata={"content_type": "document", "section_hint": "A"}, paragraph=0),
        _block("b", "two " * 20, metadata={"content_type": "document", "section_hint": "B"}, paragraph=1),
        _block("c", "def f():\n    return 1", metadata={"content_type": "code", "symbol_name": "f"}, paragraph=2),
    ]
    chunker = StructureAwareChunker(target_tokens=20, min_tokens=5, max_tokens=30, block_embeddings=[[1, 0], [0, 1], [1, 0]])
    chunks = chunker.chunk(blocks)
    assert len(chunks) >= 3
    assert chunks[0].metadata["chunker"] == "structure-aware"
    assert chunks[0].metadata["semantic_threshold"] == 0.35

    with pytest.raises(ValueError, match="one embedding"):
        StructureAwareChunker(block_embeddings=[[1.0]]).chunk(blocks)

    semantic = SemanticThresholdChunker([[1, 0], [0, 1], [0, 1]], target_tokens=20, min_tokens=1, max_tokens=30)
    sem_chunks = semantic.chunk(blocks)
    assert sem_chunks[0].metadata["chunker"] == "semantic-threshold"
    with pytest.raises(ValueError, match="one embedding"):
        SemanticThresholdChunker([[1.0]]).chunk(blocks)


def test_mcp_helpers_and_server_registration(monkeypatch, capsys):
    workspace = _workspace_tmp()
    config = mcp_server.McpServerConfig(workspace=workspace)
    packed = workspace / "packed.md"
    packed.write_text("abc", encoding="utf-8")

    with pytest.raises(ValueError, match="offset"):
        mcp_server.read_packed_context_tool(path="packed.md", config=config, offset=-1)
    with pytest.raises(ValueError, match="max_chars"):
        mcp_server.read_packed_context_tool(path="packed.md", config=config, max_chars=0)
    with pytest.raises(ValueError, match="does not exist"):
        mcp_server.read_packed_context_tool(path="missing.md", config=config)

    any_config = mcp_server.McpServerConfig(workspace=workspace, allow_any_path=True)
    assert mcp_server._resolve_workspace_path(workspace / "x.txt", any_config).name == "x.txt"
    assert mcp_server._run_root(any_config) == Path(".tokenpack/runs")

    payload = mcp_server._pack_result_payload(_FakePackResult("x" * (mcp_server.INLINE_MARKDOWN_LIMIT + 5)))
    assert payload["markdown_truncated"] is True
    assert payload["next_offset"] == mcp_server.INLINE_MARKDOWN_LIMIT

    registered = {}

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def decorator(fn):
                registered[fn.__name__] = fn
                return fn

            return decorator

        def run(self, transport):
            registered["transport"] = transport

    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", types.SimpleNamespace(FastMCP=FakeFastMCP))
    server = mcp_server.build_server(config)
    assert server.name == "TokenPack-RAG"
    assert {"pack_context", "read_packed_context"} <= registered.keys()

    monkeypatch.setattr(mcp_server, "build_server", lambda config: server)
    assert mcp_server.main(["--workspace", str(workspace)]) == 0
    assert registered["transport"] == "stdio"

    monkeypatch.setattr(mcp_server, "build_server", lambda config: (_ for _ in ()).throw(RuntimeError("missing mcp")))
    assert mcp_server.main(["--workspace", str(workspace)]) == 1
    assert "missing mcp" in capsys.readouterr().err


def test_mcp_pack_context_tool_success_and_conflict(monkeypatch):
    workspace = _workspace_tmp()
    source = workspace / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    conflict = workspace / "source-tp.md"
    conflict.write_text("exists", encoding="utf-8")
    config = mcp_server.McpServerConfig(workspace=workspace)

    with pytest.raises(FileExistsError, match="Output already exists"):
        mcp_server.pack_context_tool(source="source.txt", query="q", config=config)

    monkeypatch.setattr(mcp_server, "_make_mcp_embedder", lambda config: _Embedder())
    monkeypatch.setattr(mcp_server, "pack_source", lambda **kwargs: _FakePackResult("markdown"))
    payload = mcp_server.pack_context_tool(
        source="source.txt",
        query="q",
        out="custom.md",
        config=config,
        overwrite=True,
        budget=100,
    )
    assert payload["markdown"] == "markdown"


def test_reranker_cross_encoder_and_blending(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def predict(self, pairs, show_progress_bar=False):
            return [0.25 for _ in pairs]

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    reranker = CrossEncoderReranker(device="cpu", local_files_only=False)
    scored = [_scored("a", 0.1, 1), _scored("b", 0.5, 1)]
    assert reranker.score("query", scored) == [0.25, 0.25]

    applied = apply_reranker(scored, query="query", reranker=reranker, candidate_pool=1, weight=2.0)
    assert applied[1].score_components["reranker_weight"] == 1.0
    assert applied[0].score_components == {}
    assert blend_reranker_scores(scored, [], [], weight=0.5) == scored


def test_cli_subcommands_with_test_doubles(monkeypatch, capsys):
    tmp_path = _workspace_tmp()
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    index_path = tmp_path / "index.json"
    output = tmp_path / "out.json"

    monkeypatch.setattr(cli, "collect_diagnostics", lambda url: {"ollama": {"available": False}})
    assert cli.main(["doctor"]) == 0
    assert "ollama" in capsys.readouterr().out

    monkeypatch.setattr(cli, "_make_cli_embedder", lambda args, model_name: _Embedder())
    monkeypatch.setattr(cli, "ingest_path", lambda *args, **kwargs: ChunkIndex(chunks=[_chunk("a")], embeddings=[[1.0]], model_name="toy"))
    assert cli.main(["--model", "quality", "ingest", str(source), "--index", str(index_path)]) == 0
    assert "Indexed 1 chunks" in capsys.readouterr().out

    monkeypatch.setattr(cli, "load_index", lambda path: ChunkIndex(chunks=[_chunk("a")], embeddings=[[1.0]], model_name="toy"))
    monkeypatch.setattr(cli, "score_chunks", lambda *args, **kwargs: [_scored("a", 1.0, 1)])
    monkeypatch.setattr(cli, "select_chunks", lambda *args, **kwargs: _FakeSelectionResult())
    assert cli.main(["select", "--query", "alpha", "--index", str(index_path), "--output", str(output), "--json"]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["used_tokens"] == 1

    benchmark_payload = {"budget": 100, "summary": {"budget-top-k": {"evidence_recall_at_budget": 1.0}}}
    monkeypatch.setattr(cli, "run_benchmark", lambda *args, **kwargs: benchmark_payload)
    saved = {}
    monkeypatch.setattr(cli, "save_benchmark", lambda payload, path: saved.setdefault("benchmark", path))
    assert cli.main(["benchmark", "--index", str(index_path), "--output", str(output), "--markdown-output", str(tmp_path / "b.md"), "--csv-output", str(tmp_path / "b.csv")]) == 0
    assert saved["benchmark"] == str(output)

    records = [GoldRecord(query="q", answer="a", evidence_chunk_ids=["a"])]
    monkeypatch.setattr(cli, "propose_gold_records", lambda index, sample_size: records)
    gold_out = tmp_path / "gold.jsonl"
    assert cli.main(["dataset", "propose", "--index", str(index_path), "--output", str(gold_out)]) == 0
    assert "Proposed 1" in capsys.readouterr().out

    monkeypatch.setattr(cli, "load_gold_records", lambda path: records)
    monkeypatch.setattr(cli, "validate_gold_records", lambda records, index: [])
    assert cli.main(["dataset", "validate", "--index", str(index_path), "--gold", str(gold_out)]) == 0
    monkeypatch.setattr(cli, "validate_gold_records", lambda records, index: ["missing"])
    assert cli.main(["dataset", "validate", "--index", str(index_path), "--gold", str(gold_out)]) == 1

    class FakeCompression:
        origin_tokens = 10
        compressed_tokens = 4
        ratio = 2.5
        saving_rate = 0.6

    monkeypatch.setattr(cli, "export_selection", lambda *args, **kwargs: FakeCompression())
    assert cli.main(["export-context", "--selection", str(output), "--output", str(tmp_path / "ctx.md"), "--compressor", "llmlingua"]) == 0
    assert "Compression:" in capsys.readouterr().out

    monkeypatch.setattr(cli, "answer_from_selection", lambda **kwargs: {"answer": "ok"})
    monkeypatch.setattr(cli, "save_answer", lambda payload, path: saved.setdefault("answer", payload))
    assert cli.main(["answer", "--query", "q", "--selection", str(output), "--output", str(tmp_path / "answer.json")]) == 0
    assert saved["answer"]["answer"] == "ok"


def test_cli_pack_and_budget_helpers(monkeypatch, capsys):
    tmp_path = _workspace_tmp()
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")

    with pytest.raises(ValueError, match="budgets"):
        cli._parse_budgets(",", 10)
    assert cli._parse_budgets("10, 20", 1) == [10, 20]

    with pytest.raises(SystemExit, match="Source does not exist"):
        cli.main(["pack", str(tmp_path / "missing.txt"), "--query", "q"])

    out = tmp_path / "packed.md"
    out.write_text("exists", encoding="utf-8")
    with pytest.raises(SystemExit, match="Output already exists"):
        cli.main(["pack", str(source), "--query", "q", "--out", str(out)])

    monkeypatch.setattr(cli, "_make_cli_embedder", lambda args, model_name: _Embedder())
    monkeypatch.setattr(cli, "pack_source", lambda **kwargs: _FakePackResult("markdown"))
    monkeypatch.setattr(cli, "format_pack_summary", lambda result: ["summary line"])
    assert cli.main(["pack", str(source), "--query", "q", "--out", str(out), "--overwrite", "--quiet"]) == 0
    assert "summary line" in capsys.readouterr().out

    monkeypatch.setattr(cli, "pack_source", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad pack")))
    with pytest.raises(SystemExit, match="bad pack"):
        cli.main(["pack", str(source), "--query", "q", "--out", str(out), "--overwrite", "--quiet"])


def test_token_counter_fallback_without_tiktoken(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("offline")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    counter = TokenCounter()

    assert counter._encoding is None
    assert counter.count("") == 0
    assert counter.count("alpha, beta!") == 4


def test_scoring_edge_helpers_and_redundancy_branches():
    base_meta = {
        "content_type": "code",
        "symbol_name": "train_model",
        "symbol_kind": "function",
        "language": "python",
        "start_line": 1,
        "end_line": 5,
        "section_hint": "Training API",
    }
    chunks = [
        _chunk(
            "code-a",
            text="import optimizer\n\ndef train_model(data):\n    return optimizer.fit(data)",
            weight=40,
            metadata=base_meta,
        ),
        _chunk(
            "code-b",
            paragraph=1,
            text="import optimizer\n\ndef train_model(batch):\n    return optimizer.fit(batch)",
            weight=45,
            metadata={**base_meta, "start_line": 3, "end_line": 7},
        ),
        _chunk(
            "section",
            paragraph=2,
            text="Training API prose mentions optimizer behavior.",
            weight=20,
            metadata={"section_hint": "Training API"},
        ),
    ]
    embeddings = [[1.0, 0.0], [0.96, 0.04], [0.0, 1.0]]

    scored = prod_scoring.score_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        query_text="optimizer train_model Training API",
        redundancy_penalty=0.5,
        redundancy_candidate_pool=None,
    )

    assert scored[1].redundancy_penalty > 0
    assert "novelty" in scored[1].score_components
    assert prod_scoring._bm25_scores("", chunks) == [0.0, 0.0, 0.0]
    assert prod_scoring._query_coverage("", chunks) == [0.0, 0.0, 0.0]
    assert prod_scoring._normalized_signal([]) == []
    assert prod_scoring._normalized_signal([0.0, -1.0]) == [0.0, 0.0]
    assert prod_scoring._minmax([]) == []
    assert prod_scoring._lexical_overlap("", "alpha") == 0.0
    assert prod_scoring._has_dependency_hint("import optimizer", set()) is False
    assert prod_scoring._has_dependency_hint("print('nothing')", {"optimizer"}) is False
    assert prod_scoring._has_dependency_hint("from optimizer import Adam", {"optimizer"}) is True

    other_source = _chunk("other", metadata=base_meta, source="other.py")
    language_match = _chunk(
        "language",
        metadata={**base_meta, "symbol_name": "other_symbol", "section_hint": "", "start_line": 20, "end_line": 25},
    )
    no_lines = _chunk(
        "no-lines",
        metadata={**base_meta, "symbol_name": "third_symbol", "section_hint": "", "start_line": "1"},
    )

    assert prod_scoring._structural_overlap(chunks[0], other_source) == 0.0
    assert prod_scoring._structural_overlap(chunks[0], chunks[1]) == 1.0
    assert prod_scoring._structural_overlap(chunks[0], chunks[2]) == 0.75
    assert prod_scoring._structural_overlap(chunks[0], language_match) == 0.25
    assert prod_scoring._line_overlap(chunks[0].metadata, no_lines.metadata) == 0.0
    assert prod_scoring._line_overlap(chunks[0].metadata, chunks[1].metadata) > 0.0


def test_experimental_scoring_helper_edges():
    code_meta = {
        "content_type": "code",
        "symbol_name": "build_index",
        "symbol_kind": "function",
        "language": "python",
        "start_line": 10,
        "end_line": 20,
        "section_hint": "Index Builder",
    }
    chunks = [
        _chunk(
            "code-a",
            text="import tokenpack\n\ndef build_index(source):\n    return tokenpack.ingest(source)",
            weight=80,
            metadata=code_meta,
        ),
        _chunk(
            "code-b",
            paragraph=1,
            text="import tokenpack\n\ndef build_index(path):\n    return tokenpack.ingest(path)",
            weight=90,
            metadata={**code_meta, "start_line": 18, "end_line": 25},
        ),
        _chunk("blank", paragraph=2, text="!!!", weight=5),
    ]
    embeddings = [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]]

    with pytest.raises(ValueError, match="Unknown scoring profile"):
        exp_scoring.score_experimental_chunks([1.0, 0.0], chunks, embeddings, scoring="mystery")

    scored = exp_scoring.score_experimental_chunks(
        [1.0, 0.0],
        chunks,
        embeddings,
        scoring="query-support",
        query_text="build_index tokenpack source",
        redundancy_penalty=0.5,
        redundancy_candidate_pool=None,
    )
    assert scored[1].redundancy_penalty > 0

    assert exp_scoring._query_coverage("", chunks) == [0.0, 0.0, 0.0]
    assert exp_scoring._support_likelihood("a an to", chunks) == [0.0, 0.0, 0.0]
    assert exp_scoring._phrase_overlap("tiny", chunks) == [0.0, 0.0, 0.0]
    assert exp_scoring._term_proximity("tokenpack", chunks) == [0.0, 0.0, 0.0]
    assert exp_scoring._split_decision_query("Which one?\n\nA. Alpha choice\n\n2) Beta choice") == (
        "Which one?",
        ["Alpha choice", "Beta choice"],
    )
    assert exp_scoring._candidate_decision_signals([], chunks) == ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert exp_scoring._structural_prior("build_index tokenpack Index Builder", chunks)[0] > 0.7
    assert exp_scoring._has_dependency_hint("import tokenpack", set()) is False
    assert exp_scoring._has_dependency_hint("print('nothing')", {"tokenpack"}) is False
    assert exp_scoring._has_dependency_hint("use tokenpack", {"tokenpack"}) is True
    assert exp_scoring._position_bias([chunks[0]]) == [1.0]
    assert exp_scoring._length_utility([]) == []
    assert exp_scoring._term_specificity([]) == []
    assert exp_scoring._term_specificity([chunks[2]]) == [0.0]
    assert exp_scoring._normalized_signal([]) == []
    assert exp_scoring._normalized_signal([0.0, -0.2]) == [0.0, 0.0]
    assert exp_scoring._minmax([]) == []
    assert exp_scoring._lexical_overlap("", "alpha") == 0.0

    other_source = _chunk("other", metadata=code_meta, source="other.py")
    same_section = _chunk("section", metadata={"section_hint": "Index Builder"})
    same_language = _chunk(
        "language",
        metadata={**code_meta, "symbol_name": "other_symbol", "section_hint": "", "start_line": 40, "end_line": 50},
    )
    no_lines = _chunk(
        "no-lines",
        metadata={**code_meta, "symbol_name": "third_symbol", "section_hint": "", "start_line": "10"},
    )

    assert exp_scoring._structural_overlap(chunks[0], other_source) == 0.0
    assert exp_scoring._structural_overlap(chunks[0], chunks[1]) == 1.0
    assert exp_scoring._structural_overlap(chunks[0], same_section) == 0.75
    assert exp_scoring._structural_overlap(chunks[0], same_language) == 0.25
    assert exp_scoring._line_overlap(chunks[0].metadata, no_lines.metadata) == 0.0
    assert exp_scoring._line_overlap(chunks[0].metadata, chunks[1].metadata) > 0.0


def test_pack_source_compression_and_budget_edges(monkeypatch):
    tmp_path = _workspace_tmp()
    source = tmp_path / "source.txt"
    source.write_text("alpha", encoding="utf-8")
    progress_messages: list[str] = []
    index = ChunkIndex(
        chunks=[_chunk("a", text="alpha evidence", weight=10)],
        embeddings=[[1.0, 0.0]],
        model_name="toy",
    )
    compression = CompressionResult(
        compressed_prompt="compressed alpha",
        origin_tokens=10,
        compressed_tokens=4,
        ratio=2.5,
        metadata={"backend": "fake"},
    )

    monkeypatch.setattr(packing_module, "ingest_path", lambda *args, **kwargs: index)
    monkeypatch.setattr(packing_module, "score_chunks", lambda *args, **kwargs: [_scored("a", 1.0, 10)])
    monkeypatch.setattr(packing_module, "select_chunks", lambda *args, **kwargs: _FakeSelectionResult())
    monkeypatch.setattr(
        packing_module,
        "render_compressed_context",
        lambda chunks, config, include_headers: ("compressed alpha\n", compression),
    )

    result = packing_module.pack_source(
        source=source,
        query="alpha",
        embedder=_Embedder(),
        out=tmp_path / "packed.md",
        overwrite=True,
        budget=100,
        reserve_output=10,
        compress="llmlingua",
        output_detail="debug",
        longllmlingua=True,
        llmlingua2=True,
        allow_download=True,
        compression_context_filter=True,
        compression_sentence_filter=True,
        no_compression_token_filter=True,
        progress=progress_messages.append,
        run_root=tmp_path / "runs",
    )

    assert result.compression_result is compression
    assert result.to_metadata()["compression_tokens"] == 4
    assert "Compression:" in "\n".join(packing_module.format_pack_summary(result))
    assert "compressed alpha" in result.output_path.read_text(encoding="utf-8")
    assert any("Compressing selected context" in message for message in progress_messages)

    assert packing_module._render_pack_markdown_header(
        source=source,
        output_path=result.output_path,
        query="q",
        budget=result.budget,
        selected_chunks=1,
        selected_tokens=10,
        index_path=result.index_path,
        selection_path=result.selection_path,
        compression="none",
        compression_result=None,
        output_detail="none",
    ) == ""

    manual_budget = packing_module._resolve_pack_budget(
        source_tokens=100,
        budget=50,
        budget_ratio=0.5,
        min_budget=10,
        max_budget=100,
        reserve_output=5,
    )
    manual_result = packing_module.PackResult(
        source=source,
        output_path=result.output_path,
        index_path=result.index_path,
        selection_path=result.selection_path,
        markdown="",
        budget=manual_budget,
        selected_chunks=1,
        selected_tokens=10,
        scoring="evidence-hybrid",
        selector="budget-top-k",
        compression="none",
    )
    assert any(line.startswith("Manual budget") for line in packing_module.format_pack_summary(manual_result))

    with pytest.raises(ValueError, match="Source does not exist"):
        packing_module.pack_source(source=tmp_path / "missing.txt", query="q", embedder=_Embedder())
    with pytest.raises(FileExistsError, match="Output already exists"):
        packing_module.pack_source(source=source, query="q", embedder=_Embedder(), out=result.output_path)
    with pytest.raises(ValueError, match="Unknown compressor"):
        packing_module.pack_source(source=source, query="q", embedder=_Embedder(), out=tmp_path / "x.md", compress="zip")
    with pytest.raises(ValueError, match="Unknown output detail"):
        packing_module.pack_source(source=source, query="q", embedder=_Embedder(), out=tmp_path / "x.md", output_detail="loud")

    invalid_budget_args = [
        {"source_tokens": -1, "budget": None, "budget_ratio": 0.5, "min_budget": 10, "max_budget": 100, "reserve_output": None},
        {"source_tokens": 10, "budget": None, "budget_ratio": 0.0, "min_budget": 10, "max_budget": 100, "reserve_output": None},
        {"source_tokens": 10, "budget": None, "budget_ratio": 0.5, "min_budget": 0, "max_budget": 100, "reserve_output": None},
        {"source_tokens": 10, "budget": None, "budget_ratio": 0.5, "min_budget": 100, "max_budget": 10, "reserve_output": None},
        {"source_tokens": 10, "budget": 0, "budget_ratio": 0.5, "min_budget": 10, "max_budget": 100, "reserve_output": None},
        {"source_tokens": 10, "budget": 50, "budget_ratio": 0.5, "min_budget": 10, "max_budget": 100, "reserve_output": -1},
    ]
    for kwargs in invalid_budget_args:
        with pytest.raises(ValueError):
            packing_module._resolve_pack_budget(**kwargs)

    assert packing_module._resolve_pack_budget(
        source_tokens=1,
        budget=None,
        budget_ratio=0.5,
        min_budget=10,
        max_budget=100,
        reserve_output=None,
    ).cap_reason == "min-budget"
    assert packing_module._resolve_pack_budget(
        source_tokens=10_000,
        budget=None,
        budget_ratio=0.5,
        min_budget=10,
        max_budget=100,
        reserve_output=None,
    ).cap_reason == "max-budget"


def test_chunking_edge_branches_for_large_blocks_and_boundaries():
    with pytest.raises(ValueError, match="min_tokens"):
        StructureAwareChunker(target_tokens=10, min_tokens=11, max_tokens=12)

    word_counter = _WordCounter()
    chunker = StructureAwareChunker(target_tokens=4, min_tokens=1, max_tokens=4, token_counter=word_counter)
    assert chunker._split_block_units(_block("empty", "   ")) == []
    assert chunker._split_block_units(_block("code", "one\n\ntwo", metadata={"content_type": "code"})) == ["one", "two"]
    assert chunker._split_oversized_unit("") == [""]

    prose = _block("prose", "one two three. four five.", source="prose.txt")
    prose_splits = chunker._split_large_block(0, prose)
    assert [item.text for item in prose_splits] == ["one two three.", "four five."]

    code = _block("code", "alpha\nbeta\ngamma", metadata={"content_type": "code"}, source="code.py")
    code_splits = StructureAwareChunker(target_tokens=2, min_tokens=1, max_tokens=3, token_counter=word_counter)._split_large_block(0, code)
    assert "\n" in code_splits[0].text

    oversized = _block("oversized", "one two three four five six", source="long.txt")
    oversized_splits = chunker._split_large_block(0, oversized)
    assert len(oversized_splits) == 2

    mixed = chunker._make_chunk(
        [
            (0, _block("doc", "alpha", metadata={"content_type": "document", "start_line": 7}), 1),
            (1, _block("code", "beta", metadata={"content_type": "code", "end_line": 11}), 1),
        ]
    )
    assert mixed.metadata["content_type"] == "mixed"
    assert mixed.metadata["start_line"] == 7
    assert mixed.metadata["end_line"] == 11

    symbol_long = _block(
        "symbol",
        "def alpha():\n    return one two three four five six",
        metadata={"content_type": "code", "symbol_name": "alpha"},
        source="code.py",
    )
    normal_long = _block("normal", "one two three four five six seven", source="doc.txt")
    chunks = chunker.chunk([symbol_long, normal_long])
    assert len(chunks) >= 3

    left = _block("left", "alpha", metadata={"content_type": "document", "section_hint": "Intro"}, source="a.txt")
    right_source = _block("right", "beta", metadata={"content_type": "document", "section_hint": "Intro"}, source="b.txt")
    right_type = _block("right", "beta", metadata={"content_type": "code", "section_hint": "Intro"}, source="a.txt")
    right_section = _block("right", "beta", metadata={"content_type": "document", "section_hint": "Body"}, source="a.txt")
    assert chunker._structural_boundary(left, right_source) is True
    assert chunker._structural_boundary(left, right_type) is True
    assert chunker._structural_boundary(left, right_section) is True

    semantic = StructureAwareChunker(
        target_tokens=4,
        min_tokens=1,
        max_tokens=4,
        token_counter=word_counter,
        block_embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    assert semantic._semantic_boundary(1, right_source, [(0, left, 1)]) is False
    assert semantic._semantic_boundary(1, right_type, [(0, right_type, 1)]) is False
    assert semantic._semantic_boundary(1, right_type, [(0, left, 1)]) is False

    sem_chunker = SemanticThresholdChunker(
        [[1, 0], [1, 0], [0, 1], [1, 0]],
        target_tokens=3,
        min_tokens=1,
        max_tokens=4,
        token_counter=word_counter,
    )
    sem_blocks = [
        _block("large", "one two three four five six", source="a.txt"),
        _block("a", "one", source="a.txt"),
        _block("b", "two", source="b.txt"),
        _block("c", "three four five", source="b.txt"),
    ]
    sem_chunks = sem_chunker.chunk(sem_blocks)
    assert len(sem_chunks) >= 4
    assert sem_chunker._topic_shift(1, [(0, sem_blocks[0], 1), (1, sem_blocks[1], 1)]) is False


def test_loader_remaining_dispatch_and_optional_dependency_edges(monkeypatch):
    tmp_path = _workspace_tmp()
    pptx = tmp_path / "deck.pptx"
    xlsx = tmp_path / "sheet.xlsx"
    pptx.write_text("fake", encoding="utf-8")
    xlsx.write_text("fake", encoding="utf-8")

    monkeypatch.setattr(loaders_module, "load_office_blocks", lambda path, document_index=0: [_block("office", "loaded", source=str(path))])
    assert load_blocks(pptx)[0].text == "loaded"

    monkeypatch.setattr(loaders_module, "load_pptx_blocks", lambda path, document_index=0: [_block("pptx", "slides", source=str(path))])
    monkeypatch.setattr(loaders_module, "load_xlsx_blocks", lambda path, document_index=0: [_block("xlsx", "rows", source=str(path))])
    assert load_office_blocks(pptx)[0].text == "slides"
    assert load_office_blocks(xlsx)[0].text == "rows"

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"docx", "pptx", "openpyxl"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="DOCX support"):
        load_docx_blocks(tmp_path / "missing.docx")
    with pytest.raises(RuntimeError, match="PPTX support"):
        load_pptx_blocks(tmp_path / "missing.pptx")
    with pytest.raises(RuntimeError, match="XLSX support"):
        load_xlsx_blocks(tmp_path / "missing.xlsx")

    parser = _HTMLTextExtractor()
    parser.feed("<script><b>ignored</b></script><p>visible</p>")
    assert "visible" in parser.text()
    assert "ignored" not in parser.text()

    data = tmp_path / "data.json"
    data.write_text(json.dumps(["alpha", "", {"nested": ""}]), encoding="utf-8")
    blocks = load_json_blocks(data)
    assert blocks[0].metadata["json_path"] == "[0]"

    csv_path = tmp_path / "empty-row.csv"
    csv_path.write_text("name,score\n,\nalpha,1\n", encoding="utf-8")
    assert len(load_csv_blocks(csv_path)) == 1

    js = tmp_path / "module.js"
    js.write_text("// module prefix\nconst setup = true;\nfunction run() {\n  return setup;\n}\n", encoding="utf-8")
    code_blocks = load_code_blocks(js)
    assert code_blocks[0].metadata["start_line"] == 1
    assert "symbol_name" not in code_blocks[0].metadata
    assert code_blocks[1].metadata["symbol_name"] == "run"


def test_benchmark_profiles_models_dataset_and_small_helpers(monkeypatch):
    index = ChunkIndex(
        chunks=[
            _chunk("a", text="alpha evidence first sentence.", weight=4),
            _chunk("b", paragraph=1, text="beta evidence second sentence.", weight=4),
            _chunk("c", paragraph=2, text="gamma evidence third sentence.", weight=4),
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        model_name="toy",
    )
    records = [GoldRecord(query="alpha", answer="alpha", evidence_chunk_ids=["a"], source_path="doc.txt")]

    monkeypatch.setattr(benchmark_module, "propose_gold_records", lambda index, sample_size=12: records)
    monkeypatch.setattr(
        benchmark_module,
        "run_gold_benchmark",
        lambda **kwargs: {"budgets": [{"budget": kwargs["budgets"][0], "queries": [{"query": "alpha"}]}]},
    )

    assert benchmark_module.synthetic_queries(index, sample_size=1)[0]["evidence_chunk_id"] == "a"
    smoke = benchmark_module.run_benchmark(index, _Embedder(), budget=10, reserve_output=2)
    assert smoke["mode"] == "smoke"
    out = _workspace_tmp() / "reports" / "bench.json"
    benchmark_module.save_benchmark(smoke, out)
    assert json.loads(out.read_text(encoding="utf-8"))["mode"] == "smoke"

    manual = resolve_chunk_size_config("manual", 10, 2, 12)
    assert manual.target_tokens == 10
    with pytest.raises(ValueError, match="Unknown chunk size preset"):
        resolve_chunk_size_config("huge", 10, 2, 12)

    payload = GoldRecord.from_dict({"query": "q", "evidence_chunk_id": "a"})
    assert payload.evidence_chunk_ids == ["a"]
    gold_path = _workspace_tmp() / "gold.jsonl"
    save_gold_records([payload], gold_path)
    gold_path.write_text("\n" + gold_path.read_text(encoding="utf-8"), encoding="utf-8")
    assert load_gold_records(gold_path)[0].query == "q"
    assert validate_gold_records([payload], index) == []
    assert _spaced_indices(0, 3) == []
    assert _spaced_indices(10, 3) == [0, 3, 6]

    block = TextBlock(text="alpha", source_path="doc.txt", document_index=1, bbox=(1.0, 2.0, 3.0, 4.0))
    block_payload = block.to_dict()
    assert block_payload["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert TextBlock.from_dict(block_payload).bbox == (1.0, 2.0, 3.0, 4.0)
    selection = SelectionResult("s", 10, 4, 1.5, [_scored("a", 1.0, 4)], 0.01)
    assert selection.to_dict()["selected"][0]["chunk"]["id"] == "a"


def test_embeddings_compression_generation_mcp_and_pipeline_edges(monkeypatch):
    class LocalSentenceTransformer:
        def __init__(self, model_name, local_files_only=True):
            self.local_files_only = local_files_only

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            return [[1, 2] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=LocalSentenceTransformer),
    )
    embedder = SentenceTransformerEmbedder("model", local_files_only=None)
    assert embedder.embed(["alpha"]) == [[1.0, 2.0]]
    assert normalize([0.0, 0.0]) == [0.0, 0.0]
    assert normalize([3.0, 4.0]) == [0.6, 0.8]
    assert cosine([], [1.0]) == 0.0

    assert CompressionResult("", 0, 5, 1.0, {}).saving_rate == 0.0
    assert _ratio_from_payload("badx", 10, 2) == 5.0
    assert _llmlingua_kwargs(CompressionConfig(compressor="llmlingua", longllmlingua=True, question="q"))["rank_method"] == "longllmlingua"

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"llmlingua", "huggingface_hub"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="LLMLingua is not installed"):
        _make_llmlingua_backend(CompressionConfig(compressor="llmlingua"))
    with pytest.raises(RuntimeError, match="huggingface_hub"):
        _resolve_local_model_path("missing/model")

    selection_path = _workspace_tmp() / "selection.json"
    selection_path.write_text(json.dumps({"selected": [{"chunk": _chunk("a").to_dict()}]}), encoding="utf-8")
    monkeypatch.setattr(generation_module, "_cerebras_answer", lambda prompt, model: "cerebras ok")
    monkeypatch.setattr(generation_module, "_groq_answer", lambda prompt, model: "groq ok")
    assert answer_from_selection("q", selection_path, provider="cerebras")["answer"] == "cerebras ok"
    assert answer_from_selection("q", selection_path, provider="groq")["answer"] == "groq ok"

    workspace = _workspace_tmp()
    config = mcp_server.McpServerConfig(workspace=workspace)
    packed = workspace / "packed.md"
    packed.write_text("abcdefghij", encoding="utf-8")
    assert mcp_server.read_packed_context_tool(path="packed.md", config=config, offset=2, max_chars=3)["text"] == "cde"
    assert mcp_server._run_root(mcp_server.McpServerConfig(workspace=workspace, allow_any_path=True)) == Path(".tokenpack/runs")
    mcp_server._make_mcp_embedder.cache_clear()
    monkeypatch.setattr(mcp_server, "make_embedder", lambda **kwargs: _Embedder())
    assert mcp_server._make_mcp_embedder(config).model_name == "toy"

    class FakeMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self):
            def register(func):
                self.tools[func.__name__] = func
                return func

            return register

    monkeypatch.setattr(builtins, "__import__", lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError(name)) if name == "mcp.server.fastmcp" else original_import(name, *args, **kwargs))
    with pytest.raises(RuntimeError, match="MCP support"):
        mcp_server.build_server(config)

    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeMCP
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)
    monkeypatch.setattr(builtins, "__import__", original_import)
    server = mcp_server.build_server(config)
    monkeypatch.setattr(mcp_server, "pack_context_tool", lambda **kwargs: {"markdown": "ok"})
    assert server.tools["pack_context"]("source.txt", "q")["markdown"] == "ok"
    assert server.tools["read_packed_context"]("packed.md", offset=0, max_chars=2)["text"] == "ab"

    with pytest.raises(ValueError, match="Unknown chunker"):
        pipeline_module.ingest_path("source.txt", "index.json", _Embedder(), chunker_name="unknown")


def test_cli_reporting_reranking_selectors_and_score_leftovers(monkeypatch, capsys):
    tmp_path = _workspace_tmp()
    index_path = tmp_path / "index.json"
    output = tmp_path / "selection.json"

    monkeypatch.setattr(cli, "_make_cli_embedder", lambda args, model_name: _Embedder())
    monkeypatch.setattr(cli, "load_index", lambda path: ChunkIndex(chunks=[_chunk("a")], embeddings=[[1.0]], model_name="toy"))
    monkeypatch.setattr(cli, "score_chunks", lambda *args, **kwargs: [_scored("a", 1.0, 1)])
    monkeypatch.setattr(cli, "select_chunks", lambda *args, **kwargs: _FakeSelectionResult())
    assert cli.main(["select", "--query", "alpha", "--index", str(index_path), "--output", str(output)]) == 0
    assert "Selection saved" in capsys.readouterr().out

    records = [GoldRecord(query="q", answer="a", evidence_chunk_ids=["a"])]
    monkeypatch.setattr(cli, "load_gold_records", lambda path: records)
    monkeypatch.setattr(cli, "validate_gold_records", lambda records, index: ["bad"])
    with pytest.raises(SystemExit, match="Gold validation failed"):
        cli.main(["benchmark", "--index", str(index_path), "--gold", str(tmp_path / "gold.jsonl")])
    monkeypatch.setattr(cli, "validate_gold_records", lambda records, index: [])
    monkeypatch.setattr(
        cli,
        "run_gold_benchmark",
        lambda *args, **kwargs: {"mode": "gold", "budgets": [{"summary": {"budget-top-k": {"evidence_recall_at_budget": 1.0}}}]},
    )
    monkeypatch.setattr(cli, "save_benchmark", lambda payload, path: output.write_text(json.dumps(payload), encoding="utf-8"))
    assert cli.main(["benchmark", "--index", str(index_path), "--gold", str(tmp_path / "gold.jsonl"), "--output", str(output)]) == 0
    assert cli._parse_budgets(None, 12) == [12]
    assert cli._make_cli_embedder(types.SimpleNamespace(offline_models=True), "toy").model_name == "toy"
    cli._pack_progress("working")
    assert "[tokenpack] working" in capsys.readouterr().err

    assert reporting_module._fmt(object()).startswith("<object object")
    assert reranking_module._minmax([]) == []

    class EmptyReranker:
        def predict(self, pairs, show_progress_bar=False):
            return []

    cross = CrossEncoderReranker.__new__(CrossEncoderReranker)
    cross._model = EmptyReranker()
    assert cross.score("q", []) == []

    scored = [
        ScoredChunk(_chunk("first", paragraph=0, weight=4), value=1.0, raw_similarity=0.1, weight=4, embedding=[1.0, 0.0]),
        ScoredChunk(_chunk("huge", paragraph=1, weight=99), value=0.9, raw_similarity=0.9, weight=99, embedding=[1.0, 0.0]),
        ScoredChunk(_chunk("second", paragraph=2, weight=4), value=0.8, raw_similarity=0.8, weight=4, embedding=[0.0, 1.0]),
    ]
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        selectors_module.select_chunks(scored, strategy="unknown", budget=5)
    assert selectors_module.select_chunks(scored, strategy="document-prefix", budget=4).selected[0].chunk.id == "first"
    assert selectors_module.select_chunks(scored, strategy="mmr", budget=4, embeddings=None).selected[0].chunk.id == "first"
    assert selectors_module.select_chunks(scored[1:2], strategy="mmr", budget=4).selected == []
    assert selectors_module._knapsack(scored, budget=0) == []
    many = [ScoredChunk(_chunk(f"m{i}", weight=1), value=1.0, raw_similarity=1.0, weight=1) for i in range(1001)]
    assert selectors_module._knapsack(many, budget=9991, token_granularity=1)
    assert selectors_module._coverage_greedy(scored, budget=0, coverage_query="alpha") == []

    left = _chunk(
        "left",
        metadata={"content_type": "code", "language": "python", "symbol_name": "left", "start_line": 1, "end_line": 5},
    )
    right = _chunk(
        "right",
        metadata={"content_type": "code", "language": "ruby", "symbol_name": "right", "start_line": 3, "end_line": 7},
    )
    assert prod_scoring._structural_overlap(left, right) > 0.0
    assert exp_scoring._structural_overlap(left, right) > 0.0
    exp_scoring.score_experimental_chunks(
        [1.0, 0.0],
        [_chunk("a"), _chunk("b"), _chunk("c")],
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        scoring="cosine",
        redundancy_penalty=0.1,
        redundancy_candidate_pool=2,
    )


def _workspace_tmp() -> Path:
    root = Path(".test-tmp")
    root.mkdir(exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    return path


def _chunk(
    chunk_id: str,
    *,
    paragraph: int = 0,
    text: str | None = None,
    weight: int = 1,
    metadata: dict | None = None,
    source: str = "doc.txt",
    document_index: int = 0,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text or f"Text {chunk_id}",
        source_path=source,
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


def _block(
    label: str,
    text: str,
    *,
    metadata: dict | None = None,
    paragraph: int = 0,
    source: str = "doc.txt",
) -> TextBlock:
    return TextBlock(
        text=text,
        source_path=source,
        document_index=0,
        paragraph_index=paragraph,
        char_start=0,
        char_end=len(text),
        metadata=metadata or {},
    )


def _scored(chunk_id: str, value: float, weight: int) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, weight=weight), value=value, raw_similarity=value, weight=weight)


class _WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class _Embedder:
    model_name = "toy"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class _WeirdPromptCompressor:
    def compress_prompt(self, context, **kwargs):
        return {"compressed_prompt": "alpha", "origin_tokens": "bad", "compressed_tokens": "bad", "ratio": "bad"}


class _FakePackResult:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown

    def to_metadata(self):
        return {"output_path": "packed.md"}


class _FakeSelectionResult:
    selected = [_scored("a", 1.0, 1)]
    used_tokens = 1
    total_value = 1.0

    def to_dict(self):
        return {"selected": [], "used_tokens": 1, "total_value": 1.0}
