from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenpack.benchmark import run_benchmark, run_gold_benchmark, save_benchmark
from tokenpack.chunk_profiles import resolve_chunk_size_config
from tokenpack.compression import CompressionConfig
from tokenpack.dataset import load_gold_records, propose_gold_records, save_gold_records, validate_gold_records
from tokenpack.doctor import collect_diagnostics
from tokenpack.embeddings import DEFAULT_EMBEDDING_MODEL, make_embedder
from tokenpack.export import export_selection
from tokenpack.generation import answer_from_selection, save_answer
from tokenpack.index import load_index
from tokenpack.packing import (
    AUTO_BUDGET_RATIO,
    AUTO_MAX_BUDGET,
    AUTO_MIN_BUDGET,
    DEFAULT_COMPRESSION_MODEL,
    _infer_pack_output_path,
    _resolve_pack_budget,
    format_pack_summary,
    pack_source,
)
from tokenpack.pipeline import ingest_path
from tokenpack.reporting import save_csv_report, save_markdown_report
from tokenpack.scoring import DEFAULT_SCORING_PROFILE, score_chunks
from tokenpack.selectors import select_chunks

DEFAULT_INDEX = ".tokenpack/index.json"
DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL
QUALITY_MODEL = "BAAI/bge-m3"

SELECTION_STRATEGIES = [
    "document-prefix",
    "full-document",
    "top-k",
    "production-rag",
    "budget-top-k",
    "greedy-value",
    "greedy-density",
    "mmr",
    "knapsack",
    "knapsack-redundancy",
    "knapsack-coverage",
    "knapsack-augment",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenpack-rag",
        description="TokenPack-RAG CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--offline-models",
        action="store_true",
        help="Use only locally cached sentence-transformers files; avoids Hugging Face network checks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack = subparsers.add_parser(
        "pack",
        help="One-command ingest, select, and export packed context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pack.add_argument("source", help="File or folder to pack.")
    pack.add_argument("--query", required=True, help="Question or task used to select relevant context.")
    pack.add_argument("--out", "--output", dest="out", help="Output Markdown path. Defaults to <source>-tp.md.")
    pack.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    pack.add_argument("--budget", type=int, help="Manual total context budget. If omitted, TokenPack estimates one.")
    pack.add_argument("--budget-ratio", type=float, default=AUTO_BUDGET_RATIO, help="Source-token ratio used by auto budget.")
    pack.add_argument("--min-budget", type=int, default=AUTO_MIN_BUDGET, help="Minimum auto budget.")
    pack.add_argument("--max-budget", type=int, default=AUTO_MAX_BUDGET, help="Maximum auto budget.")
    pack.add_argument("--reserve-output", type=int, help="Tokens reserved for the downstream answer.")
    pack.add_argument("--index-out", help="Optional path for the generated index JSON.")
    pack.add_argument("--selection-out", help="Optional path for the generated selection JSON.")
    pack.add_argument("--candidate-pool", type=int, default=250, help="Number of top-scored chunks considered by selection.")
    pack.add_argument(
        "--chunk-size-preset",
        choices=["manual", "default", "low-budget"],
        default="low-budget",
        help="Named chunk-size profile.",
    )
    pack.add_argument("--target-tokens", type=int, default=650, help="Target tokens per chunk when using manual sizing.")
    pack.add_argument("--min-tokens", type=int, default=120, help="Minimum tokens per chunk when using manual sizing.")
    pack.add_argument("--max-tokens", type=int, default=900, help="Maximum tokens per chunk when using manual sizing.")
    pack.add_argument("--source-type", choices=["auto", "document", "code", "mixed"], default="auto", help="Loader hint.")
    pack.add_argument(
        "--output-detail",
        choices=["clean", "debug", "none"],
        default="clean",
        help=(
            "How much metadata to include in the packed Markdown. "
            "clean is LLM-ready; debug includes chunk ids, token counts, and artifact paths; none writes only context."
        ),
    )
    pack.add_argument("--quiet", action="store_true", help="Suppress pack progress messages.")
    pack.add_argument("--compress", choices=["none", "llmlingua"], default="none", help="Optional second-stage compression.")
    pack.add_argument("--compression-model", default=DEFAULT_COMPRESSION_MODEL, help="LLMLingua model name or local path.")
    pack.add_argument("--compression-rate", type=float, default=0.85, help="LLMLingua compression rate.")
    pack.add_argument("--compression-target-tokens", type=int, default=-1, help="LLMLingua target token count; -1 lets the backend decide.")
    pack.add_argument("--compression-instruction", default="", help="Optional compression instruction.")
    pack.add_argument("--longllmlingua", action="store_true", help="Use LongLLMLingua query-conditioned compression mode.")
    pack.add_argument("--llmlingua2", action="store_true", help="Enable LLMLingua-2 backend mode.")
    pack.add_argument("--compression-device-map", default="cpu", help="Device map passed to LLMLingua.")
    pack.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face network access for optional compression models.",
    )
    pack.add_argument("--compression-context-filter", action="store_true", help="Enable LLMLingua context-level filtering.")
    pack.add_argument("--compression-sentence-filter", action="store_true", help="Enable LLMLingua sentence-level filtering.")
    pack.add_argument("--no-compression-token-filter", action="store_true", help="Disable LLMLingua token-level filtering.")

    ingest = subparsers.add_parser("ingest", help="Load documents, chunk them, and build an embedding index.")
    ingest.add_argument("source")
    ingest.add_argument("--index", default=DEFAULT_INDEX)
    ingest.add_argument("--target-tokens", type=int, default=650)
    ingest.add_argument("--min-tokens", type=int, default=120)
    ingest.add_argument("--max-tokens", type=int, default=900)
    ingest.add_argument(
        "--chunk-size-preset",
        choices=["manual", "default", "low-budget"],
        default="low-budget",
        help="Override chunk token limits with a named preset; low-budget uses smaller evidence-sized chunks.",
    )
    ingest.add_argument("--source-type", choices=["auto", "document", "code", "mixed"], default="auto")

    select = subparsers.add_parser("select", help="Select chunks for a query under a token budget.")
    select.add_argument("--query", required=True)
    select.add_argument("--index", default=DEFAULT_INDEX)
    select.add_argument("--budget", type=int, default=50_000)
    select.add_argument("--reserve-output", type=int, default=4_000)
    select.add_argument("--strategy", choices=SELECTION_STRATEGIES, default="budget-top-k")
    select.add_argument("--candidate-pool", type=int, default=250)
    select.add_argument("--relevance-threshold", type=float, default=0.0)
    select.add_argument("--redundancy-penalty", type=float, default=0.35)
    select.add_argument("--output", default=".tokenpack/selection.json")
    select.add_argument("--json", action="store_true", help="Print full JSON result.")

    benchmark = subparsers.add_parser("benchmark", help="Compare top-k, MMR, and knapsack strategies.")
    benchmark.add_argument("--index", default=DEFAULT_INDEX)
    benchmark.add_argument("--budget", type=int, default=50_000)
    benchmark.add_argument("--budgets", help="Comma-separated context budgets, e.g. 2000,5000,50000.")
    benchmark.add_argument("--reserve-output", type=int, default=4_000)
    benchmark.add_argument("--sample-size", type=int, default=12)
    benchmark.add_argument("--candidate-pool", type=int, default=250)
    benchmark.add_argument("--gold", help="Gold evidence JSONL file.")
    benchmark.add_argument("--output", default=".tokenpack/benchmark.json")
    benchmark.add_argument("--markdown-output", help="Optional Markdown summary report path.")
    benchmark.add_argument("--csv-output", help="Optional CSV summary report path.")

    dataset = subparsers.add_parser("dataset", help="Create and validate gold evidence datasets.")
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    propose = dataset_subparsers.add_parser("propose", help="Propose reviewable gold JSONL records from an index.")
    propose.add_argument("--index", default=DEFAULT_INDEX)
    propose.add_argument("--sample-size", type=int, default=12)
    propose.add_argument("--output", default=".tokenpack/gold-proposed.jsonl")
    validate = dataset_subparsers.add_parser("validate", help="Validate a gold JSONL file against an index.")
    validate.add_argument("--index", default=DEFAULT_INDEX)
    validate.add_argument("--gold", required=True)

    export = subparsers.add_parser("export-context", help="Render selected chunks in original order.")
    export.add_argument("--selection", default=".tokenpack/selection.json")
    export.add_argument("--output", default=".tokenpack/context.txt")
    export.add_argument("--no-headers", action="store_true")
    export.add_argument("--compressor", choices=["none", "llmlingua"], default="none")
    export.add_argument("--compression-model", default=DEFAULT_COMPRESSION_MODEL)
    export.add_argument("--compression-rate", type=float, default=0.5)
    export.add_argument("--compression-target-tokens", type=int, default=-1)
    export.add_argument("--compression-question", default="")
    export.add_argument("--compression-instruction", default="")
    export.add_argument("--longllmlingua", action="store_true")
    export.add_argument("--llmlingua2", action="store_true")
    export.add_argument("--compression-device-map", default="cpu")
    export.add_argument(
        "--compression-allow-download",
        action="store_true",
        help="Allow Hugging Face network access when the compression model is not cached locally.",
    )
    export.add_argument("--compression-context-filter", action="store_true")
    export.add_argument("--compression-sentence-filter", action="store_true")
    export.add_argument(
        "--no-compression-token-filter",
        action="store_true",
        help="Disable LLMLingua token-level filtering; mostly useful for ablations.",
    )

    answer = subparsers.add_parser("answer", help="Optionally generate an answer from a saved selection.")
    answer.add_argument("--query", required=True)
    answer.add_argument("--selection", default=".tokenpack/selection.json")
    answer.add_argument("--provider", choices=["none", "ollama", "local", "cerebras", "groq"], default="none")
    answer.add_argument("--model", default="qwen3:0.6b")
    answer.add_argument("--ollama-url", default="http://localhost:11434")
    answer.add_argument("--output", default=".tokenpack/answer.json")

    doctor = subparsers.add_parser("doctor", help="Check local TokenPack, embedding, and Ollama readiness.")
    doctor.add_argument("--ollama-url", default="http://localhost:11434")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_name = QUALITY_MODEL if args.model == "quality" else args.model
    if args.command == "doctor":
        print(json.dumps(collect_diagnostics(args.ollama_url), ensure_ascii=False, indent=2))
        return 0

    if args.command == "pack":
        return _run_pack(args, model_name)

    if args.command == "ingest":
        embedder = _make_cli_embedder(args, model_name)
        chunk_size = resolve_chunk_size_config(
            args.chunk_size_preset,
            args.target_tokens,
            args.min_tokens,
            args.max_tokens,
        )
        index = ingest_path(
            args.source,
            args.index,
            embedder=embedder,
            target_tokens=chunk_size.target_tokens,
            min_tokens=chunk_size.min_tokens,
            max_tokens=chunk_size.max_tokens,
            chunker_name="structure-aware",
            source_type=args.source_type,
        )
        print(f"Indexed {len(index.chunks)} chunks with {index.model_name}: {args.index}")
        return 0

    if args.command == "select":
        embedder = _make_cli_embedder(args, model_name)
        index = load_index(args.index)
        query_embedding = embedder.embed([args.query])[0]
        penalty = args.redundancy_penalty if args.strategy == "knapsack-redundancy" else 0.0
        scored = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            redundancy_penalty=penalty,
            scoring=DEFAULT_SCORING_PROFILE,
            query_text=args.query,
            redundancy_candidate_pool=args.candidate_pool,
        )
        effective_budget = max(0, args.budget - args.reserve_output)
        result = select_chunks(
            scored,
            strategy=args.strategy,
            budget=effective_budget,
            candidate_pool=args.candidate_pool,
            relevance_threshold=args.relevance_threshold,
            embeddings=index.embeddings,
            coverage_query=args.query,
        )
        payload = result.to_dict() | {"scoring": DEFAULT_SCORING_PROFILE}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(
                f"{args.strategy}: selected {len(result.selected)} chunks, "
                f"{result.used_tokens}/{effective_budget} tokens, "
                f"scoring={DEFAULT_SCORING_PROFILE}, value={result.total_value:.3f}"
            )
            print(f"Selection saved: {args.output}")
        return 0

    if args.command == "benchmark":
        embedder = _make_cli_embedder(args, model_name)
        index = load_index(args.index)
        if args.gold:
            records = load_gold_records(args.gold)
            errors = validate_gold_records(records, index)
            if errors:
                raise SystemExit("Gold validation failed:\n" + "\n".join(errors))
            payload = run_gold_benchmark(
                index,
                embedder=embedder,
                records=records,
                budgets=_parse_budgets(args.budgets, args.budget),
                reserve_output=args.reserve_output,
                candidate_pool=args.candidate_pool,
                scoring=DEFAULT_SCORING_PROFILE,
            )
        else:
            payload = run_benchmark(
                index,
                embedder=embedder,
                budget=args.budget,
                reserve_output=args.reserve_output,
                sample_size=args.sample_size,
                candidate_pool=args.candidate_pool,
                scoring=DEFAULT_SCORING_PROFILE,
            )
        save_benchmark(payload, args.output)
        if args.markdown_output:
            save_markdown_report(payload, args.markdown_output)
        if args.csv_output:
            save_csv_report(payload, args.csv_output)
        summary = payload["budgets"][0]["summary"] if "budgets" in payload else payload["summary"]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Benchmark saved: {args.output}")
        return 0

    if args.command == "dataset":
        index = load_index(args.index)
        if args.dataset_command == "propose":
            records = propose_gold_records(index, sample_size=args.sample_size)
            save_gold_records(records, args.output)
            print(f"Proposed {len(records)} reviewable gold records: {args.output}")
            return 0
        if args.dataset_command == "validate":
            records = load_gold_records(args.gold)
            errors = validate_gold_records(records, index)
            if errors:
                print("Gold validation failed:")
                for error in errors:
                    print(f"- {error}")
                return 1
            print(f"Gold validation passed: {len(records)} records")
            return 0

    if args.command == "export-context":
        compression_config = CompressionConfig(
            compressor=args.compressor,
            model_name=args.compression_model,
            rate=args.compression_rate,
            target_tokens=args.compression_target_tokens,
            question=args.compression_question,
            instruction=args.compression_instruction,
            longllmlingua=args.longllmlingua,
            llmlingua2=args.llmlingua2,
            use_context_level_filter=args.compression_context_filter,
            use_sentence_level_filter=args.compression_sentence_filter,
            use_token_level_filter=not args.no_compression_token_filter,
            device_map=args.compression_device_map,
            local_files_only=not args.compression_allow_download,
        )
        compression_result = export_selection(
            args.selection,
            args.output,
            include_headers=not args.no_headers,
            compression_config=compression_config,
        )
        print(f"Context exported: {args.output}")
        if compression_result is not None:
            print(
                "Compression: "
                f"{compression_result.origin_tokens}->{compression_result.compressed_tokens} tokens, "
                f"ratio={compression_result.ratio:.2f}x, "
                f"saving={compression_result.saving_rate:.1%}"
            )
        return 0

    if args.command == "answer":
        payload = answer_from_selection(
            query=args.query,
            selection_path=args.selection,
            provider=args.provider,
            model=args.model,
            ollama_url=args.ollama_url,
        )
        save_answer(payload, args.output)
        print(f"Answer saved: {args.output}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def _parse_budgets(value: str | None, default_budget: int) -> list[int]:
    if not value:
        return [default_budget]
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise ValueError("--budgets must include at least one integer budget.")
    return budgets


def _run_pack(args: argparse.Namespace, model_name: str) -> int:
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    output_path = _infer_pack_output_path(source, args.out)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}\nUse --overwrite or choose --out.")
    progress = None if args.quiet else _pack_progress
    if progress is not None:
        mode = "local cache only" if args.offline_models else "local cache first; download if missing"
        progress(f"Loading embedding model: {model_name} ({mode}).")
    embedder = _make_cli_embedder(args, model_name)
    try:
        result = pack_source(
            source=args.source,
            query=args.query,
            out=args.out,
            overwrite=args.overwrite,
            budget=args.budget,
            budget_ratio=args.budget_ratio,
            min_budget=args.min_budget,
            max_budget=args.max_budget,
            reserve_output=args.reserve_output,
            index_out=args.index_out,
            selection_out=args.selection_out,
            candidate_pool=args.candidate_pool,
            chunk_size_preset=args.chunk_size_preset,
            target_tokens=args.target_tokens,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            source_type=args.source_type,
            compress=args.compress,
            compression_model=args.compression_model,
            compression_rate=args.compression_rate,
            compression_target_tokens=args.compression_target_tokens,
            compression_instruction=args.compression_instruction,
            longllmlingua=args.longllmlingua,
            llmlingua2=args.llmlingua2,
            compression_device_map=args.compression_device_map,
            allow_download=args.allow_download,
            compression_context_filter=args.compression_context_filter,
            compression_sentence_filter=args.compression_sentence_filter,
            no_compression_token_filter=args.no_compression_token_filter,
            output_detail=args.output_detail,
            progress=progress,
            embedder=embedder,
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    for line in format_pack_summary(result):
        print(line)
    return 0


def _make_cli_embedder(args: argparse.Namespace, model_name: str):
    return make_embedder(
        model_name=model_name,
        local_files_only=True if args.offline_models else None,
    )


def _pack_progress(message: str) -> None:
    print(f"[tokenpack] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
