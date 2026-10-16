from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_utils import pairwise_rows, read_jsonl, summarize_rows, write_csv, write_jsonl  # noqa: E402
from tokenpack.chunking import StructureAwareChunker  # noqa: E402
from tokenpack.compression import CompressionConfig, compress_chunks  # noqa: E402
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.evidence_grouping import fit_render_to_token_budget, render_source_evidence_groups  # noqa: E402
from tokenpack.evidence_neighbors import select_neighbor_augmented  # noqa: E402
from tokenpack.models import Chunk, ScoredChunk, TextBlock  # noqa: E402
from tokenpack.reranking import CrossEncoderReranker, apply_reranker  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.selectors import select_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


from submission.evaluation.timing import timing_record

DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "longbench_v2_pilot"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "longbench_v2_pilot"
LONG_BENCH_REPO = "zai-org/LongBench-v2"
PIPELINES = [
    "full-context",
    "production-rag-50",
    "tokenpack-50",
    "only-longllmlingua-rate050",
    "tokenpack-50+longllmlingua-rate050",
    "tokenpack-60+longllmlingua-rate050",
    "tokenpack-50+longllmlingua-rate065",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, run, and summarize a LongBench v2 TokenPack pilot.")
    parser.add_argument("--data-file", help="Local LongBench v2 JSON/JSONL file. If omitted, streams from HF.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--tasks-jsonl", help="Override task output path.")
    parser.add_argument("--results-jsonl", help="Summarize an existing result JSONL.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--source-min-tokens", type=int, default=8_000)
    parser.add_argument("--source-max-tokens", type=int, default=24_000)
    parser.add_argument("--max-scanned", type=int, default=180)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-allow-download", action="store_true")
    parser.add_argument("--chunker", choices=["structure-aware"], default="structure-aware")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="evidence-hybrid",
    )
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    parser.add_argument(
        "--context-order",
        choices=["score", "source", "score-then-source"],
        default="score",
        help="How selected TokenPack chunks are rendered for generation.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=["budget-top-k", "knapsack-redundancy", "knapsack-coverage"],
        default="budget-top-k",
    )
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument(
        "--phase2-ablation", action="store_true",
        help="Build four opt-in 50%% budget evidence-neighbor/grouping diagnostic arms; never alters tokenpack-50.",
    )
    parser.add_argument(
        "--phase2-neighbor-fraction", type=float, default=0.15,
        help="Fraction of chunk-token budget reserved for genuine source-adjacent evidence.",
    )
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--longllmlingua-model",
        default="gpt2",
        help="Causal LM used by LongLLMLingua. Use a stronger model such as microsoft/phi-2 for a paper-grade run.",
    )
    parser.add_argument("--compression-rate", type=float, default=0.50)
    parser.add_argument("--compression-device-map", default="cpu")
    parser.add_argument("--compression-allow-download", action="store_true")
    parser.add_argument("--reranker", choices=["none", "cross-encoder"], default="none")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--reranker-candidate-pool", type=int, default=80)
    parser.add_argument("--reranker-weight", type=float, default=0.35)
    parser.add_argument("--reranker-allow-download", action="store_true")
    parser.add_argument("--cascade-frontier", action="store_true")
    parser.add_argument(
        "--diagnostic-selectors",
        action="store_true",
        help="Build selector/scoring diagnostic methods: similarity-knapsack, hybrid-greedy, and hybrid-knapsack.",
    )
    parser.add_argument("--skip-compression", action="store_true", help="Build only full-context and TokenPack tasks.")
    parser.add_argument(
        "--fast-pilot", action="store_true",
        help="Build 5 no-GPT2 methods: full, production-rag-50, tokenpack-50, head-tail-50, tokenpack-25. Requires --skip-compression.",
    )
    parser.add_argument(
        "--rag25-only", action="store_true",
        help="Build just the exact production-rag-25 method, selecting directly from the full ranked candidate pool at 25% budget.",
    )
    parser.add_argument(
        "--checkpoint-dir", default="",
        help="Write/reuse one verified checkpoint per completed case; allows interruption-safe resume.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-modal", action="store_true")
    parser.add_argument("--shard-size", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--latency-mode",
        action="store_true",
        help="Run Modal generation one prompt at a time and record per-request wall-clock latency.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = Path(args.tasks_jsonl) if args.tasks_jsonl else output_dir / "longbench_generation_tasks.jsonl"

    if args.results_jsonl:
        _summarize_results(Path(args.results_jsonl), output_dir)
        return 0

    tasks, report = build_tasks(args)
    write_jsonl(tasks, tasks_path)
    report_path = output_dir / "task_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(tasks)} LongBench tasks to {tasks_path}")
    print(f"Wrote task report to {report_path}")

    modal_command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(THIS_DIR / "app.py"),
        "--tasks-jsonl",
        str(tasks_path),
        "--output-jsonl",
        str(output_dir / "longbench_generation_results.jsonl"),
        "--shard-size",
        str(args.shard_size),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.latency_mode:
        modal_command.append("--latency-mode")
    if args.run_modal:
        env = dict(**{key: value for key, value in __import__("os").environ.items()}, PYTHONIOENCODING="utf-8")
        subprocess.run(modal_command, check=True, env=env)
    else:
        print("Modal command:")
        print(" ".join(modal_command))
        if args.dry_run:
            print("Dry run complete; no Modal job launched.")
    return 0


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _apply_default_args(args)
    if args.phase2_ablation and (
        not args.skip_compression or args.fast_pilot or args.rag25_only
        or args.diagnostic_selectors or args.cascade_frontier
        or args.selection_strategy != "budget-top-k"
        or args.context_order != "score"
        or abs(args.budget_ratio - 0.5) > 1e-9
    ):
        raise ValueError(
            "--phase2-ablation requires --skip-compression, standard budget-top-k, "
            "--context-order score, --budget-ratio 0.50 and no conflicting ablation switches."
        )
    if args.fast_pilot and (not args.skip_compression or args.diagnostic_selectors or args.cascade_frontier):
        raise ValueError("--fast-pilot requires --skip-compression and cannot be combined with other ablations")
    if args.rag25_only and (
        not args.skip_compression or args.fast_pilot or args.diagnostic_selectors
        or args.cascade_frontier or args.reranker != "none"
    ):
        raise ValueError("--rag25-only requires --skip-compression, --reranker none and no other ablations")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_signature = _checkpoint_config_signature(args)
    token_counter = TokenCounter()
    embedder = make_embedder(
        model_name=args.embedding_model,
        local_files_only=not args.embedding_allow_download,
    )
    compressor_backend = None
    reranker_backend = _make_reranker(args)
    tasks: list[dict[str, Any]] = []
    selected_cases = 0
    scanned = 0
    skipped_too_short = 0
    skipped_too_long = 0

    for row in _load_rows(args):
        if scanned >= args.max_scanned or selected_cases >= args.limit:
            break
        scanned += 1
        case = _normalize_row(row)
        # Include question, choices and hidden gold as well as the document:
        # cached contexts must never be reused for a changed benchmark case.
        source_hash = hashlib.sha256(
            json.dumps(case, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        checkpoint_path = (
            checkpoint_dir / (hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest() + ".json")
            if checkpoint_dir is not None else None
        )
        if checkpoint_path is not None and checkpoint_path.exists():
            cached = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (cached.get("config_signature") != config_signature
                    or cached.get("source_sha256") != source_hash
                    or cached.get("case_id") != case["case_id"]):
                raise ValueError("Checkpoint differs from this source or method configuration; refusing unsafe reuse.")
            expected_methods = (
                8 if args.phase2_ablation else
                1 if args.rag25_only else
                5 if args.fast_pilot else
                3 if args.skip_compression else 5
            )
            if len(cached["tasks"]) != expected_methods:
                raise ValueError("Checkpoint contains an incorrect number of methods.")
            tasks.extend(cached["tasks"])
            selected_cases += 1
            print(f"[{selected_cases}/{args.limit}] CACHE HIT {case['case_id']}", flush=True)
            continue
        source_tokens = token_counter.count(case["context"])
        if source_tokens < args.source_min_tokens:
            skipped_too_short += 1
            continue
        if source_tokens > args.source_max_tokens:
            skipped_too_long += 1
            continue

        document_started = time.perf_counter()
        render_timings = {}
        def timed_render(fn, *values, **kwargs):
            start = time.perf_counter()
            text = fn(*values, **kwargs)
            render_timings[id(text)] = time.perf_counter() - start
            return text
        chunks = _chunk_context(
            case_id=case["case_id"],
            context=case["context"],
            chunker_name=args.chunker,
            token_counter=token_counter,
            embedder=embedder,
            semantic_threshold=0.35,
            target_tokens=args.target_tokens,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        if not chunks:
            continue
        query = _selection_query(case)
        chunk_embeddings = embedder.embed([chunk.text for chunk in chunks])
        document_seconds = time.perf_counter() - document_started
        query_started = time.perf_counter()
        query_embedding = embedder.embed([query])[0]
        query_seconds = time.perf_counter() - query_started
        scoring_started = time.perf_counter()
        scored = score_chunks(
            query_embedding,
            chunks,
            chunk_embeddings,
            scoring=args.scoring,
            query_text=query,
            redundancy_candidate_pool=args.candidate_pool,
        )
        if reranker_backend is not None:
            scored = apply_reranker(
                scored,
                query=query,
                reranker=reranker_backend,
                candidate_pool=args.reranker_candidate_pool,
                weight=args.reranker_weight,
            )
        scoring_seconds = time.perf_counter() - scoring_started
        budget_tokens = max(1, int(source_tokens * args.budget_ratio))
        selection = select_chunks(
            scored,
            strategy=args.selection_strategy,
            budget=budget_tokens,
            candidate_pool=args.candidate_pool,
            coverage_query=query,
        )
        production_rag = select_chunks(
            scored,
            strategy="production-rag",
            budget=budget_tokens,
            candidate_pool=args.candidate_pool,
        )
        diagnostic_contexts: dict[str, tuple[str, int, dict[str, Any]]] = {}
        if args.diagnostic_selectors:
            similarity_scored = _with_similarity_values(scored)
            similarity_knapsack = select_chunks(
                similarity_scored,
                strategy="knapsack",
                budget=budget_tokens,
                candidate_pool=args.candidate_pool,
            )
            hybrid_greedy = select_chunks(
                scored,
                strategy="budget-top-k",
                budget=budget_tokens,
                candidate_pool=args.candidate_pool,
            )
            hybrid_knapsack = (
                selection
                if args.selection_strategy == "knapsack-redundancy"
                else select_chunks(
                    scored,
                    strategy="knapsack-redundancy",
                    budget=budget_tokens,
                    candidate_pool=args.candidate_pool,
                    coverage_query=query,
                )
            )
            diagnostic_contexts = {
                "similarity-knapsack-50": (
                    timed_render(_render_similarity_context, similarity_knapsack.selected),
                    similarity_knapsack.used_tokens,
                    {"selection_seconds": similarity_knapsack.elapsed_seconds, "compression_seconds": 0.0},
                ),
                "hybrid-greedy-50": (
                    timed_render(_render_selected_context, hybrid_greedy.selected, order=args.context_order),
                    hybrid_greedy.used_tokens,
                    {"selection_seconds": hybrid_greedy.elapsed_seconds, "compression_seconds": 0.0},
                ),
                "hybrid-knapsack-50": (
                    timed_render(_render_selected_context, hybrid_knapsack.selected, order=args.context_order),
                    hybrid_knapsack.used_tokens,
                    {"selection_seconds": hybrid_knapsack.elapsed_seconds, "compression_seconds": 0.0},
                ),
            }
        selected_chunks = [item.chunk for item in selection.selected]

        contexts: dict[str, tuple[str, int, dict[str, Any]]] = {
            "full-context": (
                case["context"],
                source_tokens,
                {"selection_seconds": 0.0, "compression_seconds": 0.0},
            ),
            "production-rag-50": (
                timed_render(_render_production_rag_context, production_rag.selected),
                production_rag.used_tokens,
                {"selection_seconds": production_rag.elapsed_seconds, "compression_seconds": 0.0},
            ),
        }
        if args.rag25_only:
            quarter_budget = max(1, int(source_tokens * 0.25))
            rag_25 = select_chunks(
                scored, strategy="production-rag", budget=quarter_budget,
                candidate_pool=args.candidate_pool, coverage_query=query,
            )
            contexts = {
                "production-rag-25": (
                    timed_render(_render_production_rag_context, rag_25.selected),
                    rag_25.used_tokens,
                    {"selection_seconds": rag_25.elapsed_seconds, "compression_seconds": 0.0,
                     "budget_tokens": quarter_budget},
                )
            }
        if args.diagnostic_selectors:
            contexts.update(diagnostic_contexts)
        elif not args.rag25_only:
            contexts["tokenpack-50"] = (
                timed_render(_render_selected_context, selection.selected, order=args.context_order),
                selection.used_tokens,
                {"selection_seconds": selection.elapsed_seconds, "compression_seconds": 0.0},
            )
        if args.phase2_ablation:
            # Factorial 2x2: current selection vs source-neighbour augmentation,
            # current score-order output vs provenance/adjacency grouping.
            # All four NEW arms strictly count rendered headers inside the budget.
            # Original "tokenpack-50" and "production-rag-50" are untouched.
            # A fifth arm budget-matches Production RAG for a fairer comparison.
            rag_render_started = time.perf_counter()
            rag_rendered, rag_retained, rag_dropped = fit_render_to_token_budget(
                production_rag.selected,
                render=_render_production_rag_context,
                token_counter=token_counter,
                budget=budget_tokens,
                drop_priority=lambda item: item.raw_similarity,
            )
            render_timings[id(rag_rendered)] = time.perf_counter() - rag_render_started
            contexts["production-rag-50-p2-budgeted"] = (
                rag_rendered,
                sum(item.weight for item in rag_retained),
                {
                    "selection_seconds": production_rag.elapsed_seconds,
                    "compression_seconds": 0.0,
                    "phase2_ablation": True,
                    "phase2_neighbor": False,
                    "phase2_grouped": False,
                    "phase2_seed_chunk_count": len(production_rag.selected),
                    "phase2_kept_chunk_count": len(rag_retained),
                    "phase2_trimmed_header_overhead": rag_dropped,
                    "phase2_neighbor_fraction": 0.0,
                },
            )
            neighbor_selection = select_neighbor_augmented(
                scored,
                budget_tokens,
                candidate_pool=args.candidate_pool,
                neighbor_fraction=args.phase2_neighbor_fraction,
            )
            phase2_arms = (
                ("tokenpack-50-p2-control", selection.selected, False, selection.elapsed_seconds),
                ("tokenpack-50-p2-grouped", selection.selected, True, selection.elapsed_seconds),
                ("tokenpack-50-p2-neighbor", neighbor_selection.selected, False, neighbor_selection.elapsed_seconds),
                ("tokenpack-50-p2-neighbor-grouped", neighbor_selection.selected, True, neighbor_selection.elapsed_seconds),
            )
            for name, selected_items, grouped, selection_time in phase2_arms:
                render_started = time.perf_counter()
                rendered, retained, dropped = _phase2_render_budgeted(
                    selected_items,
                    grouped=grouped,
                    token_counter=token_counter,
                    budget_tokens=budget_tokens,
                )
                render_timings[id(rendered)] = time.perf_counter() - render_started
                contexts[name] = (
                    rendered,
                    sum(item.weight for item in retained),
                    {
                        "selection_seconds": selection_time,
                        "compression_seconds": 0.0,
                        "phase2_ablation": True,
                        "phase2_neighbor": "neighbor" in name,
                        "phase2_grouped": grouped,
                        "phase2_seed_chunk_count": len(selected_items),
                        "phase2_kept_chunk_count": len(retained),
                        "phase2_trimmed_header_overhead": dropped,
                        "phase2_neighbor_fraction": args.phase2_neighbor_fraction,
                    },
                )
        if args.fast_pilot:
            quarter_budget = max(1, int(source_tokens * 0.25))
            quarter_selection = select_chunks(
                scored, strategy=args.selection_strategy, budget=quarter_budget,
                candidate_pool=args.candidate_pool, coverage_query=query,
            )
            contexts["tokenpack-25"] = (
                timed_render(_render_selected_context, quarter_selection.selected, order=args.context_order),
                quarter_selection.used_tokens,
                {"selection_seconds": quarter_selection.elapsed_seconds,
                 "compression_seconds": 0.0, "budget_tokens": quarter_budget},
            )
            head_tail = timed_render(
                _render_head_tail_context, case["context"],
                budget_tokens, token_counter,
            )
            contexts["head-tail-50"] = (
                head_tail, token_counter.count(head_tail),
                {"selection_seconds": 0.0, "compression_seconds": 0.0},
            )

        if not args.skip_compression:
            if compressor_backend is None:
                compressor_backend = _make_compressor_backend(args)
            long_full, full_seconds = _compress(
                chunks=chunks,
                args=args,
                question=query,
                backend=compressor_backend,
                token_counter=token_counter,
            )
            long_selected, selected_seconds = _compress(
                chunks=selected_chunks,
                args=args,
                question=query,
                backend=compressor_backend,
                token_counter=token_counter,
            )
            contexts["only-longllmlingua-rate050"] = (
                long_full.compressed_prompt,
                long_full.compressed_tokens,
                {"selection_seconds": 0.0, "compression_seconds": full_seconds},
            )
            contexts["tokenpack-50+longllmlingua-rate050"] = (
                long_selected.compressed_prompt,
                long_selected.compressed_tokens,
                {"selection_seconds": selection.elapsed_seconds, "compression_seconds": selected_seconds},
            )
            if args.cascade_frontier:
                selection_60 = select_chunks(
                    scored,
                    strategy=args.selection_strategy,
                    budget=max(1, int(source_tokens * 0.60)),
                    candidate_pool=args.candidate_pool,
                    coverage_query=query,
                )
                long_selected_60, selected_60_seconds = _compress(
                    chunks=[item.chunk for item in selection_60.selected],
                    args=args,
                    question=query,
                    backend=compressor_backend,
                    token_counter=token_counter,
                    compression_rate=0.50,
                )
                long_selected_65, selected_65_seconds = _compress(
                    chunks=selected_chunks,
                    args=args,
                    question=query,
                    backend=compressor_backend,
                    token_counter=token_counter,
                    compression_rate=0.65,
                )
                contexts["tokenpack-60+longllmlingua-rate050"] = (
                    long_selected_60.compressed_prompt,
                    long_selected_60.compressed_tokens,
                    {
                        "selection_seconds": selection_60.elapsed_seconds,
                        "compression_seconds": selected_60_seconds,
                        "cascade_budget_ratio": 0.60,
                        "cascade_compression_rate": 0.50,
                    },
                )
                contexts["tokenpack-50+longllmlingua-rate065"] = (
                    long_selected_65.compressed_prompt,
                    long_selected_65.compressed_tokens,
                    {
                        "selection_seconds": selection.elapsed_seconds,
                        "compression_seconds": selected_65_seconds,
                        "cascade_budget_ratio": args.budget_ratio,
                        "cascade_compression_rate": 0.65,
                    },
                )

        case_tasks: list[dict[str, Any]] = []
        for method, (context, context_tokens, metadata) in contexts.items():
            full = method == "full-context"
            compression_only = method.startswith("only-")
            compressed = "llmlingua" in method
            skipped = []
            if full:
                skipped.extend(("document_preparation", "query_embedding", "scoring", "selection", "render", "compression"))
            elif compression_only:
                skipped.extend(("query_embedding", "scoring", "selection"))
            if not compressed and not full:
                skipped.append("compression")
            timing = timing_record(document_preparation=document_seconds,
                                   query_embedding=query_seconds, scoring=scoring_seconds,
                                   selection=metadata.get("selection_seconds"),
                                   render=render_timings.get(id(context)),
                                   compression=metadata.get("compression_seconds"),
                                   skipped=skipped, document_id=case["case_id"])
            # Compressed rendering is included in compressor timing, not measured twice.
            if compressed:
                timing["stages"]["render"] = {"status": "included_in_compression", "seconds": None}
            actual_context_tokens = token_counter.count(context)
            case_tasks.append(
                {
                    "task_id": f"{case['case_id']}::{method}",
                    "case_id": case["case_id"],
                    "method": method,
                    "domain": case["domain"],
                    "sub_domain": case["sub_domain"],
                    "difficulty": case["difficulty"],
                    "length": case["length"],
                    "question": case["question"],
                    "choice_A": case["choice_A"],
                    "choice_B": case["choice_B"],
                    "choice_C": case["choice_C"],
                    "choice_D": case["choice_D"],
                    "answer": case["answer"],
                    "context": context,
                    "source_tokens": source_tokens,
                    "context_tokens": actual_context_tokens,
                    "selected_content_tokens": context_tokens,
                    "prompt_tokens": None,
                    "schema_version": "2", "metric_version": "2.0",
                    "timing": timing,
                    "context_tokenizer": token_counter.encoding_name if token_counter._encoding is not None else "regex",
                    "tokenizer_revision": "unknown",
                    "budget_tokens": budget_tokens,
                    "token_saving_vs_full": 1.0 - actual_context_tokens / max(1, source_tokens),
                    **metadata,
                }
            )
        tasks.extend(case_tasks)
        selected_cases += 1
        if checkpoint_path is not None:
            payload = {
                "case_id": case["case_id"], "source_sha256": source_hash,
                "config_signature": config_signature, "tasks": case_tasks,
            }
            temporary = checkpoint_path.with_suffix(".writing")
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint_path)
        print(f"[{selected_cases}/{args.limit}] READY {case['case_id']} "
              f"methods={len(case_tasks)} source_tokens={source_tokens}", flush=True)

    by_method: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_method.setdefault(str(task["method"]), []).append(task)
    report = {
        "schema_version": "2", "metric_version": "2.0",
        "resolved_config": vars(args),
        "config_hash": hashlib.sha256(json.dumps(vars(args), sort_keys=True, default=str).encode()).hexdigest(),
        "model_revision": "unknown",
        "tasks": len(tasks),
        "cases": selected_cases,
        "scanned": scanned,
        "skipped_too_short": skipped_too_short,
        "skipped_too_long": skipped_too_long,
        "source_token_window": [args.source_min_tokens, args.source_max_tokens],
        "scoring": args.scoring,
        "methods": sorted(by_method),
        "selection_strategy": args.selection_strategy,
        "context_order": args.context_order,
        "production_rag_baseline": True,
        "diagnostic_selectors": bool(args.diagnostic_selectors),
        "phase2_ablation": bool(args.phase2_ablation),
        "phase2_neighbor_fraction": args.phase2_neighbor_fraction,
        "reranker": args.reranker,
        "reranker_model": args.reranker_model,
        "reranker_candidate_pool": args.reranker_candidate_pool,
        "reranker_weight": args.reranker_weight,
        "cascade_frontier": args.cascade_frontier,
        "fast_pilot": args.fast_pilot,
        "rag25_only": args.rag25_only,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "per_method": {
            method: {
                "tasks": len(rows),
                "avg_source_tokens": _mean(row["source_tokens"] for row in rows),
                "avg_context_tokens": _mean(row["context_tokens"] for row in rows),
                "avg_token_saving_vs_full": _mean(row["token_saving_vs_full"] for row in rows),
            }
            for method, rows in sorted(by_method.items())
        },
    }
    return tasks, report


def _load_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.data_file:
        path = Path(args.data_file)
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield json.loads(line)
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            yield from payload
            return
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            yield from payload["data"]
            return
        raise ValueError(f"Unsupported LongBench data file format: {path}")

    from datasets import load_dataset

    yield from load_dataset(LONG_BENCH_REPO, split="train", streaming=True)


def _normalize_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "case_id": str(row.get("_id") or row.get("id") or ""),
        "domain": str(row.get("domain") or ""),
        "sub_domain": str(row.get("sub_domain") or ""),
        "difficulty": str(row.get("difficulty") or ""),
        "length": str(row.get("length") or ""),
        "question": str(row.get("question") or ""),
        "choice_A": str(row.get("choice_A") or ""),
        "choice_B": str(row.get("choice_B") or ""),
        "choice_C": str(row.get("choice_C") or ""),
        "choice_D": str(row.get("choice_D") or ""),
        "answer": str(row.get("answer") or "").strip().upper(),
        "context": str(row.get("context") or ""),
    }


def _selection_query(case: dict[str, str]) -> str:
    return (
        f"{case['question']}\n"
        f"A. {case['choice_A']}\n"
        f"B. {case['choice_B']}\n"
        f"C. {case['choice_C']}\n"
        f"D. {case['choice_D']}"
    )


def _apply_default_args(args: argparse.Namespace) -> None:
    defaults = {
        "fast_pilot": False,
        "rag25_only": False,
        "checkpoint_dir": "",
        "context_order": "score",
        "reranker": "none",
        "reranker_model": "BAAI/bge-reranker-base",
        "reranker_candidate_pool": 80,
        "reranker_weight": 0.35,
        "reranker_allow_download": False,
        "cascade_frontier": False,
        "compression_rate": 0.50,
        "longllmlingua_model": "gpt2",
        "compression_device_map": "cpu",
        "compression_allow_download": False,
        "embedding_allow_download": False,
        "diagnostic_selectors": False,
        "phase2_ablation": False,
        "phase2_neighbor_fraction": 0.15,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def _checkpoint_config_signature(args: argparse.Namespace) -> str:
    # Paths, reporting switches and limits do not change a case's context.
    ignored = {
        "data_file", "output_dir", "work_dir", "tasks_jsonl", "results_jsonl",
        "checkpoint_dir", "limit", "max_scanned", "dry_run", "run_modal",
        "shard_size", "batch_size", "latency_mode",
    }
    material = {k: v for k, v in vars(args).items() if k not in ignored}
    if not getattr(args, "phase2_ablation", False):
        # Preserve the old default checkpoint hash for pre-Phase-2 runs.
        # Experimental flags become material only when the ablation is enabled.
        material.pop("phase2_ablation", None)
        material.pop("phase2_neighbor_fraction", None)
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def _render_head_tail_context(context: str, budget: int, token_counter: TokenCounter) -> str:
    if token_counter._encoding is None:
        raise ValueError("Head/tail benchmark requires exact cl100k tokenizer, not regex fallback")
    token_ids = token_counter._encoding.encode(context, disallowed_special=())
    if len(token_ids) <= budget:
        return context
    front = budget // 2
    rear = budget - front
    return (
        token_counter._encoding.decode(token_ids[:front])
        + "\n\n[HEAD/TAIL REFERENCE: SOURCE MIDDLE OMITTED]\n\n"
        + token_counter._encoding.decode(token_ids[-rear:])
    )


def _make_reranker(args: argparse.Namespace):
    if args.reranker == "none":
        return None
    if args.reranker == "cross-encoder":
        return CrossEncoderReranker(
            args.reranker_model,
            local_files_only=not args.reranker_allow_download,
        )
    raise ValueError(f"Unknown reranker: {args.reranker}")


def _chunk_context(
    *,
    case_id: str,
    context: str,
    chunker_name: str,
    token_counter: TokenCounter,
    embedder: Any | None = None,
    semantic_threshold: float = 0.35,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    blocks = _blocks_from_context(case_id, context)
    if chunker_name != "structure-aware":
        raise ValueError(f"Unknown chunker: {chunker_name}")
    block_embeddings = embedder.embed([block.text for block in blocks]) if embedder is not None else None
    chunker = StructureAwareChunker(
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        token_counter=token_counter,
        block_embeddings=block_embeddings,
        semantic_threshold=semantic_threshold,
    )
    return chunker.chunk(blocks)


def _blocks_from_context(case_id: str, context: str) -> list[TextBlock]:
    parts = [part.strip() for part in re.split(r"\n\s*\n+", context) if part.strip()]
    if not parts:
        parts = [context.strip()] if context.strip() else []
    blocks: list[TextBlock] = []
    cursor = 0
    for index, part in enumerate(parts):
        start = context.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        cursor = end
        blocks.append(
            TextBlock(
                text=part,
                source_path=f"longbench:{case_id}",
                document_index=0,
                paragraph_index=index,
                char_start=start,
                char_end=end,
                metadata={"content_type": "document"},
            )
        )
    return blocks


def _render_selected_context(items: list[ScoredChunk], *, order: str = "score") -> str:
    if order == "score":
        return _render_ordered_chunks(sorted(items, key=lambda item: item.value, reverse=True))
    if order == "source":
        return _render_ordered_chunks(sorted(items, key=lambda item: item.chunk.order_key))
    if order == "score-then-source":
        priority = sorted(items, key=lambda item: item.value, reverse=True)[:6]
        priority_ids = {item.chunk.id for item in priority}
        remaining = [item for item in sorted(items, key=lambda item: item.chunk.order_key) if item.chunk.id not in priority_ids]
        parts: list[str] = []
        if priority:
            parts.append("[Priority Evidence]")
            parts.append(_render_ordered_chunks(priority, label="Priority Chunk").strip())
        if remaining:
            parts.append("[Context Evidence]")
            parts.append(_render_ordered_chunks(remaining, label="Chunk").strip())
        return "\n\n".join(part for part in parts if part).strip() + "\n"
    raise ValueError(f"Unknown context order: {order}")


def _phase2_render_budgeted(
    items: list[ScoredChunk], *,
    grouped: bool, token_counter: TokenCounter, budget_tokens: int,
) -> tuple[str, list[ScoredChunk], int]:
    """Budget INCLUDING markup; never silently truncate a source chunk."""
    renderer = (
        render_source_evidence_groups if grouped
        else lambda kept: _render_selected_context(kept, order="score")
    )
    return fit_render_to_token_budget(
        items, render=renderer, token_counter=token_counter, budget=budget_tokens,
    )


def _render_production_rag_context(items: list[ScoredChunk]) -> str:
    return _render_ordered_chunks(
        sorted(items, key=lambda item: item.raw_similarity, reverse=True),
        label="Retrieved Chunk",
    )


def _render_similarity_context(items: list[ScoredChunk]) -> str:
    return _render_ordered_chunks(
        sorted(items, key=lambda item: item.raw_similarity, reverse=True),
        label="Similarity-Knapsack Chunk",
    )


def _with_similarity_values(items: list[ScoredChunk]) -> list[ScoredChunk]:
    return [
        replace(
            item,
            value=max(0.0, item.raw_similarity),
            score_components={
                **item.score_components,
                "diagnostic_value": max(0.0, item.raw_similarity),
                "diagnostic_profile": "raw_similarity",
            },
        )
        for item in items
    ]


def _render_score_sorted_context(items: list[ScoredChunk]) -> str:
    return _render_selected_context(items, order="score")


def _render_ordered_chunks(items: list[ScoredChunk], *, label: str = "Chunk") -> str:
    parts: list[str] = []
    for number, item in enumerate(items, start=1):
        chunk = item.chunk
        parts.append(
            f"[{label} "
            f"{number}: id={chunk.id}, source={chunk.source_path}, tokens={chunk.token_count}, "
            f"score={item.value:.4f}, density={item.value / max(1, item.weight):.6f}]"
        )
        parts.append(chunk.text)
    return "\n\n".join(parts).strip() + "\n"


def _compress(
    *,
    chunks: list[Chunk],
    args: argparse.Namespace,
    question: str,
    backend: Any,
    token_counter: TokenCounter,
    compression_rate: float | None = None,
):
    import time

    compressor_chunks = _chunks_for_compression(chunks, token_counter=token_counter, max_tokens=350)
    started = time.perf_counter()
    result = compress_chunks(
        compressor_chunks,
        CompressionConfig(
            compressor="llmlingua",
            model_name=args.longllmlingua_model,
            rate=args.compression_rate if compression_rate is None else compression_rate,
            question=question,
            longllmlingua=True,
            llmlingua2=False,
            use_context_level_filter=True,
            use_token_level_filter=False,
            device_map=args.compression_device_map,
            local_files_only=not args.compression_allow_download,
        ),
        backend=backend,
        token_counter=token_counter,
    )
    return result, time.perf_counter() - started


def _chunks_for_compression(chunks: list[Chunk], *, token_counter: TokenCounter, max_tokens: int) -> list[Chunk]:
    prepared: list[Chunk] = []
    for chunk in chunks:
        text = _normalize_for_compressor(chunk.text)
        if not text:
            continue
        pieces = _split_text_for_compressor(text, token_counter=token_counter, max_tokens=max_tokens)
        for offset, piece in enumerate(pieces):
            prepared.append(
                Chunk(
                    id=f"{chunk.id}-llmlingua-{offset}",
                    text=piece,
                    source_path=chunk.source_path,
                    document_index=chunk.document_index,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    start_paragraph=chunk.start_paragraph,
                    end_paragraph=chunk.end_paragraph,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_count=token_counter.count(piece),
                    block_ids=list(chunk.block_ids),
                    metadata=dict(chunk.metadata),
                )
            )
    return prepared


def _normalize_for_compressor(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_text_for_compressor(text: str, *, token_counter: TokenCounter, max_tokens: int) -> list[str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences or [text]:
        sentence_tokens = token_counter.count(sentence)
        if sentence_tokens > max_tokens:
            if current:
                pieces.append(" ".join(current).strip())
                current = []
            pieces.extend(_split_words(sentence, token_counter=token_counter, max_tokens=max_tokens))
            continue
        candidate = " ".join([*current, sentence]).strip()
        if current and token_counter.count(candidate) > max_tokens:
            pieces.append(" ".join(current).strip())
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current).strip())
    return [piece for piece in pieces if piece]


def _split_words(text: str, *, token_counter: TokenCounter, max_tokens: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join([*current, word]).strip()
        if current and token_counter.count(candidate) > max_tokens:
            pieces.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current).strip())
    return pieces


def _make_compressor_backend(args: argparse.Namespace):
    from llmlingua import PromptCompressor

    model_name = args.longllmlingua_model
    if not args.compression_allow_download:
        from tokenpack.compression import _resolve_local_model_path

        model_name = _resolve_local_model_path(model_name)
    return PromptCompressor(
        model_name=model_name,
        device_map=args.compression_device_map,
        model_config={"local_files_only": not args.compression_allow_download},
        use_llmlingua2=False,
    )


def _summarize_results(results_jsonl: Path, output_dir: Path) -> None:
    rows = read_jsonl(results_jsonl)
    tasks_path = results_jsonl.parent / "longbench_generation_tasks.jsonl"
    expected = read_jsonl(tasks_path) if tasks_path.exists() else None
    summary = summarize_rows(rows, expected=expected)
    pairwise = pairwise_rows(rows, expected=expected)
    pairwise_vs_production_rag = pairwise_rows(rows, baseline="production-rag-50", expected=expected)
    write_csv(summary, output_dir / "longbench_generation_summary.csv")
    write_csv(pairwise, output_dir / "longbench_generation_pairwise.csv")
    write_csv(pairwise_vs_production_rag, output_dir / "longbench_generation_pairwise_vs_production_rag.csv")
    _write_readout(summary, pairwise, output_dir / "longbench_generation_readout.md")
    _write_ablation_readouts(summary, pairwise, output_dir, _load_task_report(output_dir))
    print(f"Wrote summary assets to {output_dir}")


def _write_readout(summary, pairwise, path):
    from submission.evaluation.reporting import write_readout
    write_readout(summary, pairwise, path)


def _write_ablation_readouts(
    summary: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    if report.get("context_order") not in ("", None, "score"):
        _write_readout(summary, pairwise, output_dir / "ordering_ablation_readout.md")
    if report.get("reranker") not in ("", None, "none"):
        _write_readout(summary, pairwise, output_dir / "reranker_ablation_readout.md")
    if report.get("cascade_frontier") is True:
        _write_readout(summary, pairwise, output_dir / "cascade_frontier_readout.md")


def _load_task_report(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "task_report.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values]
    return sum(numeric) / len(numeric) if numeric else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
