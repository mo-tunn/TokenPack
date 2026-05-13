from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
THIS_DIR = Path(__file__).resolve().parent
for path in (SRC, EXPERIMENTS, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_utils import (  # noqa: E402
    METHODS,
    answer_template_for_variant,
    load_prompt_templates,
    pairwise_tokenpack_wins,
    prompt_token_estimate,
    read_jsonl,
    render_answer_prompt,
    stable_case_id,
    summarize_rows,
    write_jsonl,
    write_latex_table,
    write_summary_csv,
)
from qasper_selector_eval import (  # noqa: E402
    _blocks_from_qasper_row,
    _build_index,
    _load_qasper_rows,
    _parse_budgets,
    _questions_from_qasper_row,
)
from tokenpack.chunk_profiles import resolve_chunk_size_config  # noqa: E402
from tokenpack.compression import CompressionConfig, compress_chunks  # noqa: E402
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.export import render_context  # noqa: E402
from tokenpack.models import ScoredChunk  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.selectors import select_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "modal_generation_eval"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "modal_generation_eval"
DEFAULT_SPLIT_SOURCE = (
    ROOT / "submission" / "results" / "qasper_compression_eval_200q_rate050_full" / "qasper_compression_eval_raw.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and summarize Modal QASPER generation-quality evaluation tasks.")
    parser.add_argument("--data-file", help="Local QASPER parquet/json/jsonl file. If omitted, pandas reads the HF URL.")
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--question-ids-from", default=str(DEFAULT_SPLIT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--tasks-jsonl", help="Override task output path.")
    parser.add_argument("--results-jsonl", help="Summarize an existing Modal result JSONL.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-papers", type=int, default=10_000)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-allow-download", action="store_true")
    parser.add_argument("--chunker", choices=["semantic-threshold", "structure-aware"], default="structure-aware")
    parser.add_argument("--scoring", choices=list(SCORING_PROFILES), default="evidence-hybrid")
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--chunk-size-preset", choices=["manual", "default", "low-budget"], default="low-budget")
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--llmlingua2-model", default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
    parser.add_argument("--compression-device-map", default="cpu")
    parser.add_argument("--compression-allow-download", action="store_true")
    parser.add_argument("--skip-llmlingua2", action="store_true", help="Build only the three non-compression methods.")
    parser.add_argument(
        "--tokenpack-variants",
        default="original",
        help=(
            "Comma-separated TokenPack presentation variants: original, score-sorted, "
            "density-sorted, score-sorted-strong, score-sorted-grounded, score-sorted-extractive."
        ),
    )
    parser.add_argument(
        "--tokenpack-only",
        action="store_true",
        help="Build only the requested TokenPack variants, omitting full-document, top-k, and LLMLingua-2 tasks.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build tasks and sizing report, but do not launch Modal.")
    parser.add_argument("--run-modal", action="store_true", help="After building tasks, launch `modal run app.py`.")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = Path(args.tasks_jsonl) if args.tasks_jsonl else output_dir / "qasper_generation_tasks.jsonl"

    if args.results_jsonl:
        _summarize_results(Path(args.results_jsonl), output_dir)
        return 0

    tasks = build_tasks(args)
    write_jsonl(tasks, tasks_path)
    _write_task_report(tasks, output_dir / "task_report.json")
    print(f"Wrote {len(tasks)} generation tasks to {tasks_path}")
    print(f"Wrote task report to {output_dir / 'task_report.json'}")

    modal_command = [
        sys.executable,
        "-m",
        "modal",
        "run",
        str(THIS_DIR / "app.py"),
        "--tasks-jsonl",
        str(tasks_path),
        "--output-jsonl",
        str(output_dir / "qasper_generation_judged.jsonl"),
        "--shard-size",
        str(args.shard_size),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.run_modal:
        subprocess.run(modal_command, check=True)
    else:
        print("Modal command:")
        print(" ".join(modal_command))
        if args.dry_run:
            print("Dry run complete; no Modal job launched.")
    return 0


def build_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    desired_order = _load_question_order(Path(args.question_ids_from)) if args.question_ids_from else []
    desired = set(desired_order)
    token_counter = TokenCounter()
    embedder = make_embedder(
        model_name=args.embedding_model,
        local_files_only=not args.embedding_allow_download,
    )
    chunk_size = resolve_chunk_size_config(
        args.chunk_size_preset,
        target_tokens=args.target_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    rows = list(_load_qasper_rows(args.data_file, args.split))
    tasks_by_case: dict[str, list[dict[str, Any]]] = {}
    compressor_backend: Any | None = None
    processed_papers = 0

    for row in rows:
        if processed_papers >= args.max_papers:
            break
        paper_id = str(row.get("id") or row.get("paper_id") or processed_papers)
        questions = _questions_from_qasper_row(row)
        if desired and not any((paper_id, question.question_id) in desired for question in questions):
            processed_papers += 1
            continue
        blocks = _blocks_from_qasper_row(row, document_index=processed_papers, token_counter=token_counter)
        if not blocks or not questions:
            processed_papers += 1
            continue
        index = _build_index(
            paper_id=paper_id,
            blocks=blocks,
            embedder=embedder,
            work_dir=Path(args.work_dir),
            chunker_name=args.chunker,
            target_tokens=chunk_size.target_tokens,
            min_tokens=chunk_size.min_tokens,
            max_tokens=chunk_size.max_tokens,
            semantic_threshold=args.semantic_threshold,
            token_counter=token_counter,
        )
        source_text = "\n\n".join(block.text for block in blocks if block.text.strip())
        source_tokens = token_counter.count(source_text)
        budget_spec = _parse_budgets(None, str(args.budget_ratio), source_tokens)[0]
        processed_papers += 1

        for question in questions:
            if desired and (paper_id, question.question_id) not in desired:
                continue
            if not desired and len(tasks_by_case) >= args.limit:
                break
            query_embedding = embedder.embed([question.question])[0]
            scored = score_chunks(
                query_embedding,
                index.chunks,
                index.embeddings,
                scoring=args.scoring,
                query_text=question.question,
            )
            tokenpack = select_chunks(
                scored,
                strategy="budget-top-k",
                budget=budget_spec.tokens,
                candidate_pool=args.candidate_pool,
            )
            production_rag = select_chunks(
                scored,
                strategy="production-rag",
                budget=budget_spec.tokens,
                candidate_pool=args.candidate_pool,
            )
            tokenpack_variants = _parse_tokenpack_variants(args.tokenpack_variants)
            method_contexts: dict[str, tuple[str, int, str]] = {}
            if not args.tokenpack_only:
                method_contexts.update(
                    {
                        "full-document": (source_text, source_tokens, "default"),
                        "production-rag-50": (
                            _render_production_rag_context(production_rag.selected),
                            production_rag.used_tokens,
                            "default",
                        ),
                    }
                )
                if getattr(args, "include_budget_top_k", False):
                    method_contexts["budget-top-k-50"] = (
                        render_context([item.chunk for item in tokenpack.selected]),
                        tokenpack.used_tokens,
                        "default",
                    )
            method_contexts.update(_tokenpack_variant_contexts(tokenpack.selected, tokenpack_variants))
            if not args.skip_llmlingua2 and not args.tokenpack_only:
                if compressor_backend is None:
                    compressor_backend = _make_llmlingua2_backend(args)
                compressed = compress_chunks(
                    index.chunks,
                    CompressionConfig(
                        compressor="llmlingua",
                        model_name=args.llmlingua2_model,
                        rate=0.50,
                        question=question.question,
                        llmlingua2=True,
                        device_map=args.compression_device_map,
                        local_files_only=not args.compression_allow_download,
                    ),
                    backend=compressor_backend,
                    token_counter=token_counter,
                )
                method_contexts["only-llmlingua2-rate050"] = (
                    compressed.compressed_prompt,
                    compressed.compressed_tokens,
                    "default",
                )
            case_id = stable_case_id(paper_id, question.question_id)
            tasks_by_case[case_id] = [
                _task_row(
                    paper_id=paper_id,
                    question_id=question.question_id,
                    method=method,
                    question=question.question,
                    gold_answer=question.answer,
                    evidence_texts=question.evidence_texts,
                    context=context,
                    context_tokens=context_tokens,
                    source_tokens=source_tokens,
                    budget_tokens=budget_spec.tokens,
                    answer_prompt_variant=answer_prompt_variant,
                )
                for method, (context, context_tokens, answer_prompt_variant) in method_contexts.items()
            ]
            if len(tasks_by_case) >= args.limit:
                break
        if len(tasks_by_case) >= args.limit:
            break

    ordered_case_ids = [stable_case_id(paper_id, question_id) for paper_id, question_id in desired_order]
    if not ordered_case_ids:
        ordered_case_ids = list(tasks_by_case)
    tasks: list[dict[str, Any]] = []
    for case_id in ordered_case_ids:
        if case_id in tasks_by_case:
            tasks.extend(tasks_by_case[case_id])
        if len({task["case_id"] for task in tasks}) >= args.limit:
            break
    return tasks


def _task_row(
    *,
    paper_id: str,
    question_id: str,
    method: str,
    question: str,
    gold_answer: str,
    evidence_texts: list[str],
    context: str,
    context_tokens: int,
    source_tokens: int,
    budget_tokens: int,
    answer_prompt_variant: str = "default",
) -> dict[str, Any]:
    case_id = stable_case_id(paper_id, question_id)
    return {
        "task_id": f"{case_id}::{method}",
        "case_id": case_id,
        "paper_id": paper_id,
        "question_id": question_id,
        "method": method,
        "question": question,
        "gold_answer": gold_answer,
        "evidence_texts": evidence_texts,
        "context": context,
        "context_tokens": context_tokens,
        "source_tokens": source_tokens,
        "budget_tokens": budget_tokens,
        "answer_prompt_variant": answer_prompt_variant,
    }


def _parse_tokenpack_variants(raw: str) -> list[str]:
    allowed = {
        "original",
        "score-sorted",
        "density-sorted",
        "score-sorted-strong",
        "score-sorted-grounded",
        "score-sorted-extractive",
    }
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in variants if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown TokenPack variants: {', '.join(unknown)}")
    return variants or ["original"]


def _tokenpack_variant_contexts(
    selected: list[ScoredChunk],
    variants: list[str],
) -> dict[str, tuple[str, int, str]]:
    contexts: dict[str, tuple[str, int, str]] = {}
    for variant in variants:
        if variant == "original":
            contexts["tokenpack-50"] = (
                render_context([item.chunk for item in selected]),
                sum(item.weight for item in selected),
                "default",
            )
        elif variant == "score-sorted":
            contexts["tokenpack-score-sorted-50"] = (
                _render_scored_context(selected, order="score"),
                sum(item.weight for item in selected),
                "default",
            )
        elif variant == "density-sorted":
            contexts["tokenpack-density-sorted-50"] = (
                _render_scored_context(selected, order="density"),
                sum(item.weight for item in selected),
                "default",
            )
        elif variant == "score-sorted-strong":
            contexts["tokenpack-score-sorted-strong-50"] = (
                _render_scored_context(selected, order="score"),
                sum(item.weight for item in selected),
                "strict",
            )
        elif variant == "score-sorted-grounded":
            contexts["tokenpack-score-sorted-grounded-50"] = (
                _render_scored_context(selected, order="score"),
                sum(item.weight for item in selected),
                "grounded",
            )
        elif variant == "score-sorted-extractive":
            contexts["tokenpack-score-sorted-extractive-50"] = (
                _render_scored_context(selected, order="score"),
                sum(item.weight for item in selected),
                "extractive",
            )
    return contexts


def _render_scored_context(items: list[ScoredChunk], *, order: str) -> str:
    if order == "score":
        ordered = sorted(items, key=lambda item: item.value, reverse=True)
    elif order == "density":
        ordered = sorted(items, key=lambda item: item.value / max(1, item.weight), reverse=True)
    else:
        ordered = list(items)
    parts: list[str] = []
    for number, item in enumerate(ordered, start=1):
        chunk = item.chunk
        parts.append(
            "[Chunk "
            f"{number}: id={chunk.id}, source={chunk.source_path}, tokens={chunk.token_count}, "
            f"score={item.value:.4f}, density={item.value / max(1, item.weight):.6f}]"
        )
        parts.append(chunk.text)
    return "\n\n".join(parts).strip() + "\n"


def _render_production_rag_context(items: list[ScoredChunk]) -> str:
    ordered = sorted(items, key=lambda item: item.raw_similarity, reverse=True)
    parts: list[str] = []
    for number, item in enumerate(ordered, start=1):
        chunk = item.chunk
        parts.append(
            "[Retrieved Chunk "
            f"{number}: id={chunk.id}, source={chunk.source_path}, tokens={chunk.token_count}, "
            f"similarity={item.raw_similarity:.4f}]"
        )
        parts.append(chunk.text)
    return "\n\n".join(parts).strip() + "\n"


def _make_llmlingua2_backend(args: argparse.Namespace):
    from qasper_compression_eval import _make_compressor_backend

    return _make_compressor_backend(
        model_name=args.llmlingua2_model,
        device_map=args.compression_device_map,
        allow_download=args.compression_allow_download,
        use_llmlingua2=True,
    )


def _load_question_order(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("paper_id") or ""), str(row.get("question_id") or ""))
            if key[0] and key[1] and key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _write_task_report(tasks: list[dict[str, Any]], path: Path) -> None:
    templates = load_prompt_templates(THIS_DIR / "prompts")
    prompts = [
        render_answer_prompt(_answer_template_for_task(templates, task), question=str(task["question"]), context=str(task["context"]))
        for task in tasks
    ]
    by_method: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_method.setdefault(str(task["method"]), []).append(task)
    report = {
        "tasks": len(tasks),
        "questions": len({task["case_id"] for task in tasks}),
        "methods": sorted(by_method),
        "avg_answer_prompt_tokens_est": _mean(prompt_token_estimate(prompt) for prompt in prompts),
        "max_answer_prompt_tokens_est": max([prompt_token_estimate(prompt) for prompt in prompts] or [0]),
        "per_method": {
            method: {
                "tasks": len(rows),
                "avg_context_tokens": _mean(row["context_tokens"] for row in rows),
                "avg_token_saving_vs_full": _mean(
                    1.0 - float(row["context_tokens"]) / max(1.0, float(row["source_tokens"])) for row in rows
                ),
            }
            for method, rows in sorted(by_method.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_results(results_jsonl: Path, output_dir: Path) -> None:
    rows = read_jsonl(results_jsonl)
    summary = summarize_rows(rows)
    pairwise = pairwise_tokenpack_wins(rows)
    write_summary_csv(summary, output_dir / "qasper_generation_quality_summary.csv")
    write_summary_csv(pairwise, output_dir / "qasper_generation_quality_pairwise.csv")
    write_latex_table(summary, output_dir / "qasper_generation_quality_table.tex")
    print(f"Wrote summary assets to {output_dir}")


def _mean(values: Any) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _answer_template_for_task(templates: Any, task: dict[str, Any]) -> str:
    return answer_template_for_variant(templates, str(task.get("answer_prompt_variant") or "default"))


if __name__ == "__main__":
    raise SystemExit(main())
