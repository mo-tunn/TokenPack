from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qasper_selector_eval import (  # noqa: E402
    BudgetSpec,
    QasperQuestion,
    _blocks_from_qasper_row,
    _build_index,
    _csv_list,
    _evidence_recall,
    _load_qasper_rows,
    _parse_budgets,
    _questions_from_qasper_row,
    _token_f1,
)
from tokenpack.chunk_profiles import resolve_chunk_size_config  # noqa: E402
from tokenpack.compression import CompressionConfig, compress_chunks  # noqa: E402
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.models import Chunk, SelectionResult  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.selectors import select_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "qasper_compression_eval"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "qasper_compression_eval"
DEFAULT_PIPELINES = "only-tokenpack,only-llmlingua2,tokenpack+llmlingua2"
PIPELINE_ALIASES = {
    "only-tokenpack": "only-tokenpack",
    "tokenpack": "only-tokenpack",
    "only-llmlingua2": "only-llmlingua2",
    "llmlingua2": "only-llmlingua2",
    "only-longllmlingua": "only-longllmlingua",
    "longllmlingua": "only-longllmlingua",
    "tokenpack+llmlingua2": "tokenpack+llmlingua2",
    "tokenpack-llmlingua2": "tokenpack+llmlingua2",
    "tokenpack+longllmlingua": "tokenpack+longllmlingua",
    "tokenpack-longllmlingua": "tokenpack+longllmlingua",
}


@dataclass(frozen=True, slots=True)
class CompressionSetting:
    variant: str
    mode: str
    rate: float | None = None
    target_tokens: int | None = None

    @property
    def label(self) -> str:
        if self.mode == "rate":
            return f"rate={self.rate:.2f}"
        return f"target={self.target_tokens}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate QASPER evidence retention for Only TokenPack, Only LLMLingua, "
            "and TokenPack -> LLMLingua pipelines."
        )
    )
    parser.add_argument("--data-file", help="Local QASPER parquet/json/jsonl file.")
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--backend", default="hash", choices=["auto", "hash", "sentence-transformers"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["paragraph", "semantic-threshold", "structure-aware"], default="structure-aware")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="evidence-hybrid",
    )
    parser.add_argument("--pipelines", default=DEFAULT_PIPELINES)
    parser.add_argument("--selection-strategies", default="knapsack-redundancy")
    parser.add_argument("--budget-ratios", default="0.50")
    parser.add_argument("--budgets")
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--chunk-size-preset", choices=["manual", "default", "low-budget"], default="low-budget")
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument(
        "--llmlingua2-model",
        default="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    )
    parser.add_argument("--longllmlingua-model")
    parser.add_argument("--compression-rates", default="0.50")
    parser.add_argument("--compression-targets")
    parser.add_argument("--compression-device-map", default="cpu")
    parser.add_argument("--compression-allow-download", action="store_true")
    parser.add_argument("--compression-instruction", default="")
    parser.add_argument("--compression-no-question", action="store_true")
    parser.add_argument("--compression-context-level-filter", action="store_true")
    parser.add_argument("--compression-sentence-level-filter", action="store_true")
    parser.add_argument("--disable-token-level-filter", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    pipelines = _parse_pipeline_aliases(args.pipelines)
    selection_strategies = _csv_list(args.selection_strategies)
    compression_settings = _parse_compression_settings(
        raw_rates=args.compression_rates,
        raw_targets=args.compression_targets,
        pipelines=pipelines,
    )
    if any("llmlingua" in pipeline for pipeline in pipelines) and not compression_settings:
        raise ValueError("Compression pipelines require at least one --compression-rates or --compression-targets value.")

    rows = list(_load_qasper_rows(args.data_file, args.split))
    token_counter = TokenCounter()
    embedder = make_embedder(backend=args.backend, model_name=args.model, local_files_only=True)
    chunk_size = resolve_chunk_size_config(args.chunk_size_preset, args.target_tokens, args.min_tokens, args.max_tokens)
    compressor_pool = _CompressorPool(args)

    raw_rows: list[dict[str, Any]] = []
    processed_papers = 0
    processed_questions = 0

    for row in rows:
        if processed_papers >= args.max_papers or processed_questions >= args.max_questions:
            break
        paper_id = str(row.get("id") or row.get("paper_id") or processed_papers)
        blocks = _blocks_from_qasper_row(row, document_index=processed_papers, token_counter=token_counter)
        questions = _questions_from_qasper_row(row)
        if not blocks or not questions:
            continue
        index = _build_index(
            paper_id=paper_id,
            blocks=blocks,
            embedder=embedder,
            work_dir=work_dir,
            chunker_name=args.chunker,
            target_tokens=chunk_size.target_tokens,
            min_tokens=chunk_size.min_tokens,
            max_tokens=chunk_size.max_tokens,
            semantic_threshold=args.semantic_threshold,
            token_counter=token_counter,
        )
        ordered_chunks = sorted(index.chunks, key=lambda chunk: chunk.order_key)
        source_tokens = sum(chunk.token_count for chunk in ordered_chunks)
        full_context_text = " ".join(chunk.text for chunk in ordered_chunks)
        full_context_evidence_cache: dict[str, float] = {}
        full_context_answer_f1_cache: dict[str, float] = {}
        budgets = _parse_budgets(args.budgets, args.budget_ratios, source_tokens)
        processed_papers += 1

        for question in questions:
            if processed_questions >= args.max_questions:
                break
            processed_questions += 1
            full_context_evidence_cache[question.question_id] = _evidence_recall(question.evidence_texts, full_context_text)
            full_context_answer_f1_cache[question.question_id] = _token_f1(full_context_text, question.answer)
            query_embedding = embedder.embed([question.question])[0]
            scored = score_chunks(
                query_embedding,
                index.chunks,
                index.embeddings,
                scoring=args.scoring,
                query_text=question.question,
                redundancy_candidate_pool=args.candidate_pool,
            )
            selection_cache = _selection_cache(scored, budgets, selection_strategies, args.candidate_pool)

            for pipeline in pipelines:
                if pipeline == "only-tokenpack":
                    for budget_spec in budgets:
                        for selection_strategy in selection_strategies:
                            selection = selection_cache[(selection_strategy, budget_spec.label)]
                            raw_rows.append(
                                _build_result_row(
                                    paper_id=paper_id,
                                    question=question,
                                    pipeline=pipeline,
                                    selection_strategy=selection_strategy,
                                    budget_spec=budget_spec,
                                    compression_setting=None,
                                    source_tokens=source_tokens,
                                    selected_tokens=selection.used_tokens,
                                    final_tokens=selection.used_tokens,
                                    selected_text=" ".join(item.chunk.text for item in selection.selected),
                                    final_text=" ".join(item.chunk.text for item in selection.selected),
                                    selection_seconds=selection.elapsed_seconds,
                                    compression_seconds=0.0,
                                )
                            )
                    continue

                if pipeline in {"only-llmlingua2", "only-longllmlingua"}:
                    variant = "llmlingua2" if pipeline == "only-llmlingua2" else "longllmlingua"
                    for compression_setting in _variant_settings(compression_settings, variant):
                        compressed, compression_seconds = _compress_context(
                            chunks=ordered_chunks,
                            args=args,
                            question=question.question,
                            compression_setting=compression_setting,
                            compressor_pool=compressor_pool,
                            token_counter=token_counter,
                        )
                        raw_rows.append(
                            _build_result_row(
                                paper_id=paper_id,
                                question=question,
                                pipeline=pipeline,
                                selection_strategy="full-document",
                                budget_spec=None,
                                compression_setting=compression_setting,
                                source_tokens=source_tokens,
                                selected_tokens=source_tokens,
                                final_tokens=compressed.compressed_tokens,
                                selected_text=full_context_text,
                                final_text=compressed.compressed_prompt,
                                selection_seconds=0.0,
                                compression_seconds=compression_seconds,
                            )
                        )
                    continue

                if pipeline in {"tokenpack+llmlingua2", "tokenpack+longllmlingua"}:
                    variant = "llmlingua2" if pipeline == "tokenpack+llmlingua2" else "longllmlingua"
                    for budget_spec in budgets:
                        for selection_strategy in selection_strategies:
                            selection = selection_cache[(selection_strategy, budget_spec.label)]
                            selected_chunks = [item.chunk for item in selection.selected]
                            selected_text = " ".join(chunk.text for chunk in selected_chunks)
                            for compression_setting in _variant_settings(compression_settings, variant):
                                compressed, compression_seconds = _compress_context(
                                    chunks=selected_chunks,
                                    args=args,
                                    question=question.question,
                                    compression_setting=compression_setting,
                                    compressor_pool=compressor_pool,
                                    token_counter=token_counter,
                                )
                                raw_rows.append(
                                    _build_result_row(
                                        paper_id=paper_id,
                                        question=question,
                                        pipeline=pipeline,
                                        selection_strategy=selection_strategy,
                                        budget_spec=budget_spec,
                                        compression_setting=compression_setting,
                                        source_tokens=source_tokens,
                                        selected_tokens=selection.used_tokens,
                                        final_tokens=compressed.compressed_tokens,
                                        selected_text=selected_text,
                                        final_text=compressed.compressed_prompt,
                                        selection_seconds=selection.elapsed_seconds,
                                        compression_seconds=compression_seconds,
                                    )
                                )
                    continue

                raise ValueError(f"Unsupported pipeline: {pipeline}")

    summary_rows = _summarize(raw_rows, processed_papers=processed_papers, processed_questions=processed_questions)
    raw_path = output_dir / "qasper_compression_eval_raw.csv"
    summary_path = output_dir / "qasper_compression_eval_summary.csv"
    _write_csv(raw_rows, raw_path)
    _write_csv(summary_rows, summary_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Processed papers={processed_papers}, questions={processed_questions}, runs={len(raw_rows)}")
    return 0


def _parse_pipeline_aliases(raw: str) -> list[str]:
    pipelines: list[str] = []
    for item in _csv_list(raw):
        canonical = PIPELINE_ALIASES.get(item.strip().lower())
        if canonical is None:
            raise ValueError(
                "Unknown pipeline "
                f"{item!r}. Supported values: {', '.join(sorted(set(PIPELINE_ALIASES.values())))}"
            )
        if canonical not in pipelines:
            pipelines.append(canonical)
    return pipelines


def _parse_compression_settings(
    raw_rates: str,
    raw_targets: str | None,
    pipelines: list[str],
) -> list[CompressionSetting]:
    needed_variants = {
        "llmlingua2"
        if "llmlingua2" in pipeline
        else "longllmlingua"
        for pipeline in pipelines
        if "llmlingua" in pipeline
    }
    settings: list[CompressionSetting] = []
    rates = [float(value) for value in _csv_list(raw_rates)] if raw_rates.strip() else []
    targets = [int(value) for value in _csv_list(raw_targets)] if raw_targets else []
    for variant in sorted(needed_variants):
        for rate in rates:
            settings.append(CompressionSetting(variant=variant, mode="rate", rate=rate))
        for target_tokens in targets:
            settings.append(CompressionSetting(variant=variant, mode="target", target_tokens=target_tokens))
    return settings


def _variant_settings(settings: list[CompressionSetting], variant: str) -> list[CompressionSetting]:
    return [setting for setting in settings if setting.variant == variant]


def _selection_cache(
    scored,
    budgets: list[BudgetSpec],
    selection_strategies: list[str],
    candidate_pool: int,
) -> dict[tuple[str, str], SelectionResult]:
    cache: dict[tuple[str, str], SelectionResult] = {}
    for budget_spec in budgets:
        for selection_strategy in selection_strategies:
            cache[(selection_strategy, budget_spec.label)] = select_chunks(
                scored,
                strategy=selection_strategy,
                budget=budget_spec.tokens,
                candidate_pool=candidate_pool,
            )
    return cache


def _build_result_row(
    *,
    paper_id: str,
    question: QasperQuestion,
    pipeline: str,
    selection_strategy: str,
    budget_spec: BudgetSpec | None,
    compression_setting: CompressionSetting | None,
    source_tokens: int,
    selected_tokens: int,
    final_tokens: int,
    selected_text: str,
    final_text: str,
    selection_seconds: float,
    compression_seconds: float,
) -> dict[str, Any]:
    selected_evidence_recall = _evidence_recall(question.evidence_texts, selected_text)
    final_evidence_recall = _evidence_recall(question.evidence_texts, final_text)
    selected_answer_token_f1 = _token_f1(selected_text, question.answer)
    final_answer_token_f1 = _token_f1(final_text, question.answer)
    total_seconds = selection_seconds + compression_seconds
    compression_variant = compression_setting.variant if compression_setting else "none"
    compression_mode = compression_setting.mode if compression_setting else "none"
    compression_rate = compression_setting.rate if compression_setting and compression_setting.mode == "rate" else ""
    compression_target_tokens = (
        compression_setting.target_tokens
        if compression_setting and compression_setting.mode == "target"
        else ""
    )
    budget_label = budget_spec.label if budget_spec else ""
    budget_tokens = budget_spec.tokens if budget_spec else ""
    return {
        "paper_id": paper_id,
        "question_id": question.question_id,
        "pipeline": pipeline,
        "setting_label": _setting_label(pipeline, selection_strategy, budget_spec, compression_setting),
        "selection_strategy": selection_strategy,
        "budget": budget_label,
        "budget_tokens": budget_tokens,
        "compression_variant": compression_variant,
        "compression_mode": compression_mode,
        "compression_rate": compression_rate,
        "compression_target_tokens": compression_target_tokens,
        "source_tokens": source_tokens,
        "selected_tokens": selected_tokens,
        "final_tokens": final_tokens,
        "selection_saving_vs_full": 1.0 - selected_tokens / max(1, source_tokens),
        "compression_saving_vs_selected": 1.0 - final_tokens / max(1, selected_tokens),
        "total_saving_vs_full": 1.0 - final_tokens / max(1, source_tokens),
        "selected_evidence_recall": selected_evidence_recall,
        "final_evidence_recall": final_evidence_recall,
        "evidence_recall_delta_after_compression": final_evidence_recall - selected_evidence_recall,
        "selected_complete_evidence": 1.0 if selected_evidence_recall >= 0.80 else 0.0,
        "final_complete_evidence": 1.0 if final_evidence_recall >= 0.80 else 0.0,
        "selected_answer_token_f1": selected_answer_token_f1,
        "final_answer_token_f1": final_answer_token_f1,
        "answer_token_f1_delta_after_compression": final_answer_token_f1 - selected_answer_token_f1,
        "compression_helped_evidence": 1.0 if final_evidence_recall > selected_evidence_recall else 0.0,
        "compression_hurt_evidence": 1.0 if final_evidence_recall < selected_evidence_recall else 0.0,
        "compression_helped_answer_token_f1": 1.0 if final_answer_token_f1 > selected_answer_token_f1 else 0.0,
        "compression_hurt_answer_token_f1": 1.0 if final_answer_token_f1 < selected_answer_token_f1 else 0.0,
        "selection_seconds": selection_seconds,
        "compression_seconds": compression_seconds,
        "total_seconds": total_seconds,
    }


def _setting_label(
    pipeline: str,
    selection_strategy: str,
    budget_spec: BudgetSpec | None,
    compression_setting: CompressionSetting | None,
) -> str:
    parts = [pipeline]
    if selection_strategy and selection_strategy != "full-document":
        parts.append(selection_strategy)
    if budget_spec is not None:
        parts.append(f"budget={budget_spec.label}")
    if compression_setting is not None:
        parts.append(compression_setting.label)
    return " / ".join(parts)


def _compress_context(
    *,
    chunks: list[Chunk],
    args: argparse.Namespace,
    question: str,
    compression_setting: CompressionSetting,
    compressor_pool: "_CompressorPool",
    token_counter: TokenCounter,
):
    config = _compression_config(args, question, compression_setting)
    backend = compressor_pool.get(compression_setting.variant)
    started = time.perf_counter()
    result = compress_chunks(
        chunks,
        config,
        backend=backend,
        token_counter=token_counter,
    )
    elapsed = time.perf_counter() - started
    return result, elapsed


def _compression_config(
    args: argparse.Namespace,
    question: str,
    compression_setting: CompressionSetting,
) -> CompressionConfig:
    target_tokens = compression_setting.target_tokens if compression_setting.mode == "target" else -1
    rate = compression_setting.rate if compression_setting.mode == "rate" else 0.5
    use_question = "" if args.compression_no_question else question
    return CompressionConfig(
        compressor="llmlingua",
        model_name=_model_name_for_variant(args, compression_setting.variant),
        rate=rate,
        target_tokens=target_tokens,
        question=use_question,
        instruction=args.compression_instruction,
        longllmlingua=compression_setting.variant == "longllmlingua",
        llmlingua2=compression_setting.variant == "llmlingua2",
        use_context_level_filter=args.compression_context_level_filter,
        use_sentence_level_filter=args.compression_sentence_level_filter,
        use_token_level_filter=not args.disable_token_level_filter,
        device_map=args.compression_device_map,
        local_files_only=not args.compression_allow_download,
    )


def _model_name_for_variant(args: argparse.Namespace, variant: str) -> str:
    if variant == "llmlingua2":
        return args.llmlingua2_model
    return args.longllmlingua_model or args.llmlingua2_model


class _CompressorPool:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._backends: dict[str, Any] = {}

    def get(self, variant: str):
        backend = self._backends.get(variant)
        if backend is None:
            backend = _make_compressor_backend(
                model_name=_model_name_for_variant(self._args, variant),
                device_map=self._args.compression_device_map,
                allow_download=self._args.compression_allow_download,
                use_llmlingua2=variant == "llmlingua2",
            )
            self._backends[variant] = backend
        return backend


def _make_compressor_backend(
    *,
    model_name: str,
    device_map: str,
    allow_download: bool,
    use_llmlingua2: bool,
):
    from llmlingua import PromptCompressor

    resolved_name = model_name
    if not allow_download:
        from tokenpack.compression import _resolve_local_model_path

        resolved_name = _resolve_local_model_path(model_name)
    return PromptCompressor(
        model_name=resolved_name,
        device_map=device_map,
        model_config={"local_files_only": not allow_download},
        use_llmlingua2=use_llmlingua2,
    )


def _summarize(
    rows: list[dict[str, Any]],
    *,
    processed_papers: int,
    processed_questions: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["pipeline"]), str(row["setting_label"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (_, _), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        sample = group_rows[0]
        summary.append(
            {
                "pipeline": sample["pipeline"],
                "setting_label": sample["setting_label"],
                "selection_strategy": sample["selection_strategy"],
                "budget": sample["budget"],
                "budget_tokens": sample["budget_tokens"],
                "compression_variant": sample["compression_variant"],
                "compression_mode": sample["compression_mode"],
                "compression_rate": sample["compression_rate"],
                "compression_target_tokens": sample["compression_target_tokens"],
                "processed_papers": processed_papers,
                "processed_questions": processed_questions,
                "runs": len(group_rows),
                "avg_source_tokens": _avg(group_rows, "source_tokens"),
                "avg_selected_tokens": _avg(group_rows, "selected_tokens"),
                "avg_final_tokens": _avg(group_rows, "final_tokens"),
                "selection_saving_vs_full": _avg(group_rows, "selection_saving_vs_full"),
                "compression_saving_vs_selected": _avg(group_rows, "compression_saving_vs_selected"),
                "total_saving_vs_full": _avg(group_rows, "total_saving_vs_full"),
                "selected_evidence_recall": _avg(group_rows, "selected_evidence_recall"),
                "final_evidence_recall": _avg(group_rows, "final_evidence_recall"),
                "evidence_recall_delta_after_compression": _avg(
                    group_rows, "evidence_recall_delta_after_compression"
                ),
                "selected_complete_evidence_rate": _avg(group_rows, "selected_complete_evidence"),
                "final_complete_evidence_rate": _avg(group_rows, "final_complete_evidence"),
                "selected_answer_token_f1": _avg(group_rows, "selected_answer_token_f1"),
                "final_answer_token_f1": _avg(group_rows, "final_answer_token_f1"),
                "answer_token_f1_delta_after_compression": _avg(
                    group_rows, "answer_token_f1_delta_after_compression"
                ),
                "compression_helped_evidence_rate": _avg(group_rows, "compression_helped_evidence"),
                "compression_hurt_evidence_rate": _avg(group_rows, "compression_hurt_evidence"),
                "compression_helped_answer_token_f1_rate": _avg(
                    group_rows, "compression_helped_answer_token_f1"
                ),
                "compression_hurt_answer_token_f1_rate": _avg(
                    group_rows, "compression_hurt_answer_token_f1"
                ),
                "avg_selection_seconds": _avg(group_rows, "selection_seconds"),
                "avg_compression_seconds": _avg(group_rows, "compression_seconds"),
                "avg_total_seconds": _avg(group_rows, "total_seconds"),
            }
        )
    return summary


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
