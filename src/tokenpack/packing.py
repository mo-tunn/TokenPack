from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tokenpack.chunk_profiles import resolve_chunk_size_config
from tokenpack.compression import CompressionConfig, CompressionResult
from tokenpack.embeddings import Embedder
from tokenpack.export import render_compressed_context, render_context
from tokenpack.pipeline import ingest_path
from tokenpack.scoring import DEFAULT_SCORING_PROFILE, score_chunks
from tokenpack.selectors import select_chunks

PACK_RUN_ROOT = ".tokenpack/runs"
AUTO_BUDGET_RATIO = 0.50
AUTO_MIN_BUDGET = 1_200
AUTO_MAX_BUDGET = 64_000
DEFAULT_COMPRESSION_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
DEFAULT_SELECTOR = "budget-top-k"


@dataclass(frozen=True, slots=True)
class PackBudget:
    source_tokens: int
    mode: str
    budget: int
    effective_budget: int
    reserve_output: int
    budget_ratio: float
    min_budget: int
    max_budget: int
    cap_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PackResult:
    source: Path
    output_path: Path
    index_path: Path
    selection_path: Path
    markdown: str
    budget: PackBudget
    selected_chunks: int
    selected_tokens: int
    scoring: str
    selector: str
    compression: str
    compression_result: CompressionResult | None = None

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": str(self.source),
            "output_path": str(self.output_path),
            "index_path": str(self.index_path),
            "selection_path": str(self.selection_path),
            "source_tokens": self.budget.source_tokens,
            "budget_mode": self.budget.mode,
            "budget_ratio": self.budget.budget_ratio,
            "budget": self.budget.budget,
            "reserve_output": self.budget.reserve_output,
            "selection_budget": self.budget.effective_budget,
            "effective_budget": self.budget.effective_budget,
            "selected_chunks": self.selected_chunks,
            "selected_tokens": self.selected_tokens,
            "scoring": self.scoring,
            "selector": self.selector,
            "compression": self.compression,
            "cap_reason": self.budget.cap_reason,
        }
        if self.compression_result is not None:
            payload.update(
                {
                    "compression_origin_tokens": self.compression_result.origin_tokens,
                    "compression_tokens": self.compression_result.compressed_tokens,
                    "compression_saving_rate": self.compression_result.saving_rate,
                }
            )
        return payload


def pack_source(
    *,
    source: str | Path,
    query: str,
    embedder: Embedder,
    out: str | Path | None = None,
    overwrite: bool = False,
    budget: int | None = None,
    budget_ratio: float = AUTO_BUDGET_RATIO,
    min_budget: int = AUTO_MIN_BUDGET,
    max_budget: int = AUTO_MAX_BUDGET,
    reserve_output: int | None = None,
    index_out: str | Path | None = None,
    selection_out: str | Path | None = None,
    candidate_pool: int = 250,
    chunk_size_preset: str = "low-budget",
    target_tokens: int = 650,
    min_tokens: int = 120,
    max_tokens: int = 900,
    source_type: str = "auto",
    compress: str = "none",
    compression_model: str = DEFAULT_COMPRESSION_MODEL,
    compression_rate: float = 0.85,
    compression_target_tokens: int = -1,
    compression_instruction: str = "",
    longllmlingua: bool = False,
    llmlingua2: bool = False,
    compression_device_map: str = "cpu",
    allow_download: bool = False,
    compression_context_filter: bool = False,
    compression_sentence_filter: bool = False,
    no_compression_token_filter: bool = False,
    run_root: str | Path = PACK_RUN_ROOT,
) -> PackResult:
    source_path = Path(source)
    if not source_path.exists():
        raise ValueError(f"Source does not exist: {source_path}")
    output_path = _infer_pack_output_path(source_path, str(out) if out is not None else None)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}\nUse --overwrite or choose --out.")
    if compress not in {"none", "llmlingua"}:
        raise ValueError(f"Unknown compressor: {compress}")

    run_dir = _pack_run_dir(source_path, run_root=run_root)
    index_path = Path(index_out) if index_out else run_dir / "index.json"
    selection_path = Path(selection_out) if selection_out else run_dir / "selection.json"

    chunk_size = resolve_chunk_size_config(
        chunk_size_preset,
        target_tokens,
        min_tokens,
        max_tokens,
    )
    index = ingest_path(
        source_path,
        index_path,
        embedder=embedder,
        target_tokens=chunk_size.target_tokens,
        min_tokens=chunk_size.min_tokens,
        max_tokens=chunk_size.max_tokens,
        chunker_name="structure-aware",
        source_type=source_type,
    )
    source_tokens = sum(max(0, chunk.token_count) for chunk in index.chunks)
    resolved_budget = _resolve_pack_budget(
        source_tokens=source_tokens,
        budget=budget,
        budget_ratio=budget_ratio,
        min_budget=min_budget,
        max_budget=max_budget,
        reserve_output=reserve_output,
    )

    query_embedding = embedder.embed([query])[0]
    scored = score_chunks(
        query_embedding,
        index.chunks,
        index.embeddings,
        redundancy_penalty=0.0,
        scoring=DEFAULT_SCORING_PROFILE,
        query_text=query,
        redundancy_candidate_pool=candidate_pool,
    )
    result = select_chunks(
        scored,
        strategy=DEFAULT_SELECTOR,
        budget=resolved_budget.effective_budget,
        candidate_pool=candidate_pool,
        embeddings=index.embeddings,
        coverage_query=query,
    )
    payload = result.to_dict() | {
        "scoring": DEFAULT_SCORING_PROFILE,
        "source": str(source_path),
        "output": str(output_path),
        "source_tokens": resolved_budget.source_tokens,
        "budget_mode": resolved_budget.mode,
        "budget_ratio": resolved_budget.budget_ratio,
        "budget": resolved_budget.budget,
        "reserve_output": resolved_budget.reserve_output,
        "effective_budget": resolved_budget.effective_budget,
        "min_budget": resolved_budget.min_budget,
        "max_budget": resolved_budget.max_budget,
        "cap_reason": resolved_budget.cap_reason,
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    chunks = [item.chunk for item in result.selected]
    compression_result = None
    if compress == "llmlingua":
        compression_config = CompressionConfig(
            compressor="llmlingua",
            model_name=compression_model,
            rate=compression_rate,
            target_tokens=compression_target_tokens,
            question=query,
            instruction=compression_instruction,
            longllmlingua=longllmlingua,
            llmlingua2=llmlingua2,
            use_context_level_filter=compression_context_filter,
            use_sentence_level_filter=compression_sentence_filter,
            use_token_level_filter=not no_compression_token_filter,
            device_map=compression_device_map,
            local_files_only=not allow_download,
        )
        context, compression_result = render_compressed_context(chunks, compression_config, include_headers=True)
    else:
        context = render_context(chunks, include_headers=True)

    markdown = (
        _render_pack_markdown_header(
            source=source_path,
            output_path=output_path,
            query=query,
            budget=resolved_budget,
            selected_chunks=len(result.selected),
            selected_tokens=result.used_tokens,
            index_path=index_path,
            selection_path=selection_path,
            compression=compress,
            compression_result=compression_result,
        )
        + context
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    return PackResult(
        source=source_path,
        output_path=output_path,
        index_path=index_path,
        selection_path=selection_path,
        markdown=markdown,
        budget=resolved_budget,
        selected_chunks=len(result.selected),
        selected_tokens=result.used_tokens,
        scoring=DEFAULT_SCORING_PROFILE,
        selector=DEFAULT_SELECTOR,
        compression=compress,
        compression_result=compression_result,
    )


def format_pack_summary(result: PackResult) -> list[str]:
    lines = [
        f"Source: {result.source}",
        f"Output: {result.output_path}",
        f"Source tokens: {_fmt_int(result.budget.source_tokens)}",
    ]
    if result.budget.mode == "auto":
        ratio = f"{result.budget.budget_ratio:.0%}"
        cap = f", capped by {result.budget.cap_reason}" if result.budget.cap_reason else ""
        lines.append(f"Auto budget: {_fmt_int(result.budget.budget)} tokens (ratio={ratio}{cap})")
    else:
        lines.append(f"Manual budget: {_fmt_int(result.budget.budget)} tokens")
    lines.extend(
        [
            f"Reserved for answer: {_fmt_int(result.budget.reserve_output)}",
            f"Selection budget: {_fmt_int(result.budget.effective_budget)}",
            f"Selected: {result.selected_chunks} chunks / {_fmt_int(result.selected_tokens)} tokens",
        ]
    )
    if result.compression_result is not None:
        lines.append(
            "Compression: "
            f"{_fmt_int(result.compression_result.origin_tokens)}->"
            f"{_fmt_int(result.compression_result.compressed_tokens)} tokens, "
            f"saving={result.compression_result.saving_rate:.1%}"
        )
    return lines


def _infer_pack_output_path(source: str | Path, out: str | None = None) -> Path:
    if out:
        return Path(out)
    source_path = Path(source)
    if source_path.is_dir():
        name = source_path.name or "tokenpack"
        return source_path.parent / f"{name}-tp.md"
    return source_path.with_name(f"{source_path.stem}-tp.md")


def _pack_run_dir(source: Path, run_root: str | Path = PACK_RUN_ROOT) -> Path:
    stem = source.name if source.is_dir() else source.stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-") or "source"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return Path(run_root) / f"{safe_stem}-{timestamp}"


def _resolve_pack_budget(
    *,
    source_tokens: int,
    budget: int | None,
    budget_ratio: float,
    min_budget: int,
    max_budget: int,
    reserve_output: int | None,
) -> PackBudget:
    if source_tokens < 0:
        raise ValueError("source_tokens must be non-negative.")
    if budget_ratio <= 0:
        raise ValueError("--budget-ratio must be greater than 0.")
    if min_budget <= 0:
        raise ValueError("--min-budget must be greater than 0.")
    if max_budget < min_budget:
        raise ValueError("--max-budget must be greater than or equal to --min-budget.")

    cap_reason = None
    if budget is None:
        raw_budget = math.ceil(source_tokens * budget_ratio)
        resolved_budget = max(min_budget, min(max_budget, raw_budget))
        if raw_budget > max_budget:
            cap_reason = "max-budget"
        elif raw_budget < min_budget:
            cap_reason = "min-budget"
        mode = "auto"
    else:
        if budget <= 0:
            raise ValueError("--budget must be greater than 0.")
        resolved_budget = budget
        mode = "manual"

    resolved_reserve = reserve_output if reserve_output is not None else _auto_reserve_output(resolved_budget)
    if resolved_reserve < 0:
        raise ValueError("--reserve-output must be greater than or equal to 0.")
    effective_budget = max(0, resolved_budget - resolved_reserve)
    return PackBudget(
        source_tokens=source_tokens,
        mode=mode,
        budget=resolved_budget,
        effective_budget=effective_budget,
        reserve_output=resolved_reserve,
        budget_ratio=budget_ratio,
        min_budget=min_budget,
        max_budget=max_budget,
        cap_reason=cap_reason,
    )


def _auto_reserve_output(budget: int) -> int:
    return min(4_000, max(512, int(budget * 0.10)))


def _render_pack_markdown_header(
    *,
    source: Path,
    output_path: Path,
    query: str,
    budget: PackBudget,
    selected_chunks: int,
    selected_tokens: int,
    index_path: Path,
    selection_path: Path,
    compression: str,
    compression_result,
) -> str:
    compression_value = compression
    if compression_result is not None:
        compression_value = (
            f"{compression} ({compression_result.origin_tokens}->{compression_result.compressed_tokens} tokens, "
            f"saving={compression_result.saving_rate:.1%})"
        )
    rows = [
        ("Source", str(source)),
        ("Output", str(output_path)),
        ("Query", query),
        ("Source tokens", _fmt_int(budget.source_tokens)),
        ("Budget mode", budget.mode),
        ("Budget", _fmt_int(budget.budget)),
        ("Reserved for answer", _fmt_int(budget.reserve_output)),
        ("Selection budget", _fmt_int(budget.effective_budget)),
        ("Selected", f"{selected_chunks} chunks / {_fmt_int(selected_tokens)} tokens"),
        ("Scoring", DEFAULT_SCORING_PROFILE),
        ("Selector", DEFAULT_SELECTOR),
        ("Compression", compression_value),
        ("Index artifact", str(index_path)),
        ("Selection artifact", str(selection_path)),
    ]
    if budget.mode == "auto":
        ratio = f"{budget.budget_ratio:.0%}"
        detail = f"ratio={ratio}"
        if budget.cap_reason:
            detail += f", capped by {budget.cap_reason}"
        rows.insert(5, ("Auto budget detail", detail))
    table = "\n".join(f"| {_md_table_cell(key)} | {_md_table_cell(value)} |" for key, value in rows)
    return (
        "# TokenPack Packed Context\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        f"{table}\n\n"
        "## Context\n\n"
    )


def _md_table_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _fmt_int(value: int) -> str:
    return f"{value:,}"
