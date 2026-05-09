from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chunking_ablation import (  # type: ignore
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DEFAULT_WORK_DIR,
    _avg,
    _build_index,
    _content_terms,
    _fmt,
    _latex_text,
    _load_experiment_blocks,
    _parse_budgets,
    _redundancy_score,
    _templates_from_blocks,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenpack.embeddings import make_embedder
from tokenpack.scoring import SCORING_PROFILES, score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter


STRATEGIES = ["budget-top-k", "greedy-density", "knapsack"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare budget-safe selectors under the same scoring function.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--backend", default="hash", choices=["auto", "hash", "sentence-transformers"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["paragraph", "semantic-threshold"], default="semantic-threshold")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="hybrid",
    )
    parser.add_argument("--budget-ratios", default="0.01,0.03,0.05")
    parser.add_argument("--budgets", help="Optional comma-separated absolute budgets; overrides --budget-ratios.")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--base-block-tokens", type=int, default=90)
    parser.add_argument("--min-evidence-terms", type=int, default=12)
    parser.add_argument("--max-documents", type=int, default=8)
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter()
    embedder = make_embedder(backend=args.backend, model_name=args.model, local_files_only=True)
    blocks = _load_experiment_blocks(
        Path(args.source),
        max_documents=args.max_documents,
        base_block_tokens=args.base_block_tokens,
        token_counter=token_counter,
    )
    if not blocks:
        raise SystemExit("No supported source blocks found for selector ablation.")

    total_source_tokens = sum(token_counter.count(block.text) for block in blocks)
    budgets = _parse_budgets(args.budgets, args.budget_ratios, total_source_tokens)
    templates = _templates_from_blocks(
        blocks,
        sample_size=args.sample_size,
        min_evidence_terms=args.min_evidence_terms,
    )
    if not templates:
        raise SystemExit("No evidence templates could be derived from the source blocks.")

    index = _build_index(
        chunker_name=args.chunker,
        blocks=blocks,
        embedder=embedder,
        work_dir=work_dir,
        target_tokens=args.target_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        semantic_threshold=args.semantic_threshold,
        token_counter=token_counter,
    )

    rows: list[dict[str, object]] = []
    for strategy in STRATEGIES:
        for budget in budgets:
            metrics = _evaluate_strategy(
                index=index,
                templates=templates,
                embedder=embedder,
                budget=budget,
                candidate_pool=args.candidate_pool,
                scoring=args.scoring,
                strategy=strategy,
            )
            rows.append(
                {
                    "strategy": strategy,
                    "chunker": args.chunker,
                    "scoring": args.scoring,
                    "budget_ratio": budget / max(1, total_source_tokens),
                    "budget": budget,
                    "document_count": len({block.source_path for block in blocks}),
                    "evidence_count": len(templates),
                    "chunk_count": len(index.chunks),
                    "avg_chunk_tokens": _avg(chunk.token_count for chunk in index.chunks),
                    **metrics,
                }
            )

    csv_path = output_dir / "selector_ablation.csv"
    table_path = output_dir / "selector_ablation_table.tex"
    paper_table_path = ROOT / "submission" / "paper" / "tables" / "selector_ablation_table.tex"
    _write_csv(rows, csv_path)
    _write_latex(rows, table_path)
    _write_latex(rows, paper_table_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")
    return 0


def _evaluate_strategy(
    index,
    templates,
    embedder,
    budget: int,
    candidate_pool: int,
    scoring: str,
    strategy: str,
) -> dict[str, float]:
    totals = {
        "evidence_term_recall": 0.0,
        "complete_evidence_rate": 0.0,
        "avg_used_tokens": 0.0,
        "budget_utilization": 0.0,
        "over_budget_rate": 0.0,
        "total_value": 0.0,
        "redundancy_score": 0.0,
        "latency_seconds": 0.0,
    }
    for template in templates:
        query_embedding = embedder.embed([template.query])[0]
        scored = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            scoring=scoring,
            query_text=template.query,
        )
        started = time.perf_counter()
        result = select_chunks(
            scored,
            strategy=strategy,
            budget=budget,
            candidate_pool=candidate_pool,
            embeddings=index.embeddings,
        )
        elapsed = time.perf_counter() - started
        selected_terms = set(_content_terms(" ".join(item.chunk.text for item in result.selected)))
        recall = len(template.evidence_terms & selected_terms) / max(1, len(template.evidence_terms))
        totals["evidence_term_recall"] += recall
        totals["complete_evidence_rate"] += 1.0 if recall >= 0.80 else 0.0
        totals["avg_used_tokens"] += result.used_tokens
        totals["budget_utilization"] += result.used_tokens / max(1, budget)
        totals["over_budget_rate"] += 1.0 if result.used_tokens > budget else 0.0
        totals["total_value"] += result.total_value
        totals["redundancy_score"] += _redundancy_score([item.chunk for item in result.selected])
        totals["latency_seconds"] += elapsed

    runs = max(1, len(templates))
    return {key: value / runs for key, value in totals.items()}


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "strategy",
        "chunker",
        "scoring",
        "budget_ratio",
        "budget",
        "document_count",
        "evidence_count",
        "chunk_count",
        "avg_chunk_tokens",
        "evidence_term_recall",
        "complete_evidence_rate",
        "avg_used_tokens",
        "budget_utilization",
        "over_budget_rate",
        "total_value",
        "redundancy_score",
        "latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Selector Ablation With Fixed Chunking and Value Function}",
        r"\label{tab:selector-ablation}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Selector & Budget & Term Recall & Complete & Avg. Used & Util. & Value & Lat. \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_text(str(row["strategy"])),
                    f"{float(row['budget_ratio']) * 100:.0f}\\%",
                    _fmt(row["evidence_term_recall"]),
                    _fmt(row["complete_evidence_rate"]),
                    _fmt(row["avg_used_tokens"]),
                    _fmt(row["budget_utilization"]),
                    _fmt(row["total_value"]),
                    _fmt(row["latency_seconds"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
