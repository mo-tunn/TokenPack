from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chunking_ablation import (  # type: ignore
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
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenpack.embeddings import make_embedder
from tokenpack.export import render_context
from tokenpack.generation import _cerebras_answer, _default_ollama_model, _groq_answer, _ollama_answer, _openai_answer
from tokenpack.scoring import SCORING_PROFILES, score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter


DEFAULT_GOLD = ROOT / "submission" / "gold" / "resources_reviewed_qa.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "reviewed_qa"
STRATEGIES = ["budget-top-k", "greedy-density", "knapsack"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selectors on reviewed resources QA records.")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--backend", default="hash", choices=["auto", "hash", "sentence-transformers"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["paragraph", "semantic-threshold", "structure-aware"], default="semantic-threshold")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="hybrid",
    )
    parser.add_argument("--budget-ratios", default="0.01,0.03,0.05")
    parser.add_argument("--budgets", help="Optional comma-separated absolute budgets; overrides --budget-ratios.")
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--base-block-tokens", type=int, default=90)
    parser.add_argument("--max-documents", type=int, default=8)
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--write-contexts", action="store_true")
    parser.add_argument("--provider", choices=["none", "openai", "ollama", "local", "cerebras", "groq"], default="none")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    records = _load_records(Path(args.gold), include_unreviewed=args.include_unreviewed)
    if not records:
        raise SystemExit("No reviewed QA records found. Run review_resources_qa.py first, or pass --include-unreviewed.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = output_dir / "reviewed_qa_answers.jsonl"
    if args.provider != "none" and answer_path.exists():
        answer_path.unlink()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter()
    embedder = make_embedder(backend=args.backend, model_name=args.model, local_files_only=True)
    blocks = _load_experiment_blocks(
        Path(args.source),
        max_documents=args.max_documents,
        base_block_tokens=args.base_block_tokens,
        token_counter=token_counter,
    )
    total_source_tokens = sum(token_counter.count(block.text) for block in blocks)
    budgets = _parse_budgets(args.budgets, args.budget_ratios, total_source_tokens)
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
            rows.append(
                {
                    "strategy": strategy,
                    "chunker": args.chunker,
                    "scoring": args.scoring,
                    "budget_ratio": budget / max(1, total_source_tokens),
                    "budget": budget,
                    "qa_count": len(records),
                    "chunk_count": len(index.chunks),
                    "avg_chunk_tokens": _avg(chunk.token_count for chunk in index.chunks),
                    **_evaluate(
                        records=records,
                        index=index,
                        embedder=embedder,
                        strategy=strategy,
                        budget=budget,
                        candidate_pool=args.candidate_pool,
                        scoring=args.scoring,
                        context_dir=output_dir / "contexts" if args.write_contexts else None,
                        answer_path=answer_path if args.provider != "none" else None,
                        provider=args.provider,
                        model=args.llm_model,
                        ollama_url=args.ollama_url,
                    ),
                }
            )

    csv_path = output_dir / "reviewed_qa_selector_eval.csv"
    table_path = output_dir / "reviewed_qa_selector_eval_table.tex"
    _write_csv(rows, csv_path)
    _write_latex(rows, table_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")
    return 0


def _load_records(path: Path, include_unreviewed: bool) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            if include_unreviewed or metadata.get("reviewed") is True:
                records.append(record)
    return records


def _evaluate(
    records,
    index,
    embedder,
    strategy,
    budget,
    candidate_pool,
    scoring,
    context_dir: Path | None,
    answer_path: Path | None,
    provider: str,
    model: str,
    ollama_url: str,
) -> dict[str, float]:
    totals = {
        "evidence_term_recall": 0.0,
        "complete_evidence_rate": 0.0,
        "answer_f1": 0.0,
        "avg_used_tokens": 0.0,
        "budget_utilization": 0.0,
        "over_budget_rate": 0.0,
        "total_value": 0.0,
        "redundancy_score": 0.0,
        "latency_seconds": 0.0,
    }
    if context_dir:
        context_dir.mkdir(parents=True, exist_ok=True)
    answer_rows: list[dict] = []

    for record_index, record in enumerate(records, start=1):
        query = str(record["query"])
        evidence_terms = set()
        for evidence in record.get("evidence", []):
            evidence_terms.update(evidence.get("evidence_terms") or _content_terms(str(evidence.get("text", ""))))
        query_embedding = embedder.embed([query])[0]
        scored = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            scoring=scoring,
            query_text=query,
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
        recall = len(evidence_terms & selected_terms) / max(1, len(evidence_terms))
        totals["evidence_term_recall"] += recall
        totals["complete_evidence_rate"] += 1.0 if recall >= 0.80 else 0.0
        totals["avg_used_tokens"] += result.used_tokens
        totals["budget_utilization"] += result.used_tokens / max(1, budget)
        totals["over_budget_rate"] += 1.0 if result.used_tokens > budget else 0.0
        totals["total_value"] += result.total_value
        totals["redundancy_score"] += _redundancy_score([item.chunk for item in result.selected])
        totals["latency_seconds"] += elapsed

        context = render_context([item.chunk for item in result.selected])
        if context_dir:
            context_path = context_dir / f"q{record_index:03d}_{strategy}_{budget}.txt"
            context_path.write_text(
                f"QUESTION: {query}\n\nGOLD ANSWER: {record.get('answer', '')}\n\n"
                f"SELECTED CONTEXT:\n{context}",
                encoding="utf-8",
            )
        if provider != "none":
            answer = _generate_answer(
                provider=provider,
                model=model,
                ollama_url=ollama_url,
                query=query,
                context=context,
            )
            answer_f1 = _token_f1(answer, str(record.get("answer", "")))
            totals["answer_f1"] += answer_f1
            answer_rows.append(
                {
                    "record_index": record_index,
                    "strategy": strategy,
                    "budget": budget,
                    "query": query,
                    "gold_answer": record.get("answer", ""),
                    "model_answer": answer,
                    "answer_f1": answer_f1,
                    "used_tokens": result.used_tokens,
                }
            )

    runs = max(1, len(records))
    if answer_path and answer_rows:
        with answer_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in answer_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {key: value / runs for key, value in totals.items()}


def _generate_answer(provider: str, model: str, ollama_url: str, query: str, context: str) -> str:
    prompt = f"Use only the context below to answer the question.\n\nContext:\n{context}\nQuestion: {query}\nAnswer:"
    if provider == "openai":
        return _openai_answer(prompt, model=model)
    if provider in {"ollama", "local"}:
        local_model = _default_ollama_model(model)
        return _ollama_answer(prompt, model=local_model, base_url=ollama_url)
    if provider == "cerebras":
        return _cerebras_answer(prompt, model=model)
    if provider == "groq":
        return _groq_answer(prompt, model=model)
    raise ValueError(f"Unknown answer provider: {provider}")


def _token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = _answer_tokens(predicted)
    expected_tokens = _answer_tokens(expected)
    if not predicted_tokens or not expected_tokens:
        return 0.0
    expected_counts: dict[str, int] = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    overlap = 0
    for token in predicted_tokens:
        count = expected_counts.get(token, 0)
        if count > 0:
            overlap += 1
            expected_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _answer_tokens(text: str) -> list[str]:
    return [token for token in _content_terms(text) if token]


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "strategy",
        "chunker",
        "scoring",
        "budget_ratio",
        "budget",
        "qa_count",
        "chunk_count",
        "avg_chunk_tokens",
        "evidence_term_recall",
        "complete_evidence_rate",
        "answer_f1",
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
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Reviewed QA Selector Evaluation}",
        r"\label{tab:reviewed-qa-selector}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Selector & Budget & Recall & Complete & Ans. F1 & Value & Lat. \\",
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
                    _fmt(row["answer_f1"]),
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
