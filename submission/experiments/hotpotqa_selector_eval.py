from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qasper_selector_eval import (  # noqa: E402
    BudgetSpec,
    _build_index,
    _csv_list,
    _parse_budgets,
    _tokens,
)
from tokenpack.chunk_profiles import resolve_chunk_size_config  # noqa: E402
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.models import TextBlock  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.selectors import select_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "hotpotqa"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "hotpotqa"
STRATEGIES = ["budget-top-k", "greedy-density", "knapsack", "knapsack-redundancy", "knapsack-augment"]
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")
STOPWORDS = {
    "and",
    "are",
    "from",
    "have",
    "how",
    "into",
    "its",
    "not",
    "the",
    "their",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(slots=True)
class HotpotQuestion:
    example_id: str
    question: str
    answer: str
    evidence_texts: list[str]
    supporting_titles: set[str]
    source_text: str
    source_tokens: int
    blocks: list[TextBlock]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selectors on HotpotQA supporting facts.")
    parser.add_argument("--data-file", help="Optional local HotpotQA JSON/JSONL file.")
    parser.add_argument("--split", default="validation", choices=["validation", "train"])
    parser.add_argument("--backend", default="hash", choices=["auto", "hash", "sentence-transformers"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["paragraph", "semantic-threshold", "structure-aware"], default="paragraph")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="evidence-hybrid",
    )
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--budget-ratios", default="0.10,0.20,0.30,0.40")
    parser.add_argument("--budgets")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--block-level", choices=["title", "sentence"], default="title")
    parser.add_argument("--levels", default="medium,hard", help="Comma-separated HotpotQA difficulty levels.")
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--chunk-size-preset",
        choices=["manual", "default", "low-budget"],
        default="manual",
    )
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
    chunk_size = resolve_chunk_size_config(
        args.chunk_size_preset,
        args.target_tokens,
        args.min_tokens,
        args.max_tokens,
    )
    levels = {item.strip() for item in args.levels.split(",") if item.strip()}
    strategies = _csv_list(args.strategies)
    rows = list(_load_hotpot_rows(args.data_file, args.split))

    totals: dict[tuple[str, str], dict[str, float]] = {}
    processed = 0
    for row in rows:
        if processed >= args.max_examples:
            break
        if levels and str(row.get("level", "")) not in levels:
            continue
        question = _question_from_row(row, processed, args.block_level, token_counter)
        if not question.evidence_texts or not question.blocks:
            continue
        index = _build_index(
            paper_id=question.example_id,
            blocks=question.blocks,
            embedder=embedder,
            work_dir=work_dir,
            chunker_name=args.chunker,
            target_tokens=chunk_size.target_tokens,
            min_tokens=chunk_size.min_tokens,
            max_tokens=chunk_size.max_tokens,
            semantic_threshold=args.semantic_threshold,
            token_counter=token_counter,
        )
        query_embedding = embedder.embed([question.question])[0]
        scored = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            scoring=args.scoring,
            query_text=question.question,
            redundancy_candidate_pool=args.candidate_pool,
        )
        budgets = _parse_budgets(args.budgets, args.budget_ratios, question.source_tokens)
        processed += 1
        for budget_spec in budgets:
            for strategy in strategies:
                metrics = _evaluate_question(question, scored, strategy, budget_spec, args.candidate_pool)
                bucket = totals.setdefault(
                    (strategy, budget_spec.label),
                    {
                        "budget_sort": budget_spec.sort_key,
                        "avg_budget_tokens": 0.0,
                        "supporting_fact_recall": 0.0,
                        "complete_support_rate": 0.0,
                        "supporting_title_recall": 0.0,
                        "answer_token_recall": 0.0,
                        "avg_used_tokens": 0.0,
                        "token_ratio_vs_full": 0.0,
                        "cost_saving_vs_full": 0.0,
                        "budget_utilization": 0.0,
                        "over_budget_rate": 0.0,
                        "avg_total_value": 0.0,
                        "latency_seconds": 0.0,
                        "runs": 0.0,
                    },
                )
                bucket["avg_budget_tokens"] += float(budget_spec.tokens)
                for key, value in metrics.items():
                    bucket[key] += value
                bucket["runs"] += 1.0

    summary = _summarize(totals, args, processed)
    csv_path = output_dir / "hotpotqa_selector_eval.csv"
    _write_csv(summary, csv_path)
    print(f"Wrote {csv_path}")
    print(f"Processed examples={processed}")
    return 0


def _load_hotpot_rows(data_file: str | None, split: str) -> Iterable[dict[str, Any]]:
    if data_file:
        path = Path(data_file)
        if path.suffix.lower() == ".arrow":
            from datasets import Dataset  # type: ignore

            return Dataset.from_file(str(path)).to_list()
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else list(payload.values())

    from datasets import load_dataset  # type: ignore

    cached = _cached_hotpot_arrow(split)
    if cached is not None:
        from datasets import Dataset  # type: ignore

        return Dataset.from_file(str(cached)).to_list()
    return load_dataset("hotpotqa/hotpot_qa", "distractor", split=split).to_list()


def _cached_hotpot_arrow(split: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "hotpotqa___hotpot_qa"
    if not cache_root.exists():
        return None
    matches = sorted(cache_root.rglob(f"hotpot_qa-{split}.arrow"))
    return matches[-1] if matches else None


def _question_from_row(
    row: dict[str, Any],
    document_index: int,
    block_level: str,
    token_counter: TokenCounter,
) -> HotpotQuestion:
    titles = [str(title) for title in row["context"]["title"]]
    sentences_by_title = row["context"]["sentences"]
    supporting_titles = [str(title) for title in row["supporting_facts"]["title"]]
    supporting_sent_ids = [int(sent_id) for sent_id in row["supporting_facts"]["sent_id"]]
    evidence_texts: list[str] = []
    supporting_title_set = set(supporting_titles)
    title_to_sentences = {title: list(map(str, sentences)) for title, sentences in zip(titles, sentences_by_title, strict=True)}
    for title, sent_id in zip(supporting_titles, supporting_sent_ids, strict=True):
        sentences = title_to_sentences.get(title, [])
        if 0 <= sent_id < len(sentences):
            evidence_texts.append(_clean_text(f"{title}. {sentences[sent_id]}"))

    blocks: list[TextBlock] = []
    paragraph_index = 0
    if block_level == "title":
        for title, sentences in title_to_sentences.items():
            text = _clean_text(f"{title}. {' '.join(sentences)}")
            if not text:
                continue
            blocks.append(_make_block(text, title, None, document_index, paragraph_index, token_counter))
            paragraph_index += 1
    else:
        for title, sentences in title_to_sentences.items():
            for sent_id, sentence in enumerate(sentences):
                text = _clean_text(f"{title}. {sentence}")
                if not text:
                    continue
                blocks.append(_make_block(text, title, sent_id, document_index, paragraph_index, token_counter))
                paragraph_index += 1

    source_text = " ".join(block.text for block in blocks)
    source_tokens = sum(int(block.metadata.get("token_count", token_counter.count(block.text))) for block in blocks)
    return HotpotQuestion(
        example_id=str(row.get("id") or row.get("_id") or document_index),
        question=str(row["question"]),
        answer=str(row["answer"]),
        evidence_texts=evidence_texts,
        supporting_titles=supporting_title_set,
        source_text=source_text,
        source_tokens=source_tokens,
        blocks=blocks,
    )


def _make_block(
    text: str,
    title: str,
    sent_id: int | None,
    document_index: int,
    paragraph_index: int,
    token_counter: TokenCounter,
) -> TextBlock:
    metadata: dict[str, Any] = {"title": title, "content_type": "document", "token_count": token_counter.count(text)}
    if sent_id is not None:
        metadata["sent_id"] = sent_id
    return TextBlock(
        text=text,
        source_path=f"hotpotqa:{document_index}",
        document_index=document_index,
        paragraph_index=paragraph_index,
        char_start=0,
        char_end=len(text),
        metadata=metadata,
    )


def _evaluate_question(
    question: HotpotQuestion,
    scored,
    strategy: str,
    budget: BudgetSpec,
    candidate_pool: int,
) -> dict[str, float]:
    started = time.perf_counter()
    result = select_chunks(scored, strategy=strategy, budget=budget.tokens, candidate_pool=candidate_pool)
    elapsed = time.perf_counter() - started
    selected_text = " ".join(item.chunk.text for item in result.selected)
    support_recall = _evidence_recall(question.evidence_texts, selected_text)
    selected_titles = _selected_titles(result.selected)
    title_recall = len(question.supporting_titles & selected_titles) / max(1, len(question.supporting_titles))
    answer_recall = _answer_token_recall(selected_text, question.answer)
    return {
        "supporting_fact_recall": support_recall,
        "complete_support_rate": 1.0 if support_recall >= 0.999 else 0.0,
        "supporting_title_recall": title_recall,
        "answer_token_recall": answer_recall,
        "avg_used_tokens": float(result.used_tokens),
        "token_ratio_vs_full": result.used_tokens / max(1, question.source_tokens),
        "cost_saving_vs_full": 1.0 - result.used_tokens / max(1, question.source_tokens),
        "budget_utilization": result.used_tokens / max(1, budget.tokens),
        "over_budget_rate": 1.0 if result.used_tokens > budget.tokens else 0.0,
        "avg_total_value": result.total_value,
        "latency_seconds": elapsed,
    }


def _selected_titles(selected) -> set[str]:
    titles: set[str] = set()
    for item in selected:
        title = item.chunk.metadata.get("title")
        if title:
            titles.add(str(title))
    return titles


def _evidence_recall(evidence_texts: list[str], selected_text: str) -> float:
    selected_norm = _normalize(selected_text)
    if not evidence_texts:
        return 0.0
    hits = 0.0
    for evidence in evidence_texts:
        evidence_norm = _normalize(evidence)
        if evidence_norm in selected_norm:
            hits += 1.0
            continue
        evidence_terms = set(_content_tokens(evidence_norm))
        selected_terms = set(_content_tokens(selected_norm))
        hits += len(evidence_terms & selected_terms) / max(1, len(evidence_terms))
    return hits / max(1, len(evidence_texts))


def _answer_token_recall(context: str, answer: str) -> float:
    answer_terms = set(_content_tokens(answer))
    if not answer_terms:
        return 1.0 if str(answer).strip().lower() in {"yes", "no"} and str(answer).strip().lower() in context.lower() else 0.0
    context_terms = set(_content_tokens(context))
    return len(answer_terms & context_terms) / max(1, len(answer_terms))


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(str(text).lower()) if token not in STOPWORDS]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _summarize(
    totals: dict[tuple[str, str], dict[str, float]],
    args: argparse.Namespace,
    processed_examples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (strategy, budget), values in sorted(totals.items(), key=lambda item: (item[1]["budget_sort"], item[0][0])):
        runs = max(1.0, values["runs"])
        rows.append(
            {
                "strategy": strategy,
                "block_level": args.block_level,
                "chunker": args.chunker,
                "scoring": args.scoring,
                "budget": budget,
                "avg_budget_tokens": values["avg_budget_tokens"] / runs,
                "processed_examples": processed_examples,
                "runs": int(values["runs"]),
                "supporting_fact_recall": values["supporting_fact_recall"] / runs,
                "complete_support_rate": values["complete_support_rate"] / runs,
                "supporting_title_recall": values["supporting_title_recall"] / runs,
                "answer_token_recall": values["answer_token_recall"] / runs,
                "avg_used_tokens": values["avg_used_tokens"] / runs,
                "token_ratio_vs_full": values["token_ratio_vs_full"] / runs,
                "cost_saving_vs_full": values["cost_saving_vs_full"] / runs,
                "budget_utilization": values["budget_utilization"] / runs,
                "over_budget_rate": values["over_budget_rate"] / runs,
                "avg_total_value": values["avg_total_value"] / runs,
                "latency_seconds": values["latency_seconds"] / runs,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
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
