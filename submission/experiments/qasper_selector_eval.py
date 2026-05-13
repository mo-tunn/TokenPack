from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenpack.chunk_profiles import resolve_chunk_size_config
from tokenpack.chunking import SemanticThresholdChunker, StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.index import ChunkIndex
from tokenpack.models import Chunk, TextBlock
from tokenpack.scoring import SCORING_PROFILES, score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "qasper"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "qasper"
DEFAULT_PARQUET_URLS = {
    "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/test/0000.parquet",
    "train": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/train/0000.parquet",
}
STRATEGIES = ["production-rag", "budget-top-k", "greedy-density", "knapsack", "knapsack-redundancy"]
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")


@dataclass(slots=True)
class QasperQuestion:
    paper_id: str
    title: str
    question_id: str
    question: str
    answer: str
    evidence_texts: list[str]


@dataclass(slots=True)
class BudgetSpec:
    label: str
    tokens: int
    sort_key: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selectors on QASPER evidence annotations.")
    parser.add_argument("--data-file", help="Local QASPER parquet/json/jsonl file. Prefer converted parquet.")
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["semantic-threshold", "structure-aware"], default="structure-aware")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="evidence-hybrid",
    )
    parser.add_argument("--budget-ratios", default="0.05,0.10,0.20")
    parser.add_argument("--budgets", help="Optional absolute budgets; overrides --budget-ratios.")
    parser.add_argument("--max-papers", type=int, default=40)
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument(
        "--chunk-size-preset",
        choices=["manual", "default", "low-budget"],
        default="manual",
        help="Override target/min/max token limits with a named chunking preset.",
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument(
        "--no-paper-table",
        action="store_true",
        help="Write only to --output-dir and leave submission/paper/tables untouched.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_load_qasper_rows(args.data_file, args.split))
    if not rows:
        raise SystemExit("No QASPER rows loaded.")

    token_counter = TokenCounter()
    embedder = make_embedder(model_name=args.model, local_files_only=True)
    chunk_size = resolve_chunk_size_config(
        args.chunk_size_preset,
        args.target_tokens,
        args.min_tokens,
        args.max_tokens,
    )
    strategies = _csv_list(args.strategies)
    per_strategy: dict[tuple[str, str], dict[str, float]] = {}
    processed_questions = 0
    processed_papers = 0

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
        source_tokens = sum(block.token_count if hasattr(block, "token_count") else token_counter.count(block.text) for block in blocks)
        budgets = _parse_budgets(args.budgets, args.budget_ratios, source_tokens)
        processed_papers += 1

        for question in questions:
            if processed_questions >= args.max_questions:
                break
            processed_questions += 1
            query_embedding = embedder.embed([question.question])[0]
            scored = score_chunks(
                query_embedding,
                index.chunks,
                index.embeddings,
                scoring=args.scoring,
                query_text=question.question,
                redundancy_candidate_pool=args.candidate_pool,
            )
            for budget_spec in budgets:
                for strategy in strategies:
                    metrics = _evaluate_question(
                        qasper_question=question,
                        scored=scored,
                        strategy=strategy,
                        budget=budget_spec.tokens,
                        candidate_pool=args.candidate_pool,
                    )
                    bucket = per_strategy.setdefault(
                        (strategy, budget_spec.label),
                        {
                            "budget_sort": budget_spec.sort_key,
                            "avg_budget_tokens": 0.0,
                            "evidence_recall": 0.0,
                            "complete_evidence_rate": 0.0,
                            "answer_token_f1": 0.0,
                            "avg_used_tokens": 0.0,
                            "budget_utilization": 0.0,
                            "over_budget_rate": 0.0,
                            "total_value": 0.0,
                            "latency_seconds": 0.0,
                            "runs": 0.0,
                        },
                    )
                    bucket["avg_budget_tokens"] += float(budget_spec.tokens)
                    for name, value in metrics.items():
                        bucket[name] += value
                    bucket["runs"] += 1.0

    summary_rows = _summarize(per_strategy, args, processed_papers, processed_questions)
    csv_path = output_dir / "qasper_selector_eval.csv"
    table_path = output_dir / "qasper_selector_eval_table.tex"
    paper_table_path = ROOT / "submission" / "paper" / "tables" / "qasper_selector_eval_table.tex"
    _write_csv(summary_rows, csv_path)
    _write_latex(summary_rows, table_path)
    if not args.no_paper_table:
        _write_latex(summary_rows, paper_table_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")
    print(f"Processed papers={processed_papers}, questions={processed_questions}")
    return 0


def _load_qasper_rows(data_file: str | None, split: str) -> Iterable[dict[str, Any]]:
    if data_file:
        path = Path(data_file)
        if path.suffix.lower() == ".parquet":
            import pandas as pd

            return pd.read_parquet(path).to_dict(orient="records")
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else list(payload.values())
        raise ValueError(f"Unsupported QASPER data file: {path}")

    import pandas as pd

    url = DEFAULT_PARQUET_URLS[split]
    return pd.read_parquet(url).to_dict(orient="records")


def _blocks_from_qasper_row(row: dict[str, Any], document_index: int, token_counter: TokenCounter) -> list[TextBlock]:
    full_text = row.get("full_text") or {}
    section_names = _as_list(full_text.get("section_name"))
    paragraphs_by_section = _as_list(full_text.get("paragraphs"))
    blocks: list[TextBlock] = []
    paragraph_index = 0
    for section_index, paragraphs in enumerate(paragraphs_by_section):
        section_name = section_names[section_index] if section_index < len(section_names) else ""
        for paragraph in _as_list(paragraphs):
            text = _clean_text(str(paragraph))
            if not text:
                continue
            if section_name:
                text = f"{section_name}. {text}"
            blocks.append(
                TextBlock(
                    text=text,
                    source_path=f"qasper:{row.get('id')}",
                    document_index=document_index,
                    page=None,
                    paragraph_index=paragraph_index,
                    char_start=0,
                    char_end=len(text),
                    metadata={"section_name": section_name, "token_count": token_counter.count(text)},
                )
            )
            paragraph_index += 1
    return blocks


def _questions_from_qasper_row(row: dict[str, Any]) -> list[QasperQuestion]:
    qas = row.get("qas") or {}
    questions = _as_list(qas.get("question"))
    question_ids = _as_list(qas.get("question_id")) or [str(index) for index in range(len(questions))]
    answers_by_question = _as_list(qas.get("answers"))
    records: list[QasperQuestion] = []
    for index, question in enumerate(questions):
        answers_payload = answers_by_question[index] if index < len(answers_by_question) else {}
        answers = _as_list(answers_payload.get("answer")) if isinstance(answers_payload, dict) else []
        if not answers:
            continue
        chosen = _choose_answer(answers)
        if chosen is None:
            continue
        evidence_texts = [_clean_text(item) for item in _as_list(chosen.get("evidence")) if _clean_text(item)]
        if not evidence_texts:
            evidence_texts = [
                _clean_text(item)
                for item in _as_list(chosen.get("highlighted_evidence"))
                if _clean_text(item)
            ]
        if not evidence_texts:
            continue
        answer_text = _answer_text(chosen)
        if not answer_text:
            continue
        records.append(
            QasperQuestion(
                paper_id=str(row.get("id")),
                title=str(row.get("title") or ""),
                question_id=str(question_ids[index] if index < len(question_ids) else index),
                question=str(question),
                answer=answer_text,
                evidence_texts=evidence_texts,
            )
        )
    return records


def _choose_answer(answers: list[dict[str, Any]]) -> dict[str, Any] | None:
    for answer in answers:
        if answer.get("unanswerable") is True:
            continue
        if _as_list(answer.get("evidence")) or _as_list(answer.get("highlighted_evidence")):
            return answer
    return None


def _answer_text(answer: dict[str, Any]) -> str:
    spans = [str(item) for item in _as_list(answer.get("extractive_spans")) if str(item).strip()]
    if spans:
        return " ".join(spans)
    free_form = str(answer.get("free_form_answer") or "").strip()
    if free_form:
        return free_form
    if answer.get("yes_no") is True:
        return "yes"
    if answer.get("yes_no") is False:
        return "no"
    return ""


def _build_index(
    paper_id: str,
    blocks: list[TextBlock],
    embedder: Any,
    work_dir: Path,
    chunker_name: str,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    semantic_threshold: float,
    token_counter: TokenCounter,
) -> ChunkIndex:
    cache = EmbeddingCache(work_dir / f"qasper-{paper_id}-{chunker_name}.embeddings.json")
    if chunker_name == "structure-aware":
        block_embeddings = cache.get_or_embed([block.text for block in blocks], embedder)
        chunker = StructureAwareChunker(
            target_tokens,
            min_tokens,
            max_tokens,
            token_counter=token_counter,
            block_embeddings=block_embeddings,
            semantic_threshold=semantic_threshold,
        )
    elif chunker_name == "semantic-threshold":
        block_embeddings = cache.get_or_embed([block.text for block in blocks], embedder)
        chunker = SemanticThresholdChunker(
            block_embeddings,
            similarity_threshold=semantic_threshold,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    else:
        raise ValueError(f"Unknown chunker: {chunker_name}")
    chunks = chunker.chunk(blocks)
    embeddings = cache.get_or_embed([chunk.text for chunk in chunks], embedder)
    return ChunkIndex(chunks=chunks, embeddings=embeddings, model_name=embedder.model_name)


def _evaluate_question(
    qasper_question: QasperQuestion,
    scored,
    strategy: str,
    budget: int,
    candidate_pool: int,
) -> dict[str, float]:
    started = time.perf_counter()
    result = select_chunks(
        scored,
        strategy=strategy,
        budget=budget,
        candidate_pool=candidate_pool,
    )
    elapsed = time.perf_counter() - started
    selected_text = " ".join(item.chunk.text for item in result.selected)
    evidence_recall = _evidence_recall(qasper_question.evidence_texts, selected_text)
    return {
        "evidence_recall": evidence_recall,
        "complete_evidence_rate": 1.0 if evidence_recall >= 0.80 else 0.0,
        "answer_token_f1": _token_f1(selected_text, qasper_question.answer),
        "avg_used_tokens": float(result.used_tokens),
        "budget_utilization": result.used_tokens / max(1, budget),
        "over_budget_rate": 1.0 if result.used_tokens > budget else 0.0,
        "total_value": result.total_value,
        "latency_seconds": elapsed,
    }


def _evidence_recall(evidence_texts: list[str], selected_text: str) -> float:
    selected_norm = _normalize(selected_text)
    if not evidence_texts:
        return 0.0
    hits = 0.0
    for evidence in evidence_texts:
        evidence_norm = _normalize(evidence)
        if not evidence_norm:
            continue
        if evidence_norm in selected_norm:
            hits += 1.0
        else:
            evidence_terms = set(_tokens(evidence_norm))
            selected_terms = set(_tokens(selected_norm))
            hits += len(evidence_terms & selected_terms) / max(1, len(evidence_terms))
    return hits / max(1, len(evidence_texts))


def _token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = _tokens(_normalize(predicted))
    expected_tokens = _tokens(_normalize(expected))
    if not predicted_tokens or not expected_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in expected_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in predicted_tokens:
        count = counts.get(token, 0)
        if count:
            overlap += 1
            counts[token] = count - 1
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if text.startswith("FLOAT SELECTED"):
        return ""
    return text


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_budgets(raw_budgets: str | None, raw_ratios: str, source_tokens: int) -> list[BudgetSpec]:
    if raw_budgets:
        specs = []
        for item in raw_budgets.split(","):
            if not item.strip():
                continue
            tokens = int(item.strip())
            specs.append(BudgetSpec(label=str(tokens), tokens=tokens, sort_key=float(tokens)))
        return specs
    specs = []
    for item in raw_ratios.split(","):
        if not item.strip():
            continue
        ratio = float(item.strip())
        specs.append(
            BudgetSpec(
                label=f"{ratio:.0%}",
                tokens=max(1, int(source_tokens * ratio)),
                sort_key=ratio,
            )
        )
    return specs


def _summarize(
    per_strategy: dict[tuple[str, str], dict[str, float]],
    args: argparse.Namespace,
    processed_papers: int,
    processed_questions: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (strategy, budget), values in sorted(per_strategy.items(), key=lambda item: (item[1]["budget_sort"], item[0][0])):
        runs = max(1.0, values["runs"])
        rows.append(
            {
                "strategy": strategy,
                "chunker": args.chunker,
                "scoring": args.scoring,
                "budget": budget,
                "avg_budget_tokens": values["avg_budget_tokens"] / runs,
                "processed_papers": processed_papers,
                "processed_questions": processed_questions,
                "runs": int(values["runs"]),
                "evidence_recall": values["evidence_recall"] / runs,
                "complete_evidence_rate": values["complete_evidence_rate"] / runs,
                "answer_token_f1": values["answer_token_f1"] / runs,
                "avg_used_tokens": values["avg_used_tokens"] / runs,
                "budget_utilization": values["budget_utilization"] / runs,
                "over_budget_rate": values["over_budget_rate"] / runs,
                "avg_total_value": values["total_value"] / runs,
                "latency_seconds": values["latency_seconds"] / runs,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "strategy",
        "chunker",
        "scoring",
        "budget",
        "avg_budget_tokens",
        "processed_papers",
        "processed_questions",
        "runs",
        "evidence_recall",
        "complete_evidence_rate",
        "answer_token_f1",
        "avg_used_tokens",
        "budget_utilization",
        "over_budget_rate",
        "avg_total_value",
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
        r"\caption{QASPER Selector Evaluation With Fixed Scoring}",
        r"\label{tab:qasper-selector}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Selector & Budget & Avg Tok. & Evidence Recall & Complete & Answer Overlap F1 & Value & Lat. \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_text(str(row["strategy"])),
                    _latex_text(str(row["budget"])),
                    _fmt(row["avg_budget_tokens"]),
                    _fmt(row["evidence_recall"]),
                    _fmt(row["complete_evidence_rate"]),
                    _fmt(row["answer_token_f1"]),
                    _fmt(row["avg_total_value"]),
                    _fmt(row["latency_seconds"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: object) -> str:
    return f"{float(value):.3f}" if not isinstance(value, int) else str(value)


def _latex_text(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


if __name__ == "__main__":
    raise SystemExit(main())
