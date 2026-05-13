from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "submission" / "experiments"
for path in (SRC, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qasper_cost_quality_curve import _answer_token_recall, _content_tokens, _evidence_recall  # noqa: E402
from qasper_selector_eval import (  # noqa: E402
    _blocks_from_qasper_row,
    _build_index,
    _load_qasper_rows,
    _parse_budgets,
    _questions_from_qasper_row,
)
from tokenpack.embeddings import make_embedder  # noqa: E402
from tokenpack.export import render_context  # noqa: E402
from tokenpack.scoring import SCORING_PROFILES, score_chunks  # noqa: E402
from tokenpack.selectors import select_chunks  # noqa: E402
from tokenpack.tokenization import TokenCounter  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results" / "qasper_ollama_smoke"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "qasper_ollama_smoke"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small QASPER LLM answer-quality smoke test with Ollama.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunker", choices=["semantic-threshold", "structure-aware"], default="structure-aware")
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="hybrid",
    )
    parser.add_argument("--strategies", default="production-rag,budget-top-k,knapsack")
    parser.add_argument("--budget-ratios", default="0.60,0.80,1.00")
    parser.add_argument("--max-questions", type=int, default=30)
    parser.add_argument("--max-papers", type=int, default=10_000)
    parser.add_argument("--candidate-pool", type=int, default=10_000)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument("--num-predict", type=int, default=90)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "qasper_ollama_smoke.jsonl"
    summary_path = output_dir / "qasper_ollama_smoke_summary.csv"

    rows = list(_load_qasper_rows(args.data_file, "validation"))
    token_counter = TokenCounter()
    embedder = make_embedder(model_name=args.embedding_model, local_files_only=True)
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]

    generated_rows: list[dict[str, Any]] = []
    processed_papers = 0
    processed_questions = 0
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for row in rows:
            if processed_papers >= args.max_papers or processed_questions >= args.max_questions:
                break
            paper_id = str(row.get("id") or row.get("paper_id") or processed_papers)
            blocks = _blocks_from_qasper_row(row, document_index=processed_papers, token_counter=token_counter)
            questions = _questions_from_qasper_row(row)
            if not blocks or not questions:
                continue
            source_text = " ".join(block.text for block in blocks)
            index = _build_index(
                paper_id=paper_id,
                blocks=blocks,
                embedder=embedder,
                work_dir=work_dir,
                chunker_name=args.chunker,
                target_tokens=args.target_tokens,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                semantic_threshold=args.semantic_threshold,
                token_counter=token_counter,
            )
            source_tokens = sum(block.metadata.get("token_count", token_counter.count(block.text)) for block in blocks)
            budget_specs = _parse_budgets(None, args.budget_ratios, source_tokens)
            processed_papers += 1

            for question in questions:
                if processed_questions >= args.max_questions:
                    break
                if not _is_meaningful_answer(question.answer):
                    continue
                if _answer_token_recall(source_text, question.answer) < 0.80:
                    continue
                processed_questions += 1
                query_embedding = embedder.embed([question.question])[0]
                scored = score_chunks(
                    query_embedding,
                    index.chunks,
                    index.embeddings,
                    scoring=args.scoring,
                    query_text=question.question,
                )
                for budget_spec in budget_specs:
                    for strategy in strategies:
                        selection = select_chunks(
                            scored,
                            strategy=strategy,
                            budget=budget_spec.tokens,
                            candidate_pool=args.candidate_pool,
                        )
                        context = render_context([item.chunk for item in selection.selected])
                        started = time.perf_counter()
                        try:
                            answer = _ollama_answer(
                                query=question.question,
                                context=context,
                                model=args.model,
                                base_url=args.ollama_url,
                                num_predict=args.num_predict,
                            )
                            status = "completed"
                            error = ""
                        except Exception as exc:
                            answer = ""
                            status = "failed"
                            error = str(exc)
                        elapsed = time.perf_counter() - started
                        selected_text = " ".join(item.chunk.text for item in selection.selected)
                        result_row = {
                            "paper_id": paper_id,
                            "question_id": question.question_id,
                            "model": args.model,
                            "chunker": args.chunker,
                            "scoring": args.scoring,
                            "strategy": strategy,
                            "budget": budget_spec.label,
                            "budget_tokens": budget_spec.tokens,
                            "used_tokens": selection.used_tokens,
                            "question": question.question,
                            "gold_answer": question.answer,
                            "answer": answer,
                            "status": status,
                            "error": error,
                            "latency_seconds": elapsed,
                            "context_answer_recall": _answer_token_recall(selected_text, question.answer),
                            "context_evidence_recall": _evidence_recall(question.evidence_texts, selected_text),
                            "answer_token_f1": _token_f1(answer, question.answer),
                            "answer_token_recall": _answer_token_recall(answer, question.answer),
                            "insufficient_answer": 1.0 if "insufficient" in answer.lower() else 0.0,
                        }
                        raw_handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
                        raw_handle.flush()
                        generated_rows.append(result_row)
                        print(
                            f"{processed_questions}/{args.max_questions} {strategy} {budget_spec.label} "
                            f"tokens={selection.used_tokens} f1={result_row['answer_token_f1']:.3f}"
                        )

    summary = _summarize(generated_rows, processed_papers=processed_papers, processed_questions=processed_questions)
    _write_csv(summary, summary_path)
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    return 0


def _ollama_answer(query: str, context: str, model: str, base_url: str, num_predict: int) -> str:
    prompt = (
        "/no_think\n"
        "Answer the question using only the provided paper context. "
        "If the answer is not present, say: The context is insufficient. "
        "Return only the final answer. Do not explain your reasoning. "
        "Be concise and do not mention the context unless it is insufficient.\n\n"
        f"Paper context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": num_predict},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    message = payload.get("message") or {}
    answer = str(message.get("content") or payload.get("response") or "").strip()
    return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()


def _token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = _content_tokens(predicted)
    expected_tokens = _content_tokens(expected)
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


def _is_meaningful_answer(answer: str) -> bool:
    text = str(answer).strip()
    if not text:
        return False
    if re.fullmatch(r"(BIBREF\d+\s*)+", text):
        return False
    if text.lower() in {"yes", "no"}:
        return True
    return len(_content_tokens(text)) >= 3


def _summarize(rows: list[dict[str, Any]], processed_papers: int, processed_questions: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["strategy"]), str(row["budget"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (strategy, budget), group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        completed = [row for row in group if row["status"] == "completed"]
        summary.append(
            {
                "strategy": strategy,
                "budget": budget,
                "processed_papers": processed_papers,
                "processed_questions": processed_questions,
                "runs": len(group),
                "completed": len(completed),
                "avg_used_tokens": _mean(row["used_tokens"] for row in group),
                "context_answer_recall": _mean(row["context_answer_recall"] for row in group),
                "context_evidence_recall": _mean(row["context_evidence_recall"] for row in group),
                "answer_token_f1": _mean(row["answer_token_f1"] for row in completed),
                "answer_token_recall": _mean(row["answer_token_recall"] for row in completed),
                "insufficient_rate": _mean(row["insufficient_answer"] for row in completed),
                "avg_latency_seconds": _mean(row["latency_seconds"] for row in completed),
            }
        )
    return summary


def _mean(values: Any) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(1, len(values))


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
