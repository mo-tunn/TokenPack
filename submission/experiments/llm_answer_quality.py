from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenpack.dataset import GoldRecord, load_gold_records, validate_gold_records
from tokenpack.embeddings import make_embedder
from tokenpack.export import render_context
from tokenpack.index import ChunkIndex, load_index
from tokenpack.models import Chunk, SelectionResult
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter


STRATEGIES = ["document-prefix", "full-document", "top-k", "knapsack"]
LOCAL_SMALL = "qwen3:0.6b"
LOCAL_MEDIUM = "qwen3:4b"
OPENAI_LARGE = "gpt-4o"
DEFAULT_PRICES = {
    "openai-gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00},
    "openai-gpt-5.4-mini": {"input_per_m": 0.75, "output_per_m": 4.50},
}


@dataclass(slots=True)
class ModelSpec:
    label: str
    provider: str
    model: str


def main() -> int:
    load_dotenv_keys(Path(".env"))
    parser = argparse.ArgumentParser(description="Run TokenPack LLM answer-quality and cost experiments.")
    parser.add_argument("--index", default="submission/gold/simple_corpus/simple-index.json")
    parser.add_argument("--gold", default="submission/gold/gold.jsonl")
    parser.add_argument("--output-dir", default="submission/results/llm_quality")
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--reserve-output", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=250)
    parser.add_argument("--models", default="small", help="Comma-separated: small,medium,large")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep rows for models not requested and replace only the requested model labels.",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(STRATEGIES),
        help="Comma-separated strategy subset: document-prefix,full-document,top-k,knapsack",
    )
    parser.add_argument("--skip-generation", action="store_true", help="Only summarize existing JSONL and cost assets.")
    parser.add_argument("--auto-score", action="store_true", help="Add a simple lexical pilot score; not a human score.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = load_index(args.index)
    records = load_gold_records(args.gold)
    errors = validate_gold_records(records, index)
    if errors:
        raise SystemExit("Gold validation failed:\n" + "\n".join(errors))

    answer_path = output_dir / "llm_answer_quality.jsonl"
    model_specs = parse_model_specs(args.models)
    strategies = parse_strategies(args.strategies)
    if not args.skip_generation:
        rows = run_answer_quality(
            index=index,
            records=records,
            model_specs=model_specs,
            strategies=strategies,
            budget=args.budget,
            reserve_output=args.reserve_output,
            candidate_pool=args.candidate_pool,
            ollama_url=args.ollama_url,
            auto_score=args.auto_score,
        )
        if args.merge_existing and answer_path.exists():
            rows = merge_existing_rows(read_jsonl(answer_path), rows, model_specs=model_specs)
        write_jsonl(rows, answer_path)
    else:
        rows = read_jsonl(answer_path) if answer_path.exists() else []

    summary = summarize_answer_rows(rows)
    write_summary_csv(summary, output_dir / "llm_answer_quality_summary.csv")
    write_answer_review_packet(rows, output_dir / "llm_answer_review_packet.md")

    cost_rows = build_cost_rows(index, records, budget=args.budget, reserve_output=args.reserve_output)
    write_cost_csv(cost_rows, output_dir / "cost_savings.csv")
    write_cost_table(cost_rows, Path("submission/paper/tables/cost_savings_table.tex"))
    write_quality_table(summary, Path("submission/paper/tables/llm_answer_quality_table.tex"))
    write_cost_plot(cost_rows, Path("submission/paper/figures/tokenpack_cost_savings.png"))
    write_cost_table(cost_rows, output_dir / "cost_savings_table.tex")
    write_quality_table(summary, output_dir / "llm_answer_quality_table.tex")
    write_cost_plot(cost_rows, output_dir / "tokenpack_cost_savings.png")
    print(f"Wrote LLM evidence assets to {output_dir}")
    return 0


def load_dotenv_keys(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        key = name.strip()
        if key in {"OPENAI_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY"} and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


def parse_model_specs(value: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for item in [part.strip().lower() for part in value.split(",") if part.strip()]:
        if item == "small":
            specs.append(ModelSpec("Small local", "ollama", LOCAL_SMALL))
        elif item == "medium":
            specs.append(ModelSpec("Medium local", "ollama", LOCAL_MEDIUM))
        elif item == "large":
            specs.append(ModelSpec("Large cloud", "openai", OPENAI_LARGE))
        else:
            raise ValueError(f"Unknown model alias: {item}")
    return specs


def parse_strategies(value: str) -> list[str]:
    strategies = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [strategy for strategy in strategies if strategy not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategy: {', '.join(unknown)}")
    return strategies


def merge_existing_rows(
    existing: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    model_specs: list[ModelSpec],
) -> list[dict[str, Any]]:
    replaced_labels = {spec.label for spec in model_specs}
    kept = [row for row in existing if row.get("model_label") not in replaced_labels]
    return kept + generated


def run_answer_quality(
    index: ChunkIndex,
    records: list[GoldRecord],
    model_specs: list[ModelSpec],
    strategies: list[str],
    budget: int,
    reserve_output: int,
    candidate_pool: int,
    ollama_url: str,
    auto_score: bool,
) -> list[dict[str, Any]]:
    effective_budget = max(0, budget - reserve_output)
    embedder = make_embedder(backend="hash")
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        query_embedding = embedder.embed([record.query])[0]
        scored = score_chunks(query_embedding, index.chunks, index.embeddings)
        selections = {
            strategy: select_chunks(
                scored,
                strategy=strategy,
                budget=effective_budget,
                candidate_pool=candidate_pool,
                embeddings=index.embeddings,
            )
            for strategy in strategies
        }
        for model_spec in model_specs:
            for strategy, selection in selections.items():
                row = build_answer_row(record_index, record, model_spec, strategy, selection, effective_budget)
                if row["status"] == "ready":
                    started = time.perf_counter()
                    try:
                        row["answer"] = generate_answer(
                            provider=model_spec.provider,
                            model=model_spec.model,
                            query=record.query,
                            chunks=[item.chunk for item in selection.selected],
                            ollama_url=ollama_url,
                        )
                        row["latency_seconds"] = time.perf_counter() - started
                        row["status"] = "completed"
                    except Exception as exc:
                        row["status"] = "skipped"
                        row["error"] = str(exc)
                if auto_score and row.get("answer"):
                    row["auto_score"] = lexical_score(row["answer"], record.answer)
                rows.append(row)
    return rows


def build_answer_row(
    record_index: int,
    record: GoldRecord,
    model_spec: ModelSpec,
    strategy: str,
    selection: SelectionResult,
    effective_budget: int,
) -> dict[str, Any]:
    selected_ids = [item.chunk.id for item in selection.selected]
    evidence_ids = set(record.evidence_chunk_ids)
    matched = sorted(evidence_ids & set(selected_ids))
    over_budget = selection.used_tokens > effective_budget
    status = "invalid-over-budget" if over_budget else "ready"
    if model_spec.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        status = "skipped"
    return {
        "record_index": record_index,
        "model_label": model_spec.label,
        "provider": model_spec.provider,
        "model": model_spec.model,
        "strategy": strategy,
        "query": record.query,
        "gold_answer": record.answer,
        "evidence_chunk_ids": record.evidence_chunk_ids,
        "selected_chunk_ids": selected_ids,
        "matched_evidence_ids": matched,
        "context_tokens": selection.used_tokens,
        "effective_budget": effective_budget,
        "over_budget": over_budget,
        "evidence_recall": len(matched) / max(1, len(evidence_ids)),
        "status": status,
        "answer": "",
        "latency_seconds": None,
        "human_score": None,
        "human_notes": "",
    }


def generate_answer(provider: str, model: str, query: str, chunks: list[Chunk], ollama_url: str) -> str:
    context = render_context(chunks)
    prompt = (
        "/no_think\n"
        "Answer the question using only the provided context. "
        "If the context does not contain the answer, say that the context is insufficient. "
        "Keep the answer to one or two sentences.\n\n"
        f"Context:\n{context}\nQuestion: {query}\nAnswer:"
    )
    if provider == "ollama":
        return call_ollama(prompt, model=model, base_url=ollama_url)
    if provider == "groq":
        return call_groq(prompt, model=model)
    if provider == "openai":
        return call_openai(prompt, model=model)
    raise ValueError(f"Unsupported provider: {provider}")


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 180},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        payload = json.loads(response.read().decode("utf-8"))
    message = payload.get("message") or {}
    return clean_answer(str(message.get("content") or payload.get("response") or ""))


def call_groq(prompt: str, model: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 180,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return clean_answer(str(payload["choices"][0]["message"]["content"]))


def call_openai(prompt: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 180,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return clean_answer(str(payload["choices"][0]["message"]["content"]))


def clean_answer(answer: str) -> str:
    without_thinking = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
    return without_thinking or answer.strip()


def lexical_score(answer: str, gold_answer: str) -> int:
    answer_words = content_words(answer)
    gold_words = content_words(gold_answer)
    if not gold_words:
        return 0
    coverage = len(answer_words & gold_words) / len(gold_words)
    if coverage >= 0.55:
        return 2
    if coverage >= 0.25:
        return 1
    return 0


def content_words(text: str) -> set[str]:
    stop = {"the", "and", "that", "with", "from", "into", "under", "using", "only", "can", "does"}
    return {word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", text.lower()) if word not in stop}


def summarize_answer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("model_label")), str(row.get("strategy"))), []).append(row)
    summary: list[dict[str, Any]] = []
    for (model_label, strategy), group in sorted(grouped.items()):
        completed = [row for row in group if row.get("status") == "completed"]
        human_scored = [row for row in completed if row.get("human_score") is not None]
        auto_scored = [row for row in completed if row.get("auto_score") is not None]
        score_source = "human" if human_scored else "auto-lexical" if auto_scored else "pending"
        score_rows = human_scored or auto_scored
        scores = [float(row.get("human_score") if human_scored else row.get("auto_score")) for row in score_rows]
        summary.append(
            {
                "model_label": model_label,
                "strategy": strategy,
                "completed": len(completed),
                "invalid_over_budget": sum(1 for row in group if row.get("status") == "invalid-over-budget"),
                "skipped": sum(1 for row in group if row.get("status") == "skipped"),
                "avg_context_tokens": mean([float(row.get("context_tokens") or 0) for row in group]),
                "avg_evidence_recall": mean([float(row.get("evidence_recall") or 0) for row in group]),
                "avg_score": mean(scores) if scores else None,
                "correct_rate": mean([1.0 if score >= 2 else 0.0 for score in scores]) if scores else None,
                "score_source": score_source,
            }
        )
    return summary


def build_cost_rows(index: ChunkIndex, records: list[GoldRecord], budget: int, reserve_output: int) -> list[dict[str, Any]]:
    effective_budget = max(0, budget - reserve_output)
    embedder = make_embedder(backend="hash")
    rows: list[dict[str, Any]] = []
    for record in records:
        query_embedding = embedder.embed([record.query])[0]
        scored = score_chunks(query_embedding, index.chunks, index.embeddings)
        for strategy in STRATEGIES:
            selection = select_chunks(
                scored,
                strategy=strategy,
                budget=effective_budget,
                candidate_pool=250,
                embeddings=index.embeddings,
            )
            rows.append(
                {
                    "strategy": strategy,
                    "context_tokens": selection.used_tokens,
                    "over_budget": selection.used_tokens > effective_budget,
                }
            )
    summary: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        selected = [row for row in rows if row["strategy"] == strategy]
        avg_tokens = mean([float(row["context_tokens"]) for row in selected])
        over_budget_rate = mean([1.0 if row["over_budget"] else 0.0 for row in selected])
        scaled_requests = 1_000_000 / max(1.0, avg_tokens)
        for price_name, prices in DEFAULT_PRICES.items():
            input_cost_per_request = avg_tokens / 1_000_000 * prices["input_per_m"]
            output_cost_per_request = reserve_output / 1_000_000 * prices["output_per_m"]
            summary.append(
                {
                    "price_model": price_name,
                    "strategy": strategy,
                    "avg_input_tokens": avg_tokens,
                    "over_budget_rate": over_budget_rate,
                    "input_cost_per_request": input_cost_per_request,
                    "output_reserve_cost_per_request": output_cost_per_request,
                    "requests_per_1m_input_tokens": scaled_requests,
                }
            )
    return add_cost_savings(summary)


def add_cost_savings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_price: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_price.setdefault(str(row["price_model"]), []).append(row)
    for group in by_price.values():
        full_doc = next(row for row in group if row["strategy"] == "full-document")
        full_cost = float(full_doc["input_cost_per_request"])
        full_tokens = float(full_doc["avg_input_tokens"])
        input_price = full_cost / max(full_tokens, 1.0) * 1_000_000
        for row in group:
            paid_tokens_per_1m = float(row["avg_input_tokens"]) / max(full_tokens, 1.0) * 1_000_000
            row["paid_tokens_per_1m_full_document_tokens"] = paid_tokens_per_1m
            row["input_cost_per_1m_full_document_tokens"] = paid_tokens_per_1m / 1_000_000 * input_price
            row["saving_vs_full_document_percent"] = 100.0 * (full_cost - float(row["input_cost_per_request"])) / max(
                full_cost, 1e-12
            )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_cost_csv(rows: list[dict[str, Any]], path: Path) -> None:
    write_summary_csv(rows, path)


def write_answer_review_packet(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# LLM Answer Quality Review Packet",
        "",
        "Score each completed answer: 0 = wrong, 1 = partially correct, 2 = correct and grounded.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## Record {index}",
                "",
                f"- Model: `{row.get('model_label')}` / `{row.get('model')}`",
                f"- Strategy: `{row.get('strategy')}`",
                f"- Status: `{row.get('status')}`",
                f"- Context tokens: `{row.get('context_tokens')}`",
                f"- Evidence recall: `{float(row.get('evidence_recall') or 0):.2f}`",
                f"- Query: {row.get('query')}",
                f"- Gold answer: {row.get('gold_answer')}",
                "",
                "Answer:",
                "",
                "```text",
                str(row.get("answer") or ""),
                "```",
                "",
                "Human score: [ ] 0  [ ] 1  [ ] 2",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_table(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [row for row in rows if row["strategy"] in {"document-prefix", "top-k", "knapsack"}]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Preliminary LLM Answer Quality Pilot on 8 Human-Reviewed Questions}",
        r"\label{tab:llm-answer-quality}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrrl}",
        r"\hline",
        r"Model & Strategy & Completed & Invalid & Avg. Tokens & Avg. Score & Source \\",
        r"\hline",
    ]
    for row in selected:
        strategy = str(row["strategy"])
        prefix = ""
        strategy_text = "TokenPack knapsack" if strategy == "knapsack" else strategy
        score = "--" if row["avg_score"] is None else f"{float(row['avg_score']):.2f}"
        lines.append(
            f"{prefix}{row['model_label']} & {strategy_text} & {row['completed']} & "
            f"{row['invalid_over_budget']} & {float(row['avg_context_tokens']):.1f} & "
            f"{score} & {row['score_source']} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"{\footnotesize Descriptive only: this pilot covers eight reviewed questions and uses automatic lexical scoring, so it is not used for the paper's main claims.}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cost_table(rows: list[dict[str, Any]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["price_model"] == "openai-gpt-4o" and row["strategy"] in {"document-prefix", "full-document", "top-k", "knapsack"}
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Estimated Input Cost Savings per 1M Baseline Tokens}",
        r"\label{tab:cost-savings}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Method & Paid Tokens & Cost (\$) & Saving vs Full Doc \\",
        r"\hline",
    ]
    for row in selected:
        strategy = str(row["strategy"])
        prefix = r"\rowcolor{tokenpackhighlight}" if strategy == "knapsack" else ""
        strategy_text = r"\textbf{TokenPack knapsack}" if strategy == "knapsack" else strategy
        lines.append(
            f"{prefix}{strategy_text} & {float(row['paid_tokens_per_1m_full_document_tokens']):.0f} & "
            f"{float(row['input_cost_per_1m_full_document_tokens']):.3f} & "
            f"{float(row['saving_vs_full_document_percent']):.1f}\\% \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cost_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["price_model"] == "openai-gpt-4o" and row["strategy"] in {"document-prefix", "full-document", "top-k", "knapsack"}
    ]
    labels = ["TokenPack\nknapsack" if row["strategy"] == "knapsack" else str(row["strategy"]).replace("-", "\n") for row in selected]
    costs = [float(row["input_cost_per_1m_full_document_tokens"]) for row in selected]
    colors = ["#9e9e9e", "#c62828", "#ef6c00", "#008a2e"]
    fig, ax = plt.subplots(figsize=(6.2, 3.5), dpi=180)
    bars = ax.bar(labels, costs, color=colors)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"${cost:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Estimated input cost per 1M baseline tokens ($)")
    ax.set_title("TokenPack Reduces Paid Input Tokens")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


if __name__ == "__main__":
    raise SystemExit(main())
