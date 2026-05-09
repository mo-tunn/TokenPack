from __future__ import annotations

import json
import time
from pathlib import Path

from tokenpack.dataset import GoldRecord, propose_gold_records
from tokenpack.embeddings import Embedder, cosine
from tokenpack.index import ChunkIndex
from tokenpack.models import ScoredChunk, SelectionResult
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks


STRATEGIES = [
    "document-prefix",
    "full-document",
    "top-k",
    "budget-top-k",
    "greedy-value",
    "greedy-density",
    "mmr",
    "knapsack",
    "knapsack-redundancy",
    "knapsack-coverage",
]


def synthetic_queries(index: ChunkIndex, sample_size: int = 12) -> list[dict]:
    return [
        {
            "query": record.query,
            "evidence_chunk_id": record.evidence_chunk_ids[0],
            "source_path": record.source_path,
        }
        for record in propose_gold_records(index, sample_size=sample_size)
    ]


def run_benchmark(
    index: ChunkIndex,
    embedder: Embedder,
    budget: int,
    reserve_output: int,
    sample_size: int = 12,
    candidate_pool: int = 250,
    scoring: str = "cosine",
) -> dict:
    """Developer smoke benchmark using auto-proposed single-evidence queries."""

    records = propose_gold_records(index, sample_size=sample_size)
    payload = run_gold_benchmark(
        index=index,
        embedder=embedder,
        records=records,
        budgets=[budget],
        reserve_output=reserve_output,
        candidate_pool=candidate_pool,
        scoring=scoring,
    )
    return payload["budgets"][0] | {
        "mode": "smoke",
        "scoring": scoring,
        "query_count": len(records),
        "queries": payload["budgets"][0]["queries"],
    }


def run_gold_benchmark(
    index: ChunkIndex,
    embedder: Embedder,
    records: list[GoldRecord],
    budgets: list[int],
    reserve_output: int,
    candidate_pool: int = 250,
    strategies: list[str] | None = None,
    redundancy_penalty: float = 0.35,
    scoring: str = "cosine",
) -> dict:
    strategy_names = strategies or STRATEGIES
    budget_runs = [
        _run_gold_for_budget(
            index=index,
            embedder=embedder,
            records=records,
            budget=budget,
            reserve_output=reserve_output,
            candidate_pool=candidate_pool,
            strategies=strategy_names,
            redundancy_penalty=redundancy_penalty,
            scoring=scoring,
        )
        for budget in budgets
    ]
    return {
        "mode": "gold",
        "scoring": scoring,
        "query_count": len(records),
        "reserve_output": reserve_output,
        "strategies": strategy_names,
        "budgets": budget_runs,
    }


def save_benchmark(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_gold_for_budget(
    index: ChunkIndex,
    embedder: Embedder,
    records: list[GoldRecord],
    budget: int,
    reserve_output: int,
    candidate_pool: int,
    strategies: list[str],
    redundancy_penalty: float,
    scoring: str,
) -> dict:
    effective_budget = max(0, budget - reserve_output)
    totals = {
        strategy: {
            "recall": 0.0,
            "precision": 0.0,
            "coverage": 0.0,
            "used_tokens": 0,
            "budget_utilization": 0.0,
            "total_value": 0.0,
            "redundancy": 0.0,
            "latency": 0.0,
            "runs": 0,
        }
        for strategy in strategies
    }
    per_query = []
    for record in records:
        query_embedding = embedder.embed([record.query])[0]
        scored = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            scoring=scoring,
            query_text=record.query,
        )
        scored_redundant = score_chunks(
            query_embedding,
            index.chunks,
            index.embeddings,
            redundancy_penalty=redundancy_penalty,
            scoring=scoring,
            query_text=record.query,
        )
        query_result = {
            "query": record.query,
            "evidence_chunk_ids": record.evidence_chunk_ids,
            "strategies": {},
        }
        for strategy in strategies:
            source_scores = scored_redundant if strategy == "knapsack-redundancy" else scored
            started = time.perf_counter()
            result = select_chunks(
                source_scores,
                strategy=strategy,
                budget=effective_budget,
                candidate_pool=candidate_pool,
                embeddings=index.embeddings,
                coverage_query=record.query,
            )
            elapsed = time.perf_counter() - started
            metrics = _selection_metrics(result, record.evidence_chunk_ids, effective_budget)
            metrics["latency_seconds"] = elapsed
            _accumulate(totals[strategy], metrics, result)
            query_result["strategies"][strategy] = metrics | {
                "selected_count": len(result.selected),
                "selected_chunk_ids": [item.chunk.id for item in result.selected],
            }
        per_query.append(query_result)

    summary = {strategy: _summarize(values) for strategy, values in totals.items()}
    return {
        "budget": budget,
        "reserve_output": reserve_output,
        "effective_budget": effective_budget,
        "query_count": len(records),
        "summary": summary,
        "queries": per_query,
    }


def _selection_metrics(result: SelectionResult, evidence_ids: list[str], budget: int) -> dict:
    selected_ids = {item.chunk.id for item in result.selected}
    evidence_set = set(evidence_ids)
    matched = selected_ids & evidence_set
    recall = len(matched) / max(1, len(evidence_set))
    precision = len(matched) / max(1, len(selected_ids))
    return {
        "evidence_recall_at_budget": recall,
        "evidence_precision": precision,
        "coverage_ratio": 1.0 if evidence_set and evidence_set.issubset(selected_ids) else 0.0,
        "used_tokens": result.used_tokens,
        "budget_utilization": result.used_tokens / max(1, budget),
        "over_budget": result.used_tokens > budget,
        "over_budget_tokens": max(0, result.used_tokens - budget),
        "total_value": result.total_value,
        "value_density": result.total_value / max(1, result.used_tokens),
        "redundancy_score": redundancy_score(result.selected),
    }


def redundancy_score(selected: list[ScoredChunk]) -> float:
    embeddings = [item.embedding for item in selected if item.embedding is not None]
    if len(embeddings) < 2:
        return 0.0
    total = 0.0
    comparisons = 0
    for left_index, left in enumerate(embeddings):
        for right in embeddings[left_index + 1 :]:
            total += max(0.0, cosine(left, right))
            comparisons += 1
    return total / max(1, comparisons)


def _accumulate(total: dict, metrics: dict, result: SelectionResult) -> None:
    total["recall"] += metrics["evidence_recall_at_budget"]
    total["precision"] += metrics["evidence_precision"]
    total["coverage"] += metrics["coverage_ratio"]
    total["used_tokens"] += result.used_tokens
    total["budget_utilization"] += metrics["budget_utilization"]
    total["over_budget"] = total.get("over_budget", 0) + int(metrics["over_budget"])
    total["over_budget_tokens"] = total.get("over_budget_tokens", 0) + metrics["over_budget_tokens"]
    total["total_value"] += result.total_value
    total["redundancy"] += metrics["redundancy_score"]
    total["latency"] += metrics["latency_seconds"]
    total["runs"] += 1


def _summarize(values: dict) -> dict:
    runs = max(1, values["runs"])
    avg_tokens = values["used_tokens"] / runs
    avg_value = values["total_value"] / runs
    return {
        "evidence_recall_at_budget": values["recall"] / runs,
        "evidence_precision": values["precision"] / runs,
        "coverage_ratio": values["coverage"] / runs,
        "avg_used_tokens": avg_tokens,
        "budget_utilization": values["budget_utilization"] / runs,
        "over_budget_rate": values.get("over_budget", 0) / runs,
        "avg_over_budget_tokens": values.get("over_budget_tokens", 0) / runs,
        "avg_total_value": avg_value,
        "value_density": avg_value / max(1.0, avg_tokens),
        "redundancy_score": values["redundancy"] / runs,
        "latency_seconds": values["latency"] / runs,
    }
