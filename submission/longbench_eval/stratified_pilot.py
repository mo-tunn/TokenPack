"""Reproducible 20/20/20 stratified LongBench v2 sampling.

Samples by official *length* labels, never by model outputs or gold answers.
No contexts are shortened. Oversized cases are tracked explicitly and excluded
from the eligible model-input population (conditional estimand).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from tokenpack.tokenization import TokenCounter

DEFAULT_DATASET = "zai-org/LongBench-v2"
STRATA = ("short", "medium", "long")
REQUIRED = ("context", "question", "choice_A", "choice_B", "choice_C", "choice_D", "answer")


def iter_source(path: Path | None) -> Iterable[dict[str, Any]]:
    if path is None:
        from datasets import load_dataset
        yield from load_dataset(DEFAULT_DATASET, split="train", streaming=True)
        return
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        file = pq.ParquetFile(path)
        for batch in file.iter_batches(batch_size=8):
            yield from batch.to_pylist()
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        yield from obj["data"] if isinstance(obj, dict) else obj
    else:
        raise ValueError("Supported local formats: .jsonl, .json, .parquet")


def sample(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    per_stratum: int,
    max_source_tokens: int,
    token_counter: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_stratum < 1 or max_source_tokens < 100:
        raise ValueError("Sample count and token guard must be positive")
    eligible: dict[str, list[tuple[str, dict[str, Any], int]]] = {s: [] for s in STRATA}
    sizes = collections.Counter()
    exclusion = collections.Counter()
    domains: dict[str, collections.Counter] = {s: collections.Counter() for s in STRATA}
    seen = set()
    all_rows = 0
    for row in rows:
        all_rows += 1
        category = str(row.get("length", "")).strip().lower()
        sizes[category] += 1
        if category not in eligible:
            exclusion["unknown_length"] += 1
            continue
        case = str(row.get("_id") or row.get("id") or "").strip()
        if not case:
            raise ValueError("Missing case ID")
        if case in seen:
            raise ValueError("Duplicate LongBench case ID")
        seen.add(case)
        for key in REQUIRED:
            if not str(row.get(key) or "").strip():
                raise ValueError(f"Missing {key} for LongBench case")
        if str(row["answer"]).strip().upper() not in "ABCD":
            raise ValueError("Invalid four-way gold answer")
        n_tokens = int(token_counter.count(str(row["context"])))
        if n_tokens > max_source_tokens:
            exclusion["over_token_guard_"+category] += 1
            continue
        domains[category][str(row.get("domain") or "unknown")] += 1
        rank = hashlib.sha256(f"stratified-longbench-v2:{seed}:{case}".encode()).hexdigest()
        eligible[category].append((rank, row, n_tokens))
    missing = {s: (per_stratum - len(eligible[s])) for s in STRATA if len(eligible[s]) < per_stratum}
    if missing:
        raise ValueError(
            "Not enough full-context-eligible cases per length. "
            f"Missing={missing}, pool sizes={dict(sizes)}, excluded={dict(exclusion)}. "
            "Do not silently relabel or truncate questions."
        )
    selected: list[dict[str, Any]] = []
    selected_info: list[dict[str, Any]] = []
    for category in STRATA:
        subset = sorted(eligible[category], key=lambda item: item[0])[:per_stratum]
        for rank, row, count in subset:
            selected.append(row)
            selected_info.append({
                "case_id": str(row.get("_id") or row.get("id")),
                "length": category,
                "source_tokens_est": count,
                "domain": str(row.get("domain") or ""),
                "difficulty": str(row.get("difficulty") or ""),
                "selection_hash": rank,
            })
    details = {
        "dataset": DEFAULT_DATASET,
        "seed": seed,
        "sample_per_stratum": per_stratum,
        "expected_cases": per_stratum * len(STRATA),
        "expected_tasks_for_5_methods": per_stratum * len(STRATA) * 5,
        "total_scanned": all_rows,
        "source_length_counts": dict(sizes),
        "eligible_counts": {s: len(eligible[s]) for s in STRATA},
        "excluded": dict(exclusion),
        "sample_counts": dict(collections.Counter(x["length"] for x in selected_info)),
        "sample_difficulties": dict(collections.Counter(x["difficulty"] for x in selected_info)),
        "sample_domains": dict(collections.Counter(x["domain"] for x in selected_info)),
        "max_source_tokens_est": max_source_tokens,
        "tokenizer": token_counter.encoding_name,
        "max_sample_tokens_est": max(x["source_tokens_est"] for x in selected_info),
        "selected": selected_info,
        "note": "Estimation is conditional on contexts fitting the token guard; no truncation or source filter beyond guard.",
    }
    return selected, details


def write_jsonl(rows: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting previous experimental samples.
    with destination.open("x", encoding="utf-8", newline="\n") as fp:
        for obj in rows:
            fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Select 20 short/20 medium/20 long LongBench v2 cases reproducibly.")
    p.add_argument("--data-file", type=Path, default=None, help="Optional local JSONL, JSON, Parquet; else Hugging Face streaming")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--per-stratum", type=int, default=20)
    p.add_argument("--max-source-tokens", type=int, default=210000, help="Estimated input safety guard; default fits 272k model context.")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Pilot output directory is not empty. Choose a new directory.")
    counter = TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("Missing tiktoken encoder; install tiktoken and retry. No regex fallback is allowed for eligibility checks.")
    selected, report = sample(
        iter_source(args.data_file), seed=args.seed,
        per_stratum=args.per_stratum, max_source_tokens=args.max_source_tokens,
        token_counter=counter,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "selected_cases.jsonl"
    write_jsonl(selected, output)
    report["source_file"] = str(args.data_file.resolve()) if args.data_file else None
    report["selected_cases_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected": len(selected), "counts": report["sample_counts"],
        "eligible": report["eligible_counts"], "excluded": report["excluded"],
        "selected_file": str(output.resolve()),
        "source_token_limit_est": args.max_source_tokens,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
