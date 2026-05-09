from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import modal

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = next(
    (
        parent / "src"
        for parent in (THIS_DIR, *THIS_DIR.parents)
        if (parent / "src" / "tokenpack").exists()
    ),
    Path("/root/src"),
)
REMOTE_EVAL_DIR = Path("/root/submission/longbench_eval")
REMOTE_SRC_DIR = Path("/root/src")
for path in (THIS_DIR, SRC_DIR, REMOTE_EVAL_DIR, REMOTE_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from client import build_tasks  # noqa: E402
from eval_utils import (  # noqa: E402
    pairwise_grounded_rows,
    pairwise_rows,
    parse_choice,
    parse_grounded_answer,
    parse_grounding_judge,
    quote_found_in_context,
    read_jsonl,
    render_grounded_prompt,
    render_grounding_judge_prompt,
    render_mc_prompt,
    summarize_grounded_rows,
    summarize_rows,
    write_csv,
    write_jsonl,
)


APP_NAME = "tokenpack-longbench-eval"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
GPU = os.environ.get("TOKENPACK_MODAL_GPU", "L40S")
SECRET_NAME = os.environ.get("TOKENPACK_MODAL_SECRET_NAME", "")
REMOTE_WORK_DIR = Path("/data/tokenpack_longbench_eval")
REMOTE_SHARD_DIR = REMOTE_WORK_DIR / "shards"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.5",
        "transformers>=4.45.0",
        "huggingface_hub>=0.24.0",
        "datasets>=2.20.0",
        "llmlingua>=0.2.2",
        "sentence-transformers>=3.0.0",
    )
    .add_local_dir(str(THIS_DIR), remote_path="/root/submission/longbench_eval")
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
)

volume = modal.Volume.from_name("tokenpack-longbench-eval", create_if_missing=True)
secrets = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []
app = modal.App(APP_NAME)

_LLM: Any | None = None
_LLM_MODEL_ID: str | None = None


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    volumes={"/data": volume},
    secrets=secrets,
)
def build_tasks_remote(args_payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf-cache")
    args = SimpleNamespace(**args_payload)
    tasks, report = build_tasks(args)
    REMOTE_WORK_DIR.mkdir(parents=True, exist_ok=True)
    tasks_path = REMOTE_WORK_DIR / "longbench_generation_tasks.jsonl"
    report_path = REMOTE_WORK_DIR / "task_report.json"
    write_jsonl(tasks, tasks_path)
    import json

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    volume.commit()
    return {"tasks": tasks, "report": report, "remote_tasks_path": str(tasks_path), "remote_report_path": str(report_path)}


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    volumes={"/data": volume},
    secrets=secrets,
)
def run_eval_shard(
    tasks: list[dict[str, Any]],
    shard_id: int,
    model_id: str = DEFAULT_MODEL_ID,
    max_answer_tokens: int = 8,
    batch_size: int = 2,
) -> dict[str, Any]:
    llm = _get_llm(model_id)

    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens)
    prompts = [render_mc_prompt(task) for task in tasks]
    started = time.perf_counter()
    outputs = _generate_texts(llm, prompts, params, batch_size=batch_size)
    elapsed = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for task, raw in zip(tasks, outputs):
        prediction = parse_choice(raw)
        gold = str(task.get("answer") or "").strip().upper()
        rows.append(
            {
                **{key: value for key, value in task.items() if key != "context"},
                "status": "completed",
                "model_id": model_id,
                "raw_answer": raw,
                "prediction": prediction,
                "correct": bool(prediction and prediction == gold),
                "answer_latency_seconds": elapsed / max(1, len(tasks)),
            }
        )

    REMOTE_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = REMOTE_SHARD_DIR / f"shard-{shard_id:04d}.jsonl"
    write_jsonl(rows, shard_path)
    volume.commit()
    return {"shard_id": shard_id, "rows": rows, "remote_path": str(shard_path)}


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 60 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    volumes={"/data": volume},
    secrets=secrets,
)
def run_grounded_eval_shard(
    tasks: list[dict[str, Any]],
    shard_id: int,
    model_id: str = DEFAULT_MODEL_ID,
    max_answer_tokens: int = 192,
    max_judge_tokens: int = 160,
    batch_size: int = 1,
) -> dict[str, Any]:
    llm = _get_llm(model_id)

    from vllm import SamplingParams

    answer_params = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens)
    answer_prompts = [render_grounded_prompt(task) for task in tasks]
    started = time.perf_counter()
    raw_answers = _generate_texts(llm, answer_prompts, answer_params, batch_size=batch_size)
    answer_elapsed = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for task, raw_answer in zip(tasks, raw_answers):
        parsed = parse_grounded_answer(raw_answer)
        prediction = parsed["prediction"]
        gold = str(task.get("answer") or "").strip().upper()
        quote_found = quote_found_in_context(parsed["evidence_quote"], str(task.get("context") or ""))
        rows.append(
            {
                **{key: value for key, value in task.items() if key != "context"},
                "status": "completed",
                "model_id": model_id,
                "raw_answer": raw_answer,
                "prediction": prediction,
                "rationale": parsed["rationale"],
                "evidence_quote": parsed["evidence_quote"],
                "quote_found": quote_found,
                "correct": bool(prediction and prediction == gold),
                "answer_latency_seconds": answer_elapsed / max(1, len(tasks)),
            }
        )

    judge_params = SamplingParams(temperature=0.0, max_tokens=max_judge_tokens)
    judge_prompts = [render_grounding_judge_prompt(task, row) for task, row in zip(tasks, rows)]
    judge_started = time.perf_counter()
    raw_judges = _generate_texts(llm, judge_prompts, judge_params, batch_size=batch_size)
    judge_elapsed = time.perf_counter() - judge_started

    for row, raw_judge in zip(rows, raw_judges):
        parsed_judge = parse_grounding_judge(raw_judge)
        row.update(parsed_judge)
        row["raw_judge"] = raw_judge
        row["grounded"] = bool(
            row.get("prediction")
            and row.get("quote_found") is True
            and row.get("supported_answer") is True
            and row.get("supported_rationale") is True
            and row.get("evidence_quote_supports_answer") is True
            and row.get("unsupported_claims") is not True
        )
        row["hallucinated"] = bool(row.get("unsupported_claims") is True)
        row["strict_grounding_failure"] = bool(
            row.get("grounded") is not True
            or row.get("quote_found") is not True
            or row.get("supported_answer") is not True
            or row.get("supported_rationale") is not True
            or row.get("evidence_quote_supports_answer") is not True
        )
        row["judge_latency_seconds"] = judge_elapsed / max(1, len(rows))

    REMOTE_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = REMOTE_SHARD_DIR / f"grounded-shard-{shard_id:04d}.jsonl"
    write_jsonl(rows, shard_path)
    volume.commit()
    return {"shard_id": shard_id, "rows": rows, "remote_path": str(shard_path)}


@app.local_entrypoint()
def run(
    tasks_jsonl: str,
    output_jsonl: str = "submission/results/longbench_eval/longbench_generation.jsonl",
    model_id: str = DEFAULT_MODEL_ID,
    shard_size: int = 80,
    max_answer_tokens: int = 8,
    batch_size: int = 2,
) -> None:
    tasks = read_jsonl(Path(tasks_jsonl))
    shards = [tasks[index : index + shard_size] for index in range(0, len(tasks), shard_size)]
    all_rows: list[dict[str, Any]] = []
    shard_ids = list(range(len(shards)))
    for result in run_eval_shard.map(
        shards,
        shard_ids,
        kwargs={
            "model_id": model_id,
            "max_answer_tokens": max_answer_tokens,
            "batch_size": batch_size,
        },
        order_outputs=True,
    ):
        all_rows.extend(result["rows"])
        print(f"Completed shard {result['shard_id']} -> {result['remote_path']}")
    write_jsonl(all_rows, Path(output_jsonl))
    print(f"Wrote merged results to {output_jsonl}")


@app.local_entrypoint()
def grounded_from_tasks(
    tasks_jsonl: str,
    output_dir: str = "submission/results/longbench_v2_grounded100",
    model_id: str = DEFAULT_MODEL_ID,
    max_cases: int = 0,
    shard_size: int = 80,
    max_answer_tokens: int = 192,
    max_judge_tokens: int = 160,
    batch_size: int = 1,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(Path(tasks_jsonl))
    if max_cases > 0:
        allowed_cases: list[str] = []
        seen_cases: set[str] = set()
        for task in tasks:
            case_id = str(task.get("case_id"))
            if case_id in seen_cases:
                continue
            seen_cases.add(case_id)
            allowed_cases.append(case_id)
            if len(allowed_cases) >= max_cases:
                break
        allowed = set(allowed_cases)
        tasks = [task for task in tasks if str(task.get("case_id")) in allowed]
    shards = [tasks[index : index + shard_size] for index in range(0, len(tasks), shard_size)]
    all_rows: list[dict[str, Any]] = []
    shard_ids = list(range(len(shards)))
    for result in run_grounded_eval_shard.map(
        shards,
        shard_ids,
        kwargs={
            "model_id": model_id,
            "max_answer_tokens": max_answer_tokens,
            "max_judge_tokens": max_judge_tokens,
            "batch_size": batch_size,
        },
        order_outputs=True,
    ):
        all_rows.extend(result["rows"])
        print(f"Completed grounded shard {result['shard_id']} -> {result['remote_path']}")

    results_path = output_path / "longbench_grounded_results.jsonl"
    write_jsonl(all_rows, results_path)
    summary = summarize_grounded_rows(all_rows)
    pairwise = pairwise_grounded_rows(all_rows)
    write_csv(summary, output_path / "longbench_grounded_summary.csv")
    write_csv(pairwise, output_path / "longbench_grounded_pairwise.csv")
    _write_grounded_readout(summary, pairwise, output_path / "longbench_grounded_readout.md")
    print(f"Wrote grounded results and summaries to {output_path}")


@app.local_entrypoint()
def build_and_run(
    output_dir: str = "submission/results/longbench_v2_pilot30_modal",
    limit: int = 30,
    source_min_tokens: int = 8000,
    source_max_tokens: int = 24000,
    max_scanned: int = 503,
    backend: str = "hash",
    scoring: str = "evidence-hybrid",
    selection_strategy: str = "knapsack-redundancy",
    budget_ratio: float = 0.50,
    context_order: str = "score",
    compression_rate: float = 0.50,
    longllmlingua_model: str = "gpt2",
    compression_device_map: str = "cuda",
    reranker: str = "none",
    reranker_model: str = "BAAI/bge-reranker-base",
    reranker_candidate_pool: int = 80,
    reranker_weight: float = 0.35,
    cascade_frontier: bool = False,
    shard_size: int = 80,
    batch_size: int = 2,
    model_id: str = DEFAULT_MODEL_ID,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "data_file": None,
        "limit": limit,
        "source_min_tokens": source_min_tokens,
        "source_max_tokens": source_max_tokens,
        "max_scanned": max_scanned,
        "backend": backend,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunker": "structure-aware",
        "scoring": scoring,
        "selection_strategy": selection_strategy,
        "budget_ratio": budget_ratio,
        "context_order": context_order,
        "candidate_pool": 300,
        "target_tokens": 650,
        "min_tokens": 120,
        "max_tokens": 900,
        "longllmlingua_model": longllmlingua_model,
        "compression_rate": compression_rate,
        "compression_device_map": compression_device_map,
        "compression_allow_download": True,
        "reranker": reranker,
        "reranker_model": reranker_model,
        "reranker_candidate_pool": reranker_candidate_pool,
        "reranker_weight": reranker_weight,
        "reranker_allow_download": True,
        "cascade_frontier": cascade_frontier,
        "skip_compression": False,
    }
    build_result = build_tasks_remote.remote(args_payload)
    tasks = build_result["tasks"]
    write_jsonl(tasks, output_path / "longbench_generation_tasks.jsonl")
    import json

    (output_path / "task_report.json").write_text(
        json.dumps(build_result["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Built {len(tasks)} tasks on Modal -> {build_result['remote_tasks_path']}")

    shards = [tasks[index : index + shard_size] for index in range(0, len(tasks), shard_size)]
    all_rows: list[dict[str, Any]] = []
    shard_ids = list(range(len(shards)))
    for result in run_eval_shard.map(
        shards,
        shard_ids,
        kwargs={
            "model_id": model_id,
            "max_answer_tokens": 8,
            "batch_size": batch_size,
        },
        order_outputs=True,
    ):
        all_rows.extend(result["rows"])
        print(f"Completed shard {result['shard_id']} -> {result['remote_path']}")

    results_path = output_path / "longbench_generation_results.jsonl"
    write_jsonl(all_rows, results_path)
    summary = summarize_rows(all_rows)
    pairwise = pairwise_rows(all_rows)
    write_csv(summary, output_path / "longbench_generation_summary.csv")
    write_csv(pairwise, output_path / "longbench_generation_pairwise.csv")
    _write_readout(summary, pairwise, output_path / "longbench_generation_readout.md")
    if context_order != "score":
        _write_readout(summary, pairwise, output_path / "ordering_ablation_readout.md")
    if reranker != "none":
        _write_readout(summary, pairwise, output_path / "reranker_ablation_readout.md")
    if cascade_frontier:
        _write_readout(summary, pairwise, output_path / "cascade_frontier_readout.md")
    print(f"Wrote merged results and summaries to {output_path}")


def _get_llm(model_id: str):
    global _LLM, _LLM_MODEL_ID
    if _LLM is not None and _LLM_MODEL_ID == model_id:
        return _LLM
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf-cache")
    from vllm import LLM

    _LLM = LLM(
        model=model_id,
        download_dir="/data/hf-cache",
        dtype="auto",
        trust_remote_code=True,
        max_model_len=32768,
    )
    _LLM_MODEL_ID = model_id
    return _LLM


def _generate_texts(llm: Any, prompts: list[str], params: Any, *, batch_size: int) -> list[str]:
    texts: list[str] = []
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(prompts), safe_batch_size):
        outputs = llm.generate(prompts[start : start + safe_batch_size], params)
        texts.extend(_first_output(output) for output in outputs)
    return texts


def _first_output(output: Any) -> str:
    generations = getattr(output, "outputs", None) or []
    if not generations:
        return ""
    return str(getattr(generations[0], "text", "")).strip()


def _write_readout(summary: list[dict[str, Any]], pairwise: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# LongBench v2 Modal Pilot Readout",
        "",
        "| Method | Runs | Accuracy | Avg context toks | Saving | Parse fail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {int(row['runs'])} | {float(row['accuracy']):.3f} | "
            f"{float(row['avg_context_tokens']):.0f} | {float(row['avg_token_saving_vs_full']):.3f} | "
            f"{float(row['parse_failure_rate']):.3f} |"
        )
    lines.extend(["", "## Pairwise vs LongLLMLingua", "", "| Method | Compared | Win | Tie | Loss |", "|---|---:|---:|---:|---:|"])
    for row in pairwise:
        lines.append(
            f"| {row['method']} | {int(row['compared'])} | {int(row['wins'])} | "
            f"{int(row['ties'])} | {int(row['losses'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grounded_readout(summary: list[dict[str, Any]], pairwise: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# LongBench v2 Grounded Readout",
        "",
        "| Method | Runs | Acc. | Grounded acc. | Halluc. claims | Quote found | Correct unsupported | Saving |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {int(row['runs'])} | {float(row['accuracy']):.3f} | "
            f"{float(row['grounded_accuracy']):.3f} | {float(row['hallucination_rate']):.3f} | "
            f"{float(row['quote_found_rate']):.3f} | {float(row['correct_but_unsupported_rate']):.3f} | "
            f"{float(row['avg_token_saving_vs_full']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Grounding Diagnostic Split",
            "",
            "| Method | Unsupported claims | Strict grounding failure | Answer supported | Rationale supported | Quote supports answer | Quote missing |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['method']} | {float(row['unsupported_claim_rate']):.3f} | "
            f"{float(row['strict_grounding_failure_rate']):.3f} | {float(row['answer_supported_rate']):.3f} | "
            f"{float(row['rationale_supported_rate']):.3f} | {float(row['quote_supports_answer_rate']):.3f} | "
            f"{float(row['quote_missing_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Grounded Accuracy vs LongLLMLingua",
            "",
            "| Method | Compared | Win | Tie | Loss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise:
        lines.append(
            f"| {row['method']} | {int(row['compared'])} | {int(row['wins'])} | "
            f"{int(row['ties'])} | {int(row['losses'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
