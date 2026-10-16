from __future__ import annotations

import os
import copy
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
    .add_local_dir(str(THIS_DIR.parent / "evaluation"), remote_path="/root/submission/evaluation")
)

volume = modal.Volume.from_name("tokenpack-longbench-eval", create_if_missing=True)
secrets = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []
app = modal.App(APP_NAME)

_LLM: Any | None = None
_LLM_CACHE_KEY: tuple[str, int, bool, float] | None = None


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
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    max_answer_tokens: int = 8,
    batch_size: int = 2,
    latency_mode: bool = False,
) -> dict[str, Any]:
    llm = _get_llm(
        model_id,
        max_model_len=max_model_len,
        enable_yarn=enable_yarn,
        yarn_factor=yarn_factor,
    )

    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens)
    prompts = [render_mc_prompt(task) for task in tasks]
    if latency_mode:
        outputs, answer_latencies = _generate_texts_with_latencies(llm, prompts, params)
    else:
        started = time.perf_counter()
        outputs = _generate_texts(llm, prompts, params, batch_size=batch_size)
        elapsed = time.perf_counter() - started
        answer_latencies = [elapsed / max(1, len(tasks)) for _ in tasks]

    rows: list[dict[str, Any]] = []
    for task, raw, answer_latency in zip(tasks, outputs, answer_latencies):
        timing = copy.deepcopy(task.get("timing"))
        if timing:
            timing["stages"]["generation"] = {"status": "measured", "seconds": answer_latency}
            timing["generation_mode"] = "per_request_hot" if latency_mode else "batch_average_not_request_latency"
        prediction = parse_choice(raw)
        gold = str(task.get("answer") or "").strip().upper()
        preprocessing_seconds = (float(task["selection_seconds"]) + float(task["compression_seconds"])
                                 if task.get("selection_seconds") is not None and task.get("compression_seconds") is not None else None)
        rows.append(
            {
                **{key: value for key, value in task.items() if key != "context"},
                "schema_version": "2", "metric_version": "2.0",
                "status": "completed",
                "model_id": model_id,
                "raw_answer": raw,
                "prediction": prediction,
                "correct": bool(prediction and prediction == gold),
                "answer_latency_seconds": answer_latency,
                "preprocessing_seconds": preprocessing_seconds,
                "recorded_stage_total_seconds": preprocessing_seconds + answer_latency if preprocessing_seconds is not None else None,
                "latency_scope": "selection_compression_generation_only_not_end_to_end",
                "timing": timing,
                "prompt_tokens": len(llm.get_tokenizer().encode(render_mc_prompt(task))),
                "prompt_tokenizer": model_id,
                "prompt_tokenizer_revision": "unknown",
                "latency_mode": latency_mode,
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
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    max_answer_tokens: int = 192,
    max_judge_tokens: int = 160,
    batch_size: int = 1,
) -> dict[str, Any]:
    llm = _get_llm(
        model_id,
        max_model_len=max_model_len,
        enable_yarn=enable_yarn,
        yarn_factor=yarn_factor,
    )

    from vllm import SamplingParams

    answer_params = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens)
    answer_prompts = [render_grounded_prompt(task) for task in tasks]
    started = time.perf_counter()
    raw_answers = _generate_texts(llm, answer_prompts, answer_params, batch_size=batch_size)
    answer_elapsed = time.perf_counter() - started

    rows: list[dict[str, Any]] = []
    for task, raw_answer in zip(tasks, raw_answers):
        timing = copy.deepcopy(task.get("timing"))
        if timing:
            timing["stages"]["generation"] = {"status": "measured", "seconds": answer_elapsed / max(1, len(tasks))}
            timing["generation_mode"] = "batch_average_not_request_latency"
        parsed = parse_grounded_answer(raw_answer)
        prediction = parsed["prediction"]
        gold = str(task.get("answer") or "").strip().upper()
        quote_found = quote_found_in_context(parsed["evidence_quote"], str(task.get("context") or ""))
        rows.append(
            {
                **{key: value for key, value in task.items() if key != "context"},
                "schema_version": "2", "metric_version": "2.0",
                "status": "completed",
                "model_id": model_id,
                "raw_answer": raw_answer,
                "prediction": prediction,
                "rationale": parsed["rationale"],
                "evidence_quote": parsed["evidence_quote"],
                "quote_found": quote_found,
                "correct": bool(prediction and prediction == gold),
                "answer_latency_seconds": answer_elapsed / max(1, len(tasks)),
                "timing": timing,
                "prompt_tokens": len(llm.get_tokenizer().encode(render_grounded_prompt(task))),
                "prompt_tokenizer": model_id, "prompt_tokenizer_revision": "unknown",
                "latency_scope": "generation_batch_average_not_end_to_end",
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
        if row.get("judge_parse_failure") is True:
            row["grounded"] = None
            row["hallucinated"] = None
            row["strict_grounding_failure"] = None
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
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    shard_size: int = 80,
    max_answer_tokens: int = 8,
    batch_size: int = 2,
    latency_mode: bool = False,
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
            "max_model_len": max_model_len,
            "enable_yarn": enable_yarn,
            "yarn_factor": yarn_factor,
            "max_answer_tokens": max_answer_tokens,
            "batch_size": batch_size,
            "latency_mode": latency_mode,
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
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
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
            "max_model_len": max_model_len,
            "enable_yarn": enable_yarn,
            "yarn_factor": yarn_factor,
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
    summary = summarize_grounded_rows(all_rows, expected=tasks)
    pairwise = pairwise_grounded_rows(all_rows, expected=tasks)
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
    scoring: str = "evidence-hybrid",
    selection_strategy: str = "budget-top-k",
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
    diagnostic_selectors: bool = False,
    shard_size: int = 80,
    batch_size: int = 2,
    latency_mode: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    skip_compression: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    args_payload = {
        "data_file": None,
        "limit": limit,
        "source_min_tokens": source_min_tokens,
        "source_max_tokens": source_max_tokens,
        "max_scanned": max_scanned,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_allow_download": True,
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
        "diagnostic_selectors": diagnostic_selectors,
        "skip_compression": skip_compression,
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
            "max_model_len": max_model_len,
            "enable_yarn": enable_yarn,
            "yarn_factor": yarn_factor,
            "max_answer_tokens": 8,
            "batch_size": batch_size,
            "latency_mode": latency_mode,
        },
        order_outputs=True,
    ):
        all_rows.extend(result["rows"])
        print(f"Completed shard {result['shard_id']} -> {result['remote_path']}")

    results_path = output_path / "longbench_generation_results.jsonl"
    write_jsonl(all_rows, results_path)
    summary = summarize_rows(all_rows, expected=tasks)
    pairwise = pairwise_rows(all_rows, expected=tasks)
    pairwise_vs_production_rag = pairwise_rows(all_rows, baseline="production-rag-50", expected=tasks)
    write_csv(summary, output_path / "longbench_generation_summary.csv")
    write_csv(pairwise, output_path / "longbench_generation_pairwise.csv")
    write_csv(pairwise_vs_production_rag, output_path / "longbench_generation_pairwise_vs_production_rag.csv")
    _write_readout(summary, pairwise, output_path / "longbench_generation_readout.md")
    if context_order != "score":
        _write_readout(summary, pairwise, output_path / "ordering_ablation_readout.md")
    if reranker != "none":
        _write_readout(summary, pairwise, output_path / "reranker_ablation_readout.md")
    if cascade_frontier:
        _write_readout(summary, pairwise, output_path / "cascade_frontier_readout.md")
    print(f"Wrote merged results and summaries to {output_path}")


def _get_llm(
    model_id: str,
    *,
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
):
    global _LLM, _LLM_CACHE_KEY
    cache_key = (model_id, int(max_model_len), bool(enable_yarn), float(yarn_factor))
    if _LLM is not None and _LLM_CACHE_KEY == cache_key:
        return _LLM
    os.environ.setdefault("HF_HOME", "/data/hf-cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data/hf-cache")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
    if enable_yarn or max_model_len > 32768:
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": model_id,
        "download_dir": "/data/hf-cache",
        "dtype": "auto",
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.95,
    }
    if enable_yarn:
        llm_kwargs["hf_overrides"] = {
            "rope_theta": 1_000_000,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": float(yarn_factor),
                "original_max_position_embeddings": 32768,
            },
            "max_model_len": int(max_model_len),
        }

    _LLM = LLM(
        **llm_kwargs,
    )
    _LLM_CACHE_KEY = cache_key
    return _LLM


def _generate_texts(llm: Any, prompts: list[str], params: Any, *, batch_size: int) -> list[str]:
    texts: list[str] = []
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(prompts), safe_batch_size):
        outputs = llm.generate(prompts[start : start + safe_batch_size], params)
        texts.extend(_first_output(output) for output in outputs)
    return texts


def _generate_texts_with_latencies(llm: Any, prompts: list[str], params: Any) -> tuple[list[str], list[float]]:
    texts: list[str] = []
    latencies: list[float] = []
    for prompt in prompts:
        started = time.perf_counter()
        try:
            outputs = llm.generate([prompt], params)
            texts.extend(_first_output(output) for output in outputs)
        except Exception:
            texts.append("")
        latencies.append(time.perf_counter() - started)
    return texts, latencies


def _first_output(output: Any) -> str:
    generations = getattr(output, "outputs", None) or []
    if not generations:
        return ""
    return str(getattr(generations[0], "text", "")).strip()


def _write_readout(summary, pairwise, path):
    from submission.evaluation.reporting import write_readout
    write_readout(summary, pairwise, path)


def _write_grounded_readout(summary, pairwise, path):
    from submission.evaluation.reporting import write_readout
    write_readout(summary, pairwise, path)
