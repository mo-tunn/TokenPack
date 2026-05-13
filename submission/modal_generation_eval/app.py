from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

THIS_DIR = Path(__file__).resolve().parent
ROOT = next(
    (
        parent
        for parent in (THIS_DIR, *THIS_DIR.parents)
        if (parent / "src" / "tokenpack").exists()
    ),
    THIS_DIR.parents[1],
)
LOCAL_SRC = ROOT / "src"
LOCAL_EXPERIMENTS = ROOT / "submission" / "experiments"
REMOTE_EVAL_DIR = Path("/root/submission/modal_generation_eval")
REMOTE_SRC = Path("/root/src")
REMOTE_EXPERIMENTS = Path("/root/submission/experiments")
for path in (LOCAL_SRC, LOCAL_EXPERIMENTS, THIS_DIR, REMOTE_SRC, REMOTE_EXPERIMENTS, REMOTE_EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_utils import (  # noqa: E402
    answer_template_for_variant,
    load_prompt_templates,
    pairwise_tokenpack_wins,
    parse_judge_json,
    prompt_token_estimate,
    read_jsonl,
    render_answer_prompt,
    render_judge_prompt,
    summarize_rows,
    write_jsonl,
    write_latex_table,
    write_summary_csv,
)


APP_NAME = "tokenpack-qasper-generation-eval"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
GPU = os.environ.get("TOKENPACK_MODAL_GPU", "L40S")
SECRET_NAME = os.environ.get("TOKENPACK_MODAL_SECRET_NAME", "")
REMOTE_WORK_DIR = Path("/data/tokenpack_modal_eval")
REMOTE_SHARD_DIR = REMOTE_WORK_DIR / "shards"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.5",
        "transformers>=4.45.0",
        "huggingface_hub>=0.24.0",
        "sentence-transformers>=3.0.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
    )
    .add_local_dir(str(LOCAL_SRC), remote_path="/root/src")
    .add_local_dir(str(LOCAL_EXPERIMENTS), remote_path="/root/submission/experiments")
    .add_local_dir(str(THIS_DIR), remote_path="/root/submission/modal_generation_eval")
)

volume = modal.Volume.from_name("tokenpack-modal-generation-eval", create_if_missing=True)
secrets = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []
app = modal.App(APP_NAME)


_LLM: Any | None = None
_LLM_CACHE_KEY: tuple[str, int, bool, float] | None = None


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    volumes={"/data": volume},
    secrets=secrets,
)
def build_tasks_remote(args_payload: dict[str, Any]) -> dict[str, Any]:
    import argparse

    from client import _write_task_report, build_tasks  # noqa: E402

    REMOTE_WORK_DIR.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(**args_payload)
    tasks = build_tasks(args)
    report_path = REMOTE_WORK_DIR / "task_report.json"
    _write_task_report(tasks, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    volume.commit()
    return {"tasks": tasks, "report": report}


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
    max_answer_tokens: int = 220,
    max_judge_tokens: int = 220,
    batch_size: int = 4,
) -> dict[str, Any]:
    prompt_dir = REMOTE_EVAL_DIR / "prompts" if (REMOTE_EVAL_DIR / "prompts").exists() else THIS_DIR / "prompts"
    templates = load_prompt_templates(prompt_dir)
    llm = _get_llm(
        model_id,
        max_model_len=max_model_len,
        enable_yarn=enable_yarn,
        yarn_factor=yarn_factor,
    )

    from vllm import SamplingParams

    answer_params = SamplingParams(temperature=0.0, max_tokens=max_answer_tokens)
    judge_params = SamplingParams(temperature=0.0, max_tokens=max_judge_tokens)

    answer_prompts = [
        render_answer_prompt(
            _answer_template_for_task(templates, task),
            question=str(task["question"]),
            context=str(task["context"]),
        )
        for task in tasks
    ]
    answer_started = time.perf_counter()
    answer_outputs = _generate_texts(llm, answer_prompts, answer_params, batch_size=batch_size)
    answer_elapsed = time.perf_counter() - answer_started
    answers = answer_outputs

    judge_prompts = [
        render_judge_prompt(
            templates.judge,
            question=str(task["question"]),
            gold_answer=str(task["gold_answer"]),
            evidence_texts=[str(item) for item in task.get("evidence_texts", [])],
            model_answer=answer,
        )
        for task, answer in zip(tasks, answers)
    ]
    judge_started = time.perf_counter()
    judge_outputs = _generate_texts(llm, judge_prompts, judge_params, batch_size=batch_size)
    judge_elapsed = time.perf_counter() - judge_started

    rows: list[dict[str, Any]] = []
    for task, answer, judge_raw in zip(tasks, answers, judge_outputs):
        judge = parse_judge_json(judge_raw)
        rows.append(
            {
                **{key: value for key, value in task.items() if key != "context"},
                "status": "completed",
                "model_id": model_id,
                "answer": answer,
                "answer_tokens": prompt_token_estimate(answer),
                "answer_prompt_tokens_est": prompt_token_estimate(
                    render_answer_prompt(
                        _answer_template_for_task(templates, task),
                        question=str(task["question"]),
                        context=str(task["context"]),
                    )
                ),
                "answer_latency_seconds": answer_elapsed / max(1, len(tasks)),
                "judge_latency_seconds": judge_elapsed / max(1, len(tasks)),
                **judge,
            }
        )

    REMOTE_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = REMOTE_SHARD_DIR / f"shard-{shard_id:04d}.jsonl"
    write_jsonl(rows, shard_path)
    volume.commit()
    return {"shard_id": shard_id, "rows": rows, "remote_path": str(shard_path)}


@app.local_entrypoint()
def run(
    tasks_jsonl: str,
    output_jsonl: str = "submission/results/modal_generation_eval/qasper_generation_judged.jsonl",
    model_id: str = DEFAULT_MODEL_ID,
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    shard_size: int = 100,
    max_answer_tokens: int = 220,
    max_judge_tokens: int = 220,
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
        print(f"Completed shard {result['shard_id']} -> {result['remote_path']}")
    write_jsonl(all_rows, Path(output_jsonl))
    print(f"Wrote merged results to {output_jsonl}")


@app.local_entrypoint()
def build_and_run(
    output_dir: str = "submission/results/modal_generation_eval",
    data_file: str = "",
    split: str = "validation",
    question_ids_from: str = "",
    limit: int = 10,
    max_papers: int = 10_000,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunker: str = "structure-aware",
    scoring: str = "evidence-hybrid",
    budget_ratio: float = 0.50,
    candidate_pool: int = 300,
    target_tokens: int = 650,
    min_tokens: int = 120,
    max_tokens: int = 900,
    chunk_size_preset: str = "low-budget",
    semantic_threshold: float = 0.35,
    skip_llmlingua2: bool = True,
    tokenpack_variants: str = "original",
    tokenpack_only: bool = False,
    include_budget_top_k: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    max_model_len: int = 32768,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    shard_size: int = 100,
    batch_size: int = 2,
    max_answer_tokens: int = 220,
    max_judge_tokens: int = 220,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    args_payload = {
        "data_file": data_file or None,
        "split": split,
        "question_ids_from": question_ids_from,
        "output_dir": str(REMOTE_WORK_DIR / "build_output"),
        "work_dir": str(REMOTE_WORK_DIR / "build_work"),
        "tasks_jsonl": None,
        "results_jsonl": None,
        "limit": limit,
        "max_papers": max_papers,
        "embedding_model": embedding_model,
        "embedding_allow_download": True,
        "chunker": chunker,
        "scoring": scoring,
        "budget_ratio": budget_ratio,
        "candidate_pool": candidate_pool,
        "target_tokens": target_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "chunk_size_preset": chunk_size_preset,
        "semantic_threshold": semantic_threshold,
        "llmlingua2_model": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        "compression_device_map": "cpu",
        "compression_allow_download": False,
        "skip_llmlingua2": skip_llmlingua2,
        "tokenpack_variants": tokenpack_variants,
        "tokenpack_only": tokenpack_only,
        "dry_run": False,
        "run_modal": False,
        "shard_size": shard_size,
        "batch_size": batch_size,
        "include_budget_top_k": include_budget_top_k,
    }
    build_result = build_tasks_remote.remote(args_payload)
    tasks = build_result["tasks"]
    tasks_path = output_path / "qasper_generation_tasks.jsonl"
    results_path = output_path / "qasper_generation_judged.jsonl"
    write_jsonl(tasks, tasks_path)
    (output_path / "task_report.json").write_text(
        json.dumps(build_result["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Built {len(tasks)} tasks on Modal -> {tasks_path}")

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
            "max_judge_tokens": max_judge_tokens,
            "batch_size": batch_size,
        },
        order_outputs=True,
    ):
        all_rows.extend(result["rows"])
        print(f"Completed shard {result['shard_id']} -> {result['remote_path']}")

    write_jsonl(all_rows, results_path)
    summary = summarize_rows(all_rows)
    pairwise = pairwise_tokenpack_wins(all_rows)
    write_summary_csv(summary, output_path / "qasper_generation_quality_summary.csv")
    write_summary_csv(pairwise, output_path / "qasper_generation_quality_pairwise.csv")
    write_latex_table(summary, output_path / "qasper_generation_quality_table.tex")
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

    _LLM = LLM(**llm_kwargs)
    _LLM_CACHE_KEY = cache_key
    return _LLM


def _answer_template_for_task(templates: Any, task: dict[str, Any]) -> str:
    return answer_template_for_variant(templates, str(task.get("answer_prompt_variant") or "default"))


def _generate_texts(llm: Any, prompts: list[str], params: Any, *, batch_size: int) -> list[str]:
    texts: list[str] = []
    safe_batch_size = max(1, int(batch_size))
    for start in range(0, len(prompts), safe_batch_size):
        batch = prompts[start : start + safe_batch_size]
        try:
            outputs = llm.generate(batch, params)
            texts.extend(_first_output(output) for output in outputs)
        except Exception:
            if len(batch) == 1:
                texts.append("")
                continue
            for prompt in batch:
                try:
                    outputs = llm.generate([prompt], params)
                    texts.extend(_first_output(output) for output in outputs)
                except Exception:
                    texts.append("")
    return texts


def _first_output(output: Any) -> str:
    generations = getattr(output, "outputs", None) or []
    if not generations:
        return ""
    return str(getattr(generations[0], "text", "")).strip()
