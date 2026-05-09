from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import modal

THIS_DIR = Path(__file__).resolve().parent
REMOTE_EVAL_DIR = Path("/root/submission/modal_generation_eval")
for path in (THIS_DIR, REMOTE_EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_utils import (  # noqa: E402
    answer_template_for_variant,
    load_prompt_templates,
    parse_judge_json,
    prompt_token_estimate,
    read_jsonl,
    render_answer_prompt,
    render_judge_prompt,
    write_jsonl,
)


APP_NAME = "tokenpack-qasper-generation-eval"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
GPU = os.environ.get("TOKENPACK_MODAL_GPU", "L40S")
SECRET_NAME = os.environ.get("TOKENPACK_MODAL_SECRET_NAME", "")
REMOTE_WORK_DIR = Path("/data/tokenpack_modal_eval")
REMOTE_SHARD_DIR = REMOTE_WORK_DIR / "shards"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.8.5", "transformers>=4.45.0", "huggingface_hub>=0.24.0")
    .add_local_dir(str(THIS_DIR), remote_path="/root/submission/modal_generation_eval")
)

volume = modal.Volume.from_name("tokenpack-modal-generation-eval", create_if_missing=True)
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
def run_eval_shard(
    tasks: list[dict[str, Any]],
    shard_id: int,
    model_id: str = DEFAULT_MODEL_ID,
    max_answer_tokens: int = 220,
    max_judge_tokens: int = 220,
    batch_size: int = 4,
) -> dict[str, Any]:
    prompt_dir = REMOTE_EVAL_DIR / "prompts" if (REMOTE_EVAL_DIR / "prompts").exists() else THIS_DIR / "prompts"
    templates = load_prompt_templates(prompt_dir)
    llm = _get_llm(model_id)

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


def _answer_template_for_task(templates: Any, task: dict[str, Any]) -> str:
    return answer_template_for_variant(templates, str(task.get("answer_prompt_variant") or "default"))


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
