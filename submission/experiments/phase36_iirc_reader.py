"""Run and freeze paired Qwen 27B IIRC answer reads on both frozen contexts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase36_iirc_core import MODEL, digest_payload, reader_payload
from submission.experiments.phase36_iirc_register import OUT
from submission.experiments.phase36_iirc_verifier import APP, BASE, infer_once
from submission.experiments.phase7_modal27b_matched8 import atomic_write, append_fsync, read_rows

CONTEXTS = OUT / "gold_blind_paired_contexts.jsonl"
TASKS = OUT / "reader_tasks.jsonl"
PLAN = OUT / "READER_PRE_REMOTE_FREEZE.json"
ANSWERS = OUT / "reader_responses.jsonl"
ERROR = OUT / "reader_error.jsonl"
INFLIGHT = OUT / "reader_inflight.json"
FROZEN = OUT / "READER_RESPONSES_FROZEN.json"
MAX_SESSION_SECONDS = 1500


def prepare() -> list[dict]:
    prereg = json.loads((OUT / "PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(Path(__file__)) == prereg["code_sha256"]["phase36_iirc_reader.py"]
    freeze = json.loads((OUT / "CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert freeze["gold_unopened"] is True
    assert sha_file(CONTEXTS) == freeze["sha256"]
    contexts = read_rows(CONTEXTS)
    assert len(contexts) == prereg["sample"]["questions"]
    ids = [row["id"] for row in contexts]
    assert ids == json.loads((OUT / "ID_MANIFEST_PRE_GOLD.json").read_text(encoding="utf8"))["ids_in_frozen_dataset_order"]
    tasks = []
    for row in contexts:
        assert set(prereg["comparison"]["arms"]) <= set(row["contexts"])
        for arm in prereg["comparison"]["arms"]:
            context = row["contexts"][arm]
            assert context["tokens"] <= context["cap"] == row["cap"]
            tasks.append({
                "task_id": row["id"] + "|" + arm,
                "id": row["id"],
                "arm": arm,
                "question": row["question"],
                "context": context["text"],
                "context_tokens_cl100k": context["tokens"],
                "cap": context["cap"],
                "contexts_sha256": freeze["sha256"],
            })
    if TASKS.exists():
        assert read_rows(TASKS) == tasks
    else:
        with TASKS.open("x", encoding="utf8") as stream:
            for row in tasks:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
    spec = {
        "n": len(tasks),
        "questions": len(contexts),
        "arms": prereg["comparison"]["arms"],
        "contexts_sha256": freeze["sha256"],
        "tasks_sha256": sha_file(TASKS),
        "reader_code_sha256": sha_file(Path(__file__)),
        "reader_core_sha256": sha_file(ROOT / "submission/experiments/phase36_iirc_core.py"),
        "relay_sha256": sha_file(ROOT / "submission/experiments/phase36_iirc_relay_source.py"),
        "model": MODEL,
        "endpoint": BASE,
        "same_reader_payload_for_both_arms": True,
        "gold_or_annotations_loaded": False,
        "max_paid_calls": len(tasks),
        "max_parallel_calls": 4,
        "per_session_elapsed_guard_seconds": MAX_SESSION_SECONDS,
        "identical_payload_reuse": "Exact payload SHA256 match reuses its first frozen response verbatim.",
        "automatic_retry_after_remote_error": False,
    }
    if PLAN.exists():
        assert json.loads(PLAN.read_text(encoding="utf8")) == spec
    else:
        atomic_write(PLAN, spec)
    return tasks


def save_reuse(task: dict, source: dict, prompt_sha: str) -> dict:
    row = {
        "task_id": task["task_id"],
        "id": task["id"],
        "arm": task["arm"],
        "ok": True,
        "answer": source["answer"],
        "prompt_sha256": prompt_sha,
        "paid_remote_call": False,
        "reused_identical_prompt_from": source["task_id"],
        "input_tokens": source.get("input_tokens"),
        "output_tokens": source.get("output_tokens"),
        "finish_reason": source.get("finish_reason"),
    }
    append_fsync(ANSWERS, row)
    return row


def execute(max_new: int | None = None, concurrency: int = 4) -> None:
    tasks = prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Preserve and audit the unresolved reader error/inflight record before continuing")
    if not 1 <= concurrency <= 8:
        raise ValueError("Concurrency must be in [1, 8]")
    history = read_rows(ANSWERS)
    saved = {}
    by_prompt = {}
    allowed = {task["task_id"] for task in tasks}
    for row in history:
        assert row.get("ok") and row["task_id"] in allowed and row["task_id"] not in saved
        saved[row["task_id"]] = row
        by_prompt.setdefault(row["prompt_sha256"], row)
    expected_hashes = {
        task["task_id"]: digest_payload(reader_payload(task)) for task in tasks
    }
    assert all(saved[task_id].get("prompt_sha256") == expected_hashes[task_id] for task_id in saved)
    limit = len(tasks) if max_new is None else max(0, int(max_new))
    newly_started = 0
    task_index = 0
    stop = False
    print("PHASE36_READER_START", len(saved), "/", len(tasks), "concurrency", concurrency, flush=True)

    while task_index < len(tasks) and newly_started < limit and not stop:
        session_start = time.monotonic()
        with APP.run():
            while (task_index < len(tasks) and newly_started < limit
                   and time.monotonic() - session_start < MAX_SESSION_SECONDS and not stop):
                batch_by_hash = {}
                aliases = {}
                while task_index < len(tasks) and len(batch_by_hash) < concurrency and newly_started + len(batch_by_hash) < limit:
                    task = tasks[task_index]
                    task_index += 1
                    if task["task_id"] in saved:
                        continue
                    request = reader_payload(task)
                    prompt_sha = digest_payload(request)
                    if prompt_sha in by_prompt:
                        row = save_reuse(task, by_prompt[prompt_sha], prompt_sha)
                        saved[task["task_id"]] = row
                        continue
                    if prompt_sha in batch_by_hash:
                        aliases.setdefault(prompt_sha, []).append(task)
                        continue
                    batch_by_hash[prompt_sha] = (task, request)

                if not batch_by_hash:
                    continue
                inflight_rows = []
                for prompt_sha, (task, _) in batch_by_hash.items():
                    inflight_rows.append({
                        "task_id": task["task_id"],
                        "id": task["id"],
                        "prompt_sha256": prompt_sha,
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    })
                atomic_write(INFLIGHT, {
                    "batch": inflight_rows,
                    "aliases": {
                        prompt_sha: [task["task_id"] for task in values]
                        for prompt_sha, values in aliases.items()
                    },
                    "started_utc": inflight_rows[0]["utc"],
                })
                handles = []
                for prompt_sha, (task, request) in batch_by_hash.items():
                    handles.append((prompt_sha, task, infer_once.spawn(request)))
                    newly_started += 1
                unresolved = {entry[0] for entry in handles}
                batch_failed = False
                for prompt_sha, task, handle in handles:
                    try:
                        result = handle.get()
                    except BaseException as exc:
                        append_fsync(ERROR, {
                            "task_id": task["task_id"],
                            "id": task["id"],
                            "error_type": type(exc).__name__,
                            "prompt_sha256": prompt_sha,
                        })
                        batch_failed = True
                    else:
                        if not result.get("ok"):
                            append_fsync(ERROR, {
                                "task_id": task["task_id"],
                                "id": task["id"],
                                "phase": result.get("phase"),
                                "http_status": result.get("http_status"),
                                "exception": result.get("exception"),
                                "prompt_sha256": prompt_sha,
                            })
                            batch_failed = True
                        else:
                            entry = {
                                "task_id": task["task_id"],
                                "id": task["id"],
                                "arm": task["arm"],
                                "ok": True,
                                "answer": str(result.get("answer") or "").strip(),
                                "prompt_sha256": prompt_sha,
                                "paid_remote_call": True,
                                "input_tokens": result.get("input_tokens"),
                                "output_tokens": result.get("output_tokens"),
                                "finish_reason": result.get("finish_reason"),
                                "elapsed_s": result.get("elapsed_s"),
                            }
                            append_fsync(ANSWERS, entry)
                            saved[task["task_id"]] = entry
                            by_prompt[prompt_sha] = entry
                            for alias in aliases.get(prompt_sha, []):
                                reused = save_reuse(alias, entry, prompt_sha)
                                saved[alias["task_id"]] = reused
                    unresolved.discard(prompt_sha)
                    remaining = [row for row in inflight_rows if row["prompt_sha256"] in unresolved]
                    if remaining:
                        atomic_write(INFLIGHT, {
                            "batch": remaining,
                            "aliases": {
                                key: [task["task_id"] for task in values]
                                for key, values in aliases.items() if key in unresolved
                            },
                            "started_utc": inflight_rows[0]["utc"],
                        })
                    else:
                        INFLIGHT.unlink(missing_ok=True)
                    if len(saved) % 24 == 0:
                        print("PHASE36_READER_PROGRESS", len(saved), "/", len(tasks), "paid_this_run", newly_started, flush=True)
                stop = batch_failed
        if stop:
            break
    print("PHASE36_READER_SESSION_END", len(saved), "/", len(tasks), "started_this_run", newly_started, flush=True)


def freeze() -> None:
    tasks = prepare()
    rows = read_rows(ANSWERS)
    by_id = {row["task_id"]: row for row in rows}
    assert len(by_id) == len(rows) == len(tasks)
    assert set(by_id) == {task["task_id"] for task in tasks}
    assert all(by_id[task["task_id"]].get("ok") for task in tasks)
    assert not ERROR.exists() and not INFLIGHT.exists()
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    ordered = [by_id[task["task_id"]] for task in tasks]
    atomic_write(FROZEN, {
        "n": len(ordered),
        "question_n": len(ordered) // 2,
        "responses_sha256": sha_file(ANSWERS),
        "task_sha256": sha_file(TASKS),
        "pre_remote_sha256": sha_file(PLAN),
        "actual_paid_reader_calls": sum(bool(row.get("paid_remote_call")) for row in ordered),
        "identical_prompt_reuse_n": sum(not bool(row.get("paid_remote_call")) for row in ordered),
        "empty_answers": sum(not row["answer"] for row in ordered),
        "gold_seen": False,
    })
    print("PHASE36_READER_FROZEN", len(ordered), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--freeze", action="store_true")
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        print("PHASE36_READER_PREPARED", len(prepare()), flush=True)
    elif args.execute:
        execute(args.max_new, args.concurrency)
    else:
        freeze()


if __name__ == "__main__":
    main()
