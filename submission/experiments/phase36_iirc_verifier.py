"""Run and freeze the full gold-blind IIRC Phase28 citation verifier."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase36_iirc_core import MODEL, digest_payload, verifier_payload
from submission.experiments.phase36_iirc_register import OUT

from submission.experiments.phase7_modal27b_matched8 import (
    append_fsync,
    atomic_write,
    read_rows,
)

RELAY_SOURCE = ROOT / "submission/experiments/phase36_iirc_relay_source.py"
RELAY_STAGE = Path(tempfile.gettempdir()) / "tokenpack_phase36_iirc_relay"
RELAY_STAGE.mkdir(parents=True, exist_ok=True)
RELAY_LOCAL = RELAY_STAGE / "phase36_iirc_relay.py"
if not RELAY_LOCAL.exists() or sha_file(RELAY_LOCAL) != sha_file(RELAY_SOURCE):
    shutil.copyfile(RELAY_SOURCE, RELAY_LOCAL)
sys.path.insert(0, str(RELAY_STAGE))
from phase36_iirc_relay import APP, BASE, infer_once  # noqa: E402

TASKS = OUT / "semantic_citation_tasks.jsonl"
PLAN = OUT / "VERIFIER_PRE_REMOTE_FREEZE.json"
ANSWERS = OUT / "semantic_verifier_responses.jsonl"
ERROR = OUT / "semantic_verifier_error.jsonl"
INFLIGHT = OUT / "semantic_verifier_inflight.json"
FROZEN = OUT / "VERIFIER_RESPONSES_FROZEN.json"
MAX_SESSION_SECONDS = 1500


def append(path: Path, row: dict) -> None:
    append_fsync(path, row)


def prepare() -> list[dict]:
    prereg = json.loads((OUT / "PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(Path(__file__)) == prereg["code_sha256"]["phase36_iirc_verifier.py"]
    candidates = json.loads((OUT / "CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    assert candidates["gold_unopened"] is True
    assert sha_file(TASKS) == candidates["tasks_sha256"]
    tasks = read_rows(TASKS)
    ids = json.loads((OUT / "ID_MANIFEST_PRE_GOLD.json").read_text(encoding="utf8"))["ids_in_frozen_dataset_order"]
    assert len(tasks) == len(ids) == prereg["sample"]["questions"]
    assert [task["id"] for task in tasks] == ids
    spec = {
        "n": len(tasks),
        "task_sha256": sha_file(TASKS),
        "candidate_source_sha256": candidates["scored_sha256"],
        "verifier_code_sha256": sha_file(Path(__file__)),
        "core_code_sha256": sha_file(ROOT / "submission/experiments/phase36_iirc_core.py"),
        "phase28_selector_sha256": sha_file(ROOT / "submission/experiments/phase28_semantic_core.py"),
        "relay_code_sha256": sha_file(RELAY_SOURCE),
        "remote_model": MODEL,
        "endpoint": BASE,
        "gold_uploaded": False,
        "annotation_fields_uploaded": False,
        "max_paid_calls": len(tasks),
        "max_parallel_calls": 4,
        "per_session_elapsed_guard_seconds": MAX_SESSION_SECONDS,
        "automatic_retry_after_remote_error": False,
    }
    if PLAN.exists():
        assert json.loads(PLAN.read_text(encoding="utf8")) == spec
    else:
        atomic_write(PLAN, spec)
    return tasks


def execute(max_new: int | None = None, concurrency: int = 4) -> None:
    tasks = prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Preserve and audit the unresolved verifier error/inflight record before continuing")
    if not 1 <= concurrency <= 8:
        raise ValueError("Concurrency must be in [1, 8]")
    saved_rows = read_rows(ANSWERS)
    saved = {}
    allowed = {task["id"] for task in tasks}
    for row in saved_rows:
        assert row.get("ok") and row["id"] in allowed and row["id"] not in saved
        saved[row["id"]] = row
    expected_hash = {
        task["id"]: digest_payload(verifier_payload(task, MODEL)) for task in tasks
    }
    assert all(saved[qid].get("prompt_sha256") == expected_hash[qid] for qid in saved)
    limit = len(tasks) - len(saved) if max_new is None else max(0, int(max_new))
    newly_started = 0
    stop = False
    print("PHASE36_VERIFIER_START", len(saved), "/", len(tasks), "concurrency", concurrency, flush=True)

    while len(saved) < len(tasks) and newly_started < limit and not stop:
        session_start = time.monotonic()
        with APP.run():
            while time.monotonic() - session_start < MAX_SESSION_SECONDS and newly_started < limit and not stop:
                pending = [task for task in tasks if task["id"] not in saved]
                if not pending:
                    break
                batch = pending[: min(concurrency, limit - newly_started)]
                requests = []
                inflight_rows = []
                for task in batch:
                    request = verifier_payload(task, MODEL)
                    prompt_sha = digest_payload(request)
                    requests.append((task, request, prompt_sha))
                    inflight_rows.append({
                        "id": task["id"],
                        "prompt_sha256": prompt_sha,
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    })
                atomic_write(INFLIGHT, {
                    "batch": inflight_rows,
                    "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                })
                handles = []
                for task, request, prompt_sha in requests:
                    handles.append((task, request, prompt_sha, infer_once.spawn(request)))
                    newly_started += 1
                unresolved = {entry["id"] for entry in inflight_rows}
                batch_failed = False
                for task, request, prompt_sha, handle in handles:
                    try:
                        result = handle.get()
                    except BaseException as exc:
                        append(ERROR, {"id": task["id"], "error_type": type(exc).__name__, "prompt_sha256": prompt_sha})
                        batch_failed = True
                    else:
                        if not result.get("ok"):
                            append(ERROR, {
                                "id": task["id"],
                                "phase": result.get("phase"),
                                "http_status": result.get("http_status"),
                                "exception": result.get("exception"),
                                "prompt_sha256": prompt_sha,
                            })
                            batch_failed = True
                        else:
                            raw = str(result.get("answer") or "")
                            from submission.experiments.phase36_iirc_core import parse_ids
                            citation_ids, parse_status = parse_ids(raw, task)
                            entry = {
                                "id": task["id"],
                                "ok": True,
                                "raw_output": raw,
                                "citation_ids": citation_ids,
                                "parse_status": parse_status,
                                "prompt_sha256": prompt_sha,
                                "paid_remote_call": True,
                                "input_tokens": result.get("input_tokens"),
                                "output_tokens": result.get("output_tokens"),
                                "finish_reason": result.get("finish_reason"),
                                "elapsed_s": result.get("elapsed_s"),
                            }
                            append(ANSWERS, entry)
                            saved[task["id"]] = entry
                    unresolved.discard(task["id"])
                    if unresolved:
                        atomic_write(INFLIGHT, {
                            "batch": [row for row in inflight_rows if row["id"] in unresolved],
                            "started_utc": inflight_rows[0]["utc"],
                        })
                    else:
                        INFLIGHT.unlink(missing_ok=True)
                    if len(saved) % 16 == 0 or (not unresolved and len(saved) == len(tasks)):
                        print("PHASE36_VERIFIER_PROGRESS", len(saved), "/", len(tasks), "started_this_run", newly_started, flush=True)
                stop = batch_failed
        if stop:
            break

    print("PHASE36_VERIFIER_SESSION_END", len(saved), "/", len(tasks), "started_this_run", newly_started, flush=True)


def freeze() -> None:
    tasks = prepare()
    rows = read_rows(ANSWERS)
    by_id = {row["id"]: row for row in rows}
    assert len(rows) == len(by_id) == len(tasks)
    assert set(by_id) == {task["id"] for task in tasks}
    assert all(by_id[task["id"]].get("ok") for task in tasks)
    assert not ERROR.exists() and not INFLIGHT.exists()
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    ordered = [by_id[task["id"]] for task in tasks]
    status_counts = {}
    for row in ordered:
        status_counts[row["parse_status"]] = status_counts.get(row["parse_status"], 0) + 1
    atomic_write(FROZEN, {
        "n": len(ordered),
        "responses_sha256": sha_file(ANSWERS),
        "tasks_sha256": sha_file(TASKS),
        "pre_remote_sha256": sha_file(PLAN),
        "ordered_response_ids_sha256": hashlib.sha256("\n".join(row["id"] for row in ordered).encode("utf8")).hexdigest(),
        "parse_status_counts": status_counts,
        "actual_paid_calls": sum(bool(row.get("paid_remote_call")) for row in ordered),
        "gold_unopened": True,
    })
    print("PHASE36_VERIFIER_FROZEN", len(ordered), json.dumps(status_counts), flush=True)


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
        print("PHASE36_VERIFIER_PREPARED", len(prepare()), flush=True)
    elif args.execute:
        execute(args.max_new, args.concurrency)
    else:
        freeze()


if __name__ == "__main__":
    main()
