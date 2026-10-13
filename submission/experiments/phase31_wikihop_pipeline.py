"""Phase31 WikiHop EM-only replication pipeline.

The registered WikiHop dev sample has no gold source-sentence indices, so this
pipeline reports answer EM only. Original support passages are treated as
atomic source units; this adaptation is explicitly not a Phase28 support
coverage replication.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"; os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase13_planner import APP, BASE, MODEL, infer_once
from submission.experiments.phase28_semantic_core import make_task, parse_ids, payload as verifier_payload, select
from submission.experiments.phase8_atomic_contexts import encode_scores, render
from submission.experiments.phase27_joint_chain_core import baseline_indices
from submission.experiments.phase5b_pretrained_rerank_eval import FOLDER, verify_local_model
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from submission.experiments.phase7_modal27b_matched8 import (
    append_fsync, atomic_write, make_payload as reader_payload, read_rows,
)
from tokenpack.tokenization import TokenCounter

OUT = ROOT / "submission/results/phase31_wikihop_em_replication_20261013"
ARCHIVE = ROOT / ".test-tmp/qangaroo_v1.1.zip"
SOURCE_MEMBER = "qangaroo_v1.1/wikihop/dev.json"
ARMS = ("atomic-sentence-rank", "semantic-joint-citations")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def paths(split: str) -> dict[str, Path]:
    d = OUT / split
    return {"dir": d, "scored": d / "scored_original_supports.jsonl", "verifier_tasks": d / "semantic_citation_tasks.jsonl",
            "cand_freeze": d / "CANDIDATES_FROZEN.json", "verifier": d / "semantic_verifier_responses.jsonl",
            "verifier_plan": d / "VERIFIER_PRE_REMOTE_FREEZE.json", "verifier_frozen": d / "VERIFIER_RESPONSES_FROZEN.json",
            "contexts": d / "gold_blind_paired_contexts.jsonl", "context_freeze": d / "CONTEXTS_FROZEN.json",
            "reader_tasks": d / "reader_tasks.jsonl", "reader_plan": d / "READER_PRE_REMOTE_FREEZE.json",
            "reader": d / "reader_responses.jsonl", "reader_error": d / "reader_error.jsonl",
            "reader_inflight": d / "reader_inflight.json", "reader_frozen": d / "READER_RESPONSES_FROZEN.json",
            "result": d / ("PILOT_POST_GOLD_RESULTS.json" if split == "pilot" else "VALIDATION_POST_GOLD_RESULTS.json")}


def selected_ids(split: str) -> list[str]:
    manifest = json.loads((OUT / "ID_MANIFEST_PRE_GOLD.json").read_text(encoding="utf8"))
    return [x["id"] for x in manifest[split]]


def read_source_rows(ids: set[str], include_answer: bool = False) -> dict[str, dict]:
    # Gold is deliberately accessed only by score(), after reader freeze.
    with zipfile.ZipFile(ARCHIVE) as z:
        all_rows = json.loads(z.read(SOURCE_MEMBER))
    out = {}
    for row in all_rows:
        if row["id"] not in ids:
            continue
        if include_answer:
            out[row["id"]] = {"id": row["id"], "query": row["query"], "supports": row["supports"],
                               "candidates": row["candidates"], "answer": row["answer"]}
        else:
            out[row["id"]] = {"id": row["id"], "query": row["query"], "supports": row["supports"],
                               "candidates": row["candidates"]}
    assert set(out) == ids
    return out


def make_atoms(row: dict) -> list[dict]:
    atoms = []
    for j, text in enumerate(row["supports"]):
        text = str(text).strip()
        if not text:
            continue
        atoms.append({"article": j, "title": f"WikiHop support {j:02d}", "sent_id": 0,
                      "sentence": text, "input_text": text})
    assert atoms
    return atoms


def prepare_candidates(split: str) -> None:
    p = paths(split); p["dir"].mkdir(parents=True, exist_ok=False)
    ids = selected_ids(split); data = read_source_rows(set(ids), include_answer=False)
    verify_local_model()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(FOLDER, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(FOLDER, local_files_only=True).to("cuda").eval()
    counter = TokenCounter(); assert counter._encoding is not None
    cases = []
    for ident in ids:
        row = data[ident]; atoms = make_atoms(row)
        total = sum(counter.count(a["sentence"]) for a in atoms); cap = max(1, math.floor(total * .25))
        scores = encode_scores(model, tokenizer, row["query"], [a["input_text"] for a in atoms], max_length=SCORER_MAX_INPUT_TOKENS)
        task = make_task(ident, row["query"], atoms, scores, cap, total)
        cases.append({"id": ident, "question": row["query"], "source_token_total": total, "cap": cap,
                      "atoms": [{k: a[k] for k in ("article", "title", "sent_id", "sentence")} for a in atoms],
                      "scores": [float(x) for x in scores], "gold_unopened": True})
        with p["verifier_tasks"].open("a", encoding="utf8") as f:
            f.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
    p["scored"].write_text("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in cases) + "\n", encoding="utf8")
    p["cand_freeze"].write_text(json.dumps({"n": len(ids), "source_sha256": sha_file(p["scored"]),
        "task_sha256": sha_file(p["verifier_tasks"]), "gold_unopened": True,
        "source_archive_sha256": sha_file(ARCHIVE), "candidate_level": "original support passage atomic unit",
        "token_cap_violations": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE31_CANDIDATES_FROZEN", split, len(ids), flush=True)


def prepare_verifier(split: str) -> list[dict]:
    p = paths(split); cases = read_rows(p["scored"]); tasks = read_rows(p["verifier_tasks"])
    assert len(cases) == len(tasks) == len(selected_ids(split))
    spec = {"split": split, "n": len(tasks), "tasks_sha256": sha_file(p["verifier_tasks"]),
            "candidate_sha256": sha_file(p["scored"]), "code_sha256": sha_file(Path(__file__)),
            "model": MODEL, "endpoint": BASE, "gold_loaded": False, "max_paid_calls": len(tasks)}
    if p["verifier_plan"].exists(): assert json.loads(p["verifier_plan"].read_text(encoding="utf8")) == spec
    else: p["verifier_plan"].write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return tasks


def execute_verifier(split: str, max_new: int | None = None) -> None:
    p = paths(split); tasks = prepare_verifier(split); rows = read_rows(p["verifier"])
    if p["verifier"].exists() and rows:
        assert [r["id"] for r in rows] == [t["id"] for t in tasks[:len(rows)]]
    saved = {r["id"]: r for r in rows}; limit = len(tasks) if max_new is None else max_new
    paid = 0; started = time.monotonic()
    print("PHASE31_VERIFIER_START", split, len(saved), "/", len(tasks), flush=True)
    with APP.run():
        for task in tasks:
            if task["id"] in saved: continue
            if paid >= limit or time.monotonic() - started + 120 >= 1800: break
            req = verifier_payload(task, MODEL); ph = digest(req)
            atomic_write(p["dir"] / "verifier_inflight.json", {"id": task["id"], "prompt_sha256": ph})
            try: result = infer_once.remote(req)
            except BaseException as exc:
                append_fsync(p["dir"] / "verifier_error.jsonl", {"id": task["id"], "error": type(exc).__name__}); break
            paid += 1
            if not result.get("ok"):
                append_fsync(p["dir"] / "verifier_error.jsonl", {"id": task["id"], "error": result.get("exception")}); break
            raw = str(result.get("answer") or "")
            citation_ids, status = parse_ids(raw, task)
            rec = {"id": task["id"], "ok": True, "raw_output": raw, "citation_ids": citation_ids,
                   "parse_status": status, "prompt_sha256": ph, "paid": True}
            append_fsync(p["verifier"], rec); (p["dir"] / "verifier_inflight.json").unlink(); saved[task["id"]] = rec
            if len(saved) % 12 == 0: print("PHASE31_VERIFIER_PROGRESS", split, len(saved), "/", len(tasks), flush=True)
    print("PHASE31_VERIFIER_SESSION_END", split, len(saved), "/", len(tasks), "paid", paid, flush=True)


def freeze_verifier(split: str) -> None:
    p = paths(split); tasks = prepare_verifier(split); rows = read_rows(p["verifier"])
    assert len(rows) == len(tasks) and [r["id"] for r in rows] == [t["id"] for t in tasks]
    assert all(r.get("ok") for r in rows) and not (p["dir"] / "verifier_error.jsonl").exists() and not (p["dir"] / "verifier_inflight.json").exists()
    p["verifier_frozen"].write_text(json.dumps({"n": len(rows), "sha256": sha_file(p["verifier"]),
        "task_sha256": sha_file(p["verifier_tasks"]), "gold_unopened": True,
        "paid_calls": sum(bool(r.get("paid")) for r in rows)}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE31_VERIFIER_FROZEN", split, len(rows), flush=True)


def assemble(split: str) -> None:
    p = paths(split); cases = read_rows(p["scored"]); tasks = read_rows(p["verifier_tasks"]); answers = read_rows(p["verifier"])
    counter = TokenCounter(); out = []
    stats = {"n": 0, "source_sets_changed": 0, "token_cap_violations": 0, "fallback": 0}
    for case, task, answer in zip(cases, tasks, answers, strict=True):
        atoms, scores, cap = case["atoms"], case["scores"], case["cap"]
        base, bi = baseline_indices(atoms, scores, cap, counter)
        treat, ti, status = select(task, atoms, scores, answer["raw_output"], counter)
        selected = {}
        for arm, details, inds in ((ARMS[0], base, bi), (ARMS[1], treat, ti)):
            selected[arm] = {"text": details["text"], "tokens": details["tokens"], "cap": cap,
                             "source_article_count": details["source_articles"], "selected_sentence_count": len(inds),
                             "selected_original_pairs": [{"title": atoms[i]["title"], "sent_id": atoms[i]["sent_id"], "article": atoms[i]["article"]} for i in inds]}
        changed = selected[ARMS[0]]["text"] != selected[ARMS[1]]["text"]
        out.append({"id": case["id"], "question": case["question"], "source_total": case["source_token_total"], "cap": cap,
                    "contexts": selected, "semantic_citation_parse_status": answer["parse_status"],
                    "semantic_seed_status": status, "source_sentence_set_changed": changed, "source_has_no_gold_annotations": True})
        stats["n"] += 1; stats["source_sets_changed"] += int(changed)
        stats["token_cap_violations"] += int(base["tokens"] > cap or treat["tokens"] > cap); stats["fallback"] += int(status != "verified_citation_anchors_selected")
    p["contexts"].write_text("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in out) + "\n", encoding="utf8")
    p["context_freeze"].write_text(json.dumps({"n": len(out), "sha256": sha_file(p["contexts"]), "gold_unopened": True, "stats": stats}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE31_CONTEXTS_FROZEN", split, json.dumps(stats), flush=True)


def prepare_reader(split: str) -> list[dict]:
    p = paths(split); rows = read_rows(p["contexts"]); tasks = []
    for row in rows:
        for arm in ARMS:
            tasks.append({"task_id": row["id"] + "|" + arm, "id": row["id"], "arm": arm, "question": row["question"], "context": row["contexts"][arm]["text"]})
    if p["reader_tasks"].exists(): assert read_rows(p["reader_tasks"]) == tasks
    else: p["reader_tasks"].write_text("\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in tasks) + "\n", encoding="utf8")
    spec = {"split": split, "n": len(tasks), "tasks_sha256": sha_file(p["reader_tasks"]), "contexts_sha256": sha_file(p["contexts"]),
            "code_sha256": sha_file(Path(__file__)), "gold_loaded": False, "gold_sent_remote": False, "max_paid_calls": len(tasks)}
    if p["reader_plan"].exists(): assert json.loads(p["reader_plan"].read_text(encoding="utf8")) == spec
    else: p["reader_plan"].write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return tasks


def execute_reader(split: str, max_new: int | None = None) -> None:
    p = paths(split); tasks = prepare_reader(split); rows = read_rows(p["reader"]); saved = {r["task_id"]: r for r in rows}
    assert [r["task_id"] for r in rows] == [t["task_id"] for t in tasks[:len(rows)]]
    limit = len(tasks) if max_new is None else max_new; paid = 0; started = time.monotonic()
    print("PHASE31_READER_START", split, len(saved), "/", len(tasks), flush=True)
    prior = {}
    for task, r in zip(tasks[:len(rows)], rows, strict=True):
        prior[digest(reader_payload(task))] = r
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved: continue
            if paid >= limit or time.monotonic() - started + 120 >= 1800: break
            req = reader_payload(task); ph = digest(req)
            if ph in prior:
                rec = {"task_id": task["task_id"], "id": task["id"], "arm": task["arm"], "ok": True, "answer": prior[ph]["answer"], "prompt_sha256": ph, "paid": False, "reused_identical_prompt_from": prior[ph]["task_id"]}
                append_fsync(p["reader"], rec); saved[task["task_id"]] = rec; continue
            atomic_write(p["reader_inflight"], {"task_id": task["task_id"], "prompt_sha256": ph})
            try: result = infer_once.remote(req)
            except BaseException as exc:
                append_fsync(p["reader_error"], {"task_id": task["task_id"], "error": type(exc).__name__}); break
            paid += 1
            if not result.get("ok"):
                append_fsync(p["reader_error"], {"task_id": task["task_id"], "error": result.get("exception")}); break
            rec = {"task_id": task["task_id"], "id": task["id"], "arm": task["arm"], "ok": True, "answer": str(result.get("answer") or "").strip(), "prompt_sha256": ph, "paid": True}
            append_fsync(p["reader"], rec); p["reader_inflight"].unlink(); saved[task["task_id"]] = rec; prior[ph] = rec
            if len(saved) % 12 == 0: print("PHASE31_READER_PROGRESS", split, len(saved), "/", len(tasks), "paid", paid, flush=True)
    print("PHASE31_READER_SESSION_END", split, len(saved), "/", len(tasks), "paid", paid, flush=True)


def freeze_reader(split: str) -> None:
    p = paths(split); tasks = prepare_reader(split); rows = read_rows(p["reader"])
    assert len(rows) == len(tasks) and [r["task_id"] for r in rows] == [t["task_id"] for t in tasks]
    assert all(r.get("ok") for r in rows) and not p["reader_error"].exists() and not p["reader_inflight"].exists()
    p["reader_frozen"].write_text(json.dumps({"n": len(rows), "responses_sha256": sha_file(p["reader"]), "task_sha256": sha_file(p["reader_tasks"]),
        "pre_remote_sha256": sha_file(p["reader_plan"]), "actual_paid_reader_calls": sum(bool(r.get("paid")) for r in rows),
        "reuse": sum(not bool(r.get("paid")) for r in rows), "gold_seen": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE31_READER_FROZEN", split, len(rows), flush=True)


def normalize(text: object) -> str:
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split())


def answer_metrics(pred: object, truth: object) -> tuple[int, float]:
    a, b = normalize(pred), normalize(truth); return int(a == b), float(a == b)


def score(split: str) -> None:
    p = paths(split); frozen = json.loads(p["reader_frozen"].read_text(encoding="utf8")); assert frozen["gold_seen"] is False
    tasks, responses, contexts = read_rows(p["reader_tasks"]), read_rows(p["reader"]), read_rows(p["contexts"])
    data = read_source_rows(set(x["id"] for x in tasks), include_answer=True); idx = {x["task_id"]: x for x in responses}
    cases = []
    for row in contexts:
        g = data[row["id"]]["answer"]; pair = {"id": row["id"]}
        for arm in ARMS:
            pred = idx[row["id"] + "|" + arm]["answer"]; em, f1 = answer_metrics(pred, g)
            pair[arm] = {"EM": em, "F1": f1, "prediction": pred, "tokens": row["contexts"][arm]["tokens"]}
        cases.append(pair)
    wins = sum(x[ARMS[1]]["EM"] > x[ARMS[0]]["EM"] for x in cases); losses = sum(x[ARMS[1]]["EM"] < x[ARMS[0]]["EM"] for x in cases)
    n = len(cases)
    token_ok = json.loads(p["context_freeze"].read_text(encoding="utf8"))["stats"]["token_cap_violations"] == 0
    decision = "PILOT_GO_TO_VALIDATION" if split == "pilot" and wins - losses >= 3 and token_ok else ("PILOT_STOP_NO_VALIDATION" if split == "pilot" else "VALIDATION_COMPLETE")
    result = {"split": split, "decision": decision, "n": n, "paired_EM": {"win": wins, "loss": losses, "net": wins - losses,
        "two_sided_exact_mcnemar_p": float(binomtest(min(wins, losses), wins + losses, .5).pvalue) if wins + losses else 1.0},
        "arms": {arm: {"EM_correct": int(sum(x[arm]["EM"] for x in cases)), "EM_fraction": float(sum(x[arm]["EM"] for x in cases) / n), "mean_tokens": float(np.mean([x[arm]["tokens"] for x in cases]))} for arm in ARMS},
        "support_metric": None, "gold_first_opened_after_reader_freeze": True, "gold_sent_remote": False,
        "actual_paid_verifier_calls": json.loads(p["verifier_frozen"].read_text(encoding="utf8"))["paid_calls"],
        "actual_paid_reader_calls": frozen["actual_paid_reader_calls"], "token_cap_violations": json.loads(p["context_freeze"].read_text(encoding="utf8"))["stats"]["token_cap_violations"], "per_case": cases}
    p["result"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE31_POST_GOLD", json.dumps({"split": split, "decision": decision, "EM": result["paired_EM"], "arms": result["arms"]}, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--split", choices=["pilot", "validation"], required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare-candidates", action="store_true"); g.add_argument("--execute-verifier", action="store_true"); g.add_argument("--freeze-verifier", action="store_true")
    g.add_argument("--assemble", action="store_true"); g.add_argument("--prepare-reader", action="store_true"); g.add_argument("--execute-reader", action="store_true")
    g.add_argument("--freeze-reader", action="store_true"); g.add_argument("--score", action="store_true")
    p.add_argument("--max-new", type=int, default=None); a = p.parse_args()
    if a.prepare_candidates: prepare_candidates(a.split)
    elif a.execute_verifier: execute_verifier(a.split, a.max_new)
    elif a.freeze_verifier: freeze_verifier(a.split)
    elif a.assemble: assemble(a.split)
    elif a.prepare_reader: prepare_reader(a.split)
    elif a.execute_reader: execute_reader(a.split, a.max_new)
    elif a.freeze_reader: freeze_reader(a.split)
    else: score(a.split)


if __name__ == "__main__": main()
