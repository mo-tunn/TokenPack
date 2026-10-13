"""Phase 30C: exploratory reader guard ablation on a fresh frozen subset.

Phase 30B showed that a chain-first prompt overused ``unknown``.  This study
uses 44 previously unused cases (36 HotpotQA + 8 MuSiQue, selected from
unchanged source-sentence contexts) and tests a safer instruction: answer
directly when the context supports a plausible answer, reserving ``unknown``
for genuinely unsupported questions.  The selector and contexts are fixed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase13_planner import APP, BASE, MODEL, infer_once
from submission.experiments.phase7_modal27b_matched8 import append_fsync, atomic_write, read_rows

OUT = ROOT / "submission/results/phase30c_reader_guard_ablation_20261013"
HOT_CONTEXTS = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013/gold_blind_paired_contexts.jsonl"
HOT_FREEZE = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013/CONTEXTS_FROZEN.json"
HOT_RESULT = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013/VALIDATION_POST_GOLD_RESULTS.json"
MUS_CONTEXTS = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot/gold_blind_paired_contexts.jsonl"
MUS_FREEZE = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot/CONTEXTS_FROZEN.json"
MUS_RESULT = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot/PILOT_POST_GOLD_RESULTS.json"
HOT_DATA = ROOT / "submission/data/phase6_hotpotqa_validation_20261011/distractor/validation-00000-of-00001.parquet"
MUS_DATA = ROOT / "submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"
TASKS = OUT / "reader_tasks.jsonl"
PLAN = OUT / "READER_PRE_REMOTE_FREEZE.json"
RESPONSES = OUT / "reader_responses.jsonl"
ERROR = OUT / "reader_error.jsonl"
INFLIGHT = OUT / "reader_inflight.json"
FROZEN = OUT / "READER_RESPONSES_FROZEN.json"
INPUTS = OUT / "CONTEXT_INPUTS_FROZEN.json"
RESULT = OUT / "ABLATION_POST_GOLD_RESULTS.json"
SEED = 20261013
MAX_GENERATED = 72
SYSTEM = (
    "Answer the question using only the context. Identify the best-supported "
    "answer privately if helpful, then reply only with the shortest correct "
    "answer phrase, preserving names, numbers, dates, and spelling from the "
    "context. When the context supports a plausible answer, give it directly. "
    "Use unknown only when no answer is supported. Do not explain your reasoning."
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def load_contexts(dataset: str, path: Path, freeze_path: Path) -> list[dict]:
    freeze = json.loads(freeze_path.read_text(encoding="utf8"))
    assert sha_file(path) == freeze["sha256"]
    rows = read_rows(path)
    out = []
    for row in rows:
        if row.get("source_sentence_set_changed"):
            continue
        arm = row["contexts"]["semantic-joint-citations"]
        out.append({"case_key": dataset + "/" + row["id"], "dataset": dataset, "id": row["id"],
                    "question": row["question"], "context": arm["text"], "context_tokens": arm["tokens"],
                    "context_sha256": digest(arm["text"]), "source_sentence_set_changed": False})
    return out


def select(rows: list[dict], n: int, label: str) -> list[dict]:
    return sorted(rows, key=lambda r: digest(label + "|" + r["case_key"]))[:n]


def payload(task: dict) -> dict:
    user = "Context:\n" + task["context"] + "\n\nQuestion:\n" + task["question"] + "\n\nAnswer in one short phrase:"
    return {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": 20261012,
            "max_tokens": MAX_GENERATED, "reasoning_effort": "none", "stream": False}


def prepare() -> list[dict]:
    hot = select(load_contexts("HotpotQA", HOT_CONTEXTS, HOT_FREEZE), 36, "phase30c-hotpot-20261013")
    mus = select(load_contexts("MuSiQue", MUS_CONTEXTS, MUS_FREEZE), 8, "phase30c-musique-20261013")
    cases = hot + mus
    assert len(hot) == 36 and len(mus) == 8
    tasks = []
    for case in cases:
        task = dict(case)
        task["task_id"] = case["case_key"] + "|semantic-joint-citations|reader-guard-v1"
        tasks.append(task)
    OUT.mkdir(parents=True, exist_ok=True)
    if TASKS.exists():
        assert read_rows(TASKS) == tasks
    else:
        with TASKS.open("x", encoding="utf8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
    spec = {"study": "Phase30C fresh reader guard ablation", "n": 44,
            "selection_rule": "smallest SHA256(label|case_key) among unchanged source-sentence contexts",
            "datasets": {"HotpotQA": 36, "MuSiQue": 8}, "selector_frozen": "semantic-joint-citations",
            "tasks_sha256": sha_file(TASKS), "reader_code_sha256": sha_file(Path(__file__)),
            "relay_sha256": sha_file(ROOT / "submission/experiments/phase13_graph_modal_relay_source.py"),
            "model": MODEL, "endpoint": BASE, "system_prompt_sha256": hashlib.sha256(SYSTEM.encode("utf8")).hexdigest(),
            "gold_loaded": False, "gold_sent_remote": False, "max_paid_calls": 44,
            "same_cases_as_phase30b": False, "exploratory_not_confirmatory": True}
    if PLAN.exists():
        assert json.loads(PLAN.read_text(encoding="utf8")) == spec
    else:
        PLAN.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    INPUTS.write_text(json.dumps({"n": 44, "cases": cases, "gold_loaded": False,
        "hotpot_context_sha256": sha_file(HOT_CONTEXTS), "musique_context_sha256": sha_file(MUS_CONTEXTS)},
        ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return tasks


def execute(max_new: int | None = None) -> None:
    tasks = prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved reader error or inflight record")
    rows = read_rows(RESPONSES)
    saved = {}
    for row in rows:
        assert row.get("ok") and row["task_id"] not in saved
        saved[row["task_id"]] = row
    assert [r["task_id"] for r in rows] == [t["task_id"] for t in tasks[:len(rows)]]
    print("PHASE30C_READER_START", len(saved), "/", len(tasks), flush=True)
    if len(saved) == len(tasks):
        return
    limit = len(tasks) if max_new is None else min(max_new, len(tasks))
    paid = 0
    started = time.monotonic()
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved:
                continue
            elapsed = time.monotonic() - started
            if paid >= limit or elapsed + 120 >= 1800 or (elapsed + 120) * 2.60 / 3600 >= 1.35:
                print("PHASE30C_READER_GUARD", len(saved), "/", len(tasks), flush=True)
                break
            req = payload(task)
            ph = digest(req)
            atomic_write(INFLIGHT, {"task_id": task["task_id"], "prompt_sha256": ph,
                                    "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                result = infer_once.remote(req)
            except BaseException as exc:
                append_fsync(ERROR, {"task_id": task["task_id"], "error_type": type(exc).__name__})
                print("PHASE30C_READER_EXCEPTION", type(exc).__name__, flush=True)
                break
            paid += 1
            if not result.get("ok"):
                append_fsync(ERROR, {"task_id": task["task_id"], "phase": result.get("phase"),
                                     "http_status": result.get("http_status"), "exception": result.get("exception")})
                print("PHASE30C_READER_ERROR", task["task_id"], flush=True)
                break
            rec = {"task_id": task["task_id"], "case_key": task["case_key"], "dataset": task["dataset"],
                   "id": task["id"], "reader_variant": "reader-guard-v1", "ok": True,
                   "answer": str(result.get("answer") or "").strip(), "prompt_sha256": ph, "paid": True,
                   "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
                   "finish_reason": result.get("finish_reason")}
            append_fsync(RESPONSES, rec)
            INFLIGHT.unlink()
            saved[task["task_id"]] = rec
            if len(saved) % 8 == 0:
                print("PHASE30C_READER_PROGRESS", len(saved), "/", len(tasks), "paid", paid, flush=True)
    print("PHASE30C_READER_SESSION_END", len(saved), "/", len(tasks), "paid", paid, flush=True)


def freeze() -> None:
    tasks = prepare()
    rows = read_rows(RESPONSES)
    assert len(rows) == len(tasks) and [r["task_id"] for r in rows] == [t["task_id"] for t in tasks]
    assert all(r.get("ok") and r.get("paid") for r in rows) and not ERROR.exists() and not INFLIGHT.exists()
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    FROZEN.write_text(json.dumps({"n": 44, "question_n": 44, "responses_sha256": sha_file(RESPONSES),
        "task_sha256": sha_file(TASKS), "pre_remote_sha256": sha_file(PLAN), "actual_paid_reader_calls": 44,
        "identical_prompt_reuse_n": 0, "empty_predictions": sum(not r["answer"] for r in rows),
        "gold_seen": False, "selector_contexts_frozen": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE30C_READER_FROZEN", len(rows), flush=True)


def normalize(text: object) -> str:
    value = "".join(ch for ch in str(text).lower() if ch not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_metrics(pred: object, truth: object) -> tuple[int, float]:
    left, right = normalize(pred), normalize(truth)
    exact = int(left == right)
    if (left in ("yes", "no", "noanswer") or right in ("yes", "no", "noanswer")) and left != right:
        return exact, 0.0
    a, b = left.split(), right.split()
    if not a or not b:
        return exact, float(a == b)
    overlap = sum((Counter(a) & Counter(b)).values())
    if not overlap:
        return exact, 0.0
    p, r = overlap / len(a), overlap / len(b)
    return exact, 2 * p * r / (p + r)


def gold_answers() -> dict[str, str]:
    tasks = read_rows(TASKS)
    hot_ids = {t["id"] for t in tasks if t["dataset"] == "HotpotQA"}
    mus_ids = {t["id"] for t in tasks if t["dataset"] == "MuSiQue"}
    hot = {r["id"]: r["answer"] for r in pq.ParquetFile(HOT_DATA).read(columns=["id", "answer"]).to_pylist() if r["id"] in hot_ids}
    mus = {r["id"]: r["answer"] for r in pq.ParquetFile(MUS_DATA).read(columns=["id", "answer"]).to_pylist() if r["id"] in mus_ids}
    assert len(hot) == 36 and len(mus) == 8
    return {"HotpotQA/" + k: v for k, v in hot.items()} | {"MuSiQue/" + k: v for k, v in mus.items()}


def baseline_metrics() -> dict[str, dict]:
    selected = {t["case_key"] for t in read_rows(TASKS)}
    out = {}
    for dataset, path in [("HotpotQA", HOT_RESULT), ("MuSiQue", MUS_RESULT)]:
        result = json.loads(path.read_text(encoding="utf8"))
        for case in result["per_case"]:
            key = dataset + "/" + case["id"]
            if key in selected:
                arm = case["semantic-joint-citations"]
                out[key] = {"EM": float(arm["EM"]), "F1": float(arm["F1"])}
    assert len(out) == 44
    return out


def paired(rows: list[dict], metric: str) -> dict:
    wins = sum(r["new"][metric] > r["baseline"][metric] for r in rows)
    losses = sum(r["new"][metric] < r["baseline"][metric] for r in rows)
    deltas = [r["new"][metric] - r["baseline"][metric] for r in rows]
    rng = np.random.default_rng(20261031 + (1 if metric == "EM" else 2))
    boot = np.array([np.mean([deltas[i] for i in rng.integers(0, len(deltas), len(deltas))]) for _ in range(10000)])
    return {"mean_delta": float(np.mean(deltas)), "bootstrap95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "win": wins, "loss": losses, "net": wins - losses,
            "two_sided_exact_mcnemar_p": float(binomtest(min(wins, losses), wins + losses, 0.5).pvalue) if wins + losses else 1.0}


def score() -> None:
    if not FROZEN.exists() or RESULT.exists():
        if not FROZEN.exists():
            raise RuntimeError("Freeze reader responses before scoring")
        raise FileExistsError(RESULT)
    frozen = json.loads(FROZEN.read_text(encoding="utf8"))
    protocol = json.loads(PLAN.read_text(encoding="utf8"))
    assert frozen["gold_seen"] is False and sha_file(RESPONSES) == frozen["responses_sha256"]
    assert sha_file(TASKS) == frozen["task_sha256"] == protocol["tasks_sha256"]
    assert sha_file(PLAN) == frozen["pre_remote_sha256"]
    tasks = read_rows(TASKS)
    records = read_rows(RESPONSES)
    assert len(records) == len(tasks) == 44
    assert all(r["prompt_sha256"] == digest(payload(t)) for t, r in zip(tasks, records, strict=True))
    gold = gold_answers()
    baseline = baseline_metrics()
    rows = []
    for task, record in zip(tasks, records, strict=True):
        key = task["case_key"]
        em, f1 = answer_metrics(record["answer"], gold[key])
        rows.append({"case_key": key, "dataset": task["dataset"], "id": task["id"],
                     "baseline": baseline[key], "new": {"EM": em, "F1": f1, "empty": int(not record["answer"])},
                     "answer": record["answer"]})
    result = {"decision": "EXPLORATORY_READER_GUARD_COMPLETE", "n": 44, "datasets": {"HotpotQA": 36, "MuSiQue": 8},
              "same_semantic_context": True, "selector_unchanged": True, "gold_first_opened_after_reader_freeze": True,
              "not_independent_confirmatory_test": True, "paired_EM": paired(rows, "EM"), "paired_F1": paired(rows, "F1"),
              "new_unknown": sum(normalize(r["answer"]) == "unknown" for r in rows),
              "previously_correct_to_unknown": sum(r["baseline"]["EM"] == 1 and normalize(r["answer"]) == "unknown" for r in rows),
              "actual_paid_reader_calls": 44, "remote_calls": 44, "reader_sha256": frozen["responses_sha256"],
              "task_sha256": frozen["task_sha256"], "gold_sent_remote": False,
              "baseline_result_sha256": {"HotpotQA": sha_file(HOT_RESULT), "MuSiQue": sha_file(MUS_RESULT)},
              "prompt_hashes_recomputed": 44, "per_case": rows}
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    write_report(result)
    print("PHASE30C_POST_GOLD", json.dumps({"EM": result["paired_EM"], "F1": result["paired_F1"], "paid": 44}, ensure_ascii=False), flush=True)


def write_report(result: dict) -> None:
    lines = ["# Faz 30C — Okuyucu `unknown` koruma ablation", "", "**Tür:** Keşifsel, yeni donmuş alt örneklem  ",
             "**Kapsam:** 36 HotpotQA + 8 MuSiQue; Faz 30B’de kullanılmamış bağlamlar  ", "**Yeni okuyucu çağrısı:** 44", "",
             "## Tasarım", "",
             "Faz 30B’de gözlenen `unknown` kayıplarını azaltmak için selector ve semantic-joint-citations bağlamları sabit tutuldu. Faz 28 ve Faz 29 kayıtlarında kaynak cümlesi kümesi değişmemiş sorular arasından kimlik başına SHA-256 ile önceden belirlenmiş 36 HotpotQA ve 8 MuSiQue seçildi. Yeni istem, desteklenen bir cevap varsa doğrudan vermeyi ve `unknown`ı yalnızca cevapsız durumlara ayırmayı söyler.", "",
             "Bu çalışma aynı public-validation arşivlerini kullandığı için confirmatory veya bağımsız benchmark sonucu değildir. Gold cevaplar yanıtlar dondurulduktan sonra okundu.", "",
             "## Sonuç", "",
             f"- EM: semantic okuyucu {int(sum(r['baseline']['EM'] for r in result['per_case']))}/44 → reader-guard-v1 {int(sum(r['new']['EM'] for r in result['per_case']))}/44; net {result['paired_EM']['net']}, exact McNemar p = {result['paired_EM']['two_sided_exact_mcnemar_p']:.12g}.",
             f"- F1 farkı: {result['paired_F1']['mean_delta']:.4f}; bootstrap %95 GA [{result['paired_F1']['bootstrap95'][0]:.4f}, {result['paired_F1']['bootstrap95'][1]:.4f}].",
             f"- Yeni `unknown`: {result['new_unknown']}; daha önce doğru olup `unknown`a dönen: {result['previously_correct_to_unknown']}.", ""]
    for dataset in ("HotpotQA", "MuSiQue"):
        sub = [r for r in result["per_case"] if r["dataset"] == dataset]
        lines.append(f"- {dataset}: EM {int(sum(r['baseline']['EM'] for r in sub))}/{len(sub)} → {int(sum(r['new']['EM'] for r in sub))}/{len(sub)}; F1 {np.mean([r['baseline']['F1'] for r in sub]):.4f} → {np.mean([r['new']['F1'] for r in sub]):.4f}.")
    lines += ["", "## Yorum", "", "Bu sonuç yalnızca okuyucu isteminin abstention davranışını ölçer. Pozitif sonuç, Faz 30B’deki kaybın istem kaynaklı olduğunu düşündürür; sonuç nötr veya negatifse okuyucu değişikliğini bırakıp selector kazanımını raporlamak gerekir.", "", "Sonuç Faz 28’in confirmatory kararını değiştirmez."]
    (OUT / "PHASE30C_TURKCE_RAPORU.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true"); g.add_argument("--execute", action="store_true")
    g.add_argument("--freeze", action="store_true"); g.add_argument("--score", action="store_true")
    p.add_argument("--max-new", type=int, default=None)
    a = p.parse_args()
    if a.prepare: print("PREPARED", len(prepare()))
    elif a.execute: execute(a.max_new)
    elif a.freeze: freeze()
    else: score()


if __name__ == "__main__":
    main()
