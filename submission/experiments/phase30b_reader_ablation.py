"""Phase 30B exploratory reader ablation on frozen changed-context cases.

The semantic-joint-citations selector and rendered contexts are fixed.  The
only new component is a reader prompt that asks the model to internally connect
the evidence before returning the same short answer format.  Cases are limited
to the 36 Phase 28 and 8 Phase 29 cases whose source sentence set changed.
This is a diagnostic study on existing public-validation cases, not a new
confirmatory benchmark claim.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
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


OUT = ROOT / "submission/results/phase30b_reader_ablation_20261013"
HOT_CONTEXTS = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013/gold_blind_paired_contexts.jsonl"
HOT_CONTEXT_FREEZE = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013/CONTEXTS_FROZEN.json"
HOT_RESULT = ROOT / "submission/results/phase28_hotpot_semantic_citation_validation_20261013/VALIDATION_POST_GOLD_RESULTS.json"
MUS_CONTEXTS = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot/gold_blind_paired_contexts.jsonl"
MUS_CONTEXT_FREEZE = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot/CONTEXTS_FROZEN.json"
MUS_RESULT = ROOT / "submission/results/phase29_musique_semantic_citation_crossbench_20261013/pilot/PILOT_POST_GOLD_RESULTS.json"
HOT_DATA = ROOT / "submission/data/phase6_hotpotqa_validation_20261011/distractor/validation-00000-of-00001.parquet"
MUS_DATA = ROOT / "submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"

CONTEXT_INPUTS = OUT / "CONTEXT_INPUTS_FROZEN.json"
TASKS = OUT / "reader_tasks.jsonl"
PLAN = OUT / "READER_PRE_REMOTE_FREEZE.json"
RESPONSES = OUT / "reader_responses.jsonl"
ERROR = OUT / "reader_error.jsonl"
INFLIGHT = OUT / "reader_inflight.json"
FROZEN = OUT / "READER_RESPONSES_FROZEN.json"
RESULT = OUT / "ABLATION_POST_GOLD_RESULTS.json"

SYSTEM = (
    "Answer the question using only the context. First silently identify the "
    "facts needed to connect the question to its answer. Then reply only with "
    "the shortest correct answer phrase, preserving names, numbers, dates, and "
    "spelling from the context. Do not explain your reasoning. If the context "
    "does not support an answer, reply unknown."
)
MAX_GENERATED = 72
SEED = 20261012


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def payload(task: dict) -> dict:
    user = "Context:\n" + task["context"] + "\n\nQuestion:\n" + task["question"] + "\n\nAnswer in one short phrase:"
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "min_p": 0,
        "seed": SEED,
        "max_tokens": MAX_GENERATED,
        "reasoning_effort": "none",
        "stream": False,
    }


def load_changed_contexts(dataset: str, context_path: Path, freeze_path: Path) -> list[dict]:
    freeze = json.loads(freeze_path.read_text(encoding="utf8"))
    if sha_file(context_path) != freeze["sha256"]:
        raise AssertionError(f"Context hash changed: {context_path}")
    rows = read_rows(context_path)
    changed = []
    for row in rows:
        if not row.get("source_sentence_set_changed"):
            continue
        arm = row["contexts"]["semantic-joint-citations"]
        changed.append({
            "case_key": dataset + "/" + row["id"],
            "dataset": dataset,
            "id": row["id"],
            "question": row["question"],
            "context": arm["text"],
            "context_tokens": arm["tokens"],
            "context_sha256": digest(arm["text"]),
            "source_sentence_set_changed": True,
        })
    return changed


def prepare() -> list[dict]:
    hot = load_changed_contexts("HotpotQA", HOT_CONTEXTS, HOT_CONTEXT_FREEZE)
    mus = load_changed_contexts("MuSiQue", MUS_CONTEXTS, MUS_CONTEXT_FREEZE)
    cases = hot + mus
    assert len(hot) == 36 and len(mus) == 8 and len(cases) == 44
    tasks = []
    for case in cases:
        task = dict(case)
        task["task_id"] = case["case_key"] + "|semantic-joint-citations|reader-chain-v1"
        tasks.append(task)
    OUT.mkdir(parents=True, exist_ok=True)
    if TASKS.exists():
        if read_rows(TASKS) != tasks:
            raise AssertionError("Reader tasks were modified")
    else:
        with TASKS.open("x", encoding="utf8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
    spec = {
        "study": "Phase30B frozen semantic context reader ablation",
        "n": len(tasks),
        "datasets": {"HotpotQA_changed_contexts": 36, "MuSiQue_changed_contexts": 8},
        "selector_frozen": "semantic-joint-citations",
        "context_input_sha256": {
            "HotpotQA_CONTEXTS_FROZEN": sha_file(HOT_CONTEXT_FREEZE),
            "MuSiQue_CONTEXTS_FROZEN": sha_file(MUS_CONTEXT_FREEZE),
            "HotpotQA_contexts": sha_file(HOT_CONTEXTS),
            "MuSiQue_contexts": sha_file(MUS_CONTEXTS),
        },
        "tasks_sha256": sha_file(TASKS),
        "reader_code_sha256": sha_file(Path(__file__)),
        "relay_sha256": sha_file(ROOT / "submission/experiments/phase13_graph_modal_relay_source.py"),
        "model": MODEL,
        "endpoint": BASE,
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode("utf8")).hexdigest(),
        "baseline_reader_prompt": "Phase28/29 existing fixed short-answer prompt",
        "new_reader_prompt": "reader-chain-v1; silent evidence connection, short answer only",
        "same_temperature_top_k_top_p_seed_max_tokens": True,
        "gold_loaded": False,
        "gold_sent_remote": False,
        "max_paid_calls": len(tasks),
        "exploratory_same_cases_as_frozen_phase28_phase29": True,
    }
    if PLAN.exists():
        if json.loads(PLAN.read_text(encoding="utf8")) != spec:
            raise AssertionError("Reader pre-remote protocol was modified")
    else:
        PLAN.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    CONTEXT_INPUTS.write_text(json.dumps({"n": len(cases), "cases": cases,
        "hotpot_n": len(hot), "musique_n": len(mus), "gold_loaded": False,
        "hotpot_context_sha256": sha_file(HOT_CONTEXTS),
        "musique_context_sha256": sha_file(MUS_CONTEXTS)}, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return tasks


def execute(max_new: int | None = None) -> None:
    tasks = prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Existing reader error/inflight record must be audited before continuing")
    rows = read_rows(RESPONSES)
    saved = {}
    for row in rows:
        if not row.get("ok") or row["task_id"] in saved:
            raise AssertionError("Invalid or duplicate response record")
        saved[row["task_id"]] = row
    assert [row["task_id"] for row in rows] == [task["task_id"] for task in tasks[:len(rows)]]
    print("PHASE30B_READER_START", len(saved), "/", len(tasks), flush=True)
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
                print("PHASE30B_READER_GUARD", len(saved), "/", len(tasks), flush=True)
                break
            request = payload(task)
            prompt_hash = digest(request)
            atomic_write(INFLIGHT, {"task_id": task["task_id"], "prompt_sha256": prompt_hash,
                                    "utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                result = infer_once.remote(request)
            except BaseException as exc:
                append_fsync(ERROR, {"task_id": task["task_id"], "error_type": type(exc).__name__})
                print("PHASE30B_READER_EXCEPTION", type(exc).__name__, flush=True)
                break
            paid += 1
            if not result.get("ok"):
                append_fsync(ERROR, {"task_id": task["task_id"], "phase": result.get("phase"),
                                     "http_status": result.get("http_status"), "exception": result.get("exception")})
                print("PHASE30B_READER_ERROR", task["task_id"], flush=True)
                break
            record = {"task_id": task["task_id"], "case_key": task["case_key"],
                      "dataset": task["dataset"], "id": task["id"], "arm": "semantic-joint-citations",
                      "reader_variant": "reader-chain-v1", "ok": True,
                      "answer": str(result.get("answer") or "").strip(),
                      "prompt_sha256": prompt_hash, "paid": True,
                      "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
                      "finish_reason": result.get("finish_reason")}
            append_fsync(RESPONSES, record)
            INFLIGHT.unlink()
            saved[task["task_id"]] = record
            if len(saved) % 8 == 0:
                print("PHASE30B_READER_PROGRESS", len(saved), "/", len(tasks), "paid", paid, flush=True)
    print("PHASE30B_READER_SESSION_END", len(saved), "/", len(tasks), "paid", paid, flush=True)


def freeze() -> None:
    tasks = prepare()
    rows = read_rows(RESPONSES)
    assert len(rows) == len(tasks)
    assert [row["task_id"] for row in rows] == [task["task_id"] for task in tasks]
    assert all(row.get("ok") for row in rows)
    assert not ERROR.exists() and not INFLIGHT.exists()
    assert all(row.get("paid") for row in rows)
    if FROZEN.exists():
        raise FileExistsError(FROZEN)
    FROZEN.write_text(json.dumps({
        "n": len(rows), "question_n": len(rows), "responses_sha256": sha_file(RESPONSES),
        "task_sha256": sha_file(TASKS), "pre_remote_sha256": sha_file(PLAN),
        "actual_paid_reader_calls": sum(bool(row.get("paid")) for row in rows),
        "identical_prompt_reuse_n": 0, "empty_predictions": sum(not row["answer"] for row in rows),
        "gold_seen": False, "selector_contexts_frozen": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("PHASE30B_READER_FROZEN", len(rows), flush=True)


def normalize(text: object) -> str:
    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
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
    precision, recall = overlap / len(a), overlap / len(b)
    return exact, 2 * precision * recall / (precision + recall)


def read_answers() -> dict[str, str]:
    rows = read_rows(RESPONSES)
    assert len(rows) == 44 and all(row.get("ok") for row in rows)
    return {row["case_key"]: row["answer"] for row in rows}


def gold_answers() -> dict[str, str]:
    hot_ids = {case["case_key"].split("/", 1)[1] for case in read_rows(TASKS) if case["dataset"] == "HotpotQA"}
    mus_ids = {case["case_key"].split("/", 1)[1] for case in read_rows(TASKS) if case["dataset"] == "MuSiQue"}
    hot = {row["id"]: row["answer"] for row in pq.ParquetFile(HOT_DATA).read(columns=["id", "answer"]).to_pylist()
           if row["id"] in hot_ids}
    mus = {row["id"]: row["answer"] for row in pq.ParquetFile(MUS_DATA).read(columns=["id", "answer"]).to_pylist()
           if row["id"] in mus_ids}
    assert len(hot) == 36 and len(mus) == 8
    return {"HotpotQA/" + key: value for key, value in hot.items()} | {"MuSiQue/" + key: value for key, value in mus.items()}


def baseline_metrics() -> dict[str, dict]:
    out = {}
    for dataset, path in [("HotpotQA", HOT_RESULT), ("MuSiQue", MUS_RESULT)]:
        result = json.loads(path.read_text(encoding="utf8"))
        for case in result["per_case"]:
            if case.get("source_sentence_set_changed"):
                arm = case["semantic-joint-citations"]
                out[dataset + "/" + case["id"]] = {"EM": float(arm["EM"]), "F1": float(arm["F1"]),
                                                     "source_result_sha256": sha_file(path)}
    assert len(out) == 44
    return out


def paired(rows: list[dict], metric: str) -> dict:
    wins = sum(row["new"][metric] > row["baseline"][metric] for row in rows)
    losses = sum(row["new"][metric] < row["baseline"][metric] for row in rows)
    deltas = [row["new"][metric] - row["baseline"][metric] for row in rows]
    rng = np.random.default_rng(20261030 + (1 if metric == "EM" else 2))
    bootstrap = np.array([np.mean([deltas[i] for i in rng.integers(0, len(deltas), len(deltas))]) for _ in range(10000)])
    return {"mean_delta": float(np.mean(deltas)), "bootstrap95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
            "win": wins, "loss": losses, "net": wins - losses,
            "two_sided_exact_mcnemar_p": float(binomtest(min(wins, losses), wins + losses, 0.5).pvalue) if wins + losses else 1.0}


def score() -> None:
    if not FROZEN.exists():
        raise RuntimeError("Reader responses must be frozen before gold scoring")
    if RESULT.exists():
        raise FileExistsError(RESULT)
    frozen = json.loads(FROZEN.read_text(encoding="utf8"))
    assert frozen["gold_seen"] is False and frozen["n"] == 44
    protocol = json.loads(PLAN.read_text(encoding="utf8"))
    assert sha_file(RESPONSES) == frozen["responses_sha256"]
    assert sha_file(TASKS) == frozen["task_sha256"] == protocol["tasks_sha256"]
    assert sha_file(PLAN) == frozen["pre_remote_sha256"]
    input_manifest = json.loads(CONTEXT_INPUTS.read_text(encoding="utf8"))
    assert input_manifest["n"] == 44 and input_manifest["gold_loaded"] is False
    assert sha_file(HOT_CONTEXTS) == input_manifest["hotpot_context_sha256"]
    assert sha_file(MUS_CONTEXTS) == input_manifest["musique_context_sha256"]
    tasks = read_rows(TASKS)
    answers = read_answers()
    records = read_rows(RESPONSES)
    assert [record["task_id"] for record in records] == [task["task_id"] for task in tasks]
    assert all(record["prompt_sha256"] == digest(payload(task)) for task, record in zip(tasks, records, strict=True))
    manifest_cases = {case["case_key"]: case for case in input_manifest["cases"]}
    assert len(manifest_cases) == 44
    for task in tasks:
        case = manifest_cases[task["case_key"]]
        assert task["context"] == case["context"] and task["question"] == case["question"]
        assert task["context_sha256"] == digest(task["context"])
    gold = gold_answers()
    baseline = baseline_metrics()
    rows = []
    for task in tasks:
        key = task["case_key"]
        new_em, new_f1 = answer_metrics(answers[key], gold[key])
        old = baseline[key]
        rows.append({"case_key": key, "dataset": task["dataset"], "id": task["id"],
                     "baseline": old, "new": {"EM": new_em, "F1": new_f1, "empty": int(not answers[key])},
                     "answer": answers[key]})
    result = {
        "decision": "EXPLORATORY_READER_ABLATION_COMPLETE",
        "n": len(rows), "datasets": {"HotpotQA": 36, "MuSiQue": 8},
        "same_semantic_context": True, "selector_unchanged": True,
        "gold_first_opened_after_reader_freeze": True,
        "not_independent_confirmatory_test": True,
        "paired_EM": paired(rows, "EM"), "paired_F1": paired(rows, "F1"),
        "baseline_empty": int(sum(row["baseline"].get("empty", 0) for row in rows)),
        "new_empty": int(sum(row["new"]["empty"] for row in rows)),
        "new_unknown_responses": int(sum(normalize(row["answer"]) == "unknown" for row in rows)),
        "previously_correct_to_unknown": int(sum(row["baseline"]["EM"] == 1 and normalize(row["answer"]) == "unknown" for row in rows)),
        "actual_paid_reader_calls": frozen["actual_paid_reader_calls"],
        "remote_calls": frozen["actual_paid_reader_calls"],
        "context_inputs_sha256": sha_file(CONTEXT_INPUTS), "reader_sha256": frozen["responses_sha256"],
        "baseline_result_sha256": {"HotpotQA": sha_file(HOT_RESULT), "MuSiQue": sha_file(MUS_RESULT)},
        "input_integrity": {"task_hash_valid": True, "response_hash_valid": True,
                            "protocol_hash_valid": True, "context_hashes_valid": True,
                            "prompt_hashes_recomputed": len(records),
                            "same_frozen_contexts_verified": True},
        "gold_sent_remote": False, "qasper_official_test_untouched": True,
        "per_case": rows,
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    write_report(result)
    print("PHASE30B_POST_GOLD", json.dumps({"n": len(rows), "EM": result["paired_EM"],
        "F1": result["paired_F1"], "paid_reader": result["actual_paid_reader_calls"]}, ensure_ascii=False), flush=True)


def write_report(result: dict) -> None:
    by_dataset = {}
    for dataset in ("HotpotQA", "MuSiQue"):
        subset = [row for row in result["per_case"] if row["dataset"] == dataset]
        by_dataset[dataset] = {"n": len(subset),
            "baseline_EM": int(sum(row["baseline"]["EM"] for row in subset)),
            "new_EM": int(sum(row["new"]["EM"] for row in subset)),
            "baseline_F1": float(np.mean([row["baseline"]["F1"] for row in subset])),
            "new_F1": float(np.mean([row["new"]["F1"] for row in subset])),
            "EM_win": int(sum(row["new"]["EM"] > row["baseline"]["EM"] for row in subset)),
            "EM_loss": int(sum(row["new"]["EM"] < row["baseline"]["EM"] for row in subset))}
    lines = [
        "# Faz 30B — Donmuş semantic bağlamlarda okuyucu ablation",
        "",
        "**Tarih:** 13 Ekim 2026  ",
        "**Tür:** Keşifsel okuyucu tanı çalışması  ",
        f"**Kayıt:** {result['n']} değişmiş kaynak-cümlesi bağlamı  ",
        f"**Yeni ücretli okuyucu çağrısı:** {result['actual_paid_reader_calls']}",
        "",
        "## Tasarım",
        "",
        "Semantic-joint-citations tarafından seçilmiş ve Faz 28/Faz 29’da kaynak cümlesi kümesi değişmiş bağlamlar aynen korunmuştur. Selector, soru kimlikleri, bağlamlar, model, sıcaklık, seed ve çıktı bütçesi değiştirilmemiştir. Tek değişken, okuyucu istemine kanıtları içsel olarak bağlamayı ve yalnız kısa cevap döndürmeyi söyleyen `reader-chain-v1` yönergesidir.",
        "",
        "Çalışma aynı mevcut public-validation sorularını kullandığı için bağımsız doğrulama değildir. Gold cevaplar 44 yeni okuyucu kaydı tamamlanıp SHA-256 ile dondurulduktan sonra yerel olarak okunmuştur.",
        "",
        "## Sonuç",
        "",
        f"- EM: semantic okuyucu {int(sum(row['baseline']['EM'] for row in result['per_case']))}/{result['n']} → reader-chain-v1 {int(sum(row['new']['EM'] for row in result['per_case']))}/{result['n']}; net {result['paired_EM']['net']}, exact McNemar p = {result['paired_EM']['two_sided_exact_mcnemar_p']:.12g}.",
        f"- F1: ortalama {np.mean([row['baseline']['F1'] for row in result['per_case']]):.4f} → {np.mean([row['new']['F1'] for row in result['per_case']]):.4f}; fark {result['paired_F1']['mean_delta']:.4f}, bootstrap %95 GA [{result['paired_F1']['bootstrap95'][0]:.4f}, {result['paired_F1']['bootstrap95'][1]:.4f}].",
        f"- Reader-chain-v1 boş cevapları: {result['new_empty']}; `unknown` yanıtları: {result['new_unknown_responses']}.",
        f"- Baseline’da doğru olup reader-chain-v1 tarafından `unknown` yapılan cevap: {result['previously_correct_to_unknown']}.",
        "",
        "| Veri kümesi | n | Baseline EM | Reader-chain-v1 EM | Baseline F1 | Reader-chain-v1 F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, s in by_dataset.items():
        lines.append(f"| {dataset} | {s['n']} | {s['baseline_EM']}/{s['n']} | {s['new_EM']}/{s['n']} | {s['baseline_F1']:.4f} | {s['new_F1']:.4f} |")
    lines += [
        "",
        "## Yorum",
        "",
        "Bu sonuç yalnızca okuyucu isteminin, semantic selector’ın zaten değiştirdiği bağlamlarda cevap üretimini değiştirip değiştirmediğini ölçer. Pozitif bir sonuç okuyucu darboğazı için kanıt sağlar; negatif veya belirsiz bir sonuç selector’ın destek kazanımının cevap üretimine aktarılmadığını gösterir.",
        "",
        "Bu varyantta EM kazancı görülmedi: altı eşleşmiş EM değişiminin tamamı kayıptı. Önceden doğru olan altı cevap yeni istemde `unknown` oldu. Dolayısıyla bu özel ‘kanıtları içsel olarak bağla’ talimatı okuyucuyu daha temkinli yaptı ve tanısal örneklemde performansı düşürdü. Bu bulgu başka okuyucu varyantlarını sınamaz.",
        "",
        "Bu ablation’ın sonucu Faz 28’in confirmatory kararını değiştirmez ve yeni bir benchmark genelleme iddiası oluşturmaz.",
    ]
    (OUT / "PHASE30B_TURKCE_RAPORU.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--max-new", type=int, default=None)
    args = parser.parse_args()
    if args.prepare:
        print("PREPARED", len(prepare()))
    elif args.execute:
        execute(args.max_new)
    elif args.freeze:
        freeze()
    else:
        score()


if __name__ == "__main__":
    main()
