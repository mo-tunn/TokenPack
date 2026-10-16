"""Assemble gold-blind paired IIRC contexts under the identical 25% cap."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase28_semantic_core import parse_ids
from submission.experiments.phase36_iirc_candidates import CORPUS, CORPUS_FREEZE, source_atoms
from submission.experiments.phase36_iirc_core import ARMS
from submission.experiments.phase36_iirc_register import OUT, SCRATCH
from submission.experiments.phase7_modal27b_matched8 import atomic_write, read_rows
from submission.experiments.phase8_atomic_contexts import render
from tokenpack.tokenization import TokenCounter

SCORED = OUT / "scored_original_candidates.jsonl"
TASKS = OUT / "semantic_citation_tasks.jsonl"
VERIFIER_ANSWERS = OUT / "semantic_verifier_responses.jsonl"
VERIFIER_FROZEN = OUT / "VERIFIER_RESPONSES_FROZEN.json"
OUTPUT = OUT / "gold_blind_paired_contexts.jsonl"
PRE = OUT / "CONTEXT_ASSEMBLY_PRE_GOLD_FREEZE.json"
FROZEN = OUT / "CONTEXTS_FROZEN.json"
PROGRESS = SCRATCH / "context_checkpoints"


def rank_indices(atoms: list[dict], scores: list[float]) -> list[int]:
    return sorted(
        range(len(atoms)),
        key=lambda index: (-scores[index], atoms[index]["article"], atoms[index]["sent_id"]),
    )


def selected_details(atoms: list[dict], indexes: list[int], cap: int, counter: TokenCounter) -> dict:
    text = render(indexes, atoms)
    tokens = counter.count(text)
    if tokens > cap:
        raise AssertionError("Rendered context exceeds its preregistered hard cap")
    return {
        "text": text,
        "tokens": tokens,
        "budget": cap,
        "source_articles": len({atoms[index]["article"] for index in indexes}),
        "sentences": len(indexes),
        "candidate_count": len(atoms),
    }


def baseline_select(atoms: list[dict], scores: list[float], cap: int, counter: TokenCounter):
    """Same sorted greedy whole-sentence rank as Phase27 baseline_indices."""
    if len(atoms) != len(scores) or not atoms:
        raise AssertionError("Candidate score/source mismatch")
    selected = []
    for index in rank_indices(atoms, scores):
        if counter.count(render([*selected, index], atoms)) <= cap:
            selected.append(index)
    details = selected_details(atoms, selected, cap, counter)
    return details, selected


def treatment_select(task: dict, atoms: list[dict], scores: list[float], raw_output: str, counter: TokenCounter,
                     baseline: tuple[dict, list[int]]):
    """Phase28 selector logic, reusing the already computed identical fallback."""
    citations, status = parse_ids(raw_output, task)
    if not citations:
        return baseline[0], baseline[1], "fallback_" + status
    known = {candidate["cid"]: candidate for candidate in task["candidates"]}
    seed_indexes = [known[citation]["atom_index"] for citation in citations]
    if counter.count(render(seed_indexes, atoms)) > task["cap"]:
        return baseline[0], baseline[1], "fallback_citations_exceed_cap"

    selected = list(seed_indexes)
    used = set(selected)
    for index in rank_indices(atoms, scores):
        if index in used:
            continue
        if counter.count(render([*selected, index], atoms)) <= task["cap"]:
            selected.append(index)
            used.add(index)
    details = selected_details(atoms, selected, task["cap"], counter)
    return details, selected, "verified_citation_anchors_selected"


def main() -> None:
    if FROZEN.exists():
        raise FileExistsError("Context assembly is immutable once frozen")
    OUTPUT.unlink(missing_ok=True)
    OUTPUT.with_suffix(OUTPUT.suffix + ".partial").unlink(missing_ok=True)
    protocol = json.loads((OUT / "PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(Path(__file__)) == protocol["code_sha256"]["phase36_iirc_contexts.py"]
    candidate_freeze = json.loads((OUT / "CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    verifier_freeze = json.loads(VERIFIER_FROZEN.read_text(encoding="utf8"))
    source_freeze = json.loads(CORPUS_FREEZE.read_text(encoding="utf8"))
    assert candidate_freeze["gold_unopened"] and verifier_freeze["gold_unopened"]
    assert sha_file(SCORED) == candidate_freeze["scored_sha256"]
    assert sha_file(TASKS) == candidate_freeze["tasks_sha256"]
    assert sha_file(CORPUS) == source_freeze["source_corpus_sha256"]
    assert sha_file(VERIFIER_ANSWERS) == verifier_freeze["responses_sha256"]

    questions = read_rows(SCRATCH / "gold_blind_question_manifest.jsonl")
    source_rows = read_rows(CORPUS)
    cases = read_rows(SCORED)
    tasks = read_rows(TASKS)
    answers = read_rows(VERIFIER_ANSWERS)
    assert len(questions) == len(cases) == len(tasks) == len(answers) == protocol["sample"]["questions"]
    assert [row["qid"] for row in questions] == [row["id"] for row in cases] == [row["id"] for row in tasks]
    assert [row["id"] for row in answers] == [row["id"] for row in tasks]
    source_by_pid = {row["pid"]: row for row in source_rows}
    task_by_id = {row["id"]: row for row in tasks}
    answer_by_id = {row["id"]: row for row in answers}
    case_by_id = {row["id"]: row for row in cases}
    counter = TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("Exact cl100k_base tokenizer is unavailable")

    pre = {
        "n": len(questions),
        "protocol_sha256": sha_file(OUT / "PRE_GOLD_PROTOCOL.json"),
        "candidates_sha256": candidate_freeze["scored_sha256"],
        "verifier_sha256": verifier_freeze["responses_sha256"],
        "source_corpus_sha256": source_freeze["source_corpus_sha256"],
        "assembly_code_sha256": sha_file(Path(__file__)),
        "phase28_selector_code_sha256": sha_file(ROOT / "submission/experiments/phase28_semantic_core.py"),
        "gold_unopened": True,
        "same_source_pool_and_cap": True,
    }
    if PRE.exists():
        assert json.loads(PRE.read_text(encoding="utf8")) == pre
    else:
        atomic_write(PRE, pre)

    stats = {
        "n": 0,
        "contexts_changed": 0,
        "source_sentence_sets_changed": 0,
        "citation_parse_valid_cross_document": 0,
        "verified_citation_seed_sets": 0,
        "fallback_count": 0,
        "baseline_sentences": 0,
        "treatment_sentences": 0,
        "mean_baseline_tokens": 0.0,
        "mean_treatment_tokens": 0.0,
        "mean_cap": 0.0,
        "token_cap_violations": 0,
        "fallback_reasons": {},
    }
    PROGRESS.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [PROGRESS / f"case_{index:04d}.json" for index in range(len(questions))]
    for partial in PROGRESS.glob("case_*.json.partial"):
        partial.unlink()
    saved_records = []
    for index, checkpoint in enumerate(checkpoint_paths):
        if not checkpoint.exists():
            if any(path.exists() for path in checkpoint_paths[index + 1:]):
                raise AssertionError("Context checkpoints are not a contiguous prefix")
            break
        record = json.loads(checkpoint.read_text(encoding="utf8"))
        if record.get("index") != index or record.get("id") != questions[index]["qid"]:
            raise AssertionError("Context checkpoint does not match preregistered order")
        if record.get("assembly_sha256") != sha_file(Path(__file__)) or record.get("protocol_sha256") != sha_file(OUT / "PRE_GOLD_PROTOCOL.json"):
            raise AssertionError("Context checkpoint source code/protocol hash mismatch")
        saved_records.append(record)

    def add_stats(record: dict) -> None:
        contexts = record["contexts"]
        status = record["semantic_seed_status"]
        baseline_pairs = contexts[ARMS[0]]["selected_original_pairs"]
        treatment_pairs = contexts[ARMS[1]]["selected_original_pairs"]
        baseline_set = {(x["article"], x["title"], x["sent_id"]) for x in baseline_pairs}
        treatment_set = {(x["article"], x["title"], x["sent_id"]) for x in treatment_pairs}
        stats["n"] += 1
        stats["contexts_changed"] += int(contexts[ARMS[0]]["text"] != contexts[ARMS[1]]["text"])
        stats["source_sentence_sets_changed"] += int(baseline_set != treatment_set)
        stats["citation_parse_valid_cross_document"] += int(record["semantic_citation_parse_status"] == "valid_cross_article_citations")
        stats["verified_citation_seed_sets"] += int(status == "verified_citation_anchors_selected")
        stats["fallback_count"] += int(status != "verified_citation_anchors_selected")
        if status != "verified_citation_anchors_selected":
            stats["fallback_reasons"][status] = stats["fallback_reasons"].get(status, 0) + 1
        stats["baseline_sentences"] += len(baseline_pairs)
        stats["treatment_sentences"] += len(treatment_pairs)
        stats["mean_baseline_tokens"] += contexts[ARMS[0]]["tokens"] / len(questions)
        stats["mean_treatment_tokens"] += contexts[ARMS[1]]["tokens"] / len(questions)
        stats["mean_cap"] += record["cap"] / len(questions)
        stats["token_cap_violations"] += int(
            contexts[ARMS[0]]["tokens"] > record["cap"]
            or contexts[ARMS[1]]["tokens"] > record["cap"]
        )

    for record in saved_records:
        add_stats(record)

    for index, question in enumerate(questions[len(saved_records):], start=len(saved_records)):
            case = case_by_id[question["qid"]]
            task = task_by_id[question["qid"]]
            answer = answer_by_id[question["qid"]]
            atoms, source_total = source_atoms(source_by_pid[question["pid"]], counter)
            assert source_total == case["source_total_tokens"]
            assert len(atoms) == len(case["scores"])
            assert task["cap"] == case["cap"] == max(1, math.floor(source_total * 0.25))
            assert task["question"] == question["question"] == case["question"]
            scores = case["scores"]
            base, baseline_indexes = baseline_select(atoms, scores, case["cap"], counter)
            treat, treatment_indexes, seed_status = treatment_select(
                task, atoms, scores, answer["raw_output"], counter,
                (base, baseline_indexes),
            )
            contexts = {}
            for arm, details, selected_indexes in (
                (ARMS[0], base, baseline_indexes),
                (ARMS[1], treat, treatment_indexes),
            ):
                contexts[arm] = {
                    "text": details["text"],
                    "tokens": details["tokens"],
                    "cap": case["cap"],
                    "source_article_count": details["source_articles"],
                    "selected_sentence_count": len(selected_indexes),
                    "selected_original_pairs": [
                        {
                            "article": atoms[selected_index]["article"],
                            "title": atoms[selected_index]["title"],
                            "sent_id": atoms[selected_index]["sent_id"],
                        }
                        for selected_index in selected_indexes
                    ],
                }
            baseline_set = {
                (item["article"], item["title"], item["sent_id"])
                for item in contexts[ARMS[0]]["selected_original_pairs"]
            }
            treatment_set = {
                (item["article"], item["title"], item["sent_id"])
                for item in contexts[ARMS[1]]["selected_original_pairs"]
            }
            changed = baseline_set != treatment_set
            record = {
                "index": index,
                "id": question["qid"],
                "pid": question["pid"],
                "question": question["question"],
                "source_total_tokens": source_total,
                "cap": case["cap"],
                "contexts": contexts,
                "semantic_citation_parse_status": answer["parse_status"],
                "semantic_seed_status": seed_status,
                "source_sentence_set_changed": changed,
                "source_has_no_gold_annotations": True,
                "protocol_sha256": sha_file(OUT / "PRE_GOLD_PROTOCOL.json"),
                "assembly_sha256": sha_file(Path(__file__)),
            }
            atomic_write(checkpoint_paths[index], record)
            saved_records.append(record)
            add_stats(record)
            if (index + 1) % 24 == 0 or index + 1 == len(questions):
                print("PHASE36_CONTEXTS_PROGRESS", index + 1, len(questions), json.dumps(stats), flush=True)

    partial_output = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    with partial_output.open("x", encoding="utf8") as stream:
        for index in range(len(questions)):
            record = json.loads(checkpoint_paths[index].read_text(encoding="utf8"))
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial_output.replace(OUTPUT)

    freeze = {
        "n": len(questions),
        "sha256": sha_file(OUTPUT),
        "pre_gold_assembly_sha256": sha_file(PRE),
        "protocol_sha256": sha_file(OUT / "PRE_GOLD_PROTOCOL.json"),
        "verifier_response_sha256": verifier_freeze["responses_sha256"],
        "gold_unopened": True,
        "same_per_question_hard_caps": True,
        "stats": stats,
    }
    atomic_write(FROZEN, freeze)
    print("PHASE36_CONTEXTS_FROZEN", json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
