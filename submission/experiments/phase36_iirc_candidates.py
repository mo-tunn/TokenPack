"""Gold-blind IIRC source extraction and pinned MiniLM candidate scoring."""
from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import math
import os
import re
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase28_semantic_core import make_task
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from tokenpack.chunking import SENTENCE_RE
from tokenpack.tokenization import TokenCounter

# An indirect import in phase8_atomic_contexts -> phase5b_pretrained_rerank_eval
# imports legacy QASPER modules that temporarily set CUDA_VISIBLE_DEVICES="".
# Restore the preregistered local RTX device after the full import chain and
# before PyTorch's first CUDA availability/device-count query.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SOURCE = ROOT / ".tokenpack/phase36_iirc_20261013"
IIRC_ARCHIVE = SOURCE / "iirc.tar.gz"
ARTICLES_ARCHIVE = SOURCE / "context_articles.tar.gz"
SCRATCH = SOURCE / "working"
OUT = ROOT / "submission/results/phase36_iirc_semantic_citations_20261014"
EXPECTED_IIRC_SHA256 = "efab875854691530a3bd55809244b0656dff1e15e98e4108bae5405c63e31a9b"
EXPECTED_ARTICLES_SHA256 = "d239390a27ed3f4aa868afe126187e6d108990a7bcc2a3efb6cae5de917264c7"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
WEIGHTS_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
FOLDER = Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots" / RERANKER_REVISION
CORPUS = SCRATCH / "gold_blind_source_corpus.jsonl"
QUESTIONS = SCRATCH / "gold_blind_question_manifest.jsonl"
CORPUS_FREEZE = SCRATCH / "SOURCE_CORPUS_FROZEN.json"
PROGRESS = SCRATCH / "candidate_checkpoints"
SCORED = OUT / "scored_original_candidates.jsonl"
TASKS = OUT / "semantic_citation_tasks.jsonl"
FROZEN = OUT / "CANDIDATES_FROZEN.json"
SCORER_BATCH_SIZE = 32
BLOCK_TAGS = {"br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "tr"}


def verify_local_model() -> int:
    file = FOLDER / "model.safetensors"
    if not file.is_file():
        raise FileNotFoundError(f"Missing pinned offline weights {file}")
    if sha_file(file) != WEIGHTS_SHA256:
        raise AssertionError("Pinned MiniLM weight SHA256 mismatch")
    return file.stat().st_size


def accepted_code_hash(prereg: dict) -> tuple[str, str | None]:
    current = sha_file(Path(__file__))
    originally_frozen = prereg["code_sha256"]["phase36_iirc_candidates.py"]
    if current == originally_frozen:
        return current, None
    amendment_path = OUT / "PREREG_AMENDMENT_1.json"
    if not amendment_path.is_file():
        raise AssertionError("Candidate code differs from preregistration and no frozen amendment exists")
    amendment = json.loads(amendment_path.read_text(encoding="utf8"))
    assert amendment["parent_protocol_sha256"] == sha_file(OUT / "PRE_GOLD_PROTOCOL.json")
    assert amendment["original_candidate_code_sha256"] == originally_frozen
    assert amendment["updated_candidate_code_sha256"] == current
    return current, sha_file(amendment_path)


class VisibleText(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_visible_text(source: str) -> str:
    parser = VisibleText()
    parser.feed(source)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def read_goldblind_dev() -> list[dict]:
    with tarfile.open(IIRC_ARCHIVE, "r:gz") as archive:
        raw = json.load(archive.extractfile("iirc/dev.json"))
    projected = []
    for passage in raw:
        projected.append({
            "pid": str(passage["pid"]),
            "title": str(passage["title"]),
            "text": str(passage["text"]),
            "links": [str(link["target"]) for link in passage["links"]],
            "questions": [
                {"qid": str(q["qid"]), "question": str(q["question"])}
                for q in passage["questions"]
            ],
        })
    del raw
    return projected


def write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with partial.open("x", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def extract_sources() -> None:
    prereg = json.loads((OUT / "PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    accepted_code_hash(prereg)
    if sha_file(IIRC_ARCHIVE) != EXPECTED_IIRC_SHA256:
        raise AssertionError("IIRC archive checksum changed after preregistration")
    if sha_file(ARTICLES_ARCHIVE) != EXPECTED_ARTICLES_SHA256:
        raise AssertionError("IIRC context archive checksum changed after preregistration")
    if CORPUS_FREEZE.exists():
        raise FileExistsError("Gold-blind source corpus is already frozen; preserve it for audit.")
    for path in (CORPUS, QUESTIONS):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)
    CORPUS_FREEZE.with_suffix(CORPUS_FREEZE.suffix + ".partial").unlink(missing_ok=True)

    dev = read_goldblind_dev()
    link_targets = {target.casefold() for passage in dev for target in passage["links"]}
    with tarfile.open(ARTICLES_ARCHIVE, "r:gz") as archive:
        raw_articles = json.load(archive.extractfile("context_articles.json"))
    article_text = {
        title: html_to_visible_text(raw_articles[title])
        for title in link_targets
        if title in raw_articles and raw_articles[title]
    }
    del raw_articles

    corpus_rows = []
    questions = []
    missing = set()
    for passage in dev:
        docs = [{
            "article": 0,
            "title": passage["title"],
            "source_kind": "main",
            "text": passage["text"].strip(),
        }]
        seen = set()
        for target in passage["links"]:
            key = target.casefold()
            if key in seen:
                continue
            seen.add(key)
            text = article_text.get(key, "")
            if not text.strip():
                missing.add(key)
                continue
            docs.append({
                "article": len(docs),
                "title": target,
                "source_kind": "linked_article",
                "text": text,
            })
        corpus_rows.append({"pid": passage["pid"], "docs": docs})
        for question in passage["questions"]:
            questions.append({
                "qid": question["qid"],
                "pid": passage["pid"],
                "question": question["question"],
            })

    write_jsonl_exclusive(CORPUS, corpus_rows)
    write_jsonl_exclusive(QUESTIONS, questions)
    freeze = {
        "passage_count": len(corpus_rows),
        "question_count": len(questions),
        "source_corpus_sha256": sha_file(CORPUS),
        "question_manifest_sha256": sha_file(QUESTIONS),
        "source_archive_sha256": sha_file(IIRC_ARCHIVE),
        "context_articles_archive_sha256": sha_file(ARTICLES_ARCHIVE),
        "unique_link_targets": len(link_targets),
        "missing_unique_link_targets_n": len(missing),
        "missing_unique_link_targets_sha256": hashlib.sha256(
            "\n".join(sorted(missing)).encode("utf8")
        ).hexdigest(),
        "only_gold_blind_projection_used": True,
        "answer_context_question_links_accessed": False,
    }
    with CORPUS_FREEZE.open("x", encoding="utf8") as f:
        json.dump(freeze, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print("PHASE36_SOURCE_CORPUS_FROZEN", json.dumps(freeze), flush=True)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def source_atoms(source_row: dict, counter: TokenCounter) -> tuple[list[dict], int]:
    atoms = []
    total = 0
    for doc in source_row["docs"]:
        title = str(doc["title"])
        text = str(doc["text"]).strip()
        total += counter.count(title + "\n" + text)
        sentences = [sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()]
        if not sentences and text:
            sentences = [text]
        for sentence_id, sentence in enumerate(sentences):
            atoms.append({
                "article": int(doc["article"]),
                "title": title,
                "sent_id": sentence_id,
                "sentence": sentence,
                "input_text": title + "\n" + sentence,
            })
    if len(atoms) < 2:
        raise AssertionError(f"Source passage {source_row['pid']} yielded fewer than two sentences")
    return atoms, total


def encode_scores(model, tokenizer, question: str, texts: list[str], batch_size: int) -> list[float]:
    import math as _math
    import torch

    scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            [question] * len(batch), batch,
            padding=True,
            truncation="only_second",
            max_length=SCORER_MAX_INPUT_TOKENS,
            return_tensors="pt",
        )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded).logits.squeeze(-1).float().cpu().tolist()
        scores.extend(float(score) for score in output)
    if len(scores) != len(texts) or not all(_math.isfinite(score) for score in scores):
        raise AssertionError("MiniLM returned incomplete or non-finite sentence scores")
    return scores


def score_candidates(batch_size: int = SCORER_BATCH_SIZE) -> None:
    if not CORPUS_FREEZE.exists():
        raise FileNotFoundError("Run --extract before candidate scoring")
    prereg = json.loads((OUT / "PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    candidate_code_sha256, amendment_sha256 = accepted_code_hash(prereg)
    id_manifest = json.loads((OUT / "ID_MANIFEST_PRE_GOLD.json").read_text(encoding="utf8"))
    corpus_freeze = json.loads(CORPUS_FREEZE.read_text(encoding="utf8"))
    if sha_file(CORPUS) != corpus_freeze["source_corpus_sha256"]:
        raise AssertionError("Gold-blind source corpus changed")
    if sha_file(QUESTIONS) != corpus_freeze["question_manifest_sha256"]:
        raise AssertionError("Gold-blind question manifest changed")
    if FROZEN.exists():
        raise FileExistsError("Candidate artifacts are immutable once frozen")
    for path in (SCORED, TASKS):
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)

    source_rows = read_rows(CORPUS)
    questions = read_rows(QUESTIONS)
    assert len(source_rows) == prereg["sample"]["passages"]
    assert len(questions) == prereg["sample"]["questions"]
    assert [question["qid"] for question in questions] == id_manifest["ids_in_frozen_dataset_order"]
    source_by_pid = {row["pid"]: row for row in source_rows}

    counter = TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("Exact cl100k_base tokenizer is unavailable")
    atom_cache = {}
    source_totals = {}
    for row in source_rows:
        atoms, total = source_atoms(row, counter)
        atom_cache[row["pid"]] = atoms
        source_totals[row["pid"]] = total

    verify_local_model()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("RTX 3050 is not visible to the local MiniLM scorer")
    torch.cuda.init()
    tokenizer = AutoTokenizer.from_pretrained(FOLDER, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        FOLDER, local_files_only=True
    ).to("cuda").eval()
    if torch.cuda.get_device_properties(0).total_memory > 0:
        print("PHASE36_LOCAL_SCORER_DEVICE", torch.cuda.get_device_name(0), flush=True)

    PROGRESS.mkdir(parents=True, exist_ok=True)
    prereg_sha = sha_file(OUT / "PRE_GOLD_PROTOCOL.json")
    corpus_sha = sha_file(CORPUS)
    for partial in PROGRESS.glob("case_*.json.partial"):
        partial.unlink()
    saved_rows = []
    for checkpoint in sorted(PROGRESS.glob("case_*.json")):
        saved_rows.append(json.loads(checkpoint.read_text(encoding="utf8")))
    if len(saved_rows) > len(questions):
        raise AssertionError("More candidate checkpoints exist than preregistered questions")
    for index, row in enumerate(saved_rows):
        if row.get("index") != index or row.get("qid") != questions[index]["qid"]:
            raise AssertionError("Candidate checkpoints are not a contiguous preregistered prefix")
        if row.get("source_corpus_sha256") != corpus_sha or row.get("prereg_sha256") != prereg_sha:
            raise AssertionError("Candidate checkpoint source/protocol hash mismatch")
        if len(row.get("scores", [])) != len(atom_cache[row["pid"]]):
            raise AssertionError("Candidate checkpoint score count does not match its source pool")
    start_time = time.monotonic()
    totals = {
        "n": len(questions),
        "source_sentence_pairs_scored": 0,
        "verifier_candidates": 0,
        "mean_source_tokens": 0.0,
        "mean_cap": 0.0,
        "max_source_sentences_per_question": 0,
        "max_cap": 0,
        "token_cap_definition": "floor(0.25 * sum cl100k_base(title + newline + complete source text))",
    }
    for row in saved_rows:
        atom_count = len(atom_cache[row["pid"]])
        totals["source_sentence_pairs_scored"] += atom_count
        totals["verifier_candidates"] += len(row["task"]["candidates"])
        totals["mean_source_tokens"] += row["source_total_tokens"] / len(questions)
        totals["mean_cap"] += row["cap"] / len(questions)
        totals["max_source_sentences_per_question"] = max(
            totals["max_source_sentences_per_question"], atom_count
        )
        totals["max_cap"] = max(totals["max_cap"], row["cap"])
    with torch.inference_mode():
        for index, question_row in enumerate(questions[len(saved_rows):], start=len(saved_rows)):
            checkpoint = PROGRESS / f"case_{index:04d}.json"
            if checkpoint.exists():
                raise FileExistsError(f"Candidate checkpoint already exists: {checkpoint}")
            source = source_by_pid[question_row["pid"]]
            atoms = atom_cache[question_row["pid"]]
            source_total = source_totals[question_row["pid"]]
            cap = max(1, math.floor(source_total * 0.25))
            scores = encode_scores(
                model,
                tokenizer,
                question_row["question"],
                [atom["input_text"] for atom in atoms],
                batch_size,
            )
            task = make_task(
                question_row["qid"],
                question_row["question"],
                atoms,
                scores,
                cap,
                source_total,
            )
            row = {
                "index": index,
                "qid": question_row["qid"],
                "pid": question_row["pid"],
                "question": question_row["question"],
                "source_total_tokens": source_total,
                "cap": cap,
                "scores": scores,
                "gold_unopened": True,
                "source_corpus_sha256": corpus_sha,
                "prereg_sha256": prereg_sha,
                "task": task,
            }
            tmp = checkpoint.with_suffix(".json.partial")
            with tmp.open("x", encoding="utf8") as stream:
                json.dump(row, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(checkpoint)

            totals["source_sentence_pairs_scored"] += len(atoms)
            totals["verifier_candidates"] += len(task["candidates"])
            totals["mean_source_tokens"] += source_total / len(questions)
            totals["mean_cap"] += cap / len(questions)
            totals["max_source_sentences_per_question"] = max(
                totals["max_source_sentences_per_question"], len(atoms)
            )
            totals["max_cap"] = max(totals["max_cap"], cap)
            if (index + 1) % 12 == 0 or index + 1 == len(questions):
                print(
                    "PHASE36_CANDIDATES_PROGRESS",
                    index + 1,
                    len(questions),
                    "sentence_pairs",
                    totals["source_sentence_pairs_scored"],
                    "elapsed_s",
                    round(time.monotonic() - start_time, 1),
                    flush=True,
                )

    scored_rows = []
    task_rows = []
    for index in range(len(questions)):
        row = json.loads((PROGRESS / f"case_{index:04d}.json").read_text(encoding="utf8"))
        scored_rows.append({
            "id": row["qid"],
            "pid": row["pid"],
            "question": row["question"],
            "source_total_tokens": row["source_total_tokens"],
            "cap": row["cap"],
            "scores": row["scores"],
            "gold_unopened": True,
        })
        task_rows.append(row["task"])
    write_jsonl_exclusive(SCORED, scored_rows)
    write_jsonl_exclusive(TASKS, task_rows)
    freeze = {
        "n": len(questions),
        "scored_sha256": sha_file(SCORED),
        "tasks_sha256": sha_file(TASKS),
        "source_corpus_sha256": corpus_sha,
        "question_manifest_sha256": sha_file(QUESTIONS),
        "pre_gold_protocol_sha256": prereg_sha,
        "candidate_code_sha256": candidate_code_sha256,
        "prereg_amendment_sha256": amendment_sha256,
        "reranker_weights_sha256": prereg["model_and_resources"]["local_reranker_weights_sha256"],
        "scorer_batch_size": batch_size,
        "gold_unopened": True,
        "stats": totals,
    }
    with FROZEN.open("x", encoding="utf8") as stream:
        json.dump(freeze, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("PHASE36_CANDIDATES_FROZEN", json.dumps(totals), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extract", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--batch-size", type=int, default=SCORER_BATCH_SIZE)
    args = parser.parse_args()
    if args.extract:
        extract_sources()
    else:
        score_candidates(args.batch_size)


if __name__ == "__main__":
    main()
