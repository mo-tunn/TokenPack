"""Pre-register the full public IIRC dev-set comparison before local/model inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from submission.experiments.phase12_register import sha_file
from submission.experiments.phase5b_pretrained_rerank_eval import (
    MODEL_ID as RERANKER,
    REVISION as RERANKER_REVISION,
    WEIGHTS_SHA256,
    verify_local_model,
)
from submission.experiments.phase36_iirc_core import (
    ARMS,
    MODEL,
    READER_MAX_OUTPUT_TOKENS,
    READER_SEED,
    READER_SYSTEM,
    VERIFIER_MAX_OUTPUT_TOKENS,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
)
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from submission.experiments.phase28_semantic_core import MAX_CANDIDATES, MAX_TOP

SOURCE = ROOT / ".tokenpack/phase36_iirc_20261013"
IIRC_ARCHIVE = SOURCE / "iirc.tar.gz"
ARTICLES_ARCHIVE = SOURCE / "context_articles.tar.gz"
SCRATCH = SOURCE / "working"
OUT = ROOT / "submission/results/phase36_iirc_semantic_citations_20261014"
EXPECTED_IIRC_SHA256 = "efab875854691530a3bd55809244b0656dff1e15e98e4108bae5405c63e31a9b"
EXPECTED_ARTICLES_SHA256 = "d239390a27ed3f4aa868afe126187e6d108990a7bcc2a3efb6cae5de917264c7"
EXPECTED_PASSAGES = 430
EXPECTED_QUESTIONS = 1301


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def gold_blind_projection() -> list[dict]:
    """Read only the benchmark fields permitted for IIRC model inputs."""
    with tarfile.open(IIRC_ARCHIVE, "r:gz") as archive:
        source = json.load(archive.extractfile("iirc/dev.json"))
    result = []
    for passage in source:
        result.append({
            "pid": str(passage["pid"]),
            "title": str(passage["title"]),
            "text": str(passage["text"]),
            "links": [str(item["target"]) for item in passage["links"]],
            "questions": [
                {"qid": str(question["qid"]), "question": str(question["question"])}
                for question in passage["questions"]
            ],
        })
    del source
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Phase36 preregistration directory already exists.")
    if not IIRC_ARCHIVE.is_file() or not ARTICLES_ARCHIVE.is_file():
        raise FileNotFoundError("Expected IIRC archives are missing from .tokenpack.")
    iirc_sha = sha_file(IIRC_ARCHIVE)
    articles_sha = sha_file(ARTICLES_ARCHIVE)
    assert iirc_sha == EXPECTED_IIRC_SHA256
    assert articles_sha == EXPECTED_ARTICLES_SHA256

    rows = gold_blind_projection()
    question_rows = [q for passage in rows for q in passage["questions"]]
    ids = [q["qid"] for q in question_rows]
    assert len(rows) == EXPECTED_PASSAGES
    assert len(ids) == EXPECTED_QUESTIONS == len(set(ids))
    unique_link_titles = {title.casefold() for passage in rows for title in passage["links"]}

    local_model_size = verify_local_model()
    exp = ROOT / "submission/experiments"
    code_files = [
        "phase36_iirc_core.py",
        "phase36_iirc_register.py",
        "phase36_iirc_candidates.py",
        "phase36_iirc_verifier.py",
        "phase36_iirc_contexts.py",
        "phase36_iirc_reader.py",
        "phase36_iirc_score.py",
        "phase36_iirc_relay_source.py",
        "phase28_semantic_core.py",
        "phase27_joint_chain_core.py",
        "phase8_atomic_contexts.py",
        "phase8_atomic_protocol.py",
        "phase5b_pretrained_rerank_eval.py",
    ]
    code_hashes = {name: sha_file(exp / name) for name in code_files}

    manifest = {
        "phase": 36,
        "study": "Frozen Phase28 semantic-joint-citations comparison on all IIRC public dev questions",
        "benchmark": "IIRC official public development split; 430 source passages and 1,301 questions",
        "benchmark_status": "public labels; this is not an independent hidden test or leaderboard submission",
        "source": {
            "dataset_url": "https://allenai.org/data/iirc",
            "paper_url": "https://aclanthology.org/2020.emnlp-main.86/",
            "baseline_code_url": "https://github.com/jferguson144/IIRC-baseline",
            "iirc_archive_sha256": iirc_sha,
            "iirc_archive_bytes": IIRC_ARCHIVE.stat().st_size,
            "context_articles_archive_sha256": articles_sha,
            "context_articles_archive_bytes": ARTICLES_ARCHIVE.stat().st_size,
            "license": "CC BY 4.0",
        },
        "sample": {
            "rule": "Use every question in the official dev split in its published passage/question order; no question or answer-type filtering.",
            "passages": len(rows),
            "questions": len(ids),
            "question_ids_sha256": digest("\n".join(ids)),
            "unique_main_passage_link_titles": len(unique_link_titles),
        },
        "gold_blind_input_projection": [
            "passage.pid", "passage.title", "passage.text", "passage.links.target",
            "question.qid", "question.question",
        ],
        "gold_fields_excluded_until_all_reader_responses_freeze": [
            "question.answer", "question.context", "question.question_links",
        ],
        "source_pool": {
            "per_question": "The complete main passage plus every available context_articles document linked from passage.links; no question-specific gold link filtering, external retrieval, or label-based truncation.",
            "main_passage_included": True,
            "missing_link_documents": "Unavailable link targets are omitted and counted in the gold-blind corpus freeze; no gold field is consulted to resolve them.",
            "article_html": "Render visible HTML text while preserving anchor text and block boundaries.",
            "sentence_segmentation": "tokenpack.chunking.SENTENCE_RE (?<=[.!?])\\s+; sentence text is passed unchanged after HTML-to-visible-text conversion.",
        },
        "comparison": {
            "arms": list(ARMS),
            "baseline": "Unchanged Phase8/27 atomic-sentence-rank: pinned zero-shot cross-encoder/ms-marco-MiniLM-L6-v2 relevance scores, greedy whole-original-sentence selection.",
            "treatment": "Frozen Phase28 semantic-joint-citations: verifier sees the same original sentence pool's top-24 MiniLM sentences plus up to 12 additional best-ranked source sentences (36 maximum); it selects two or three original sentence IDs from at least two source documents; invalid, abstaining, or over-cap output falls back exactly to the baseline; valid citations seed selection and the same relevance rank fills the remaining budget.",
            "reader": "Same Qwen 27B model, reader system/user prompt, temperature 0, decoding options, seed, and max output for both arms. Duplicate identical prompts are answered once and reused verbatim.",
            "reader_system": READER_SYSTEM,
            "reader_seed": READER_SEED,
            "reader_max_output_tokens": READER_MAX_OUTPUT_TOKENS,
            "verifier_system": VERIFIER_SYSTEM,
            "verifier_user_template": VERIFIER_USER,
            "verifier_max_output_tokens": VERIFIER_MAX_OUTPUT_TOKENS,
            "verifier_candidate_top_n": MAX_TOP,
            "verifier_candidate_max_n": MAX_CANDIDATES,
            "budget_ratio": 0.25,
            "budget_definition": "floor(0.25 * sum over all available source documents of cl100k_base(title + newline + complete visible source text)); both final rendered arms must remain at or below the same per-question cap.",
            "scorer_max_input_tokens": SCORER_MAX_INPUT_TOKENS,
            "gold_or_annotation_sent_to_any_model": False,
            "training_or_method_tuning": False,
        },
        "model_and_resources": {
            "remote_model": MODEL,
            "remote_endpoint": "https://metekizilcik--tokenpack-qwen27b-longcontext-inference-server.modal.run",
            "remote_calls_max_verifier": len(ids),
            "remote_calls_max_readers": 2 * len(ids),
            "remote_calls_max_total": 3 * len(ids),
            "reuse_exact_duplicate_reader_payloads": True,
            "local_reranker": RERANKER,
            "local_reranker_revision": RERANKER_REVISION,
            "local_reranker_weights_sha256": WEIGHTS_SHA256,
            "local_reranker_bytes": local_model_size,
            "local_device_expected": "RTX 3050 Ti 4GB",
        },
        "primary_metrics": [
            "Official IIRC/DROP normalized Exact Match, macro-averaged across all 1,301 questions.",
            "Official IIRC/DROP numeracy-focused F1, macro-averaged across all 1,301 questions.",
        ],
        "metric_definition": {
            "source": "IIRC EMNLP 2020 paper and AllenNLP DROP evaluator",
            "drop_evaluator_url": "https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/tools/drop.py",
            "numeric_handling": "Official DROP number-aware bag-of-words F1 and one-to-one answer-span alignment.",
            "prediction_delimiter": "A model output containing semicolons is deterministically parsed as multiple answer spans; otherwise it is one span.",
            "unanswerable_target": "NONE, matching the IIRC paper's NULL/NONE output class.",
        },
        "inference_and_uncertainty": {
            "paired_by_question": True,
            "cluster_unit": "IIRC source passage pid (questions sharing a passage are resampled together).",
            "cluster_bootstrap_replicates": 10000,
            "cluster_bootstrap_seed": 360014,
            "interval": "two-sided percentile 95% paired cluster-bootstrap CI for treatment minus baseline",
            "em_test": "two-sided exact McNemar/binomial test on discordant questions, alpha 0.05; exploratory alongside cluster-bootstrap CI",
            "positive_improvement_claim_gate": "Both official EM and DROP F1 deltas must be positive and both paired cluster-bootstrap 95% CIs must exclude zero in the positive direction; otherwise report descriptive results without a general improvement claim.",
        },
        "gold_blind_requirement": "All verifier and reader outputs for all 1,301 questions must be persisted and SHA256-frozen before reading answer, context, or question_links.",
        "nonclaims": [
            "IIRC dev is public and is not an external hidden test.",
            "A successful IIRC dev result does not establish performance on all benchmark families or hidden test data.",
            "Qwen citation judgments are model predictions, not entailment proofs.",
        ],
        "qasper_official_sealed_test_untouched": True,
        "production_code_unchanged": True,
        "code_sha256": code_hashes,
    }

    OUT.mkdir(parents=True, exist_ok=False)
    SCRATCH.mkdir(parents=True, exist_ok=False)
    new_json(OUT / "PRE_GOLD_PROTOCOL.json", manifest)
    new_json(OUT / "ID_MANIFEST_PRE_GOLD.json", {
        "split": "public_dev",
        "ids_in_frozen_dataset_order": ids,
        "ids_sha256": manifest["sample"]["question_ids_sha256"],
        "gold_fields_read": False,
    })
    new_json(OUT / "DATASET_AUDIT_PRE_GOLD.json", {
        "status": "REGISTERED_PRE_MODEL",
        "iirc_archive_hash_valid": True,
        "context_archive_hash_valid": True,
        "passages": len(rows),
        "questions": len(ids),
        "unique_question_ids": len(ids),
        "annotation_projection_excludes_answer_context_question_links": True,
        "model_calls": 0,
        "decision": "RUN_FULL_GOLD_BLIND_PAIRED_IIRC_DEV_COMPARISON",
    })
    print(json.dumps({
        "status": "PHASE36_PREREGISTERED_BEFORE_MODEL_CALLS",
        "questions": len(ids),
        "passages": len(rows),
        "unique_link_titles": len(unique_link_titles),
        "max_remote_calls": 3 * len(ids),
        "out": str(OUT),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
