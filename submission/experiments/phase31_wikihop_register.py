"""Register an answer-only external WikiHop replication before model calls.

The official QAngaroo WikiHop dev set is a fresh benchmark family for this
project, but its annotations do not identify gold supporting sentences.  This
registration therefore precommits an EM-only external check and explicitly
does not claim Phase28-style support coverage.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / ".test-tmp/qangaroo_v1.1.zip"
OUT = ROOT / "submission/results/phase31_wikihop_em_replication_20261013"
SOURCE_NAME = "qangaroo_v1.1/wikihop/dev.json"
SEED = "TokenPack-phase31-WikiHop-EM-replication-20261013-v1"
PILOT_N = 48
VALIDATION_N = 96


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf8")).hexdigest()


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    OUT.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as z:
        rows = json.loads(z.read(SOURCE_NAME))
    assert len(rows) == 5129
    # Only fields available to the gold-blind selector enter the sample choice.
    eligible = []
    for row in rows:
        assert set(row) == {"id", "query", "answer", "candidates", "supports", "annotations"}
        eligible.append({
            "id": row["id"],
            "query_sha256": digest(row["query"]),
            "supports_sha256": digest(row["supports"]),
            "candidate_n": len(row["candidates"]),
            "support_n": len(row["supports"]),
            "annotation_n": len(row["annotations"]),
        })
    eligible.sort(key=lambda x: digest(SEED + "|" + x["id"]))
    pilot = eligible[:PILOT_N]
    validation = eligible[PILOT_N:PILOT_N + VALIDATION_N]
    assert len({x["id"] for x in pilot} & {x["id"] for x in validation}) == 0
    protocol = {
        "study": "Phase31 fresh WikiHop EM-only external replication",
        "benchmark": "QAngaroo WikiHop v1.1 official dev split",
        "source_archive_sha256": sha_file(ARCHIVE),
        "source_archive_bytes": ARCHIVE.stat().st_size,
        "source_member": SOURCE_NAME,
        "source_rows": len(rows),
        "source_license": "CC BY-SA 3.0",
        "selection_seed": SEED,
        "pilot_n": PILOT_N,
        "validation_n": VALIDATION_N,
        "pilot_gate": {"minimum_net_EM": 3, "token_cap_violations": 0},
        "confirmatory_gate": {"paired_two_sided_exact_mcnemar_alpha": 0.025,
                              "positive_net_EM_required": True,
                              "token_cap_violations": 0},
        "primary_metric": "normalized paired answer Exact Match",
        "support_metric": None,
        "support_metric_limitation": "Official WikiHop dev annotations expose relation patterns, not gold supporting source sentence indices.",
        "control": "atomic sentence/paragraph rank over original WikiHop support passages",
        "treatment": "frozen Phase28 semantic-joint-citations policy adapted to original WikiHop support passages",
        "same_reader_and_token_cap": True,
        "gold_answer_unopened_for_selection": True,
        "gold_sent_remote": False,
        "qasper_official_test_untouched": True,
        "not_a_phase28_support_replication": True,
        "code_sha256": sha_file(Path(__file__)),
    }
    (OUT / "PRE_GOLD_PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    manifest = {"source_archive_sha256": protocol["source_archive_sha256"], "seed": SEED,
                "pilot": pilot, "validation": validation,
                "gold_fields_loaded": False,
                "selection_fields": ["id", "query", "supports", "candidates", "annotations"],
                "excluded_from_selection": ["answer"]}
    (OUT / "ID_MANIFEST_PRE_GOLD.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    audit = {"status": "REGISTERED_PRE_GOLD", "archive_hash_valid": True, "rows": len(rows),
             "pilot_n": len(pilot), "validation_n": len(validation), "pilot_validation_disjoint": True,
             "answer_loaded_for_selection": False, "support_sentence_gold_available": False,
             "decision": "EM_ONLY_EXTERNAL_REPLICATION_CANDIDATE"}
    (OUT / "DATASET_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PHASE31_REGISTERED", "out": str(OUT), "archive_sha256": protocol["source_archive_sha256"],
                      "pilot_n": len(pilot), "validation_n": len(validation), "support_metric": None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
