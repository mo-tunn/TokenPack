"""Register a fresh WikiHop sentence-atom pilot before any model call.

This phase isolates the structural repair identified in Phase31: split each
WikiHop support passage into sentence atoms while preserving its source article
identity. Gold answers remain excluded from sample selection and preparation.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / ".test-tmp/qangaroo_v1.1.zip"
OUT = ROOT / "submission/results/phase32_wikihop_sentence_atoms_20261013"
SOURCE_NAME = "qangaroo_v1.1/wikihop/dev.json"
SEED = "TokenPack-phase32-WikiHop-sentence-atoms-20261013-v1"
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

    prior_manifest = ROOT / "submission/results/phase31_wikihop_em_replication_20261013/ID_MANIFEST_PRE_GOLD.json"
    prior = json.loads(prior_manifest.read_text(encoding="utf8"))
    prior_ids = {x["id"] for split in ("pilot", "validation") for x in prior[split]}

    # Only fields available to the gold-blind selector enter sample choice.
    eligible = []
    for row in rows:
        assert set(row) == {"id", "query", "answer", "candidates", "supports", "annotations"}
        if row["id"] in prior_ids:
            continue
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
    assert len(pilot) == PILOT_N and len(validation) == VALIDATION_N
    assert not ({x["id"] for x in pilot} & {x["id"] for x in validation})
    assert not ({x["id"] for x in pilot} | {x["id"] for x in validation}) & prior_ids

    protocol = {
        "study": "Phase32 WikiHop sentence-atom structural repair pilot",
        "benchmark": "QAngaroo WikiHop v1.1 official dev split",
        "source_archive_sha256": sha_file(ARCHIVE),
        "source_archive_bytes": ARCHIVE.stat().st_size,
        "source_member": SOURCE_NAME,
        "source_rows": len(rows),
        "source_license": "CC BY-SA 3.0",
        "selection_seed": SEED,
        "excluded_prior_phase31_ids": len(prior_ids),
        "pilot_n": PILOT_N,
        "validation_n": VALIDATION_N,
        "pilot_gate": {"minimum_net_EM": 3, "token_cap_violations": 0},
        "confirmatory_gate": {"paired_two_sided_exact_mcnemar_alpha": 0.025,
                              "positive_net_EM_required": True,
                              "token_cap_violations": 0},
        "primary_metric": "normalized paired answer Exact Match",
        "support_metric": None,
        "support_metric_limitation": "Official WikiHop dev annotations expose relation patterns, not gold supporting source sentence indices.",
        "control": "MiniLM score rank over sentence atoms split from original WikiHop support passages",
        "treatment": "Phase28 semantic-joint-citations over the same sentence atoms and source article identities",
        "atomization": "SENTENCE_RE (?<=[.!?])\\s+, source article index preserved, sent_id assigned within passage",
        "same_reader_and_token_cap": True,
        "gold_answer_unopened_for_selection": True,
        "gold_sent_remote": False,
        "qasper_official_test_untouched": True,
        "phase31_ids_untouched": True,
        "code_sha256": sha_file(Path(__file__)),
    }
    (OUT / "PRE_GOLD_PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    manifest = {"source_archive_sha256": protocol["source_archive_sha256"], "seed": SEED,
                "pilot": pilot, "validation": validation,
                "gold_fields_loaded": False,
                "selection_fields": ["id", "query", "supports", "candidates", "annotations"],
                "excluded_from_selection": ["answer"],
                "excluded_phase31_ids": sorted(prior_ids)}
    (OUT / "ID_MANIFEST_PRE_GOLD.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    audit = {"status": "REGISTERED_PRE_GOLD", "archive_hash_valid": True, "rows": len(rows),
             "eligible_after_phase31_exclusion": len(eligible), "pilot_n": len(pilot), "validation_n": len(validation),
             "pilot_validation_disjoint": True, "phase31_disjoint": True,
             "answer_loaded_for_selection": False, "support_sentence_gold_available": False,
             "decision": "EM_ONLY_STRUCTURAL_REPAIR_CANDIDATE"}
    (OUT / "DATASET_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": "PHASE32_REGISTERED", "out": str(OUT), "archive_sha256": protocol["source_archive_sha256"],
                      "pilot_n": len(pilot), "validation_n": len(validation), "excluded_phase31": len(prior_ids)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
