"""Frozen confirmatory validation on pre-registered 96 Phase28 reserved IDs.
No resampling after successful pilot; same source and model policy.
"""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_semantic_register import OUT as PILOT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase6_hotpot_eval import DATA
OUT=ROOT/"submission/results/phase28_hotpot_semantic_citation_validation_20261013"
def main():
    if OUT.exists():raise FileExistsError("Immutable preregistration exists")
    p=json.loads((PILOT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    result=json.loads((PILOT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    assert result["decision"]=="GO_TO_VALIDATION"
    assert result["paired_EM"]["net"]==7 and result["paired_full_support"]["net"]==13
    assert sha_file(DATA)==p["data_sha256"]
    ids=p["reserved_ids"]
    assert len(ids)==len(set(ids))==96 and not set(ids)&set(p["pilot_ids"])
    core=ROOT/"submission/experiments/phase28_semantic_core.py"
    assert sha_file(core)==p["code_hashes"]["phase28_semantic_core.py"]
    manifest={
        "study":"Pre-reserved 96 HotpotQA distractor PUBLIC validation Phase28 confirmation",
        "n":96,"pilot_ids":ids,
        "data_sha256":p["data_sha256"],
        "sampling":"Only exact Phase28 pilot PRE_GOLD_PROTOCOL.reserved_ids, not new selection",
        "validation_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "original_prereg_SHA256":sha_file(PILOT/"PRE_GOLD_PROTOCOL.json"),
        "positive_pilot_postgold_SHA256":sha_file(PILOT/"READER_POST_GOLD_RESULTS.json"),
        "same_source_core_SHA256":sha_file(core),
        "original_core_shas":p["code_hashes"],
        "benchmark":"HotpotQA PUBLIC validation: disjoint IDs but NOT independently sealed test",
        "method":"Unmodified Phase8 baseline MiniLM sentence selection vs EXACT frozen Phase28 Qwen 2-3 cross-source semantic citation selection, original whole sentences, same cl100k 25% cap",
        "only_goldblind_input_columns":["id","question","context.title","context.sentences"],
        "gold_open_after_ALL_reader_response_freeze":True,
        "max_semantic_verifier_calls":96,"max_reader_calls":192,
        "reuse_only_byte_identical_reader_prompt":True,
        "co_primary":["normalized paired EM delta on 96","full annotated original support sentence coverage delta on 96"],
        "alpha_each_bonferroni":0.025,
        "replication_rule":"Both co-primary endpoints strictly positive with paired two-sided exact McNemar p<=0.025 each, zero render cap violations",
        "secondary":["paired F1","paired sentence recall","model calls","frozen response integrity"],
        "no_gold_fed_to_model":True,
        "no_tuning_after_pilot":True,
        "no_structural_stop_in_reserved_96":True,
        "qasper_sealed_test_untouched":True,"production_unchanged":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",manifest)
    print("PHASE28_VALIDATION_PREREG",json.dumps({
       "n":96,"id_sha256":manifest["validation_sha256"],
       "alpha_each":0.025,"gold_unopened":True}),flush=True)
if __name__=="__main__":main()
