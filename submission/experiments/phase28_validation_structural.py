"""Validation structural diagnostic ONLY: no early stoppage allowed.

On exact pre-reserved 96 IDs, need complete paired reader comparison
regardless of changed context count. Gold still unopened.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_validation_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
DEST=OUT/"PRE_GOLD_STRUCTURAL_DECISION.json"
def main():
    if DEST.exists():raise FileExistsError(DEST)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    m=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    f=OUT/"gold_blind_paired_contexts.jsonl"
    assert sha_file(f)==m["sha256"]
    rows=read_rows(f)
    assert len(rows)==96 and [x["id"] for x in rows]==p["pilot_ids"]
    changed=sum(bool(x["source_sentence_set_changed"]) for x in rows)
    assert changed==m["stats"]["source_sentence_sets_changed"]
    assert m["stats"]["token_cap_violations"]==0
    record={"decision":"VALIDATION_FULL_READER_EVALUATION_REQUIRED",
       "n":96,"differing_sentence_sets":changed,
       "no_structural_futility_abort_in_validation":True,
       "both_co_primary_scored_after_reader_freeze":True,
       "gold_unopened":True,"context_sha256":m["sha256"]}
    new_json(DEST,record)
    print("PHASE28_VALIDATION_STRUCTURAL",json.dumps({
       "n":96,"changed":changed,"reader_required":True}),flush=True)
if __name__=="__main__":main()
