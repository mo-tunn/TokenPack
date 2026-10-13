"""One explicit controlled retry of initial Qwen Modal health TimeoutError.
Previous failed attempt preserved verbatim; no sample edits or gold access.
"""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase26_hotpot_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
def main():
    err=OUT/"hotpot_reader_error.jsonl"
    inflight=OUT/"hotpot_reader_inflight.json"
    assert err.exists() and inflight.exists()
    rows=read_rows(err);x=json.loads(inflight.read_text(encoding="utf8"))
    assert len(rows)==1 and rows[0]["id"]==x["id"]
    assert rows[0]["exception"]=="TimeoutError" and rows[0]["phase"]=="health"
    assert not (OUT/"hotpot_reader_responses.jsonl").exists()
    saved1=OUT/"hotpot_reader_error_INCIDENT1.jsonl"
    saved2=OUT/"hotpot_reader_inflight_INCIDENT1.json"
    assert not saved1.exists() and not saved2.exists()
    hashes={"error_before":sha_file(err),"inflight_before":sha_file(inflight)}
    os.replace(err,saved1);os.replace(inflight,saved2)
    assert hashes["error_before"]==sha_file(saved1)
    assert hashes["inflight_before"]==sha_file(saved2)
    new_json(OUT/"HOTPOT_READER_HEALTH_INCIDENT1_AUDIT.json",{
         "failed_id":x["id"],"failure":"TimeoutError",
         "endpoint_stage":"health","successes_before_retry":0,
         "original_sha256":hashes,
         "one_explicit_retry_allowed":True,
         "if_health_fails_again":"STOP without resampling or gold access",
         "gold_not_accessed":True})
    print("PHASE26_INCIDENT1_PRESERVED",x["id"],flush=True)
if __name__=="__main__":main()
