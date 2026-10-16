"""Audit first Phase24 Modal health TimeoutError and allow exactly one
explicitly documented retry, preserving original bytes. No gold reading.
"""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase24_answer_span_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
def main():
    err=OUT/"planner_error.jsonl";fly=OUT/"planner_inflight.json"
    assert err.exists() and fly.exists()
    rows=read_rows(err)
    x=json.loads(fly.read_text(encoding="utf8"))
    assert len(rows)==1 and rows[0]["id"]==x["id"]
    assert rows[0]["exception"]=="TimeoutError" and rows[0]["phase"]=="health"
    assert not (OUT/"planner_responses.jsonl").exists()
    dest1=OUT/"planner_error_INCIDENT1.jsonl"
    dest2=OUT/"planner_inflight_INCIDENT1.json"
    assert not dest1.exists() and not dest2.exists()
    original={dest1.name:sha_file(err),dest2.name:sha_file(fly)}
    os.replace(err,dest1);os.replace(fly,dest2)
    assert sha_file(dest1)==original[dest1.name]
    assert sha_file(dest2)==original[dest2.name]
    record={"first_planner_task_id":x["id"],
      "failed_phase":"remote Modal health endpoint",
      "exception":"TimeoutError",
      "before_model_inference_success":True,
      "successful_graphs":0,
      "could_have_billed_failure":True,
      "retry_rule":"ONE explicit retry only; on repeat error STOP without resampling",
      "evidence_sha256":original,
      "never_read_gold":True,"do_not_change_preregistered_ids":True}
    new_json(OUT/"PLANNER_HEALTH_INCIDENT1_AUDIT.json",record)
    print("PHASE24_PLANNER_INCIDENT_PRESERVED",record["exception"],flush=True)
if __name__=="__main__":main()
