"""Record a Phase18 first-reader TimeoutError without pretending a response
was received. Preserve byte-identical incident files and allow exactly ONE
explicit retrial of first task. A repeat failure is not retried again.
"""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase18_register import OUT
from submission.experiments.phase18_reader import (
    prepare,make_payload,ANS,INFLIGHT,ERROR)
from submission.experiments.phase12_register import sha_file,new_json
def main():
    tasks=prepare()
    assert len(tasks)==64
    assert tasks[0]["id"] and tasks[0]["arm"]=="branch-lock"
    assert not ANS.exists(),"Reader data exists; audit must be manual"
    errors=[json.loads(x) for x in ERROR.read_text(encoding="utf8").splitlines() if x.strip()]
    assert len(errors)==1 and errors[0]["task_id"]==tasks[0]["task_id"]
    assert errors[0]["exception"]=="TimeoutError" and errors[0]["phase"]=="inference"
    inflight=json.loads(INFLIGHT.read_text(encoding="utf8"))
    assert inflight["task_id"]==tasks[0]["task_id"]
    files=[(ERROR,OUT/"reader_error_INCIDENT1.jsonl"),
           (INFLIGHT,OUT/"reader_inflight_INCIDENT1.json")]
    assert not any(dest.exists() for _,dest in files)
    assert not (OUT/"READER_TIMEOUT_INCIDENT1_AUDIT.json").exists()
    before={dest.name:sha_file(src) for src,dest in files}
    for src,dest in files:
        os.replace(src,dest)
        assert sha_file(dest)==before[dest.name]
    record={
      "incident":"First remote Qwen reader call returned ok:false, inference TimeoutError, response not saved",
      "task_id":tasks[0]["task_id"],
      "remote_call_may_have_been_billed":True,
      "reader_answers_saved_before_incident":0,
      "policy":"ONE bounded explicit retry of this same task only; if second TimeoutError or any other remote error stop permanently and do not open gold",
      "evidence_sha256":before,
      "all_gold_unread":True,
      "unopened_reserved_validation":96,
      "phase18_original_context_sha256":sha_file(OUT/"gold_blind_contexts.jsonl"),
      "no_response_fabricated":True}
    new_json(OUT/"READER_TIMEOUT_INCIDENT1_AUDIT.json",record)
    print("PHASE18_TIMEOUT_AUDITED",json.dumps(record,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
