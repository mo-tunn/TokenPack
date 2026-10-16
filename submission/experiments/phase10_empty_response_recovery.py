"""One-off preregistered-reader incident correction: two *empty* model completions.
Do not alter response text to 'unknown'. Keep true empty response and both incident logs.
No gold data is read. The original reader script/plan is unmodified.
"""
import hashlib,json,os
from pathlib import Path
from submission.experiments.phase7_modal27b_matched8 import MODEL
from submission.experiments.phase10_musique_pilot import OUT

def go():
    assert not (OUT/"pilot_EMPTY_RESPONSE_AUDIT.json").exists()
    f=OUT/"pilot_remote_answers.jsonl"
    saved=[json.loads(s) for s in f.read_text(encoding="utf8").splitlines() if s.strip()]
    assert len(saved)==36
    tasks=[json.loads(s) for s in (OUT/"pilot_reader_tasks.jsonl").read_text(encoding="utf8").splitlines() if s.strip()]
    task=tasks[36]
    id=task["task_id"]
    assert id=="2hop__289517_139339|crossdoc-bridge"
    assert id not in {x["task_id"] for x in saved}
    prompt_hash=None
    for i in (1,2):
        suffix="retry"+str(i)
        err_path=OUT/("pilot_error_"+suffix+".jsonl")
        inflight_path=OUT/("pilot_inflight_"+suffix+".json")
        if i==2:
            err_path=OUT/"pilot_error.jsonl"
            inflight_path=OUT/"pilot_inflight.json"
        err=[json.loads(s) for s in err_path.read_text(encoding="utf8").splitlines() if s.strip()]
        inflight=json.loads(inflight_path.read_text(encoding="utf8"))
        assert len(err)==1 and err[0]["task_id"]==id
        assert inflight["task_id"]==id
        # In the current reader code, an empty answer is the only way all
        # error fields are null; a transport/HTTP failure populates some fields.
        assert err[0]["error_phase"] is None and err[0]["http_status"] is None and err[0]["exception_type"] is None
        if prompt_hash is None:prompt_hash=inflight["prompt_sha256"]
        assert prompt_hash==inflight["prompt_sha256"]
    second_error=OUT/"pilot_error.jsonl"
    second_inflight=OUT/"pilot_inflight.json"
    second_error.rename(OUT/"pilot_error_retry2.jsonl")
    second_inflight.rename(OUT/"pilot_inflight_retry2.json")
    rec={"task_id":id,"case_id":task["case_id"],"arm":task["arm"],
        "model":MODEL,"ok":True,"answer":"","prompt_sha256":prompt_hash,
        "paid_remote_call":True,"additional_paid_retries":1,
        "empty_response_after_two_identical_requests":True,
        "valid_nonempty_answer":False,
        "recovery":"Score literal empty prediction. No invented answer and no case exclusion."}
    with f.open("a",encoding="utf8") as out:
        out.write(json.dumps(rec,ensure_ascii=False)+"\n");out.flush();os.fsync(out.fileno())
    audit={"task_id":id,"empty_attempts":2,"prompt_sha256":prompt_hash,
           "incident_logs":["pilot_error_retry1.jsonl","pilot_error_retry2.jsonl"],
           "inflight_logs":["pilot_inflight_retry1.json","pilot_inflight_retry2.json"],
           "resolution":"Literal model blank retained as empty prediction; no case removed.",
           "additional_paid_calls_not_in_saved_response_count":1,
           "gold_not_read":True}
    with (OUT/"pilot_EMPTY_RESPONSE_AUDIT.json").open("x",encoding="utf8") as out:
        json.dump(audit,out,indent=2);out.write("\n")
    print("PHASE10_BLANK_SCORED_AS_BLANK",id,"saved_answers",len(saved)+1)
if __name__=="__main__":go()
