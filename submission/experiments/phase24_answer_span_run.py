"""Faz24 fixed extra-model answer-span contraction, gold BLIND.

64 frozen Phase14 direct answers, optional independent Qwen span call only
on eligible answers. Validates exact original substring & word boundaries.
All final predictions freeze before any scoring.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase24_answer_span_register import OUT
from submission.experiments.phase24_answer_span_core import (
    SYSTEM,USER,needs_review,validated_substring,make_contract_payload)
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,append_fsync,atomic_write)
BASELINE=OUT/"direct_reader_responses.jsonl"
CONTEXTS=OUT/"gold_blind_contexts.jsonl"
INPUTS=OUT/"span_contraction_tasks.jsonl"
PRE=OUT/"SPAN_CONTRACTION_PRE_GOLD_FREEZE.json"
RESP=OUT/"span_contraction_responses.jsonl"
ERROR=OUT/"span_contraction_error.jsonl"
INFLIGHT=OUT/"span_contraction_inflight.json"
FINAL=OUT/"FROZEN_GOLD_BLIND_FINAL_PREDICTIONS.jsonl"
FREEZE=OUT/"SPAN_FINAL_PREDICTIONS_FROZEN.json"
def digest(s):return hashlib.sha256(json.dumps(
    s,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def prepare():
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    df=json.loads((OUT/"DIRECT_READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    cf=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(BASELINE)==df["responses_sha256"]
    assert sha_file(CONTEXTS)==cf["sha256"]
    base=read_rows(BASELINE);contexts=read_rows(CONTEXTS)
    assert [x["id"] for x in base]==[x["id"] for x in contexts]==p["pilot_ids"]
    tasks=[{"id":r["id"],"question":ctx["question"],
        "context":ctx["contexts"]["verified-direct"]["text"],
        "original_answer":r["answer"],"eligible":needs_review(r["answer"])}
        for r,ctx in zip(base,contexts)]
    assert len(tasks)==64
    if INPUTS.exists():assert read_rows(INPUTS)==tasks
    else:
        with INPUTS.open("x",encoding="utf8") as f:
            for task in tasks:f.write(json.dumps(task,ensure_ascii=False)+"\n")
    expected_pilot_code=p["preemptive_model_code_sha256"]
    assert sha_file(ROOT/"submission/experiments/phase24_answer_span_core.py")==expected_pilot_code
    spec={"n":64,"eligible_n":sum(t["eligible"] for t in tasks),
          "source_baseline_sha256":df["responses_sha256"],
          "source_context_sha256":cf["sha256"],
          "tasks_sha256":sha_file(INPUTS),
          "core_pregistered_sha256":expected_pilot_code,
          "code_sha256":sha_file(Path(__file__)),
          "system_prompt_sha256":hashlib.sha256(SYSTEM.encode("utf8")).hexdigest(),
          "user_prompt_sha256":hashlib.sha256(USER.encode("utf8")).hexdigest(),
          "model":MODEL,"endpoint":BASE,
          "gold_columns_loaded":False,
          "all_eligible_answers_must_be_processed":True,
          "max_real_calls":sum(t["eligible"] for t in tasks)}
    if PRE.exists():assert json.loads(PRE.read_text(encoding="utf8"))==spec
    else:new_json(PRE,spec)
    return tasks
def execute(max_new=64):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():raise RuntimeError("Unresolved event, preserve")
    eligible=[x for x in tasks if x["eligible"]]
    saved={}
    for r in read_rows(RESP):
        assert r["id"] not in saved and r.get("ok")
        saved[r["id"]]=r
    print("PHASE24_CONTRACTION_START",len(saved),"/",len(eligible),flush=True)
    if len(saved)==len(eligible):return
    paid=0;start=time.monotonic()
    with APP.run():
        for task in eligible:
            if task["id"] in saved:continue
            elapsed=time.monotonic()-start
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE24_CONTRACTION_GUARD",len(saved),"/",len(eligible),flush=True);break
            request=make_contract_payload(
                task["question"],task["context"],task["original_answer"],MODEL)
            h=digest(request)
            atomic_write(INFLIGHT,{"id":task["id"],"prompt_sha256":h,
                 "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:response=infer_once.remote(request)
            except BaseException as exc:
                append_fsync(ERROR,{"id":task["id"],"error_type":type(exc).__name__})
                print("PHASE24_CONTRACTION_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not response.get("ok"):
                append_fsync(ERROR,{"id":task["id"],
                    "phase":response.get("phase"),"http_status":response.get("http_status"),
                    "exception":response.get("exception")})
                print("PHASE24_CONTRACTION_ENDPOINT_ERROR",task["id"],flush=True);break
            raw=str(response.get("answer") or "").strip()
            safe,status=validated_substring(task["original_answer"],raw)
            rec={"id":task["id"],"ok":True,"original_answer":task["original_answer"],
                "raw_candidate":raw,"safe_prediction":safe,"validation_status":status,
                "prompt_sha256":h,"output_tokens":response.get("output_tokens"),
                "input_tokens":response.get("input_tokens"),"paid":True}
            append_fsync(RESP,rec)
            INFLIGHT.unlink()
            saved[task["id"]]=rec
            if len(saved)%8==0:
                print("PHASE24_CONTRACTION_PROGRESS",len(saved),"/",len(eligible),flush=True)
    print("PHASE24_CONTRACTION_SESSION_END",len(saved),"/",len(eligible),flush=True)
def freeze():
    assert not FREEZE.exists() and not FINAL.exists()
    assert not ERROR.exists() and not INFLIGHT.exists()
    tasks=prepare()
    answers=read_rows(RESP)
    keys=[x["id"] for x in tasks if x["eligible"]]
    assert [r["id"] for r in answers]==keys
    assert all(r["ok"] for r in answers)
    indexed={r["id"]:r for r in answers}
    results=[];changed=[]
    for task in tasks:
        if task["eligible"]:
            rec=indexed[task["id"]]
            assert rec["original_answer"]==task["original_answer"]
            safe,status=validated_substring(task["original_answer"],rec["raw_candidate"])
            assert safe==rec["safe_prediction"] and status==rec["validation_status"]
        else:
            safe=task["original_answer"];status="not_eligible"
        row={"id":task["id"],"baseline_prediction":task["original_answer"],
          "contracted_prediction":safe,"contractor_status":status,
          "contractor_called":task["eligible"],"changed":safe!=task["original_answer"],
          "no_gold_in_decision":True}
        results.append(row)
        if row["changed"]:changed.append(task["id"])
    with FINAL.open("x",encoding="utf8") as f:
        for r in results:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    meta={"n":len(results),"eligible_n":len(keys),"real_model_calls":len(answers),
        "changed_predictions_n":len(changed),"changed_ids":changed,
        "positive_contracts_n":sum(r["contractor_status"]=="accepted_shorter_original_substring"
                                   for r in results),
        "status_histogram":{key:sum(r["contractor_status"]==key for r in results)
                            for key in sorted({r["contractor_status"] for r in results})},
        "baseline_sha256":sha_file(BASELINE),
        "contraction_model_responses_sha256":sha_file(RESP) if RESP.exists() else None,
        "pilot_final_predictions_sha256":sha_file(FINAL),
        "pre_gold_spec_sha256":sha_file(PRE),
        "no_gold_read_or_uploaded":True,
        "reserved_validation_192_unopened":True}
    new_json(FREEZE,meta)
    print("PHASE24_GOLDBLIND_FINAL_FROZEN",json.dumps({
        "n":meta["n"],"eligible":len(keys),"changed":len(changed),
        "statuses":meta["status_histogram"]}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--execute",action="store_true")
    g.add_argument("--freeze",action="store_true")
    p.add_argument("--max-new",type=int,default=64)
    opts=p.parse_args()
    if opts.prepare:print("PHASE24_CONTRACTION_PREPARED",len(prepare()))
    elif opts.execute:execute(opts.max_new)
    else:freeze()
