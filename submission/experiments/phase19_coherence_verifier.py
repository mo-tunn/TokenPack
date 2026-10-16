"""Phase19 ONE cross-edge source consistency judgment per fixed pilot question.

Receives question, hypothesized edges, max3 original source sentences per edge,
and Phase16 first/rescue verification IDs. NO gold answer/facts/evidences.
One inference call per case; invalid JSON -> all-null, no case removed.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase19_register import OUT
from submission.experiments.phase17_protocol import (
    COHERENCE_SYSTEM,MAX_CONSISTENCY_CALLS,MAX_SECONDS,
    MAX_ESTIMATED_USD,ESTIMATED_HOURLY_USD)
from submission.experiments.phase17_coherence_core import build_coherence_input
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase15_initial_verifier import verify_parse
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,append_fsync,atomic_write)
SCORED=OUT/"scored_candidate_pools.jsonl"
INITIAL=OUT/"verifier_responses.jsonl"
RESCUE=OUT/"rescue_verifier_responses.jsonl"
TASKS=OUT/"coherence_verifier_inputs.jsonl"
PLAN=OUT/"COHERENCE_PRE_REMOTE_FREEZE.json"
ANS=OUT/"coherence_verifier_responses.jsonl"
ERROR=OUT/"coherence_verifier_error.jsonl"
INFLIGHT=OUT/"coherence_verifier_inflight.json"
FROZEN=OUT/"COHERENCE_RESPONSES_FROZEN.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def model_payload(question_graph):
    return {"model":MODEL,"messages":[
        {"role":"system","content":COHERENCE_SYSTEM},
        {"role":"user","content":json.dumps(question_graph,ensure_ascii=False,separators=(",",":"))}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,
        "seed":20261012,"max_tokens":480,
        "reasoning_effort":"none","stream":False}

def prepare():
    f1=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    f2=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    f3=json.loads((OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(SCORED)==f1["candidate_sha256"]
    assert sha_file(INITIAL)==f2["responses_sha256"]
    assert sha_file(RESCUE)==f3["responses_sha256"]
    cases=read_rows(SCORED);v1=read_rows(INITIAL);v2=read_rows(RESCUE)
    m=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert [r["id"] for r in cases]==[r["id"] for r in v1]==[r["id"] for r in v2]==m["pilot_ids"]
    assert len(cases)==32
    tasks=[{"id":case["id"],"verifier_input":build_coherence_input(case,i,r)}
            for case,i,r in zip(cases,v1,v2)]
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as file:
            for x in tasks:file.write(json.dumps(x,ensure_ascii=False)+"\n")
    spec={"cases":len(tasks),
          "coherence_prompt_sha256":sha(COHERENCE_SYSTEM),
          "coherence_code_sha256":sha_file(Path(__file__)),
          "coherence_core_sha256":sha_file(
              ROOT/"submission/experiments/phase17_coherence_core.py"),
          "task_sha256":sha_file(TASKS),
          "candidate_sha256":f1["candidate_sha256"],
          "initial_sha256":f2["responses_sha256"],
          "rescue_sha256":f3["responses_sha256"],
          "modal_model":MODEL,"modal_endpoint":BASE,
          "max_real_calls":MAX_CONSISTENCY_CALLS,
          "model_receives_gold":False,
          "invalid_json_all_null_no_excluded_question":True,
          "decision_not_formal_entailment":True,
          "estimated_usd_per_session_not_invoice":MAX_ESTIMATED_USD}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks

def execute(max_new=MAX_CONSISTENCY_CALLS):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved in-flight/error file; manual audit required")
    saved={}
    for x in read_rows(ANS):
        assert x["id"] not in saved and x["ok"]
        saved[x["id"]]=x
    print("PHASE19_COHERENCE_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    start=time.monotonic();paid=0
    with APP.run():
        for task in tasks:
            if task["id"] in saved:continue
            elapsed=time.monotonic()-start
            if paid>=max_new or elapsed+120>=MAX_SECONDS or (
                elapsed+120)*ESTIMATED_HOURLY_USD/3600>=MAX_ESTIMATED_USD:
                print("PHASE19_COHERENCE_GUARD",len(saved),"/",len(tasks),flush=True);break
            payload=model_payload(task["verifier_input"])
            ph=sha(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            atomic_write(INFLIGHT,{"id":task["id"],"prompt_sha256":ph,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:resp=infer_once.remote(payload)
            except BaseException as exc:
                append_fsync(ERROR,{"id":task["id"],"error_type":type(exc).__name__})
                print("PHASE19_COHERENCE_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not resp.get("ok"):
                append_fsync(ERROR,{"id":task["id"],"phase":resp.get("phase"),
                    "http_status":resp.get("http_status"),
                    "exception_type":resp.get("exception")})
                print("PHASE19_COHERENCE_ENDPOINT_ERROR",task["id"],flush=True);break
            raw=str(resp.get("answer") or "")
            values,status=verify_parse(raw,task["verifier_input"])
            record={"id":task["id"],"ok":True,"decision_ids":values,
                 "parse_status":status,"raw_output":raw,
                 "prompt_sha256":ph,"paid_call":True,
                 "model":MODEL,
                 "input_tokens":resp.get("input_tokens"),
                 "output_tokens":resp.get("output_tokens")}
            append_fsync(ANS,record);INFLIGHT.unlink()
            saved[record["id"]]=record
            if len(saved)%8==0:
                print("PHASE19_COHERENCE_PROGRESS",len(saved),"/",len(tasks),
                      "session_paid",paid,flush=True)
    print("PHASE19_COHERENCE_SESSION_END",len(saved),"/",len(tasks),flush=True)

def freeze():
    tasks=prepare();rows=read_rows(ANS)
    assert len(rows)==len(tasks) and [x["id"] for x in rows]==[t["id"] for t in tasks]
    assert all(r["ok"] for r in rows)
    if FROZEN.exists():raise FileExistsError(FROZEN)
    valid=sum(r["parse_status"]=="valid_model_verification" for r in rows)
    chosen=sum(v is not None for r in rows for v in r["decision_ids"].values())
    total=sum(len(r["decision_ids"]) for r in rows)
    new_json(FROZEN,{"n":len(rows),"responses_sha256":sha_file(ANS),
        "inputs_sha256":sha_file(TASKS),
        "pre_remote_protocol_sha256":sha_file(PLAN),
        "valid_json":valid,"invalid_json":len(rows)-valid,
        "model_positive_edges":chosen,"hypothesized_edges":total,
        "gold_unopened":True,"model_judgment_not_formal_entailment":True})
    print("PHASE19_COHERENCE_FROZEN",valid,len(rows),chosen,total,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    args=p.parse_args()
    if args.prepare:print("PHASE19_COHERENCE_PREPARED",len(prepare()),flush=True)
    elif args.freeze:freeze()
    else:execute()
