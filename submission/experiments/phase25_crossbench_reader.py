"""Phase24 SAME Phase9 crossdoc-bridge direct Qwen reader, frozen original answer.

Only 64 public-validation pilot questions. All gold stays local and unread
until both baseline responses and eligible span-contractor outputs freeze.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase25_crossbench_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase7_modal27b_matched8 import (
    make_payload,read_rows,append_fsync,atomic_write)
CONTEXTS=OUT/"gold_blind_contexts.jsonl"
TASKS=OUT/"crossbench_reader_tasks.jsonl"
PLAN=OUT/"CROSSBENCH_READER_PRE_GOLD_FREEZE.json"
ANS=OUT/"crossbench_reader_responses.jsonl"
ERROR=OUT/"crossbench_reader_error.jsonl"
INFLIGHT=OUT/"crossbench_reader_inflight.json"
FROZEN=OUT/"CROSSBENCH_READER_RESPONSES_FROZEN.json"
def digest(req):
    return hashlib.sha256(json.dumps(
        req,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def prepare():
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    f=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(CONTEXTS)==f["context_file_sha256"] and f["totals"]["final_cap_violations"]==0
    cases=read_rows(CONTEXTS)
    assert [x["id"] for x in cases]==p["pilot_ids"]
    tasks=[{"task_id":x["id"]+"|crossdoc-bridge","id":x["id"],
       "arm":"crossdoc-bridge","question":x["question"],
       "context":x["final_text"]} for x in cases]
    assert len(tasks)==64
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as stream:
            for row in tasks:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    spec={"n":64,"source_context_sha256":f["context_file_sha256"],
          "tasks_sha256":sha_file(TASKS),
          "this_code_sha256":sha_file(Path(__file__)),
          "payload_code_sha256":sha_file(ROOT/"submission/experiments/phase7_modal27b_matched8.py"),
          "relay_code_sha256":sha_file(ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
          "same_unmodified_Phase7_direct_payload":True,"full_relay_not_300_char":True,
          "gold_sent_to_model":False,"max_calls":64}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks
def execute(max_new=64):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved remote failure: manual audit, no automatic retry")
    saved={}
    for row in read_rows(ANS):
        assert row["id"] not in saved and row.get("ok")
        saved[row["id"]]=row
    print("PHASE25_DIRECT_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    paid=0;start=time.monotonic()
    with APP.run():
        for task in tasks:
            if task["id"] in saved:continue
            elapsed=time.monotonic()-start
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE25_DIRECT_GUARD",len(saved),"/",len(tasks),flush=True);break
            req=make_payload(task)
            ph=digest(req)
            atomic_write(INFLIGHT,{"id":task["id"],"prompt_sha256":ph,
                 "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:response=infer_once.remote(req)
            except BaseException as exc:
                append_fsync(ERROR,{"id":task["id"],
                   "error_type":type(exc).__name__})
                print("PHASE25_DIRECT_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not response.get("ok"):
                append_fsync(ERROR,{"id":task["id"],
                  "phase":response.get("phase"),
                  "http_status":response.get("http_status"),
                  "exception":response.get("exception")})
                print("PHASE25_DIRECT_ENDPOINT_ERROR",task["id"],flush=True);break
            answer=str(response.get("answer") or "").strip()
            rec={"id":task["id"],"ok":True,"answer":answer,
                "prompt_sha256":ph,"paid":True,
                "input_tokens":response.get("input_tokens"),
                "output_tokens":response.get("output_tokens"),
                "finish_reason":response.get("finish_reason")}
            append_fsync(ANS,rec)
            INFLIGHT.unlink()
            saved[task["id"]]=rec
            if len(saved)%8==0:
                print("PHASE25_DIRECT_PROGRESS",len(saved),"/",len(tasks),flush=True)
    print("PHASE25_DIRECT_SESSION_END",len(saved),"/",len(tasks),flush=True)
def freeze():
    tasks=prepare()
    rows=read_rows(ANS)
    assert len(rows)==len(tasks)==64
    assert [r["id"] for r in rows]==[t["id"] for t in tasks]
    assert all(r.get("ok") for r in rows)
    assert not INFLIGHT.exists() and not ERROR.exists()
    if FROZEN.exists():raise FileExistsError(FROZEN)
    new_json(FROZEN,{"n":64,"responses_sha256":sha_file(ANS),
        "task_sha256":sha_file(TASKS),"pre_remote_sha256":sha_file(PLAN),
        "empty_answers":sum(not r["answer"] for r in rows),
        "gold_unopened":True,"single_unmodified_source_context":True})
    print("PHASE25_DIRECT_FROZEN",len(rows),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    parser.add_argument("--max-new",type=int,default=64)
    options=parser.parse_args()
    if options.prepare:print("PHASE25_DIRECT_PREPARED",len(prepare()))
    elif options.execute:execute(options.max_new)
    else:freeze()
