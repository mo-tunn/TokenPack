"""Faz14 question-only relation plan. Reuses EXACT Phase13 graph prompt & Qwen relay.
No source document or gold fields are sent during this stage.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase18_register import OUT
from submission.experiments.phase15_protocol import MAX_PLANNER as MAX_PLANNER_CALLS
from submission.experiments.phase13_planner import (
    payload, APP, infer_once, MODEL, BASE
)
from submission.experiments.phase13_graph_core import normalized_edges
from submission.experiments.phase12_register import (
    DATA,verify_data,sha_file,new_json
)
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,append_fsync,atomic_write
)
TASKS=OUT/"planner_questions.jsonl"
PLAN=OUT/"PLANNER_PRE_REMOTE_FREEZE.json"
RESP=OUT/"planner_responses.jsonl"
ERROR=OUT/"planner_error.jsonl"
INFLIGHT=OUT/"planner_inflight.json"
FROZEN=OUT/"PLANNER_RESPONSES_FROZEN.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def prepare():
    record=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert record["parquet_sha256"]==sha_file(DATA)
    ids=record["pilot_ids"]
    rows=verify_data().read(columns=["id","question"]).to_pylist()
    mapping={x["id"]:x["question"] for x in rows if x["id"] in set(ids)}
    assert set(mapping)==set(ids)
    tasks=[{"id":i,"question":mapping[i]} for i in ids]
    if TASKS.exists():assert tasks==read_rows(TASKS)
    else:
        with TASKS.open("x",encoding="utf8") as f:
            for r in tasks:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    spec={"n":len(tasks),"task_file_sha256":sha_file(TASKS),
          "code_sha256":sha_file(Path(__file__)),
          "phase13_planner_code_sha256":sha_file(
              ROOT/"submission/experiments/phase13_planner.py"),
          "graph_core_sha256":sha_file(
              ROOT/"submission/experiments/phase13_graph_core.py"),
          "model":MODEL,"endpoint":BASE,"same_phase13_payload":True,
          "remote_only_receives_question":True,
          "source_gold_not_sent":True,"maximum_paid_calls":32}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks

def execute(max_new=MAX_PLANNER_CALLS):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():raise RuntimeError("Unresolved remote incident")
    saved={}
    for x in read_rows(RESP):
        assert x["id"] not in saved and x.get("ok")
        saved[x["id"]]=x
    print("PHASE18_PLAN_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    paid=0;started=time.monotonic()
    with APP.run():
        for task in tasks:
            if task["id"] in saved:continue
            elapsed=time.monotonic()-started
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE18_PLAN_SESSION_GUARD",len(saved),flush=True);break
            req=payload(task["question"])
            digest=sha(json.dumps(req,sort_keys=True,ensure_ascii=False))
            atomic_write(INFLIGHT,{"id":task["id"],
                "prompt_sha256":digest,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:res=infer_once.remote(req)
            except BaseException as ex:
                append_fsync(ERROR,{"id":task["id"],"exception_type":type(ex).__name__})
                print("PHASE18_PLAN_REMOTE_EXCEPTION",type(ex).__name__,flush=True);break
            paid+=1
            if not res.get("ok"):
                append_fsync(ERROR,{"id":task["id"],"phase":res.get("phase"),
                    "http_status":res.get("http_status"),
                    "exception":res.get("exception")})
                print("PHASE18_PLAN_REMOTE_FAILURE",task["id"],flush=True);break
            raw=str(res.get("answer") or "")
            branches,status=normalized_edges(task["question"],raw)
            rec={"id":task["id"],"ok":True,"branches":branches,
                 "parse_status":status,"raw_output":raw,
                 "prompt_sha256":digest,"model":MODEL,
                 "paid_remote_call":True,"input_tokens":res.get("input_tokens"),
                 "output_tokens":res.get("output_tokens")}
            append_fsync(RESP,rec);INFLIGHT.unlink()
            saved[task["id"]]=rec
            if len(saved)%8==0:
                print("PHASE18_PLAN_PROGRESS",len(saved),"/",len(tasks),
                      "paid",paid,flush=True)
    print("PHASE18_PLAN_SESSION_END",len(saved),"/",len(tasks),flush=True)

def freeze():
    rows=read_rows(RESP);tasks=prepare()
    assert len(rows)==len(tasks) and [x["id"] for x in rows]==[x["id"] for x in tasks]
    assert all(x["ok"] and x["branches"] for x in rows)
    if FROZEN.exists():raise FileExistsError(FROZEN)
    valid=sum(x["parse_status"]=="valid_structured_graph" for x in rows)
    new_json(FROZEN,{"responses_sha256":sha_file(RESP),"count":len(rows),
            "plan_sha256":sha_file(PLAN),"valid_graphs":valid,
            "fallbacks":len(rows)-valid,
            "gold_opened":False,"remote_received_only_question":True})
    print("PHASE18_PLAN_FROZEN",valid,len(rows),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    args=ap.parse_args()
    if args.prepare:print("PHASE18_PLAN_PREPARED",len(prepare()),flush=True)
    elif args.freeze:freeze()
    else:execute()
