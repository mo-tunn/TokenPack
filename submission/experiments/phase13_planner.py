"""Phase13 question-only structured relation graph planning, bounded Modal Qwen27B.

Only ID and question read via Parquet column projection. Relay staged to tiny
temporary folder so no local benchmark files can be uploaded to Modal.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,shutil,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase13_register import (
    OUT, GRAPH_SYSTEM, MAX_REMOTE_PLANNER
)
from submission.experiments.phase13_graph_core import normalized_edges
from submission.experiments.phase12_register import DATA,verify_data,sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,append_fsync,atomic_write,
)
from submission.experiments.phase9_musique_protocol import DATA_SHA256
import pyarrow.parquet as pq

RELAY_SOURCE=ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"
STAGE=Path(tempfile.gettempdir())/"tokenpack_phase13_graph_relay_isolated"
STAGE.mkdir(parents=True,exist_ok=True)
RELAY_LOCAL=STAGE/"phase13_graph_qwen_relay.py"
if not RELAY_LOCAL.exists() or sha_file(RELAY_LOCAL)!=sha_file(RELAY_SOURCE):
    shutil.copyfile(RELAY_SOURCE,RELAY_LOCAL)
sys.path.insert(0,str(STAGE))
from phase13_graph_qwen_relay import APP,infer_once,MODEL,BASE

TASKS=OUT/"planner_questions.jsonl"
PLAN=OUT/"PLANNER_PRE_REMOTE_FREEZE.json"
RESP=OUT/"planner_remote.jsonl"
INFLIGHT=OUT/"planner_inflight.json"
ERROR=OUT/"planner_error.jsonl"
FROZEN=OUT/"PLANNER_RESPONSES_FROZEN.json"

def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def payload(question):
    return {"model":MODEL,"messages":[
        {"role":"system","content":GRAPH_SYSTEM},
        {"role":"user","content":"Question:\n"+question+"\n\nOutput structured evidence graph JSON:"}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,
        "seed":20261012,"max_tokens":520,
        "reasoning_effort":"none","stream":False}

def prepare():
    m=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert m["planner_system"]==GRAPH_SYSTEM
    assert m["planner_max_new_tokens"]==520
    assert m["data_sha256"]==sha_file(DATA)
    # Pinned benchmark file integrity and row count are enforced separately.
    ids=m["pilot_ids"]
    rows=verify_data().read(columns=["id","question"]).to_pylist()
    chosen={x["id"]:x["question"] for x in rows if x["id"] in set(ids)}
    assert set(chosen)==set(ids)
    tasks=[{"id":i,"question":chosen[i]} for i in ids]
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as f:
            for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+"\n")
    spec={"question_tasks_sha256":sha_file(TASKS),
        "graph_planner_code_sha256":sha_file(Path(__file__)),
        "relay_sha256":sha_file(RELAY_SOURCE),
        "source_protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "endpoint":BASE,"model":MODEL,"n":len(tasks),
        "system_sha256":sha(GRAPH_SYSTEM),
        "only_question_uploaded":True,"gold_uploaded":False,
        "max_paid_calls":32,"session_estimated_usd_guard":1.35,
        "actual_invoice_unknown":True,
        "invalid_json":"fallback original question, preserve ID"}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks

def execute(max_new=MAX_REMOTE_PLANNER):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Prior error/inflight must be audited")
    rows=read_rows(RESP);saved={}
    allowed={t["id"] for t in tasks}
    for r in rows:
        assert r["id"] in allowed and r["id"] not in saved and r["ok"]
        saved[r["id"]]=r
    print("PHASE13_GRAPH_PLANNER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    paid=0;started=time.monotonic()
    with APP.run():
        for t in tasks:
            if t["id"] in saved:continue
            elapsed=time.monotonic()-started
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE13_PLANNER_SESSION_GUARD",len(saved),"/",len(tasks),flush=True)
                break
            req=payload(t["question"])
            hashkey=sha(json.dumps(req,sort_keys=True,ensure_ascii=False))
            atomic_write(INFLIGHT,{"id":t["id"],"prompt_sha256":hashkey,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:result=infer_once.remote(req)
            except BaseException as e:
                append_fsync(ERROR,{"id":t["id"],"exception_type":type(e).__name__})
                print("PHASE13_PLANNER_REMOTE_EXCEPTION",type(e).__name__,flush=True)
                break
            paid+=1
            if not result.get("ok"):
                append_fsync(ERROR,{"id":t["id"],"phase":result.get("phase"),
                    "http_status":result.get("http_status"),
                    "exception_type":result.get("exception")})
                print("PHASE13_PLANNER_REMOTE_FAILURE",t["id"],flush=True)
                break
            raw=str(result.get("answer") or "")
            branches,status=normalized_edges(t["question"],raw)
            entry={"id":t["id"],"ok":True,"branches":branches,
                   "raw_output":raw,"parse_status":status,
                   "prompt_sha256":hashkey,"paid_remote_call":True,
                   "model":MODEL,"input_tokens":result.get("input_tokens"),
                   "output_tokens":result.get("output_tokens"),
                   "finish_reason":result.get("finish_reason")}
            append_fsync(RESP,entry);INFLIGHT.unlink()
            saved[t["id"]]=entry
            if len(saved)%8==0:
                print("PHASE13_GRAPH_PLANNER_PROGRESS",len(saved),"/",len(tasks),
                      "real_calls",paid,flush=True)
    print("PHASE13_GRAPH_PLANNER_SESSION_END",len(saved),"/",len(tasks),
          "error",ERROR.exists(),flush=True)

def freeze():
    tasks=prepare()
    responses=read_rows(RESP)
    assert len(responses)==len(tasks)
    assert [r["id"] for r in responses]==[t["id"] for t in tasks]
    assert all(r.get("ok") and r["branches"] for r in responses)
    if FROZEN.exists():raise FileExistsError(FROZEN)
    statuses={}
    for r in responses:
        statuses[r["parse_status"]]=statuses.get(r["parse_status"],0)+1
    new_json(FROZEN,{"n":len(responses),"response_sha256":sha_file(RESP),
       "pre_remote_sha256":sha_file(PLAN),
       "parse_status_counts":statuses,
       "mean_edge_count":sum(sum(len(b["edges"]) for b in r["branches"])
                        for r in responses)/len(responses),
       "cloud_received_question_only":True,"gold_opened":False})
    print("PHASE13_GRAPH_PLANNER_FROZEN",json.dumps(statuses),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--execute",action="store_true")
    g.add_argument("--freeze",action="store_true")
    p.add_argument("--max-new",type=int,default=32)
    args=p.parse_args()
    if args.prepare:print("PHASE13_PLANNER_PREPARED",len(prepare()),flush=True)
    elif args.freeze:freeze()
    else:execute(max(1,min(args.max_new,32)))
