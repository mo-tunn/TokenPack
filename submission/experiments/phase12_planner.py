"""Gold-blind Qwen planning 2Wiki validation: reads only id and question."""
from __future__ import annotations
import argparse,hashlib,json,time,datetime as dt,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import OUT,DATA,verify_data,sha_file,new_json
from submission.experiments.phase11_planner import make_query_payload,planned_queries
from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,BASE,read_rows,append_fsync,atomic_write)
import pyarrow.parquet as pq
TASKS=OUT/"planner_questions.jsonl"
RESP=OUT/"planner_remote.jsonl"
PLAN=OUT/"PLANNER_PRE_REMOTE_FREEZE.json"
INFLIGHT=OUT/"planner_inflight.json"
ERROR=OUT/"planner_error.jsonl"
FROZEN=OUT/"PLANNER_RESPONSES_FROZEN.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def prepare():
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    ids=manifest["ids"]
    pf=verify_data()
    subs={r["id"]:r["question"] for r in pf.read(columns=["id","question"]).to_pylist() if r["id"] in set(ids)}
    assert set(subs)==set(ids)
    rows=[{"id":i,"question":subs[i]} for i in ids]
    if TASKS.exists():assert rows==read_rows(TASKS)
    else:
        with TASKS.open("x",encoding="utf8") as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    spec={"questions_sha256":sha_file(TASKS),
          "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
          "model":MODEL,"prompt_function_sha256":sha_file(ROOT/"submission/experiments/phase11_planner.py"),
          "code_sha256":sha_file(Path(__file__)),
          "no_source_or_gold_sent":True,"n":len(rows),"allow_empty_raw_response":True,
          "fallback":"original question if JSON invalid, no question omitted"}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return rows

def execute(max_new=64):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():raise RuntimeError("Prior error needs manual audit")
    prior=read_rows(RESP);saved={}
    for r in prior:
        assert r["id"] not in saved and r.get("ok")
        saved[r["id"]]=r
    print("PHASE12_PLANNER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    elapsed_start=time.monotonic();paid=0
    with APP.run():
        for t in tasks:
            if t["id"] in saved:continue
            elapsed=time.monotonic()-elapsed_start
            if paid>=max_new or elapsed+120>1800 or (elapsed+120)*2.60/3600>1.35:
                print("PHASE12_PLANNER_GUARD",len(saved),"/",len(tasks),flush=True);break
            payload=make_query_payload(t["question"])
            digest=sha(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            atomic_write(INFLIGHT,{"id":t["id"],"sha":digest,
                "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:resp=infer_once.remote(payload)
            except BaseException as e:
                append_fsync(ERROR,{"id":t["id"],"exception_type":type(e).__name__})
                print("PHASE12_PLANNER_ERROR",type(e).__name__,flush=True);break
            paid+=1
            if not resp.get("ok"):
                append_fsync(ERROR,{"id":t["id"],"phase":resp.get("phase"),
                    "http_status":resp.get("http_status"),
                    "exception":resp.get("exception")})
                print("PHASE12_PLANNER_STOP_REMOTE_FAILURE",t["id"],flush=True);break
            raw=str(resp.get("answer") or "")
            queries,status=planned_queries(raw,t["question"])
            rec={"id":t["id"],"ok":True,"queries":queries,"status":status,
                "raw_model_output":raw,"prompt_sha256":digest,
                "paid_remote_call":True,"model":MODEL}
            append_fsync(RESP,rec);INFLIGHT.unlink();saved[t["id"]]=rec
            if len(saved)%8==0:
                print("PHASE12_PLANNER_PROGRESS",len(saved),"/",len(tasks),
                      "session_calls",paid,flush=True)
    print("PHASE12_PLANNER_SESSION_END",len(saved),"/",len(tasks),flush=True)

def freeze():
    tasks=prepare()
    rows=read_rows(RESP)
    assert len(rows)==len(tasks)
    assert [r["id"] for r in rows]==[t["id"] for t in tasks]
    assert all(r["ok"] and r["queries"] for r in rows)
    if FROZEN.exists():raise FileExistsError(FROZEN)
    new_json(FROZEN,{"n":len(rows),"responses_sha256":sha_file(RESP),
                     "pre_remote_sha256":sha_file(PLAN),
                     "valid_json":sum(r["status"]=="valid_json_queries" for r in rows),
                     "fallback_count":sum(r["status"]!="valid_json_queries" for r in rows),
                     "gold_not_opened":True,"source_not_uploaded":True})
    print("PHASE12_PLANNER_FROZEN",len(rows),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--execute",action="store_true")
    g.add_argument("--freeze",action="store_true")
    p.add_argument("--max-new",type=int,default=64)
    args=p.parse_args()
    if args.prepare:print("PHASE12_PLANNER_PREPARED",len(prepare()),flush=True)
    elif args.freeze:freeze()
    else:execute(max(1,min(args.max_new,64)))
