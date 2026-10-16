"""Gold-blind Qwen27B multi-hop query planner for Phase11.
Only question text is sent. No supporting paragraphs, gold answers, decompositions.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase11_register import OUT,digest,save_new
from submission.experiments.phase11_musique_protocol import *
from submission.experiments.phase9_musique_contexts import DATA,PROV
from submission.experiments.phase9_musique_protocol import DATA_SHA256
from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,BASE,read_rows,append_fsync,atomic_write)
import pyarrow.parquet as pq

def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def make_query_payload(question):
    return {"model":MODEL,"messages":[
      {"role":"system","content":PLANNER_SYSTEM},
      {"role":"user","content":"Question:\n"+question+"\n\nEvidence search queries as JSON array:"}],
      "temperature":0,"top_k":1,"top_p":1,"min_p":0,"seed":20261012,
      "max_tokens":180,"reasoning_effort":"none","stream":False}

def planned_queries(raw,question):
    text=str(raw).strip()
    try:
        if text.startswith("```"):text=re.sub(r"^\s*```(?:json)?","",text).split("```")[0].strip()
        if not text.startswith("[") and "[" in text:
            text=text[text.find("["):text.rfind("]")+1]
        rows=json.loads(text)
        if not isinstance(rows,list):raise ValueError()
        clean=[]
        for a in rows:
            if not isinstance(a,str):continue
            a=" ".join(a.split())
            if len(a)>0 and a.casefold() not in {v.casefold() for v in clean}:
                clean.append(" ".join(a.split()[:18]))
            if len(clean)>=MAX_PLANNED_QUERIES:break
        if not clean:raise ValueError()
        return clean,"valid_json_queries"
    except (ValueError,TypeError,json.JSONDecodeError):
        return [question],"fallback_original_question_bad_json"

def prepare():
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert manifest["dataset_sha256"]==DATA_SHA256 and digest(DATA)==DATA_SHA256
    ids=manifest["pilot_ids"]
    rows=pq.ParquetFile(DATA).read(columns=["id","question"]).to_pylist()
    subset={r["id"]:r["question"] for r in rows if r["id"] in set(ids)}
    assert set(subset)==set(ids)
    tasks=[{"id":id,"question":subset[id]} for id in ids]
    target=OUT/"pilot_planner_tasks.jsonl"
    if target.exists():assert read_rows(target)==tasks
    else:
        with target.open("x",encoding="utf8") as f:
            for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+"\n")
    spec={"model":MODEL,"endpoint":BASE,"n":len(tasks),
        "planner_code_sha256":digest(Path(__file__)),
        "prompt_sha256":sha(PLANNER_SYSTEM),
        "task_sha256":digest(target),
        "max_remote_calls":PLANNER_MAX_REMOTE_CALLS,"max_tokens_per_query_result":180,
        "fallback":"If returned output is invalid JSON or empty, use the original question only. No case dropped.",
        "remote_receives":"question only (no source text, gold, IDs, answers)",
        "estimated_cost_guard_usd":MAX_ESTIMATED_SESSION_USD,
        "real_invoice_unverified":True,
        "not_a_hidden_test":True}
    planfile=OUT/"pilot_planner_PRE_REMOTE_FREEZE.json"
    if planfile.exists():assert json.loads(planfile.read_text(encoding="utf8"))==spec
    else:save_new(planfile,spec)
    return tasks

def execute(maxnew=PLANNER_MAX_REMOTE_CALLS):
    tasks=prepare()
    savedfile=OUT/"pilot_planner_remote.jsonl"
    error=OUT/"pilot_planner_error.jsonl"
    inflight=OUT/"pilot_planner_inflight.json"
    if error.exists() or inflight.exists():
        raise RuntimeError("Manual incident audit required")
    saved={}
    allowed={x["id"] for x in tasks}
    for x in read_rows(savedfile):
        assert x["id"] in allowed and x["id"] not in saved and x.get("ok")
        saved[x["id"]]=x
    start=time.monotonic();paid=0
    print("PHASE11_PLANNER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    with APP.run():
        for t in tasks:
            if t["id"] in saved:continue
            elapsed=time.monotonic()-start
            if paid>=maxnew or elapsed+120>=MAX_SESSION_SECONDS or (
                elapsed+120)*ESTIMATED_USD_HOUR/3600>=MAX_ESTIMATED_SESSION_USD:
                print("PHASE11_PLANNER_GUARD_STOP",len(saved),"/",len(tasks),flush=True)
                break
            payload=make_query_payload(t["question"])
            ph=sha(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            atomic_write(inflight,{"id":t["id"],"prompt_sha256":ph,
                "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                response=infer_once.remote(payload)
            except BaseException as exc:
                append_fsync(error,{"id":t["id"],"exception_type":type(exc).__name__})
                print("PHASE11_PLANNER_STOP_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not response.get("ok"):
                append_fsync(error,{"id":t["id"],"phase":response.get("phase"),
                    "http_status":response.get("http_status"),
                    "exception_type":response.get("exception")})
                print("PHASE11_PLANNER_STOP_BAD_ENDPOINT",t["id"],flush=True)
                break
            raw=str(response.get("answer") or "")
            queries,parse_status=planned_queries(raw,t["question"])
            result={"id":t["id"],"ok":True,"model":MODEL,
                "prompt_sha256":ph,"raw_model_output":raw,"queries":queries,
                "parse_status":parse_status,"paid_remote_call":True,
                "finish_reason":response.get("finish_reason"),
                "input_tokens":response.get("input_tokens"),
                "output_tokens":response.get("output_tokens")}
            append_fsync(savedfile,result);inflight.unlink()
            saved[result["id"]]=result
            if len(saved)%8==0 or len(saved)==len(tasks):
                print("PHASE11_PLANNER_PROGRESS",len(saved),"/",len(tasks),
                    "paid",paid,"elapsed",round(time.monotonic()-start,1),flush=True)
    print("PHASE11_PLANNER_SESSION_END",len(saved),"/",len(tasks),flush=True)

def freeze():
    tasks=prepare()
    source=OUT/"pilot_planner_remote.jsonl"
    rows=read_rows(source)
    assert len(rows)==len(tasks)
    assert [r["id"] for r in rows]==[t["id"] for t in tasks]
    assert all(r["ok"] and r["queries"] for r in rows)
    target=OUT/"pilot_planner_RESPONSES_FROZEN.json"
    if target.exists():raise FileExistsError("Already frozen")
    save_new(target,{"records":len(rows),"raw_plans_sha256":digest(source),
        "pre_remote_sha256":digest(OUT/"pilot_planner_PRE_REMOTE_FREEZE.json"),
        "valid_json":sum(x["parse_status"]=="valid_json_queries" for x in rows),
        "fallbacks":sum(x["parse_status"]!="valid_json_queries" for x in rows),
        "remote_receives_gold":False,"remote_receives_source":False})
    print("PHASE11_PLANNER_FROZEN",len(rows),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    group=a.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    a.add_argument("--max-new",type=int,default=PLANNER_MAX_REMOTE_CALLS)
    opt=a.parse_args()
    if opt.prepare:print("PHASE11_PLANNER_PREPARED",len(prepare()),flush=True)
    elif opt.freeze:freeze()
    else:execute(max(1,min(PLANNER_MAX_REMOTE_CALLS,opt.max_new)))
