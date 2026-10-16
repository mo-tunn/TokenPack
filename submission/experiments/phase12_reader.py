"""Phase12 Qwen27B downstream reader: 32 fixed cases, 2 matched arms.
Gold answer opened only after all 64 responses complete and frozen.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,statistics,datetime as dt
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import OUT,DATA,verify_data,sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,make_payload,answer_metrics,read_rows,append_fsync,atomic_write)
import pyarrow.parquet as pq
ARMS=("crossdoc-bridge","hop-planner")
TASKS=OUT/"reader_tasks.jsonl"
PLAN=OUT/"READER_PRE_REMOTE_PROTOCOL.json"
ANS=OUT/"reader_responses.jsonl"
ERROR=OUT/"reader_error.jsonl"
INFLIGHT=OUT/"reader_inflight.json"
SCORED=OUT/"READER_POST_GOLD_RESULTS.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def prepare():
    proto=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    path=OUT/"gold_blind_contexts.jsonl"
    assert sha_file(path)==frozen["context_file_sha256"]
    cases=[json.loads(x) for x in path.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in cases]==proto["ids"]
    cases=cases[:proto["reader_first_n"]]
    tasks=[]
    for item in cases:
        for arm in ARMS:
            ctx=item["contexts"][arm]
            assert ctx["tokens"]<=item["budget"]==ctx["budget"]
            tasks.append({"task_id":item["id"]+"|"+arm,
                "id":item["id"],"arm":arm,"question":item["question"],"context":ctx["text"]})
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as out:
            for t in tasks:out.write(json.dumps(t,ensure_ascii=False)+"\n")
    plan={"n":len(tasks),"cases":len(cases),"arms":list(ARMS),
        "tasks_sha256":sha_file(TASKS),
        "contexts_sha256":frozen["context_file_sha256"],
        "reader_source_sha256":sha_file(ROOT/"submission/experiments/phase7_modal27b_matched8.py"),
        "script_sha256":sha_file(Path(__file__)),
        "success_empty_answer":"score literally as empty; no case replacement",
        "max_remote_calls":64,"max_session_seconds":1800,
        "estimated_usd_per_hour":2.60,"approx_session_guard_usd":1.35,
        "real_invoice_not_audited":True,"gold_sent":False}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==plan
    else:new_json(PLAN,plan)
    return tasks

def execute():
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():raise RuntimeError("Pending remote error/inflight; audit first")
    prior=read_rows(ANS);saved={}
    for r in prior:
        assert r["task_id"] not in saved and r.get("ok")
        saved[r["task_id"]]=r
    print("PHASE12_READER_START",len(saved),"/",len(tasks),flush=True)
    start=time.monotonic();paid=0
    if len(saved)==len(tasks):return
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved:continue
            payload=make_payload(task)
            digest=sha(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            duplicate=next((r for r in saved.values()
                if r["id"]==task["id"] and r["prompt_sha256"]==digest),None)
            if duplicate:
                rec={"task_id":task["task_id"],"id":task["id"],"arm":task["arm"],
                     "ok":True,"answer":duplicate["answer"],"paid":False,
                     "reused":True,"prompt_sha256":digest}
                append_fsync(ANS,rec);saved[rec["task_id"]]=rec;continue
            elapsed=time.monotonic()-start
            if paid>=64 or elapsed+120>1800 or (elapsed+120)*2.60/3600>1.35:
                print("PHASE12_READER_GUARD",len(saved),flush=True);break
            atomic_write(INFLIGHT,{"task_id":task["task_id"],
                "prompt_sha256":digest,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:answer=infer_once.remote(payload)
            except BaseException as e:
                append_fsync(ERROR,{"task_id":task["task_id"],"type":type(e).__name__})
                print("PHASE12_REMOTE_EXCEPTION",type(e).__name__,flush=True);break
            paid+=1
            if not answer.get("ok"):
                append_fsync(ERROR,{"task_id":task["task_id"],
                    "phase":answer.get("phase"),"http_status":answer.get("http_status"),
                    "exception":answer.get("exception")})
                print("PHASE12_READER_REMOTE_FAILURE",task["task_id"],flush=True);break
            text=str(answer.get("answer") or "").strip()
            rec={"task_id":task["task_id"],"id":task["id"],"arm":task["arm"],
                "ok":True,"answer":text,"empty":not bool(text),
                "paid":True,"prompt_sha256":digest,
                "finish_reason":answer.get("finish_reason")}
            append_fsync(ANS,rec);INFLIGHT.unlink();saved[rec["task_id"]]=rec
            if len(saved)%8==0:print("PHASE12_READER_PROGRESS",len(saved),"/",len(tasks),
                  "new_calls",paid,flush=True)
    print("PHASE12_READER_SESSION_END",len(saved),"/",len(tasks),flush=True)

def ci(vals):
    x=np.asarray(vals,dtype=float);rng=np.random.default_rng(20261012)
    sample=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(sample,[.025,.975])]

def score():
    tasks=prepare()
    responses=read_rows(ANS)
    if len(responses)!=len(tasks):raise RuntimeError("Reader incomplete; do not read labels")
    saved={r["task_id"]:r for r in responses}
    assert len(saved)==len(tasks) and all(r.get("ok") for r in responses)
    verify_data()
    ids={t["id"] for t in tasks}
    gold={r["id"]:r["answer"] for r in pq.ParquetFile(DATA).read(
        columns=["id","answer"]).to_pylist() if r["id"] in ids}
    assert set(gold)==ids
    by=defaultdict(dict)
    for t in tasks:
        em,f1=answer_metrics(saved[t["task_id"]]["answer"],gold[t["id"]])
        by[t["id"]][t["arm"]]={"EM":em,"F1":f1}
    assert len(by)==len(ids) and all(set(x)==set(ARMS) for x in by.values())
    arms={}
    for arm in ARMS:
        x=[a[arm] for a in by.values()]
        arms[arm]={"correct":sum(r["EM"] for r in x),
                   "EM":statistics.mean(r["EM"] for r in x),
                   "F1":statistics.mean(r["F1"] for r in x)}
    dif=[r["hop-planner"]["EM"]-r["crossdoc-bridge"]["EM"] for r in by.values()]
    f1=[r["hop-planner"]["F1"]-r["crossdoc-bridge"]["F1"] for r in by.values()]
    win=sum(v==1 for v in dif);loss=sum(v==-1 for v in dif)
    result={"benchmark":"2WikiMultiHopQA public validation","n":len(by),
      "arms":arms,"paired":{"EM_difference":statistics.mean(dif),
      "EM_CI95":ci(dif),"F1_difference":statistics.mean(f1),"F1_CI95":ci(f1),
      "hop_only":win,"bridge_only":loss,"net":win-loss,
      "mcnemar_exact_two_sided":binomtest(min(win,loss),win+loss,.5).pvalue if win+loss else 1.0},
      "remote_calls":sum(r.get("paid",False) for r in responses),
      "reused":sum(bool(r.get("reused")) for r in responses),
      "empty_predictions":sum(bool(r.get("empty")) for r in responses),
      "gold_read_after_all_output":True,
      "gold_not_sent_remotely":True,
      "response_sha256":sha_file(ANS),
      "per_case":[{"id":id,**values} for id,values in by.items()],
      "not_hidden_test":True}
    if SCORED.exists():raise FileExistsError(SCORED)
    new_json(SCORED,result)
    print("PHASE12_READER_SCORED",json.dumps({k:v for k,v in result.items() if k!="per_case"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--execute",action="store_true")
    g.add_argument("--score",action="store_true")
    a=p.parse_args()
    if a.prepare:print("PHASE12_READER_PREPARED",len(prepare()),flush=True)
    elif a.score:score()
    else:execute()
