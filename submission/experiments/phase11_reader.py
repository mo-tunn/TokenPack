"""Phase11 paired Qwen27B answer reader: only frozen context, no gold until 64 replies.
An empty but otherwise successful model completion is scored literally as empty,
never replaced by a guessed answer or cherry-picked retry.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,statistics,sys,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase11_register import OUT,DATA,digest,save_new
from submission.experiments.phase11_musique_protocol import *
from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,BASE,make_payload,answer_metrics,
    read_rows,append_fsync,atomic_write)
from submission.experiments.phase9_musique_protocol import DATA_SHA256
import numpy as np
from scipy.stats import binomtest

TASKS=OUT/"pilot_reader_tasks.jsonl"
PLAN=OUT/"pilot_reader_PRE_REMOTE_FREEZE.json"
LOG=OUT/"pilot_reader_responses.jsonl"
ERROR=OUT/"pilot_reader_error.jsonl"
INFLIGHT=OUT/"pilot_reader_inflight.json"
SCORED=OUT/"pilot_reader_SCORED_POST_GOLD.json"
READER_ARMS=("crossdoc-bridge","hop-planner")

def sha(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()

def make_tasks():
    pre=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"pilot_CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    contextpath=OUT/"pilot_gold_blind_contexts.jsonl"
    assert digest(contextpath)==frozen["contexts_sha256"]
    cases=[json.loads(x) for x in contextpath.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in cases]==pre["pilot_ids"]
    assert len(cases)==PILOT_N
    tasks=[]
    for c in cases:
        for arm in READER_ARMS:
            chosen=c["contexts"][arm]
            assert chosen["budget"]==c["budget"] and chosen["tokens"]<=c["budget"]
            tasks.append({"task_id":c["id"]+"|"+arm,"case_id":c["id"],
                "arm":arm,"question":c["question"],"context":chosen["text"]})
    assert len(tasks)==PILOT_N*len(READER_ARMS)
    return tasks

def prepare():
    tasks=make_tasks()
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as f:
            for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+"\n")
    spec={"model":MODEL,"endpoint":BASE,"n":len(tasks),
          "arms":list(READER_ARMS),"code_sha256":digest(Path(__file__)),
          "tasks_sha256":digest(TASKS),
          "contexts_sha256":digest(OUT/"pilot_gold_blind_contexts.jsonl"),
          "reader_prompt":"Existing phase7_modal27b_matched8.make_payload unchanged",
          "reader_prompt_source_sha256":digest(
              ROOT/"submission/experiments/phase7_modal27b_matched8.py"),
          "read_gold_only_after_all_answers":True,"gold_sent":False,
          "successful_empty_output":"Score literal empty; no auto retries or question removal",
          "max_paid_remote_calls":READER_PILOT_MAX_REMOTE_CALLS,
          "estimated_session_usd_guard":MAX_ESTIMATED_SESSION_USD,
          "real_invoice_not_proven":True}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:save_new(PLAN,spec)
    return tasks

def execute(max_new):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Prior remote error or inflight, requires audit")
    prior=read_rows(LOG)
    expected={t["task_id"] for t in tasks}
    saved={}
    for r in prior:
        assert r["task_id"] in expected and r["task_id"] not in saved and r.get("ok")
        assert r["model"]==MODEL
        saved[r["task_id"]]=r
    print("PHASE11_READER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    start=time.monotonic();paid=0
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved:continue
            payload=make_payload(task)
            prompt_hash=sha(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            duplicate=next((s for s in saved.values() if s["case_id"]==task["case_id"]
                and s["prompt_sha256"]==prompt_hash),None)
            if duplicate:
                rec={"task_id":task["task_id"],"case_id":task["case_id"],
                    "arm":task["arm"],"model":MODEL,"ok":True,
                    "answer":duplicate["answer"],"prompt_sha256":prompt_hash,
                    "paid_remote_call":False,"reused_from":duplicate["task_id"]}
                append_fsync(LOG,rec);saved[rec["task_id"]]=rec
                continue
            elapsed=time.monotonic()-start
            if paid>=max_new or elapsed+120>=MAX_SESSION_SECONDS or (
                elapsed+120)*ESTIMATED_USD_HOUR/3600>=MAX_ESTIMATED_SESSION_USD:
                print("PHASE11_READER_GUARD",len(saved),"/",len(tasks),flush=True);break
            atomic_write(INFLIGHT,{"task_id":task["task_id"],
                "prompt_sha256":prompt_hash,
                "timestamp":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                resp=infer_once.remote(payload)
            except BaseException as exc:
                append_fsync(ERROR,{"task_id":task["task_id"],
                    "exception_type":type(exc).__name__})
                print("PHASE11_READER_REMOTE_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not resp.get("ok"):
                append_fsync(ERROR,{"task_id":task["task_id"],
                    "phase":resp.get("phase"),"http_status":resp.get("http_status"),
                    "exception_type":resp.get("exception")})
                print("PHASE11_READER_ENDPOINT_ERROR",task["task_id"],flush=True);break
            ans=str(resp.get("answer") or "").strip()
            rec={"task_id":task["task_id"],"case_id":task["case_id"],
                 "arm":task["arm"],"model":MODEL,"ok":True,
                 "answer":ans,"empty_answer":not bool(ans),
                 "paid_remote_call":True,"prompt_sha256":prompt_hash,
                 "input_tokens":resp.get("input_tokens"),
                 "output_tokens":resp.get("output_tokens"),
                 "finish_reason":resp.get("finish_reason")}
            append_fsync(LOG,rec);INFLIGHT.unlink()
            saved[rec["task_id"]]=rec
            if len(saved)%8==0 or len(saved)==len(tasks):
                print("PHASE11_READER_PROGRESS",len(saved),"/",len(tasks),
                    "new_calls",paid,"elapsed",round(time.monotonic()-start,1),flush=True)
    print("PHASE11_READER_SESSION_END",len(saved),"/",len(tasks),
         "new_paid",paid,"error",ERROR.exists(),flush=True)

def ci95(vals):
    x=np.asarray(vals,dtype=float)
    rng=np.random.default_rng(20261012)
    z=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return [float(a) for a in np.quantile(z,[.025,.975])]

def score():
    tasks=prepare()
    saved=read_rows(LOG)
    if len(saved)!=len(tasks):
        raise RuntimeError("Incomplete Qwen answers; do not read gold")
    ans={x["task_id"]:x for x in saved}
    assert len(ans)==len(tasks) and all(x.get("ok") for x in saved)
    assert digest(DATA)==DATA_SHA256
    import pyarrow.parquet as pq
    ids={t["case_id"] for t in tasks}
    gold={x["id"]:x for x in pq.ParquetFile(DATA).read(
        columns=["id","answer","answer_aliases","answerable"]).to_pylist()
        if x["id"] in ids}
    assert set(gold)==ids
    by=defaultdict(dict)
    scored=[]
    for t in tasks:
        label=gold[t["case_id"]]
        values=[label["answer"]]+list(label.get("answer_aliases") or [])
        nums=[answer_metrics(ans[t["task_id"]]["answer"],str(v))
            for v in values if v is not None]
        entry={"case_id":t["case_id"],"arm":t["arm"],
               "EM":max(v[0] for v in nums),
               "F1":max(v[1] for v in nums)}
        scored.append(entry);by[t["case_id"]][t["arm"]]=entry
    assert len(by)==PILOT_N and all(set(x)==set(READER_ARMS) for x in by.values())
    counts={}
    for arm in READER_ARMS:
        values=[x[arm] for x in by.values()]
        counts[arm]={"correct":sum(a["EM"] for a in values),
            "EM":statistics.mean(a["EM"] for a in values),
            "F1":statistics.mean(a["F1"] for a in values)}
    diff=[x["hop-planner"]["EM"]-x["crossdoc-bridge"]["EM"] for x in by.values()]
    f1=[x["hop-planner"]["F1"]-x["crossdoc-bridge"]["F1"] for x in by.values()]
    wins=sum(v==1 for v in diff);losses=sum(v==-1 for v in diff)
    result={"dataset":"MuSiQue public validation, Phase9/10 disjoint 32 ID pilot",
        "n":PILOT_N,"counts":counts,
        "paired":{"EM_difference":statistics.mean(diff),
                  "EM_bootstrap_ci95":ci95(diff),
                  "F1_difference":statistics.mean(f1),
                  "F1_bootstrap_ci95":ci95(f1),
                  "hop_only_correct":wins,"bridge_only_correct":losses,
                  "net_correct":wins-losses,
                  "exact_mcnemar_p_two_sided":binomtest(min(wins,losses),wins+losses,.5).pvalue
                    if wins+losses else 1.0},
        "empty_successful_model_outputs":sum(bool(x.get("empty_answer")) for x in saved),
        "remote_paid_calls":sum(bool(x.get("paid_remote_call")) for x in saved),
        "same_payload_reused":sum(bool(x.get("reused_from")) for x in saved),
        "reader_response_sha256":digest(LOG),
        "gold_opened_after_all_answers":True,
        "gold_sent_to_remote":False,
        "gate_threshold_net_correct":PILOT_GO_NET_EM,
        "pilot_gate_passed":wins-losses>=PILOT_GO_NET_EM,
        "decision":"GO_VALIDATION" if wins-losses>=PILOT_GO_NET_EM else "STOP_NO_VALIDATION",
        "sample_metrics":scored,
        "statistical_note":"Exploratory small public validation pilot, not hidden independent test"}
    if SCORED.exists():raise FileExistsError("Pilot already scored")
    save_new(SCORED,result)
    print("PHASE11_READER_SCORED",json.dumps({k:v for k,v in result.items()
        if k!="sample_metrics"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    group=a.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--score",action="store_true")
    a.add_argument("--max-new",type=int,default=READER_PILOT_MAX_REMOTE_CALLS)
    opt=a.parse_args()
    if opt.prepare:print("PHASE11_READER_PREPARED",len(prepare()),flush=True)
    elif opt.score:score()
    else:execute(max(1,min(opt.max_new,READER_PILOT_MAX_REMOTE_CALLS)))
