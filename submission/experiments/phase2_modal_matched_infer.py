"""Bounded, sequential, resumable Qwen27B Phase-2 matched inference.

Only three previously unseen LongBench v2 cases, 8 prebuilt context arms
per case. The original questions/gold are NEVER serialized into the remote
API payload except for prompt question and answer options (NOT gold).

The API key remains in Modal Secret. No Codex-LB and no Freebuff. Inference
is guarded by an ESTIMATED, not enforceable, $2 wall-time stop threshold.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

import modal

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"submission/results/phase2_qwen27b_matched3_20261011"
TASKS=OUT/"phase2_tasks_private.jsonl"
LOG=OUT/"qwen27b_answers.jsonl"
INFLIGHT=OUT/"inflight_request.json"
ERRORS=OUT/"inference_errors.jsonl"
CONFIG=OUT/"run_config.json"
BASE="https://metekizilcik--tokenpack-qwen27b-longcontext-inference-server.modal.run"
MODEL="qwen3.8-27b-orcarouter-q4km"
APP=modal.App("tokenpack-phase2-matched3-qwen27b-20261011")
MAX_USD=2.0
CONSERVATIVE_ESTIMATED_USD_HOUR=2.60
MAX_CALLS=24
MAX_OUTPUT_TOKENS=128
SYSTEM="Use only the supplied context. Reply with exactly one uppercase letter A, B, C, or D. Do not provide explanations."

def utcnow()->str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read_rows(path:Path)->list[dict]:
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]

def append_fsync(path:Path,row:dict)->None:
    with path.open("a",encoding="utf8") as out:
        out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
        out.flush()
        os.fsync(out.fileno())

def atomic_write(path:Path,row:dict)->None:
    tmp=path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)

def make_payload(t:dict)->dict:
    """Never sends gold answer, exact task metadata, private evaluation notes."""
    for name in ["context","question","choice_A","choice_B","choice_C","choice_D"]:
        if not isinstance(t.get(name),str) or not t[name]:
            raise ValueError("Missing prompt field "+name)
    user="\n".join([
        "Context:",
        t["context"],"",
        "Question:",t["question"],"",
        "Options:",
        "A. "+t["choice_A"],"B. "+t["choice_B"],"C. "+t["choice_C"],"D. "+t["choice_D"],"",
        "Answer with only one uppercase letter:",
    ])
    return {
        "model":MODEL,
        "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,
        "seed":424242,"max_tokens":MAX_OUTPUT_TOKENS,
        "reasoning_effort":"none","stream":False,
    }

@APP.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    secrets=[modal.Secret.from_name("qwen-modal-secrets")],
    cpu=0.125,memory=768,timeout=540,
)
def infer_once(payload:dict)->dict:
    import os,json,time,urllib.error,urllib.request
    key=os.environ.get("LLM_API_KEY","")
    if not key:
        return {"ok":False,"phase":"secret","error":"LLM_API_KEY missing"}

    def invoke(path,data=None,timeout=23):
        raw=json.dumps(data).encode("utf8") if data is not None else None
        req=urllib.request.Request(
            BASE+path,
            method="POST" if data is not None else "GET",
            data=raw,
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
        )
        try:
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                return resp.status,json.loads(resp.read().decode("utf8"))
        except urllib.error.HTTPError as exc:
            return exc.code,{}

    t0=time.monotonic()
    readiness=0
    for _ in range(20):
        readiness+=1
        try:code,_=invoke("/health",timeout=22)
        except Exception as exc:
            return {"ok":False,"phase":"health","exception":type(exc).__name__,"health_checks":readiness}
        if code==200:break
        if code not in (502,503,504):
            return {"ok":False,"phase":"health","http_status":code,"health_checks":readiness}
        time.sleep(5)
    else:
        return {"ok":False,"phase":"health","error":"cold start exceeded 100sec"}
    started=time.monotonic()
    try:
        code,response=invoke("/v1/chat/completions",payload,timeout=390)
    except Exception as exc:
        return {"ok":False,"phase":"inference","exception":type(exc).__name__,
                "elapsed_s":round(time.monotonic()-started,2),"health_checks":readiness}
    if code!=200:
        return {"ok":False,"phase":"inference","http_status":code,
                "elapsed_s":round(time.monotonic()-started,2)}
    choices=response.get("choices") or []
    choice=choices[0] if choices else {}
    message=choice.get("message") or {}
    answer=str(message.get("content") or "").strip()
    usage=response.get("usage") or {}
    timings=response.get("timings") or {}
    return {
        "ok":True,"answer":answer[:240],"valid_choice":answer in ("A","B","C","D"),
        "model_response":str(response.get("model") or ""),
        "finish_reason":choice.get("finish_reason"),
        "reasoning_chars":len(str(message.get("reasoning_content") or "")),
        "input_tokens":usage.get("prompt_tokens"),
        "output_tokens":usage.get("completion_tokens"),
        "prompt_tokens_per_second":timings.get("prompt_per_second"),
        "elapsed_s":round(time.monotonic()-started,3),
        "gpu_health_wait_s":round(started-t0,3),
        "health_checks":readiness,
    }

def load_tasks()->list[dict]:
    a=read_rows(TASKS)
    if len(a)!=24 or len({t["task_id"] for t in a})!=24:
        raise ValueError("24 unique frozen matched tasks required")
    expected={"full-context","production-rag-50","tokenpack-50",
              "production-rag-50-p2-budgeted","tokenpack-50-p2-control",
              "tokenpack-50-p2-grouped","tokenpack-50-p2-neighbor",
              "tokenpack-50-p2-neighbor-grouped"}
    from collections import Counter
    per=Counter(t["method"] for t in a)
    if set(per)!=expected or any(n!=3 for n in per.values()):
        raise ValueError("Task method/case balance differs from frozen 3 x 8 setup")
    for t in a:
        if "-p2-" in t["method"] and t["context_tokens"]>t["budget_tokens"]:
            raise ValueError("Prepared Phase2 input exceeded budget")
    return sorted(a,key=lambda t:(t["case_id"],t["method"]))

def summarize(log:list[dict])->dict:
    per={}
    for r in log:
        entry=per.setdefault(r["method"],{"n":0,"correct":0,"valid":0,"input_tokens":0,"time_s":0.0})
        entry["n"]+=1
        entry["correct"]+=int(r.get("correct") is True)
        entry["valid"]+=int(r.get("valid_choice") is True)
        entry["input_tokens"]+=int(r.get("input_tokens") or 0)
        entry["time_s"]+=float(r.get("elapsed_s") or 0)
    return per

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true")
    p.add_argument("--max-new",type=int,default=MAX_CALLS)
    args=p.parse_args()
    if args.max_new<1 or args.max_new>MAX_CALLS:
        raise ValueError("max-new 1..24 required")
    tasks=load_tasks()
    prior=read_rows(LOG)
    done={r["task_id"] for r in prior}
    if len(done)!=len(prior):
        raise RuntimeError("Duplicate existing result!")
    for rec in prior:
        if rec.get("model")!=MODEL or rec.get("task_id") not in {t["task_id"] for t in tasks}:
            raise RuntimeError("Mixed output detected")
    if INFLIGHT.exists() or ERRORS.exists():
        raise RuntimeError("An unresolved inflight/error marker exists; operator review required")
    prep={"mode":"EXECUTE" if args.execute else "DRY_RUN",
          "task_count":len(tasks),"completed":len(prior),"remaining":len(tasks)-len(prior),
          "max_new":args.max_new,"estimated_cap_usd":MAX_USD,
          "estimated_usd_per_hour":CONSERVATIVE_ESTIMATED_USD_HOUR,
          "warning":"Cost cap is estimated process wall-clock, not Modal billing hard limit"}
    print(json.dumps(prep,ensure_ascii=False),flush=True)
    if not args.execute:return
    if CONFIG.exists():
        cfg=json.loads(CONFIG.read_text(encoding="utf8"))
        if cfg.get("model")!=MODEL or cfg.get("task_count")!=24:
            raise RuntimeError("Run configuration changed")
    else:
        cfg={"created_utc":utcnow(),"model":MODEL,"task_count":24,"gpu":"L40S",
             "context":262144,"temperature":0,"top_k":1,"top_p":1,"min_p":0,
             "seed":424242,"max_tokens":MAX_OUTPUT_TOKENS,"reasoning_effort":"none",
             "estimated_max_usd":MAX_USD,
             "spend_timing_note":"Conservative $2.60/h wall clock; actual charges must be checked in Modal"}
        atomic_write(CONFIG,cfg)
    started=time.monotonic()
    started_utc=dt.datetime.fromisoformat(cfg["created_utc"])
    count=0
    with APP.run():
        for task in tasks:
            if task["task_id"] in done:continue
            # Preserve the $2 protection across *separate* invocations too.
            # This is an upper-bound wall-time estimate, not actual billing.
            seconds_since_initial=(dt.datetime.now(dt.timezone.utc)-started_utc).total_seconds()
            estimate=(seconds_since_initial+390)/3600*CONSERVATIVE_ESTIMATED_USD_HOUR
            if estimate>=MAX_USD:
                print(json.dumps({"status":"PAUSE_BUDGET","completed":len(done),"estimated_usd":round(estimate,2)}),flush=True)
                break
            if count>=args.max_new:break
            payload=make_payload(task)
            prompt_digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
            marker={"task_id":task["task_id"],"time":utcnow(),"sha256":prompt_digest}
            atomic_write(INFLIGHT,marker)
            t0=time.monotonic()
            try:
                result=infer_once.remote(payload)
            except BaseException as exc:
                append_fsync(ERRORS,{"task_id":task["task_id"],"error_type":type(exc).__name__,"phase":"modal_rpc","timestamp":utcnow()})
                print("STOP_RPC "+str(type(exc).__name__),flush=True)
                raise
            rec={
                "task_id":task["task_id"],"case_id":task["case_id"],"method":task["method"],
                "timestamp_utc":utcnow(),"model":MODEL,"prompt_sha256":prompt_digest,
                "prepared_context_tokens":task["context_tokens"],
                "prepared_budget_tokens":task["budget_tokens"],
                "request_wall_s":round(time.monotonic()-t0,2),
                **result,
            }
            if result.get("ok") is True:
                rec["correct"]=rec["answer"]==task["answer"] if rec["valid_choice"] else False
                # Gold is evaluated only on local PC; not sent to Modal endpoint.
                append_fsync(LOG,rec)
                INFLIGHT.unlink()
                done.add(task["task_id"])
                count+=1
                print(json.dumps({
                    "completed":len(done),"of":24,"case":task["case_id"],"method":task["method"],
                    "input_tokens":rec["input_tokens"],"time_s":rec["elapsed_s"],
                    "valid":rec["valid_choice"],"correct":rec["correct"],
                    "estimated_spend_usd":round((time.monotonic()-started)*CONSERVATIVE_ESTIMATED_USD_HOUR/3600,2)
                },ensure_ascii=False),flush=True)
            else:
                append_fsync(ERRORS,rec)
                print("STOP_ERROR "+json.dumps({k:rec.get(k) for k in ("task_id","phase","http_status","exception","elapsed_s")}),flush=True)
                break
    status={"completed":len(done),"of":24,"new":count,"elapsed_wall_s":round(time.monotonic()-started,1),
            "estimated_wallclock_usd":round((time.monotonic()-started)*CONSERVATIVE_ESTIMATED_USD_HOUR/3600,2),
            "methods":summarize(read_rows(LOG))}
    atomic_write(OUT/"progress_summary.json",status)
    print("SUMMARY "+json.dumps(status,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
