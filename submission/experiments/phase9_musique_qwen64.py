"""Bounded preregistered 64-case Qwen27B reader pilot on NEW MuSiQue data.

Only fixed gold-blind contexts sent to previously established Modal endpoint.
Gold aliases/answers are not loaded until all three arms are committed.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))

from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,BASE,make_payload,answer_metrics,
    read_rows,append_fsync,atomic_write,
)
from submission.experiments.phase9_musique_contexts import (
    DATA,PROV,OUTPUT as ORIGINAL,
)
from submission.experiments.phase9_musique_protocol import (
    DATA_SHA256,READER_SAMPLES,N_QUESTIONS,
)

OUT=ROOT/"submission/results/phase9_musique_qwen27b_reader64_20261012"
TASKS=OUT/"gold_blind_tasks.jsonl"
ANSWERS=OUT/"remote_answers.jsonl"
INFLIGHT=OUT/"inflight_request.json"
ERROR=OUT/"error.jsonl"
PLAN=OUT/"READER_PROTOCOL_AND_COST.json"
SCORED=OUT/"QWEN_READER_EM_F1.json"
METHODS=("full-source","atomic-rank","crossdoc-bridge")
MAX_PAID_CALLS=READER_SAMPLES*3
MAX_SESSION_SECONDS=1200
ESTIMATED_USD_PER_HOUR=2.60
MAX_ESTIMATED_COST_USD=1.20


def checksum(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def prepare():
    freeze=json.loads((ORIGINAL/"CONTEXTS_FROZEN.json")
                      .read_text(encoding="utf8"))
    source=ORIGINAL/"gold_blind_contexts.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=freeze["case_file_sha256"]:
        raise AssertionError("Gold-blind MuSiQue contexts have been modified")
    rows=[json.loads(x) for x in source.read_text(encoding="utf8").splitlines()
          if x.strip()]
    if len(rows)!=N_QUESTIONS:raise AssertionError("Original 256 missing")
    # Deterministic first 64 from 256 sorted ID-only hash.
    rows=rows[:READER_SAMPLES]
    result=[]
    for item in rows:
        contexts={
            "full-source":item["source_context"],
            "atomic-rank":item["contexts"]["atomic-rank"]["text"],
            "crossdoc-bridge":item["contexts"]["crossdoc-bridge"]["text"],
        }
        if (item["contexts"]["atomic-rank"]["budget"]
            !=item["contexts"]["crossdoc-bridge"]["budget"]):
            raise AssertionError("Different actual token ceilings")
        for arm in METHODS:
            result.append({
                "task_id":item["id"]+"|"+arm,
                "case_id":item["id"],"arm":arm,
                "question":item["question"],
                "context":contexts[arm],
            })
    if len(result)!=READER_SAMPLES*3:
        raise AssertionError("Missing reader arms")
    OUT.mkdir(parents=True,exist_ok=True)
    if TASKS.exists():
        if read_rows(TASKS)!=result:
            raise AssertionError("Previously saved reader tasks changed")
    else:
        with TASKS.open("x",encoding="utf8") as out:
            for task in result:
                out.write(json.dumps(task,ensure_ascii=False)+"\n")
    digest=checksum(TASKS.read_text(encoding="utf8"))
    if PLAN.exists():
        if json.loads(PLAN.read_text(encoding="utf8"))["task_digest"]!=digest:
            raise AssertionError("Frozen reader protocol mismatch")
    else:
        per_case=defaultdict(set)
        for t in result:
            per_case[t["case_id"]].add(checksum(
                json.dumps(make_payload(t),sort_keys=True,ensure_ascii=False)))
        atomic_write(PLAN,{
            "model":MODEL,"endpoint":BASE,
            "data_sha256":DATA_SHA256,
            "reader_questions":READER_SAMPLES,"arms":METHODS,
            "requested_responses":len(result),
            "estimated_unique_paid_calls":sum(map(len,per_case.values())),
            "max_paid_calls":MAX_PAID_CALLS,
            "session_estimated_usd_guard":MAX_ESTIMATED_COST_USD,
            "estimated_usd_per_hour":ESTIMATED_USD_PER_HOUR,
            "actual_invoice_not_proven":True,
            "no_gold_aliases_in_modal_requests":True,
            "gold_blind_context_sha256":freeze["case_file_sha256"],
            "task_digest":digest,
            "preregistered_indices":[0,READER_SAMPLES],
            "public_validation_not_hidden_test":True,
        })
    return result


def inference_session(max_new):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Prior error or in-flight request requires manual audit")
    prior=read_rows(ANSWERS)
    saved={}
    allowed={x["task_id"] for x in tasks}
    for r in prior:
        if r["task_id"] not in allowed or r["task_id"] in saved or not r.get("ok"):
            raise AssertionError("Corrupt prior response log")
        if r["model"]!=MODEL:raise AssertionError("Mixed reader version")
        saved[r["task_id"]]=r
    start=time.monotonic()
    paid=0
    print("MUSIQUE_READER_START",len(saved),"/",len(tasks),
          "model",MODEL,"max_new_calls",max_new,flush=True)
    if len(saved)==len(tasks):return
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved:continue
            payload=make_payload(task)
            digest=checksum(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            reused=next((r for r in saved.values()
                         if r["case_id"]==task["case_id"]
                         and r["prompt_sha256"]==digest),None)
            if reused:
                record={
                    "task_id":task["task_id"],"case_id":task["case_id"],
                    "arm":task["arm"],"model":MODEL,
                    "prompt_sha256":digest,"answer":reused["answer"],
                    "ok":True,"paid_remote_call":False,
                    "reused_from":reused["task_id"],
                }
                append_fsync(ANSWERS,record)
                saved[task["task_id"]]=record
                continue
            seconds=time.monotonic()-start
            if (paid>=max_new or
                seconds+120>=MAX_SESSION_SECONDS or
                (seconds+120)*ESTIMATED_USD_PER_HOUR/3600>=MAX_ESTIMATED_COST_USD):
                print("MUSIQUE_READER_STOP_GUARD",len(saved),flush=True)
                break
            atomic_write(INFLIGHT,{
                "task_id":task["task_id"],"prompt_sha256":digest,
                "timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),
            })
            try:
                answer=infer_once.remote(payload)
            except BaseException as e:
                append_fsync(ERROR,{
                    "task_id":task["task_id"],
                    "exception_type":type(e).__name__,
                    "phase":"modal_runtime"})
                print("MUSIQUE_READER_STOP_FAILURE",type(e).__name__,flush=True)
                break
            paid+=1
            if not answer.get("ok") or not str(answer.get("answer") or "").strip():
                append_fsync(ERROR,{
                    "task_id":task["task_id"],"phase":answer.get("phase"),
                    "http_status":answer.get("http_status"),
                    "exception":answer.get("exception"),
                    "empty_answer":True,
                })
                print("MUSIQUE_READER_STOP_ERROR",task["task_id"],flush=True)
                break
            record={
                "task_id":task["task_id"],"case_id":task["case_id"],
                "arm":task["arm"],"model":MODEL,"ok":True,
                "answer":str(answer["answer"]).strip(),
                "prompt_sha256":digest,
                "paid_remote_call":True,
                "input_tokens":answer.get("input_tokens"),
                "output_tokens":answer.get("output_tokens"),
                "elapsed_s":answer.get("elapsed_s"),
                "finish_reason":answer.get("finish_reason"),
            }
            append_fsync(ANSWERS,record)
            INFLIGHT.unlink()
            saved[task["task_id"]]=record
            if len(saved)%24==0 or len(saved)==len(tasks):
                print("MUSIQUE_READER_PROGRESS",len(saved),len(tasks),
                      "paid_session",paid,
                      "elapsed_s",round(time.monotonic()-start,1),
                      flush=True)
    print("MUSIQUE_READER_SESSION_ENDED",
          "saved",len(saved),"paid_session",paid,
          "error",ERROR.exists(),"inflight",INFLIGHT.exists(),
          flush=True)


def score_answers():
    tasks=prepare()
    outputs=read_rows(ANSWERS)
    if len(outputs)!=READER_SAMPLES*len(METHODS):
        raise RuntimeError("Incomplete reader outputs, gold remains unopened")
    responses={x["task_id"]:x for x in outputs}
    if len(responses)!=len(tasks):raise AssertionError("Duplicate model answer")
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=DATA_SHA256:
        raise AssertionError("Benchmark changed")
    # Gold loaded only here, AFTER 192 context-conditioned responses exist.
    import pyarrow.parquet as pq
    ids={t["case_id"] for t in tasks}
    gold={x["id"]:x for x in pq.ParquetFile(DATA).read(
        columns=["id","answer","answer_aliases","answerable"]).to_pylist()
        if x["id"] in ids}
    if set(gold)!=ids:raise AssertionError("Wrong label sample")
    rows=[]
    for task in tasks:
        case=gold[task["case_id"]]
        pred=responses[task["task_id"]]["answer"]
        labels=[case["answer"]]+list(case.get("answer_aliases") or [])
        values=[answer_metrics(pred,truth) for truth in labels if truth is not None]
        em=max(v[0] for v in values)
        f1=max(v[1] for v in values)
        rows.append({
            "case_id":task["case_id"],"arm":task["arm"],
            "answerable":bool(case["answerable"]),
            "gold_answer":case["answer"],"gold_aliases":case.get("answer_aliases"),
            "model_answer":pred,"EM":em,"F1":round(f1,6),
        })
    by=defaultdict(dict)
    for row in rows:by[row["case_id"]][row["arm"]]=row
    if len(by)!=READER_SAMPLES or any(set(x)!=set(METHODS)
                                      for x in by.values()):
        raise AssertionError("Unpaired reader")
    arms={}
    for arm in METHODS:
        xx=[x[arm] for x in by.values()]
        arms[arm]={
            "correct":sum(x["EM"] for x in xx),"n":len(xx),
            "EM":round(statistics.mean(x["EM"] for x in xx),6),
            "F1":round(statistics.mean(x["F1"] for x in xx),6),
            "answerable_n":sum(x["answerable"] for x in xx),
        }
    full={cid for cid,x in by.items() if x["full-source"]["EM"]}
    for arm in ("atomic-rank","crossdoc-bridge"):
        arms[arm]["retained_when_full_correct"]=sum(
            by[cid][arm]["EM"] for cid in full)
        arms[arm]["n_full_correct"]=len(full)
    results={
        "dataset":"MuSiQue first 64 ID-only fixed sample within 256",
        "model":MODEL,"arms":arms,"questions":len(by),
        "paid_calls":sum(bool(r.get("paid_remote_call")) for r in outputs),
        "reused_same_prompt":sum(bool(r.get("reused_from")) for r in outputs),
        "gold_sent_to_modal":False,
        "gold_loaded_after_all_output":True,
        "official_metric_note":"MuSiQue EM/F1 with answer aliases; does not score support sufficiency",
        "case_metrics":rows,
    }
    if SCORED.exists():raise FileExistsError("Previous results archive")
    atomic_write(SCORED,results)
    print("MUSIQUE_QWEN_RESULT",json.dumps(
        {k:v for k,v in results.items() if k!="case_metrics"},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--score-only",action="store_true")
    parser.add_argument("--max-new",type=int,default=MAX_PAID_CALLS)
    args=parser.parse_args()
    if args.score_only:
        score_answers()
    elif args.execute:
        inference_session(max(1,min(args.max_new,MAX_PAID_CALLS)))
        if len(read_rows(ANSWERS))==READER_SAMPLES*len(METHODS):
            score_answers()
    else:
        tasks=prepare()
        print("MUSIQUE_READER_DRY_RUN",json.dumps({
            "n":len(tasks),"reader":MODEL,
            "gold_sent":False,
            "saved_inputs_sha256":checksum(TASKS.read_text(encoding="utf8")),
        },ensure_ascii=False),flush=True)
