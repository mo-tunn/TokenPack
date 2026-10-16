"""Gold-blind Qwen27B reader evaluation for Phase8 atomic sentence packing.

HotpotQA 152 source question IDs, never used in previous Phase7 Modal
answer calls. Three conditions per question: full-source, original
MS MARCO300 at 25%, new atomic-sentence-rank at exact same true 25% cap.
Fixed Qwen27B prompt/greedy decoding as Phase7. No gold in Modal inputs.

Responses appended + fsynced individually and are resumable. One error
halts calls; in-flight request requires operator review before resuming.
All answer gold scoring is local and ONLY after complete frozen model output.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,BASE,make_payload,answer_metrics,
    read_rows,append_fsync,atomic_write,
)
from submission.experiments.phase8_atomic_protocol import (
    EVAL_COUNT,ARMS,
)
from submission.experiments.phase7_prepare_contexts import OUTPUT as PHASE7
from submission.experiments.phase6_hotpot_eval import DATA,PROVENANCE

EXPERIMENT=ROOT/"submission/results/phase8_atomic_goldblind152_20261012"
ROOT_OUT=ROOT/"submission/results/phase8_qwen27b_atomic152_20261012"
FREEZE=EXPERIMENT/"CONTEXTS_FROZEN.json"
SOURCE=EXPERIMENT/"gold_blind_atomic_contexts.jsonl"
TASKS=ROOT_OUT/"gold_blind_tasks.jsonl"
LOG=ROOT_OUT/"remote_answers.jsonl"
INFLIGHT=ROOT_OUT/"inflight_request.json"
ERROR=ROOT_OUT/"error.jsonl"
PLAN=ROOT_OUT/"COST_GUARD_AND_PROTOCOL.json"
FINAL=ROOT_OUT/"answer_em_f1_report.json"
MODAL_ARMS=("full-source","msmarco-25","atomic-sentence-rank")
MAX_REAL_REMOTE_CALLS=456
MAX_RUN_SECONDS=1800
ESTIMATED_USD_PER_HOUR=2.60
MAX_ESTIMATED_USD_PER_RUN=2.00


def sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def prepare_tasks():
    if ARMS!=("msmarco-25","atomic-sentence-rank","atomic-bridge-first"):
        raise AssertionError("Frozen Phase8 protocol changed")
    frozen=json.loads(FREEZE.read_text(encoding="utf8"))
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=frozen["contexts_sha256"]:
        raise AssertionError("Phase8 gold-blind source contexts changed")
    cases=[json.loads(line) for line in SOURCE.read_text(encoding="utf8").splitlines()
           if line.strip()]
    if len(cases)!=EVAL_COUNT:raise AssertionError("Incomplete Phase8 contexts")
    old_freeze=json.loads((PHASE7/"CONTEXTS_COMPLETE.json").read_text(encoding="utf8"))
    previous_file=PHASE7/"gold_blind_contexts.jsonl"
    if hashlib.sha256(previous_file.read_bytes()).hexdigest()!=old_freeze["contexts_sha256"]:
        raise AssertionError("Phase7 full source contexts modified")
    previous=[json.loads(s) for s in previous_file.read_text(encoding="utf8").splitlines()
              if s.strip()]
    expected={x["id"] for x in previous[104:]}
    if len(expected)!=EVAL_COUNT or {x["id"] for x in cases}!=expected:
        raise AssertionError("Wrong Phase8 reader ID set")
    prev_by_id={x["id"]:x for x in previous}
    tasks=[]
    for case in cases:
        original=prev_by_id[case["id"]]
        if case["question"]!=original["question"]:
            raise AssertionError("Question changed")
        arm_contexts={
            "full-source":original["contexts"]["full-source"],
            "msmarco-25":case["contexts"]["msmarco-25"],
            "atomic-sentence-rank":case["contexts"]["atomic-sentence-rank"],
        }
        if arm_contexts["msmarco-25"]["budget"]!=arm_contexts["atomic-sentence-rank"]["budget"]:
            raise AssertionError("Different actual token ceilings")
        if any(x["tokens"]>x["budget"] for key,x in arm_contexts.items()
               if key!="full-source"):
            raise AssertionError("Real token budget exceeded")
        for arm in MODAL_ARMS:
            value=arm_contexts[arm]
            tasks.append({
                "task_id":case["id"]+"|"+arm,
                "case_id":case["id"],
                "arm":arm,
                "question":case["question"],
                "context":value["text"],
                "context_tokens_cl100k":value["tokens"],
            })
    assert len(tasks)==EVAL_COUNT*len(MODAL_ARMS)
    ROOT_OUT.mkdir(parents=True,exist_ok=True)
    if TASKS.exists():
        if tasks!=read_rows(TASKS):raise AssertionError("Task manifest modified")
    else:
        with TASKS.open("x",encoding="utf8") as file:
            for row in tasks:file.write(json.dumps(row,ensure_ascii=False)+"\n")
    source_hash=sha(TASKS.read_text(encoding="utf8"))
    if PLAN.exists():
        if json.loads(PLAN.read_text(encoding="utf8"))["task_sha256"]!=source_hash:
            raise AssertionError("Research plan changed")
    else:
        uniqueness=defaultdict(set)
        for row in tasks:
            uniqueness[row["case_id"]].add(sha(json.dumps(
                make_payload(row),sort_keys=True,ensure_ascii=False)))
        atomic_write(PLAN,{
            "goal":"Answer retention after atomic sentence packing",
            "questions":EVAL_COUNT,
            "frozen_gold_blind_phase8_context_hash":frozen["contexts_sha256"],
            "task_sha256":source_hash,
            "reader":MODEL,"reader_endpoint":BASE,
            "arms":MODAL_ARMS,
            "total_answer_records":len(tasks),
            "estimated_remote_unique_requests":sum(map(len,uniqueness.values())),
            "max_paid_calls":MAX_REAL_REMOTE_CALLS,
            "max_seconds_per_run":MAX_RUN_SECONDS,
            "max_estimated_wall_cost_usd":MAX_ESTIMATED_USD_PER_RUN,
            "usd_per_hour_approximation":ESTIMATED_USD_PER_HOUR,
            "not_hard_invoice_cap":True,
            "no_gold_in_model_requests":True,
            "no_hyperparameter_change_from_phase7":True,
            "previous_104_modal_questions_excluded":True,
            "development_analysis_not_independent_hidden_test":True,
        })
    return tasks


def execute(max_new_remote_calls):
    tasks=prepare_tasks()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved previous request; manual review first")
    previous=read_rows(LOG)
    seen={}
    known={t["task_id"] for t in tasks}
    for x in previous:
        if x["task_id"] in seen or x["task_id"] not in known:
            raise AssertionError("Old log has duplicate/unrecognized answer")
        if not x.get("ok") or x.get("model")!=MODEL:
            raise AssertionError("Incomplete/mixed model response")
        seen[x["task_id"]]=x
    print("PHASE8_MODAL_PLAN",json.dumps({
        "questions":EVAL_COUNT,"answers":len(tasks),
        "saved_prior":len(seen),
        "planned_max_calls":json.loads(PLAN.read_text(encoding="utf8"))["estimated_remote_unique_requests"],
        "per_run_usd_guard":MAX_ESTIMATED_USD_PER_RUN,
    },ensure_ascii=False),flush=True)
    if len(seen)==len(tasks):return
    begun=time.monotonic()
    calls=0
    with APP.run():
        for item in tasks:
            if item["task_id"] in seen:continue
            payload=make_payload(item)
            digest=sha(json.dumps(payload,sort_keys=True,ensure_ascii=False))
            matched=next((r for r in seen.values() if r["case_id"]==item["case_id"]
                          and r["prompt_sha256"]==digest),None)
            if matched is not None:
                result={
                    "task_id":item["task_id"],"case_id":item["case_id"],
                    "arm":item["arm"],"model":MODEL,"ok":True,
                    "prompt_sha256":digest,"answer":matched["answer"],
                    "reused_from":matched["task_id"],"paid_remote_call":False,
                }
                append_fsync(LOG,result)
                seen[item["task_id"]]=result
                continue
            seconds=time.monotonic()-begun
            if (calls>=max_new_remote_calls or
                seconds+120>=MAX_RUN_SECONDS or
                (seconds+120)*ESTIMATED_USD_PER_HOUR/3600>=MAX_ESTIMATED_USD_PER_RUN):
                print("STOP_BUDGET_GUARD","answers",len(seen),flush=True)
                break
            atomic_write(INFLIGHT,{
                "task_id":item["task_id"],"sha":digest,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                response=infer_once.remote(payload)
            except BaseException as exc:
                append_fsync(ERROR,{
                    "task_id":item["task_id"],
                    "phase":"modal_remote","error_type":type(exc).__name__})
                print("STOP_MODEL_ERROR",type(exc).__name__,flush=True)
                break
            calls+=1
            if not response.get("ok") or not str(response.get("answer") or "").strip():
                append_fsync(ERROR,{
                    "task_id":item["task_id"],"phase":response.get("phase"),
                    "http_status":response.get("http_status"),
                    "exception":response.get("exception"),"empty_answer":True})
                print("STOP_BAD_REPLY",item["task_id"],flush=True)
                break
            line={
                "task_id":item["task_id"],"case_id":item["case_id"],
                "arm":item["arm"],"model":MODEL,"ok":True,
                "prompt_sha256":digest,
                "answer":str(response["answer"]).strip(),
                "paid_remote_call":True,
                "model_response":response.get("model_response"),
                "input_tokens":response.get("input_tokens"),
                "output_tokens":response.get("output_tokens"),
                "elapsed_s":response.get("elapsed_s"),
                "health_wait_s":response.get("gpu_health_wait_s"),
                "finish_reason":response.get("finish_reason"),
            }
            append_fsync(LOG,line)
            INFLIGHT.unlink()
            seen[item["task_id"]]=line
            if len(seen)%24==0 or len(seen)==len(tasks):
                print("PHASE8_MODAL_PROGRESS",len(seen),len(tasks),
                      "remote_this_run",calls,
                      "elapsed_s",round(time.monotonic()-begun,1),
                      flush=True)
    print("PHASE8_MODAL_SESSION_ENDED",
          "answers",len(seen),"new_remote_calls",calls,
          "errors",ERROR.exists(),"inflight",INFLIGHT.exists(),
          flush=True)


def score_local():
    tasks=prepare_tasks()
    records=read_rows(LOG)
    if len(records)!=len(tasks):
        raise RuntimeError("Incomplete model responses: "+str(len(records)))
    prev={x["task_id"]:x for x in records}
    if len(prev)!=len(tasks) or not all(x.get("ok") for x in records):
        raise AssertionError("Duplicate/failed reader record")
    provenance=json.loads(PROVENANCE.read_text(encoding="utf8"))
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=provenance["sha256"]:
        raise AssertionError("Benchmark file changed")
    # FIRST answer-gold access, local and only after all outputs are frozen.
    import pyarrow.parquet as pq
    ids={x["case_id"] for x in tasks}
    gold={x["id"]:x["answer"] for x in pq.read_table(
        DATA,columns=["id","answer"]).to_pylist() if x["id"] in ids}
    if set(gold)!=ids:raise AssertionError("Gold labels missing")
    metric=[]
    for task in tasks:
        result=prev[task["task_id"]]
        em,f1=answer_metrics(result["answer"],gold[task["case_id"]])
        metric.append({
            "case_id":task["case_id"],"arm":task["arm"],
            "gold_answer":gold[task["case_id"]],
            "model_answer":result["answer"],
            "EM":em,"F1":round(f1,6),
        })
    cases=defaultdict(dict)
    for row in metric:cases[row["case_id"]][row["arm"]]=row
    if len(cases)!=EVAL_COUNT or any(set(x)!=set(MODAL_ARMS) for x in cases.values()):
        raise AssertionError("Not properly paired")
    def stats(arm):
        items=[x[arm] for x in cases.values()]
        return {
            "n":len(items),
            "correct":sum(x["EM"] for x in items),
            "EM":round(statistics.mean(x["EM"] for x in items),6),
            "F1":round(statistics.mean(x["F1"] for x in items),6),
        }
    full_correct={cid for cid,x in cases.items() if x["full-source"]["EM"]}
    arms={arm:stats(arm) for arm in MODAL_ARMS}
    for arm in ("msmarco-25","atomic-sentence-rank"):
        arms[arm]["retained_when_full_correct"]=sum(
            cases[cid][arm]["EM"] for cid in full_correct)
        arms[arm]["n_full_correct"]=len(full_correct)
    summary={
        "reader":MODEL,"questions":EVAL_COUNT,
        "arms":arms,
        "calls_paid":sum(bool(x.get("paid_remote_call")) for x in records),
        "reused_identical_prompts":sum(bool(x.get("reused_from")) for x in records),
        "gold_sent_remote":False,
        "gold_read_only_after_output_frozen":True,
        "development_set":True,
        "case_metrics":metric,
    }
    if FINAL.exists():raise FileExistsError("Prior reader result exists")
    atomic_write(FINAL,summary)
    print("PHASE8_READER_FINAL",json.dumps(
        {k:v for k,v in summary.items() if k!="case_metrics"},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--score-only",action="store_true")
    parser.add_argument("--max-new",type=int,default=MAX_REAL_REMOTE_CALLS)
    args=parser.parse_args()
    if args.score_only:
        score_local()
    elif args.execute:
        execute(max(1,min(args.max_new,MAX_REAL_REMOTE_CALLS)))
        if len(read_rows(LOG))==EVAL_COUNT*len(MODAL_ARMS):
            score_local()
    else:
        prepare_tasks()
        plan=json.loads(PLAN.read_text(encoding="utf8"))
        print("PHASE8_READER_DRY_RUN",json.dumps({
            "questions":EVAL_COUNT,
            "answer_records":plan["total_answer_records"],
            "max_unique_modal_requests":plan["estimated_remote_unique_requests"],
            "model":MODEL,
            "gold_in_requests":False,
        },ensure_ascii=False),flush=True)
