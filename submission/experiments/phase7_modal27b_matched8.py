"""Bounded HotpotQA reader evaluation via EXISTING Modal Qwen 27B endpoint.

Eight fixed preselected Phase7 HotpotQA questions; three contexts per case:
full-source, manual-25, msmarco-25. 24 matched outputs, with duplicate
context requests reused *within* a case to avoid unnecessary paid calls.
Answers and support labels are NEVER sent to Modal.

Runs via the existing Phase2 Modal Secret relay, not a new GPU deployment.
Append-only response log, resumable, stops on first error or cost/time guard.
Gold QA EM/F1 scoring ONLY after all model responses are finalized.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import string
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

# Modal's auto-source mounting otherwise walks the much larger local
# submission/repo. Stage only the tiny relay source into an isolated folder.
# No benchmark dataset, gold or trained weights are uploaded by Modal.
RELAY_ROOT=Path(tempfile.gettempdir())/"tokenpack_phase7_isolated_relay"
RELAY_ROOT.mkdir(parents=True,exist_ok=True)
RELAY_SOURCE=ROOT/"submission/experiments/phase7_modal27b_relay_source.py"
RELAY_LOCAL=RELAY_ROOT/"phase7_modal27b_relay.py"
if not RELAY_LOCAL.exists() or RELAY_LOCAL.read_bytes()!=RELAY_SOURCE.read_bytes():
    shutil.copyfile(RELAY_SOURCE,RELAY_LOCAL)
sys.path.insert(0,str(RELAY_ROOT))
from phase7_modal27b_relay import APP,infer_once,MODEL,BASE


def read_rows(path:Path)->list[dict]:
    if not path.exists():return []
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines()
            if x.strip()]


def append_fsync(path:Path,row:dict)->None:
    with path.open("a",encoding="utf8") as out:
        out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
        out.flush()
        os.fsync(out.fileno())


def atomic_write(path:Path,row:dict)->None:
    temp=path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    temp.replace(path)

PHASE7=ROOT/"submission/results/phase7_hotpotqa_reader256_20261012"
OUT=ROOT/"submission/results/phase7_modal_qwen27b_hotpotqa_matched8_20261012"
CONTEXTS=PHASE7/"gold_blind_contexts.jsonl"
PREP=PHASE7/"CONTEXTS_COMPLETE.json"
DATA=ROOT/"submission/data/phase6_hotpotqa_validation_20261011/distractor/validation-00000-of-00001.parquet"
PROV=DATA.parent.parent/"PROVENANCE.json"
TASK_FILE=OUT/"gold_blind_tasks.jsonl"
LOG=OUT/"remote_answers.jsonl"
ERR=OUT/"error.jsonl"
INFLIGHT=OUT/"inflight_request.json"
RESULT=OUT/"answer_em_f1_report.json"
PLAN=OUT/"cost_and_protocol.json"
ARMS=("full-source","manual-25","msmarco-25")
NUM_CASES=8
MAX_REMOTE_CALLS=24
MAX_WALL_SECONDS=720
MAX_ESTIMATED_USD=2.0
ESTIMATED_USD_PER_HOUR=2.60
MAX_GENERATED=72
SYSTEM=("Answer the question using only the context. Reply only with the "
        "short answer, e.g. name, number, yes or no. If the context "
        "is insufficient, reply unknown. Do not explain your reasoning.")


def sha(value):
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def tasks_from_prepared():
    freeze=json.loads(PREP.read_text(encoding="utf8"))
    if hashlib.sha256(CONTEXTS.read_bytes()).hexdigest()!=freeze["contexts_sha256"]:
        raise AssertionError("Phase7 prepared gold-blind source contexts modified")
    data=[json.loads(s) for s in CONTEXTS.read_text(encoding="utf8").splitlines() if s.strip()]
    if len(data)!=256:raise AssertionError("Expected 256 frozen question contexts")
    rows=[]
    for item in data[:NUM_CASES]:
        if set(ARMS)-set(item["contexts"]):
            raise AssertionError("Missing frozen contexts")
        for arm in ARMS:
            ctx=item["contexts"][arm]
            if arm!="full-source" and ctx["tokens"]>ctx["budget"]:
                raise AssertionError("Output rendered cap exceeded")
            rows.append({
                "task_id":item["id"]+"|"+arm,
                "case_id":item["id"],"arm":arm,"question":item["question"],
                "context":ctx["text"],
                "context_tokens_cl100k":ctx["tokens"],
                "source_sha256":freeze["contexts_sha256"],
            })
    return rows


def make_payload(task):
    # NEVER send gold answers, evidence labels, case ID or evaluation metadata.
    user=(
        "Context:\n"+task["context"]+
        "\n\nQuestion:\n"+task["question"]+
        "\n\nAnswer in one short phrase:"
    )
    return {
        "model":MODEL,
        "messages":[{"role":"system","content":SYSTEM},
                    {"role":"user","content":user}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,
        "seed":20261012,"max_tokens":MAX_GENERATED,
        "reasoning_effort":"none","stream":False,
    }


def normalize(s):
    text=str(s).lower()
    text="".join(ch for ch in text if ch not in string.punctuation)
    text=re.sub(r"\b(a|an|the)\b"," ",text)
    return " ".join(text.split())


def answer_metrics(pred,truth):
    a=normalize(pred)
    b=normalize(truth)
    exact=int(a==b)
    if (a in ("yes","no","noanswer") or b in ("yes","no","noanswer")) and a!=b:
        return exact,0.
    left=a.split();right=b.split()
    if not left or not right:return exact,float(left==right)
    n=sum((Counter(left)&Counter(right)).values())
    if n==0:return exact,0.
    prec=n/len(left);rec=n/len(right)
    return exact,2*prec*rec/(prec+rec)


def prepare():
    tasks=tasks_from_prepared()
    if len(tasks)!=NUM_CASES*len(ARMS):raise AssertionError("Wrong task count")
    OUT.mkdir(parents=True,exist_ok=True)
    if TASK_FILE.exists():
        saved=read_rows(TASK_FILE)
        if saved!=tasks:raise AssertionError("Saved tasks were modified")
    else:
        with TASK_FILE.open("x",encoding="utf8") as f:
            for task in tasks:
                f.write(json.dumps(task,ensure_ascii=False)+"\n")
    if PLAN.exists():
        record=json.loads(PLAN.read_text(encoding="utf8"))
        if record["task_digest"]!=sha(TASK_FILE.read_text(encoding="utf8")):
            raise AssertionError("Pilot task manifest changed")
    else:
        unique_per_case=defaultdict(set)
        for t in tasks:unique_per_case[t["case_id"]].add(sha(t["question"]+"\n"+t["context"]))
        atomic_write(PLAN,{
            "study":"HotpotQA 8 questions x full/manual25/MSMARCO25",
            "model":MODEL,"endpoint":BASE,"arms":ARMS,
            "task_count":len(tasks),
            "unique_paid_requests_if_all_succeed":sum(map(len,unique_per_case.values())),
            "task_digest":sha(TASK_FILE.read_text(encoding="utf8")),
            "max_paid_calls":MAX_REMOTE_CALLS,
            "max_elapsed_wall_seconds_per_run":MAX_WALL_SECONDS,
            "max_estimated_usd_per_run":MAX_ESTIMATED_USD,
            "estimated_rate_usd_hour":ESTIMATED_USD_PER_HOUR,
            "cost_warning":"Guard estimates wall time, not hard Modal actual invoice cap.",
            "max_generated_tokens":MAX_GENERATED,
            "system_prompt":SYSTEM,"gold_sent_to_modal":False,
            "gold_answers_read_before_inference":False,
        })
    return tasks


def execute(limit=MAX_REMOTE_CALLS):
    tasks=prepare()
    if INFLIGHT.exists() or ERR.exists():
        raise RuntimeError("Unresolved inflight/error; operator review required")
    prior=read_rows(LOG)
    seen={}
    for row in prior:
        tid=row["task_id"]
        if tid in seen or tid not in {x["task_id"] for x in tasks}:
            raise AssertionError("Duplicate or unknown saved answer")
        if row.get("model")!=MODEL:raise AssertionError("Mixed reader model")
        seen[tid]=row
    print("RUN_PLAN",json.dumps({
        "tasks":len(tasks),"prior":len(seen),
        "unique_requests":json.loads(PLAN.read_text(encoding="utf8"))["unique_paid_requests_if_all_succeed"],
        "max_estimated_usd":MAX_ESTIMATED_USD,
        "max_wall_s":MAX_WALL_SECONDS},ensure_ascii=False),flush=True)
    if len(seen)==len(tasks):
        print("ALL_ANSWERS_ALREADY_SAVED",flush=True)
        return
    start=time.monotonic()
    new_paid=0
    with APP.run():
        for task in tasks:
            if task["task_id"] in seen:continue
            # If two arms in the same case have identical question+context
            # and model settings, deterministically reuse exact model output.
            prompt_hash=sha(json.dumps(make_payload(task),sort_keys=True,
                                      ensure_ascii=False))
            reused=next((v for v in seen.values() if
                         v.get("prompt_sha256")==prompt_hash and
                         v["case_id"]==task["case_id"]),None)
            if reused:
                result={
                    "task_id":task["task_id"],"case_id":task["case_id"],
                    "arm":task["arm"],"model":MODEL,
                    "prompt_sha256":prompt_hash,
                    "answer":reused["answer"],"ok":True,
                    "reused_from":reused["task_id"],"paid_remote_call":False,
                    "output_tokens":reused.get("output_tokens"),
                    "input_tokens":reused.get("input_tokens"),
                }
                append_fsync(LOG,result)
                seen[task["task_id"]]=result
                print("REUSED_DUPLICATE_CONTEXT",task["case_id"],task["arm"],flush=True)
                continue
            spent=(time.monotonic()-start)*ESTIMATED_USD_PER_HOUR/3600
            if (new_paid>=limit or
                time.monotonic()-start>=MAX_WALL_SECONDS or
                spent>=MAX_ESTIMATED_USD or
                (time.monotonic()-start+120)*ESTIMATED_USD_PER_HOUR/3600>=MAX_ESTIMATED_USD):
                print("STOP_BUDGET_OR_TIME",len(seen),"saved",flush=True)
                break
            atomic_write(INFLIGHT,{"task_id":task["task_id"],
                                  "prompt_sha256":prompt_hash,
                                  "started_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                response=infer_once.remote(make_payload(task))
            except BaseException as ex:
                append_fsync(ERR,{"task_id":task["task_id"],
                                  "error_type":type(ex).__name__,
                                  "phase":"modal_remote"})
                print("STOP_REMOTE_FAILURE",type(ex).__name__,flush=True)
                break
            new_paid+=1
            if not response.get("ok") or not str(response.get("answer") or "").strip():
                append_fsync(ERR,{
                    "task_id":task["task_id"],
                    "error_phase":response.get("phase"),
                    "http_status":response.get("http_status"),
                    "exception":response.get("exception"),
                    "elapsed_s":response.get("elapsed_s"),
                    "reason":"empty_or_failed_result",
                })
                print("STOP_ENDPOINT_ERROR",task["task_id"],response,flush=True)
                break
            entry={
                "task_id":task["task_id"],"case_id":task["case_id"],
                "arm":task["arm"],"model":MODEL,"ok":True,
                "prompt_sha256":prompt_hash,
                "answer":str(response["answer"]).strip(),
                "paid_remote_call":True,
                "model_response":response.get("model_response"),
                "input_tokens":response.get("input_tokens"),
                "output_tokens":response.get("output_tokens"),
                "elapsed_s":response.get("elapsed_s"),
                "health_wait_s":response.get("gpu_health_wait_s"),
                "finish_reason":response.get("finish_reason"),
            }
            append_fsync(LOG,entry)
            INFLIGHT.unlink()
            seen[entry["task_id"]]=entry
            print("MODAL_SUCCESS",len(seen),"/",len(tasks),
                  "real_calls",new_paid,
                  "duration_s",entry["elapsed_s"],
                  "estimated_wall_usd",
                  round((time.monotonic()-start)*ESTIMATED_USD_PER_HOUR/3600,3),
                  flush=True)
    print("RUN_ENDED",json.dumps({
        "saved":len(seen),"required":len(tasks),
        "new_remote_calls":new_paid,
        "wall_s":round(time.monotonic()-start,1),
        "estimated_wall_usd":round(
            (time.monotonic()-start)*ESTIMATED_USD_PER_HOUR/3600,3),
        "inflight":INFLIGHT.exists(),"errors":ERR.exists()},
        ensure_ascii=False),flush=True)


def score_local():
    tasks=prepare()
    saved=read_rows(LOG)
    if len(saved)!=len(tasks):
        raise RuntimeError(f"Only {len(saved)}/{len(tasks)} answers; do not score incomplete run")
    index={x["task_id"]:x for x in saved}
    if len(index)!=len(tasks) or not all(x.get("ok") for x in saved):
        raise AssertionError("Duplicated or failed answers")
    # Gold HotpotQA answers are loaded for the first time here, locally.
    import pyarrow.parquet as pq
    provenance=json.loads(PROV.read_text(encoding="utf8"))
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=provenance["sha256"]:
        raise AssertionError("HotpotQA data changed")
    ids={t["case_id"] for t in tasks}
    gold={x["id"]:x["answer"] for x in pq.read_table(
        DATA,columns=["id","answer"]).to_pylist() if x["id"] in ids}
    if set(gold)!=ids:raise AssertionError("Some gold answers missing")
    scored=[]
    for t in tasks:
        row=index[t["task_id"]]
        em,fscore=answer_metrics(row["answer"],gold[t["case_id"]])
        scored.append({
            "case_id":t["case_id"],"method":t["arm"],
            "gold_answer":gold[t["case_id"]],"model_answer":row["answer"],
            "exact_match":em,"token_f1":round(fscore,6),
            "paid_remote_call":row.get("paid_remote_call",False),
            "reused_from":row.get("reused_from"),
        })
    by=defaultdict(dict)
    for x in scored:by[x["case_id"]][x["method"]]=x
    if len(by)!=NUM_CASES or any(set(x)!=set(ARMS) for x in by.values()):
        raise AssertionError("Not paired")
    summary={}
    for arm in ARMS:
        xs=[x[arm] for x in by.values()]
        summary[arm]={
            "correct":sum(z["exact_match"] for z in xs),
            "n":len(xs),
            "exact_match":round(statistics.mean(z["exact_match"] for z in xs),4),
            "mean_f1":round(statistics.mean(z["token_f1"] for z in xs),4),
        }
    full_correct=[cid for cid,x in by.items() if x["full-source"]["exact_match"]]
    for arm in ("manual-25","msmarco-25"):
        summary[arm]["retained_correct_among_full_correct"]=sum(
            by[cid][arm]["exact_match"] for cid in full_correct)
        summary[arm]["denominator_full_correct"]=len(full_correct)
        summary[arm]["gained_correct_when_full_wrong"]=sum(
            x[arm]["exact_match"] for x in by.values()
            if not x["full-source"]["exact_match"])
    audit={
        "reader":MODEL,"questions":NUM_CASES,
        "method_counts":summary,
        "remote_unique_calls":sum(bool(x.get("paid_remote_call")) for x in saved),
        "reused_identical_prompts":sum(bool(x.get("reused_from")) for x in saved),
        "no_gold_sent_to_remote":True,
        "gold_loaded_after_all_reader_responses":True,
        "limits":"Very small exploratory matched pilot, not a generalizable correctness estimate",
        "scored_rows":scored,
    }
    atomic_write(RESULT,audit)
    print("PILOT_SCORED",json.dumps({
        k:v for k,v in audit.items() if k!="scored_rows"},
        ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--score-only",action="store_true")
    parser.add_argument("--max-new",type=int,default=MAX_REMOTE_CALLS)
    args=parser.parse_args()
    if args.score_only:
        score_local()
    elif args.execute:
        execute(max(1,min(args.max_new,MAX_REMOTE_CALLS)))
        if len(read_rows(LOG))==NUM_CASES*len(ARMS):
            score_local()
    else:
        prepare()
        plan=json.loads(PLAN.read_text(encoding="utf8"))
        print("DRY_RUN",json.dumps({
            "task_count":plan["task_count"],
            "distinct_paid_requests":plan["unique_paid_requests_if_all_succeed"],
            "model":plan["model"],"max_estimated_usd":MAX_ESTIMATED_USD,
            "gold_sent":False},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
