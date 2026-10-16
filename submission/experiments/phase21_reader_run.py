"""Faz21 32x2 matched Qwen27B answer reader. Gold only after all 96 replies.

Both arms have byte-identical Phase14 edge-verified source context; only reader policy changes. Blank-but-successful answer is scored literally, not retried.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,statistics,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase21_reader_register import OUT
from submission.experiments.phase21_reader_protocol import (
    ARMS as METHODS,GATE_NET_EM as PILOT_GATE_EM_NET,
    MAX_READER_CALLS as MAX_REMOTE_READER,
    EVIDENCE_FIRST_SYSTEM,EVIDENCE_FIRST_USER_TEMPLATE,STRUCTURED_MAX_TOKENS
)
from submission.experiments.phase12_register import (
    DATA,verify_data,sha_file,new_json
)
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase7_modal27b_matched8 import (
    make_payload as direct_payload,
    answer_metrics,read_rows,append_fsync,atomic_write
)
import pyarrow.parquet as pq
TASKS=OUT/"reader_tasks.jsonl"
PLAN=OUT/"READER_PRE_REMOTE_FREEZE.json"
ANS=OUT/"reader_responses.jsonl"
INFLIGHT=OUT/"reader_inflight.json"
ERROR=OUT/"reader_error.jsonl"
SCORED=OUT/"READER_POST_GOLD_RESULTS.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def make_payload(task):
    if task["arm"]=="verified-direct":
        return direct_payload(task)
    assert task["arm"]=="verified-evidence-first"
    return {"model":MODEL,
        "messages":[
            {"role":"system","content":EVIDENCE_FIRST_SYSTEM},
            {"role":"user","content":EVIDENCE_FIRST_USER_TEMPLATE.format(
                context=task["context"],question=task["question"])}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,"seed":20261012,
        "max_tokens":STRUCTURED_MAX_TOKENS,"reasoning_effort":"none",
        "stream":False}

def parse_prediction(raw,task):
    if task["arm"]=="verified-direct":
        return raw.strip(),"direct_literal",[]
    text=str(raw or "").strip()
    if text.startswith("```"):
        text="\\n".join(text.splitlines()[1:]).split("```")[0].strip()
    if "{" in text and "}" in text:
        text=text[text.index("{"):text.rindex("}")+1]
    try:
        data=json.loads(text)
        answer=data["answer"];steps=data["steps"]
        if not isinstance(answer,str) or not isinstance(steps,list) or len(steps)>4:
            raise ValueError("Bad schema")
        out=[]
        for item in steps:
            if not isinstance(item,dict) or not isinstance(item.get("fact"),str) or not isinstance(item.get("quote"),str):
                raise ValueError("Bad step")
            quote=item["quote"].strip()
            out.append({"fact":item["fact"],"quote":quote,
                "quote_exact_in_context":bool(quote and quote in task["context"])})
        return answer.strip(),"valid_json",out
    except (ValueError,KeyError,TypeError,json.JSONDecodeError):
        return "","invalid_json_no_answer",[]


def prepare():
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    freeze=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    file=OUT/"gold_blind_contexts.jsonl"
    assert sha_file(file)==freeze["sha256"]
    rows=[json.loads(x) for x in file.open(encoding="utf8") if x.strip()]
    assert [r["id"] for r in rows]==protocol["pilot_ids"]
    tasks=[]
    for case in rows:
        for arm in METHODS:
            ctx=case["contexts"][arm]
            assert ctx["tokens"]<=ctx["budget"]==case["budget"]
            tasks.append({"task_id":case["id"]+"|"+arm,
                "id":case["id"],"arm":arm,
                "question":case["question"],"context":ctx["text"]})
    assert len(tasks)==32*2
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as f:
            for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+"\n")
    record={"n":len(tasks),"question_n":32,"arms":list(METHODS),
            "model":MODEL,"endpoint":BASE,
            "tasks_sha256":sha_file(TASKS),
            "context_sha256":freeze["sha256"],
            "source_code_sha256":sha_file(Path(__file__)),
            "reader_prompt_code_sha256":sha_file(
                ROOT/"submission/experiments/phase7_modal27b_matched8.py"),
            "full_length_modal_relay_code_sha256":sha_file(
                ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
            "transport_character_cap":4000,
            "phase20_300_character_relay_failure_not_reused":True,
            "max_new_paid_calls":MAX_REMOTE_READER,
            "max_session_seconds":1800,"estimate_usd_guard":1.35,
            "actual_invoice_not_verified":True,
            "successful_empty_response":"score literal blank, never drop case",
            "gold_only_opened_when_all_replies_saved":True,
            "gold_never_sent_to_cloud":True}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==record
    else:new_json(PLAN,record)
    return tasks

def execute(max_new):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved error/inflight; preserve and audit")
    allowed={t["task_id"] for t in tasks}
    saved={}
    for r in read_rows(ANS):
        assert r["task_id"] in allowed and r["task_id"] not in saved and r.get("ok")
        saved[r["task_id"]]=r
    print("PHASE21_READER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    paid=0;start=time.monotonic()
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved:continue
            request=make_payload(task)
            digest=sha(json.dumps(request,sort_keys=True,ensure_ascii=False))
            reuse=next((r for r in saved.values() if
                r["id"]==task["id"] and r["prompt_sha256"]==digest),None)
            if reuse:
                rec={"task_id":task["task_id"],"id":task["id"],"arm":task["arm"],
                    "ok":True,"answer":reuse["answer"],"prompt_sha256":digest,
                    "paid":False,"reused_from":reuse["task_id"]}
                append_fsync(ANS,rec);saved[rec["task_id"]]=rec
                continue
            elapsed=time.monotonic()-start
            if (paid>=max_new or elapsed+120>=1800
                or (elapsed+120)*2.60/3600>=1.35):
                print("PHASE21_READER_SESSION_GUARD",len(saved),"/",len(tasks),flush=True);break
            atomic_write(INFLIGHT,{"task_id":task["task_id"],
                "prompt_sha256":digest,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:answer=infer_once.remote(request)
            except BaseException as exc:
                append_fsync(ERROR,{"task_id":task["task_id"],
                    "exception_type":type(exc).__name__})
                print("PHASE21_READER_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not answer.get("ok"):
                append_fsync(ERROR,{"task_id":task["task_id"],
                    "phase":answer.get("phase"),"http_status":answer.get("http_status"),
                    "exception":answer.get("exception")})
                print("PHASE21_READER_REMOTE_FAILURE",task["task_id"],flush=True);break
            raw_prediction=str(answer.get("answer") or "").strip()
            prediction,parse_status,steps=parse_prediction(raw_prediction,task)
            rec={"task_id":task["task_id"],"id":task["id"],"arm":task["arm"],
                "ok":True,"answer":prediction,"empty_answer":not bool(prediction),
                "raw_reader_output":raw_prediction,"parse_status":parse_status,
                "source_citation_steps":steps,
                "paid":True,"prompt_sha256":digest,
                "input_tokens":answer.get("input_tokens"),
                "output_tokens":answer.get("output_tokens"),
                "finish_reason":answer.get("finish_reason")}
            append_fsync(ANS,rec);INFLIGHT.unlink()
            saved[rec["task_id"]]=rec
            if len(saved)%12==0 or len(saved)==len(tasks):
                print("PHASE21_READER_PROGRESS",len(saved),"/",len(tasks),
                    "new_remote",paid,"elapsed_s",round(time.monotonic()-start,1),flush=True)
    print("PHASE21_READER_SESSION_END",len(saved),"/",len(tasks),
        "paid",paid,"error",ERROR.exists(),flush=True)

def ci(values):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261012)
    draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws,[.025,.975])]

def paired(by,first,second):
    pairs=list(by.values())
    delta=[r[first]["EM"]-r[second]["EM"] for r in pairs]
    f1=[r[first]["F1"]-r[second]["F1"] for r in pairs]
    wins=sum(x==1 for x in delta);losses=sum(x==-1 for x in delta)
    return {"method":first+" minus "+second,
        "EM_delta":statistics.mean(delta),"EM_CI95":ci(delta),
        "F1_delta":statistics.mean(f1),"F1_CI95":ci(f1),
        "first_only_correct":wins,"second_only_correct":losses,
        "net_correct":wins-losses,
        "exact_mcnemar_p":binomtest(min(wins,losses),wins+losses,.5).pvalue
            if wins+losses else 1.0}

def score():
    tasks=prepare()
    log=read_rows(ANS)
    if len(log)!=len(tasks):raise RuntimeError("Incomplete, gold unopened")
    index={r["task_id"]:r for r in log}
    assert len(index)==len(tasks) and all(r.get("ok") for r in log)
    verify_data()
    ids={t["id"] for t in tasks}
    gold={r["id"]:r["answer"] for r in pq.ParquetFile(DATA).read(
        columns=["id","answer"]).to_pylist() if r["id"] in ids}
    assert set(gold)==ids
    by=defaultdict(dict)
    for t in tasks:
        prediction=index[t["task_id"]]["answer"]
        em,f1=answer_metrics(prediction,gold[t["id"]])
        by[t["id"]][t["arm"]]={"EM":em,"F1":f1}
    assert len(by)==32 and all(set(r)==set(METHODS) for r in by.values())
    arms={}
    for arm in METHODS:
        x=[r[arm] for r in by.values()]
        arms[arm]={"correct":sum(r["EM"] for r in x),"n":len(x),
                   "EM":statistics.mean(r["EM"] for r in x),
                   "F1":statistics.mean(r["F1"] for r in x)}
    primary=paired(by,"verified-evidence-first","verified-direct")
    secondary=primary
    result={"dataset":"2WikiMultiHopQA public validation; 32 reserved new Phase21 disjoint IDs",
        "n":32,"arms":arms,
        "primary_evidence_first_minus_direct":primary,
        "secondary_same_as_primary":secondary,
        "pilot_gate_net_correct":PILOT_GATE_EM_NET,
        "pilot_gate_passed":primary["net_correct"]>=PILOT_GATE_EM_NET,
        "decision":"GO_VALIDATION" if primary["net_correct"]>=PILOT_GATE_EM_NET else "STOP_NO_VALIDATION",
        "remote_calls":sum(bool(r.get("paid")) for r in log),
        "same_payload_reused":sum(bool(r.get("reused_from")) for r in log),
        "empty_predictions":sum(bool(r.get("empty_answer")) for r in log),
        "structured_valid_json":sum(r.get("parse_status")=="valid_json" for r in log
                    if r["arm"]=="verified-evidence-first"),
        "structured_invalid_json":sum(r.get("parse_status")=="invalid_json_no_answer" for r in log
                    if r["arm"]=="verified-evidence-first"),
        "source_quote_exact_rate":(
            sum(step["quote_exact_in_context"] for r in log
                if r["arm"]=="verified-evidence-first"
                for step in r.get("source_citation_steps",[]))/
            max(1,sum(len(r.get("source_citation_steps",[])) for r in log
                if r["arm"]=="verified-evidence-first"))),
        "zero_gold_in_remote_input":True,
        "responses_sha256":sha_file(ANS),
        "all_responses_saved_before_gold":True,"gold_sent_to_modal":False,
        "public_validation_not_test":True,
        "per_case":[{"id":id,**v} for id,v in by.items()]}
    if SCORED.exists():raise FileExistsError(SCORED)
    new_json(SCORED,result)
    print("PHASE21_READER_SCORED",json.dumps({k:v for k,v in result.items()
        if k!="per_case"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--execute",action="store_true")
    g.add_argument("--score",action="store_true")
    p.add_argument("--max-new",type=int,default=64)
    args=p.parse_args()
    if args.prepare:print("PHASE21_READER_PREPARED",len(prepare()),flush=True)
    elif args.score:score()
    else:execute(max(1,min(args.max_new,MAX_REMOTE_READER)))
