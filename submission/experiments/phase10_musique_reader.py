"""Phase10 bounded and resumable Qwen27B pilot; gold only after complete saved answers."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,statistics,sys,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase7_modal27b_matched8 import (
    APP,infer_once,MODEL,BASE,make_payload,answer_metrics,read_rows,append_fsync,atomic_write)
from submission.experiments.phase10_musique_pilot import OUT,DATA,DATA_SHA256,sha_file
from scipy.stats import binomtest
import numpy as np
ARMS={"pilot":("crossdoc-bridge","bridge-neighbor"),
      "validation":("full-source","atomic-rank","crossdoc-bridge","bridge-neighbor")}
MAX_SESSION_SECONDS=1600
ESTIMATED_USD_PER_HOUR=2.60
MAX_SESSION_EST_USD=1.35

def p(which,name):
    return OUT/(which+"_"+name)

def shat(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()

def tasks(which):
    prereg=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert prereg["source_data_sha256"]==DATA_SHA256
    frozen=json.loads(p(which,"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    ctxpath=p(which,"gold_blind_contexts.jsonl")
    assert sha_file(ctxpath)==frozen["context_file_sha256"]
    cases=[json.loads(x) for x in ctxpath.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in cases]==prereg["samples"][which]
    results=[]
    for item in cases:
        for arm in ARMS[which]:
            if arm=="full-source":txt=item["source_context"]
            else:
                c=item["contexts"][arm]
                assert c["budget"]==item["budget"] and c["tokens"]<=item["budget"]
                txt=c["text"]
            results.append({"task_id":item["id"]+"|"+arm,"case_id":item["id"],
                "arm":arm,"question":item["question"],"context":txt})
    if len(results)!=len(prereg["samples"][which])*len(ARMS[which]):
        raise AssertionError("Incomplete tasks")
    return results

def prepare(which):
    rows=tasks(which)
    f=p(which,"reader_tasks.jsonl")
    frozen=p(which,"READER_PRE_GOLD_PLAN.json")
    if f.exists():
        assert read_rows(f)==rows
    else:
        with f.open("x",encoding="utf8") as out:
            for t in rows:out.write(json.dumps(t,ensure_ascii=False)+"\n")
    digest=sha_file(f)
    if frozen.exists():
        v=json.loads(frozen.read_text(encoding="utf8"))
        assert v["tasks_sha256"]==digest
        assert v["reader_code_sha256"]==sha_file(Path(__file__))
    else:
        atomic_write(frozen,{"phase":which,"model":MODEL,"endpoint":BASE,
            "arms":ARMS[which],"cases":len(rows)//len(ARMS[which]),
            "max_paid_calls":len(rows),"tasks_sha256":digest,
            "reader_code_sha256":sha_file(Path(__file__)),
            "pilot_gate_net_correct":3,"cost_guard_estimated_usd":MAX_SESSION_EST_USD,
            "estimated_usd_hour":ESTIMATED_USD_PER_HOUR,
            "actual_invoice_unknown":True,"gold_sent_to_model":False,
            "same_temperature_prompt_model":True,
            "public_validation_not_hidden_test":True})
    return rows

def execute(which,max_new):
    rows=prepare(which)
    answers=p(which,"remote_answers.jsonl")
    error=p(which,"error.jsonl")
    inflight=p(which,"inflight.json")
    if error.exists() or inflight.exists():
        raise RuntimeError("Prior inflight/error must be audited before resuming")
    saved={}
    allowed={t["task_id"] for t in rows}
    for record in read_rows(answers):
        assert record["task_id"] in allowed and record["task_id"] not in saved and record.get("ok")
        assert record["model"]==MODEL
        saved[record["task_id"]]=record
    paid=0
    start=time.monotonic()
    print("PHASE10_READER_START",which,len(saved),"/",len(rows),flush=True)
    if len(saved)==len(rows):return
    with APP.run():
        for task in rows:
            if task["task_id"] in saved:continue
            prompt_hash=shat(json.dumps(make_payload(task),sort_keys=True,ensure_ascii=False))
            duplicate=next((x for x in saved.values() if x["case_id"]==task["case_id"]
                            and x["prompt_sha256"]==prompt_hash),None)
            if duplicate:
                rec={"task_id":task["task_id"],"case_id":task["case_id"],
                    "arm":task["arm"],"model":MODEL,"ok":True,
                    "prompt_sha256":prompt_hash,"answer":duplicate["answer"],
                    "paid_remote_call":False,"reused_from":duplicate["task_id"]}
                append_fsync(answers,rec);saved[rec["task_id"]]=rec
                continue
            seconds=time.monotonic()-start
            if paid>=max_new or seconds+120>MAX_SESSION_SECONDS or (
                seconds+120)*ESTIMATED_USD_PER_HOUR/3600>MAX_SESSION_EST_USD:
                print("PHASE10_READER_GUARD_STOP",len(saved),len(rows),flush=True)
                break
            atomic_write(inflight,{"task_id":task["task_id"],
                "prompt_sha256":prompt_hash,"started_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                response=infer_once.remote(make_payload(task))
            except BaseException as ex:
                append_fsync(error,{"task_id":task["task_id"],"exception_type":type(ex).__name__})
                print("PHASE10_READER_REMOTE_ERROR",type(ex).__name__,flush=True)
                break
            paid+=1
            if not response.get("ok") or not str(response.get("answer") or "").strip():
                append_fsync(error,{"task_id":task["task_id"],"error_phase":response.get("phase"),
                    "http_status":response.get("http_status"),
                    "exception_type":response.get("exception")})
                print("PHASE10_READER_BAD_ANSWER",task["task_id"],flush=True)
                break
            rec={"task_id":task["task_id"],"case_id":task["case_id"],
                "arm":task["arm"],"model":MODEL,"ok":True,
                "prompt_sha256":prompt_hash,"answer":str(response["answer"]).strip(),
                "paid_remote_call":True,"input_tokens":response.get("input_tokens"),
                "output_tokens":response.get("output_tokens"),
                "elapsed_s":response.get("elapsed_s"),
                "finish_reason":response.get("finish_reason")}
            append_fsync(answers,rec);inflight.unlink()
            saved[rec["task_id"]]=rec
            if len(saved)%8==0 or len(saved)==len(rows):
                print("PHASE10_READER_PROGRESS",which,len(saved),"/",len(rows),
                    "real_calls_this_session",paid,
                    "elapsed",round(time.monotonic()-start,1),flush=True)
    print("PHASE10_READER_SESSION_END",which,len(saved),"/",len(rows),
        "new_remote_calls",paid,"elapsed",round(time.monotonic()-start,1),
        "error",error.exists(),"inflight",inflight.exists(),flush=True)

def paired_interval(values):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261012)
    trials=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return [float(x) for x in np.quantile(trials,[.025,.975])]

def score(which):
    rows=prepare(which)
    answers=p(which,"remote_answers.jsonl")
    saved=read_rows(answers)
    if len(saved)!=len(rows):
        raise RuntimeError("No gold until ALL remote replies are saved")
    responses={x["task_id"]:x for x in saved}
    assert len(responses)==len(rows) and all(x.get("ok") for x in saved)
    assert sha_file(DATA)==DATA_SHA256
    import pyarrow.parquet as pq
    ids={t["case_id"] for t in rows}
    gold={r["id"]:r for r in pq.ParquetFile(DATA).read(
        columns=["id","answer","answer_aliases","answerable"]).to_pylist() if r["id"] in ids}
    assert set(gold)==ids
    scored=[]
    for task in rows:
        label=gold[task["case_id"]]
        pred=responses[task["task_id"]]["answer"]
        aliases=[label["answer"]]+list(label.get("answer_aliases") or [])
        metrics=[answer_metrics(pred,x) for x in aliases if x is not None]
        scored.append({"case_id":task["case_id"],"arm":task["arm"],
            "EM":max(z[0] for z in metrics),
            "F1":round(max(z[1] for z in metrics),6)})
    by=defaultdict(dict)
    for row in scored:by[row["case_id"]][row["arm"]]=row
    assert len(by)==len(ids) and all(set(x)==set(ARMS[which]) for x in by.values())
    report={"study":"Phase10 MuSiQue "+which,"n":len(by),"arms":{},
        "real_modal_calls":sum(bool(x.get("paid_remote_call")) for x in saved),
        "same_prompt_reused":sum(bool(x.get("reused_from")) for x in saved),
        "gold_loaded_only_after_all_answers":True,"gold_sent_remotely":False,
        "per_case_metrics":scored}
    for arm in ARMS[which]:
        x=[r[arm] for r in by.values()]
        report["arms"][arm]={"correct":sum(t["EM"] for t in x),
            "EM":statistics.mean(t["EM"] for t in x),
            "F1":statistics.mean(t["F1"] for t in x)}
    base=[x["crossdoc-bridge"] for x in by.values()]
    new=[x["bridge-neighbor"] for x in by.values()]
    dif=[a["EM"]-b["EM"] for a,b in zip(new,base)]
    dif_f1=[a["F1"]-b["F1"] for a,b in zip(new,base)]
    wins=sum(t>0 for t in dif);losses=sum(t<0 for t in dif)
    report["neighbor_minus_bridge"]={
        "net_correct":wins-losses,"wins":wins,"losses":losses,
        "EM_delta":statistics.mean(dif),
        "EM_ci95":paired_interval(dif),
        "F1_delta":statistics.mean(dif_f1),
        "F1_ci95":paired_interval(dif_f1),
        "exact_mcnemar_two_sided_p":binomtest(min(wins,losses),wins+losses,0.5).pvalue if wins+losses else 1.0}
    if which=="pilot":
        report["validation_gate_passed"]=wins-losses>=3
        report["decision"]=("VALIDATION_ALLOWED" if wins-losses>=3 else "STOP_NO_VALIDATION")
    target=p(which,"READER_SCORED_POST_GOLD.json")
    if target.exists():raise FileExistsError("Already scored")
    atomic_write(target,report)
    print("PHASE10_READER_SCORED",json.dumps({k:v for k,v in report.items()
        if k!="per_case_metrics"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--phase",choices=tuple(ARMS),default="pilot")
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--score",action="store_true")
    ap.add_argument("--max-new",type=int,default=400)
    a=ap.parse_args()
    if a.prepare:
        ts=prepare(a.phase)
        print("PHASE10_READER_PREPARED",a.phase,len(ts),MODEL,flush=True)
    elif a.score:score(a.phase)
    else:execute(a.phase,max(1,min(a.max_new,400)))
