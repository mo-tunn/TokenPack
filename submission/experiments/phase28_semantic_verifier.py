"""Phase28 Qwen source-only multi-hop sentence citation check.

48 fixed original-sentence tasks; records every raw judgement incl abstain.
No supporting_facts, answer or other gold fields accessible to script.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_semantic_register import OUT
from submission.experiments.phase28_semantic_core import payload,parse_ids,SYSTEM
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase7_modal27b_matched8 import read_rows,append_fsync,atomic_write
TASKS=OUT/"semantic_citation_tasks.jsonl"
PLAN=OUT/"VERIFIER_PRE_REMOTE_FREEZE.json"
ANS=OUT/"semantic_verifier_responses.jsonl"
ERROR=OUT/"semantic_verifier_error.jsonl"
INFLIGHT=OUT/"semantic_verifier_inflight.json"
FROZEN=OUT/"VERIFIER_RESPONSES_FROZEN.json"
def digest(req):return hashlib.sha256(json.dumps(req,
    sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def prepare():
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    m=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(TASKS)==m["task_sha256"]
    tasks=read_rows(TASKS)
    assert len(tasks)==48 and [x["id"] for x in tasks]==p["pilot_ids"]
    spec={"n":48,"task_sha256":sha_file(TASKS),
       "source_candidates_sha256":m["source_sha256"],
       "reader_code_sha256":sha_file(Path(__file__)),
       "semantic_core_sha256":sha_file(ROOT/"submission/experiments/phase28_semantic_core.py"),
       "relay_sha256":sha_file(ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
       "remote_model":MODEL,"endpoint":BASE,
       "only_source_and_question_sent":True,"never_gold":True,
       "max_real_calls":48}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks
def execute(max_new=48):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved remote request, preserve and audit")
    history=read_rows(ANS)
    assert [x["id"] for x in history]==[x["id"] for x in tasks[:len(history)]]
    assert all(x["ok"] for x in history)
    saved={x["id"] for x in history}
    print("PHASE28_VERIFIER_START",len(saved),48,flush=True)
    paid=0;t0=time.monotonic()
    if len(saved)==len(tasks):return
    with APP.run():
        for task in tasks:
            if task["id"] in saved:continue
            elapsed=time.monotonic()-t0
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE28_VERIFIER_GUARD",len(saved),48,flush=True);break
            req=payload(task,MODEL)
            h=digest(req)
            atomic_write(INFLIGHT,{"id":task["id"],"prompt_sha256":h,
                   "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:r=infer_once.remote(req)
            except BaseException as exc:
                append_fsync(ERROR,{"id":task["id"],"error_type":type(exc).__name__})
                print("PHASE28_VERIFIER_EXCEPTION",type(exc).__name__,flush=True);break
            paid+=1
            if not r.get("ok"):
                append_fsync(ERROR,{"id":task["id"],"phase":r.get("phase"),
                    "http_status":r.get("http_status"),
                    "exception":r.get("exception")})
                print("PHASE28_VERIFIER_ENDPOINT_ERROR",task["id"],flush=True);break
            raw=str(r.get("answer") or "")
            ids,status=parse_ids(raw,task)
            rec={"id":task["id"],"ok":True,
                  "raw_output":raw,"citation_ids":ids,"parse_status":status,
                  "prompt_sha256":h,"paid":True,
                  "input_tokens":r.get("input_tokens"),"output_tokens":r.get("output_tokens"),
                  "finish_reason":r.get("finish_reason")}
            append_fsync(ANS,rec)
            INFLIGHT.unlink()
            saved.add(task["id"])
            if len(saved)%8==0:
                print("PHASE28_VERIFIER_PROGRESS",len(saved),48,flush=True)
    print("PHASE28_VERIFIER_SESSION_END",len(saved),48,flush=True)
def freeze():
    tasks=prepare();rows=read_rows(ANS)
    assert len(rows)==len(tasks)==48
    assert [r["id"] for r in rows]==[t["id"] for t in tasks]
    assert all(r["ok"] for r in rows)
    assert not ERROR.exists() and not INFLIGHT.exists()
    if FROZEN.exists():raise FileExistsError(FROZEN)
    status={}
    for row in rows:status[row["parse_status"]]=status.get(row["parse_status"],0)+1
    new_json(FROZEN,{"n":48,
       "sha256":sha_file(ANS),
       "tasks_sha256":sha_file(TASKS),
       "pre_remote_sha256":sha_file(PLAN),
       "parse_status_counts":status,
       "gold_unopened":True,"source_citation_only_no_gold":True})
    print("PHASE28_VERIFIER_FROZEN",json.dumps(status),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    p.add_argument("--max-new",type=int,default=48)
    a=p.parse_args()
    if a.prepare:print("PHASE28_VERIFIER_PREPARED",len(prepare()))
    elif a.execute:execute(a.max_new)
    else:freeze()
