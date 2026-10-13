"""Faz27 gold-blind paired Qwen27B reads, SAME question and 25% cap.

Keep all 96 tasks and original answers before gold accessed. No automatic
retry after remote failure, preserve inflight evidence. Full Phase13 relay.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase27_joint_chain_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase7_modal27b_matched8 import (
    make_payload,read_rows,append_fsync,atomic_write)
CONTEXTS=OUT/"gold_blind_paired_contexts.jsonl"
TASKS=OUT/"reader_tasks.jsonl"
PLAN=OUT/"READER_PRE_REMOTE_FREEZE.json"
ANS=OUT/"reader_responses.jsonl"
ERROR=OUT/"reader_error.jsonl"
INFLIGHT=OUT/"reader_inflight.json"
FROZEN=OUT/"READER_RESPONSES_FROZEN.json"
ARMS=("atomic-sentence-rank","joint-title-link-seed")
def digest(req):
    return hashlib.sha256(json.dumps(req,sort_keys=True,
          ensure_ascii=False).encode("utf8")).hexdigest()
def prepare():
    proto=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    f=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(CONTEXTS)==f["sha256"]
    assert f["stats"]["n"]==48 and f["stats"]["cap_violations"]==0
    assert list(ARMS)==proto["arms"]
    cases=read_rows(CONTEXTS)
    assert [x["id"] for x in cases]==proto["pilot_ids"]
    tasks=[{"task_id":x["id"]+"|"+arm,"id":x["id"],
            "arm":arm,"question":x["question"],
            "context":x["contexts"][arm]["text"]}
           for x in cases for arm in ARMS]
    assert len(tasks)==96
    if TASKS.exists():assert read_rows(TASKS)==tasks
    else:
        with TASKS.open("x",encoding="utf8") as out:
            for r in tasks:out.write(json.dumps(r,ensure_ascii=False)+"\n")
    spec={"n":96,"questions":48,"arms":list(ARMS),
       "contexts_sha256":f["sha256"],"tasks_sha256":sha_file(TASKS),
       "reader_code_sha256":sha_file(Path(__file__)),
       "payload_sha256":sha_file(ROOT/"submission/experiments/phase7_modal27b_matched8.py"),
       "relay_sha256":sha_file(ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
       "baseline_and_treatment_same_reader":True,
       "no_answer_support_gold_loaded":True,
       "reader_token_generation_budget_equal":True,
       "max_paid_calls":96}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks
def execute(max_new=96):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():
        raise RuntimeError("Unresolved remote event; inspect and preserve; never skip")
    rows=read_rows(ANS)
    saved={}
    for r in rows:
        assert r.get("ok") and r["task_id"] not in saved
        saved[r["task_id"]]=r
    assert [r["task_id"] for r in rows]==[t["task_id"] for t in tasks[:len(rows)]]
    print("PHASE27_READER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    paid=0;t0=time.monotonic()
    with APP.run():
        for t in tasks:
            if t["task_id"] in saved:continue
            elapsed=time.monotonic()-t0
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE27_READER_GUARD",len(saved),"/",len(tasks),flush=True);break
            request=make_payload(t)
            ph=digest(request)
            atomic_write(INFLIGHT,{"task_id":t["task_id"],"id":t["id"],
                       "prompt_sha256":ph,
                       "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:result=infer_once.remote(request)
            except BaseException as e:
                append_fsync(ERROR,{"task_id":t["task_id"],
                                  "error_type":type(e).__name__})
                print("PHASE27_READER_EXCEPTION",type(e).__name__,flush=True);break
            paid+=1
            if not result.get("ok"):
                append_fsync(ERROR,{"task_id":t["task_id"],
                   "phase":result.get("phase"),
                   "http_status":result.get("http_status"),
                   "exception":result.get("exception")})
                print("PHASE27_READER_ERROR",t["task_id"],flush=True);break
            rec={"task_id":t["task_id"],"id":t["id"],"arm":t["arm"],
                 "ok":True,"answer":str(result.get("answer") or "").strip(),
                 "prompt_sha256":ph,"paid":True,
                 "input_tokens":result.get("input_tokens"),
                 "output_tokens":result.get("output_tokens"),
                 "finish_reason":result.get("finish_reason")}
            append_fsync(ANS,rec)
            INFLIGHT.unlink()
            saved[t["task_id"]]=rec
            if len(saved)%12==0:
                print("PHASE27_READER_PROGRESS",len(saved),"/",len(tasks),flush=True)
    print("PHASE27_READER_SESSION_END",len(saved),"/",len(tasks),flush=True)
def freeze():
    tasks=prepare();rows=read_rows(ANS)
    assert len(tasks)==len(rows)==96
    assert [t["task_id"] for t in tasks]==[r["task_id"] for r in rows]
    assert all(r["ok"] for r in rows)
    assert not INFLIGHT.exists() and not ERROR.exists()
    if FROZEN.exists():raise FileExistsError(FROZEN)
    new_json(FROZEN,{"n":96,"question_n":48,
       "responses_sha256":sha_file(ANS),
       "task_sha256":sha_file(TASKS),
       "pre_remote_sha256":sha_file(PLAN),
       "empty_predictions":sum(not r["answer"] for r in rows),
       "no_gold_seen":True,
       "only_predeclared_48_ids":True})
    print("PHASE27_READER_FROZEN",len(rows),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true")
    g.add_argument("--execute",action="store_true")
    g.add_argument("--freeze",action="store_true")
    p.add_argument("--max-new",type=int,default=96)
    a=p.parse_args()
    if a.prepare:print("PREPARED",len(prepare()))
    elif a.execute:execute(a.max_new)
    else:freeze()
