"""Phase29 gold-blind Qwen verifier for fixed original MuSiQue source-sentence candidates."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_semantic_core import payload,parse_ids
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase7_modal27b_matched8 import read_rows,append_fsync,atomic_write
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"

def digest(req): return hashlib.sha256(json.dumps(req,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()

def paths(split):
    sub=OUT/split
    return sub,sub/"semantic_citation_tasks.jsonl",sub/"VERIFIER_PRE_REMOTE_FREEZE.json",sub/"semantic_verifier_responses.jsonl",sub/"semantic_verifier_error.jsonl",sub/"semantic_verifier_inflight.json",sub/"VERIFIER_RESPONSES_FROZEN.json"

def prepare(split):
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    sub,tpath,plan,ans,err,inflight,frozen=paths(split)
    cf=json.loads((sub/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(tpath)==cf["task_sha256"]
    tasks=read_rows(tpath); ids=p["pilot_ids"] if split=="pilot" else p["reserved_ids"]
    assert len(tasks)==len(ids) and [x["id"] for x in tasks]==ids
    spec={"split":split,"n":len(tasks),"task_sha256":sha_file(tpath),
          "source_candidates_sha256":cf["source_sha256"],
          "verifier_code_sha256":sha_file(Path(__file__)),
          "semantic_core_sha256":sha_file(ROOT/"submission/experiments/phase28_semantic_core.py"),
          "relay_sha256":sha_file(ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
          "remote_model":MODEL,"endpoint":BASE,"only_question_and_source_sentences_sent":True,
          "gold_never_sent":True,"max_real_calls":len(tasks)}
    if plan.exists(): assert json.loads(plan.read_text(encoding="utf8"))==spec
    else:
        with plan.open("x",encoding="utf8") as f:
            json.dump(spec,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return tasks

def execute(split,max_new=None):
    tasks=prepare(split); sub,tpath,plan,ans,err,inflight,frozen=paths(split)
    if err.exists() or inflight.exists(): raise RuntimeError("Unresolved remote event; preserve and audit; no retry.")
    rows=read_rows(ans); saved={}
    for r in rows:
        assert r.get("ok") and r["id"] not in saved
        saved[r["id"]]=r
    assert [r["id"] for r in rows]==[t["id"] for t in tasks[:len(rows)]]
    limit=len(tasks) if max_new is None else max(0,min(max_new,len(tasks)))
    print("PHASE29_VERIFIER_START",split,len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks): return
    paid=0;t0=time.monotonic()
    with APP.run():
        for task in tasks:
            if task["id"] in saved: continue
            elapsed=time.monotonic()-t0
            if paid>=limit or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE29_VERIFIER_GUARD",split,len(saved),"/",len(tasks),flush=True);break
            req=payload(task,MODEL); ph=digest(req)
            atomic_write(inflight,{"id":task["id"],"prompt_sha256":ph,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try: result=infer_once.remote(req)
            except BaseException as e:
                append_fsync(err,{"id":task["id"],"error_type":type(e).__name__})
                print("PHASE29_VERIFIER_EXCEPTION",type(e).__name__,flush=True);break
            paid+=1
            if not result.get("ok"):
                append_fsync(err,{"id":task["id"],"phase":result.get("phase"),
                    "http_status":result.get("http_status"),"exception":result.get("exception")})
                print("PHASE29_VERIFIER_ERROR",task["id"],flush=True);break
            raw=str(result.get("answer") or "")
            ids,status=parse_ids(raw,task)
            rec={"id":task["id"],"ok":True,"raw_output":raw,"citation_ids":ids,
                 "parse_status":status,"prompt_sha256":ph,"paid":True,
                 "input_tokens":result.get("input_tokens"),"output_tokens":result.get("output_tokens"),
                 "finish_reason":result.get("finish_reason")}
            append_fsync(ans,rec);inflight.unlink();saved[task["id"]]=rec
            if len(saved)%8==0: print("PHASE29_VERIFIER_PROGRESS",split,len(saved),"/",len(tasks),flush=True)
    print("PHASE29_VERIFIER_SESSION_END",split,len(saved),"/",len(tasks),
          "paid_this_session",paid,"error",err.exists(),flush=True)

def freeze(split):
    tasks=prepare(split); sub,tpath,plan,ans,err,inflight,frozen=paths(split)
    rows=read_rows(ans)
    assert len(rows)==len(tasks) and [r["id"] for r in rows]==[t["id"] for t in tasks]
    assert all(r.get("ok") for r in rows) and not err.exists() and not inflight.exists()
    if frozen.exists(): raise FileExistsError(frozen)
    counts={}
    for r in rows: counts[r["parse_status"]]=counts.get(r["parse_status"],0)+1
    with frozen.open("x",encoding="utf8") as f:
        json.dump({"split":split,"n":len(rows),"sha256":sha_file(ans),
          "tasks_sha256":sha_file(tpath),"pre_remote_sha256":sha_file(plan),
          "parse_status_counts":counts,"gold_unopened":True,
          "source_question_only":True,"paid_calls":sum(bool(r.get("paid")) for r in rows)},
          f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print("PHASE29_VERIFIER_FROZEN",split,json.dumps(counts),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["pilot","validation"],required=True)
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--execute",action="store_true");g.add_argument("--freeze",action="store_true")
    ap.add_argument("--max-new",type=int,default=None);a=ap.parse_args()
    if a.prepare: print("PREPARED",len(prepare(a.split)))
    elif a.execute: execute(a.split,a.max_new)
    else: freeze(a.split)
