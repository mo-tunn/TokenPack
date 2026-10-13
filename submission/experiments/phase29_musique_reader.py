"""Gold-blind paired Qwen27B short-answer reads for Phase29."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase13_planner import APP,infer_once,MODEL,BASE
from submission.experiments.phase7_modal27b_matched8 import make_payload,read_rows,append_fsync,atomic_write
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
CONTEXTS_NAME="gold_blind_paired_contexts.jsonl"
ARMS=("atomic-sentence-rank","semantic-joint-citations")
def digest(req):return hashlib.sha256(json.dumps(req,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def paths(split):
    sub=OUT/split
    return sub,sub/CONTEXTS_NAME,sub/"reader_tasks.jsonl",sub/"READER_PRE_REMOTE_FREEZE.json",sub/"reader_responses.jsonl",sub/"reader_error.jsonl",sub/"reader_inflight.json",sub/"READER_RESPONSES_FROZEN.json"
def prepare(split):
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    sub,contexts,tasks_path,plan,ans,err,inflight,frozen=paths(split)
    c=json.loads((sub/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(contexts)==c["sha256"] and c["stats"]["token_cap_violations"]==0
    rows=read_rows(contexts);ids=p["pilot_ids"] if split=="pilot" else p["reserved_ids"]
    assert len(rows)==len(ids) and [x["id"] for x in rows]==ids
    tasks=[{"task_id":r["id"]+"|"+arm,"id":r["id"],"arm":arm,
            "question":r["question"],"context":r["contexts"][arm]["text"]}
           for r in rows for arm in ARMS]
    assert len(tasks)==2*len(ids)
    if tasks_path.exists():assert read_rows(tasks_path)==tasks
    else:
        with tasks_path.open("x",encoding="utf8") as f:
            for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+"\n")
    spec={"split":split,"n":len(tasks),"question_n":len(ids),"arms":list(ARMS),
      "contexts_sha256":c["sha256"],"tasks_sha256":sha_file(tasks_path),
      "reader_code_sha256":sha_file(Path(__file__)),
      "payload_sha256":sha_file(ROOT/"submission/experiments/phase7_modal27b_matched8.py"),
      "relay_sha256":sha_file(ROOT/"submission/experiments/phase13_graph_modal_relay_source.py"),
      "baseline_and_treatment_same_reader":True,"no_answer_or_support_gold_loaded":True,
      "reader_generation_budget_equal":True,"max_paid_calls":len(tasks)}
    if plan.exists():assert json.loads(plan.read_text(encoding="utf8"))==spec
    else:
        with plan.open("x",encoding="utf8") as f:json.dump(spec,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    return tasks
def execute(split,max_new=None):
    tasks=prepare(split);sub,contexts,tasks_path,plan,ans,err,inflight,frozen=paths(split)
    gate=json.loads((sub/"PRE_GOLD_STRUCTURAL_DECISION.json").read_text(encoding="utf8"))
    assert gate["decision"]=="READER_COMPARISON_PERMITTED_NOT_YET_SCORED"
    if err.exists() or inflight.exists():raise RuntimeError("Unresolved remote event; preserve evidence; no automatic retry.")
    rows=read_rows(ans);saved={}
    for r in rows:
        assert r.get("ok") and r["task_id"] not in saved;saved[r["task_id"]]=r
    assert [r["task_id"] for r in rows]==[t["task_id"] for t in tasks[:len(rows)]]
    by_prompt={}
    for r in rows:by_prompt.setdefault(r["prompt_sha256"],r)
    print("PHASE29_READER_START",split,len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    limit=len(tasks) if max_new is None else max(0,min(max_new,len(tasks)))
    paid=0;t0=time.monotonic()
    with APP.run():
        for task in tasks:
            if task["task_id"] in saved:continue
            elapsed=time.monotonic()-t0
            if paid>=limit or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE29_READER_GUARD",split,len(saved),"/",len(tasks),flush=True);break
            request=make_payload(task);ph=digest(request)
            if ph in by_prompt:
                src=by_prompt[ph]
                rec={"task_id":task["task_id"],"id":task["id"],"arm":task["arm"],
                    "ok":True,"answer":src["answer"],"prompt_sha256":ph,"paid":False,
                    "reused_identical_prompt_from":src["task_id"],
                    "input_tokens":src.get("input_tokens"),"output_tokens":src.get("output_tokens"),
                    "finish_reason":src.get("finish_reason")}
                append_fsync(ans,rec);saved[task["task_id"]]=rec
                continue
            atomic_write(inflight,{"task_id":task["task_id"],"id":task["id"],
                "prompt_sha256":ph,"utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:r=infer_once.remote(request)
            except BaseException as e:
                append_fsync(err,{"task_id":task["task_id"],"error_type":type(e).__name__})
                print("PHASE29_READER_EXCEPTION",type(e).__name__,flush=True);break
            paid+=1
            if not r.get("ok"):
                append_fsync(err,{"task_id":task["task_id"],"phase":r.get("phase"),
                    "http_status":r.get("http_status"),"exception":r.get("exception")})
                print("PHASE29_READER_ERROR",task["task_id"],flush=True);break
            rec={"task_id":task["task_id"],"id":task["id"],"arm":task["arm"],"ok":True,
                "answer":str(r.get("answer") or "").strip(),"prompt_sha256":ph,"paid":True,
                "input_tokens":r.get("input_tokens"),"output_tokens":r.get("output_tokens"),
                "finish_reason":r.get("finish_reason")}
            append_fsync(ans,rec);inflight.unlink();saved[task["task_id"]]=rec;by_prompt[ph]=rec
            if len(saved)%12==0:print("PHASE29_READER_PROGRESS",split,len(saved),"/",len(tasks),"paid_this_session",paid,flush=True)
    print("PHASE29_READER_SESSION_END",split,len(saved),"/",len(tasks),"paid",paid,flush=True)
def freeze(split):
    tasks=prepare(split);sub,contexts,tasks_path,plan,ans,err,inflight,frozen=paths(split)
    rows=read_rows(ans)
    assert len(rows)==len(tasks) and [x["task_id"] for x in rows]==[x["task_id"] for x in tasks]
    assert all(x.get("ok") for x in rows) and not err.exists() and not inflight.exists()
    assert all(x.get("paid") or x.get("reused_identical_prompt_from") for x in rows)
    if frozen.exists():raise FileExistsError(frozen)
    with frozen.open("x",encoding="utf8") as f:
        json.dump({"split":split,"n":len(rows),"question_n":len(rows)//2,
            "responses_sha256":sha_file(ans),"task_sha256":sha_file(tasks_path),
            "pre_remote_sha256":sha_file(plan),"empty_predictions":sum(not x["answer"] for x in rows),
            "actual_paid_reader_calls":sum(bool(x.get("paid")) for x in rows),
            "identical_prompt_reuse_n":sum(not x.get("paid") for x in rows),
            "no_gold_seen":True},f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print("PHASE29_READER_FROZEN",split,len(rows),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["pilot","validation"],required=True)
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--execute",action="store_true");g.add_argument("--freeze",action="store_true")
    ap.add_argument("--max-new",type=int,default=None);a=ap.parse_args()
    if a.prepare:print("PREPARED",len(prepare(a.split)))
    elif a.execute:execute(a.split,a.max_new)
    else:freeze(a.split)
