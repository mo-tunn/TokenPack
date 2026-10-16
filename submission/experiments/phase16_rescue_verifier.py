"""Conditional Phase16 second Qwen verification ONLY for first-stage abstained edges.
Top4 first pass fixed. Four NEW ranked source candidates per missing edge.
No gold source read; no extra model call for cases with no missing edges.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase16_register import OUT
from submission.experiments.phase15_protocol import MAX_RESCUE_VERIFY
from submission.experiments.phase16_initial_verifier import (
    APP,infer_once,MODEL,BASE,model_payload,verify_parse)
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,append_fsync,atomic_write)
SCORED=OUT/"scored_candidate_pools.jsonl"
INITIAL=OUT/"verifier_responses.jsonl"
OUTTASKS=OUT/"rescue_verifier_inputs.jsonl"
PLAN=OUT/"RESCUE_VERIFIER_PRE_REMOTE_FREEZE.json"
RESP=OUT/"rescue_verifier_responses.jsonl"
ERROR=OUT/"rescue_verifier_error.jsonl"
INFLIGHT=OUT/"rescue_verifier_inflight.json"
FROZEN=OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def payloads():
    frozen=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    candidate=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    assert frozen["responses_sha256"]==sha_file(INITIAL)
    assert candidate["candidate_sha256"]==sha_file(SCORED)
    rows=read_rows(SCORED);previous=read_rows(INITIAL)
    assert len(rows)==len(previous)==32
    tasks=[]
    for c,prev in zip(rows,previous):
        assert c["id"]==prev["id"] and prev["ok"]
        edges=[e for b in c["branches"] for e in b["edges"]]
        ids={f"e{i}" for i in range(len(edges))}
        assert set(prev["decision_ids"])==ids
        target=[]
        for j,edge in enumerate(edges):
            edgeid=f"e{j}"
            if prev["decision_ids"][edgeid] is not None:continue
            expansion=c["expanded_pools"][j]
            if not expansion:continue
            target.append({
                "edge_id":edgeid,"subject":edge["subject"],
                "relation":edge["relation"],"object":edge["object"],
                "retrieval_query":edge["query"],
                "candidates":[{
                    "candidate_id":p["candidate_id"],
                    "document_title":p["title"],
                    "original_sentence_id":p["sentence_id"],
                    "original_sentence":p["sentence"]}
                    for p in expansion]})
        tasks.append({"id":c["id"],"initial_decision_ids":prev["decision_ids"],
             "rescue_input":{"question":c["question"],
                 "hypothesized_edges_with_source_candidates":target}})
    return tasks
def prepare():
    tasks=payloads()
    if OUTTASKS.exists():assert read_rows(OUTTASKS)==tasks
    else:
        with OUTTASKS.open("x",encoding="utf8") as f:
            for t in tasks:f.write(json.dumps(t,ensure_ascii=False)+"\n")
    rec={"n":32,
         "targeted_cases":sum(bool(t["rescue_input"]["hypothesized_edges_with_source_candidates"]) for t in tasks),
         "inputs_sha256":sha_file(OUTTASKS),
         "initial_verifier_sha256":sha_file(INITIAL),
         "expanded_pools_sha256":sha_file(SCORED),
         "code_sha256":sha_file(Path(__file__)),
         "uses_same_phase14_verifier":True,
         "gold_unused":True,
         "max_paid_calls":MAX_RESCUE_VERIFY}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==rec
    else:new_json(PLAN,rec)
    return tasks

def execute(max_new=MAX_RESCUE_VERIFY):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():raise RuntimeError("Remote incident audit required")
    saved={}
    for r in read_rows(RESP):
        assert r["id"] not in saved and r.get("ok")
        saved[r["id"]]=r
    print("PHASE16_RESCUE_START",len(saved),"/",len(tasks),flush=True)
    paid=0;started=time.monotonic()
    with APP.run():
        for t in tasks:
            if t["id"] in saved:continue
            edges=t["rescue_input"]["hypothesized_edges_with_source_candidates"]
            if not edges:
                rec={"id":t["id"],"ok":True,"rescue_decisions":{},
                    "status":"no_abstained_edge","paid_remote_call":False}
                append_fsync(RESP,rec);saved[rec["id"]]=rec
                continue
            elapsed=time.monotonic()-started
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE16_RESCUE_GUARD",len(saved),"/",len(tasks),flush=True);break
            req=model_payload(t["rescue_input"])
            digest=sha(json.dumps(req,sort_keys=True,ensure_ascii=False))
            atomic_write(INFLIGHT,{"id":t["id"],"prompt_sha256":digest,
                "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:res=infer_once.remote(req)
            except BaseException as ex:
                append_fsync(ERROR,{"id":t["id"],"exception_type":type(ex).__name__})
                print("PHASE16_RESCUE_EXCEPTION",type(ex).__name__,flush=True);break
            paid+=1
            if not res.get("ok"):
                append_fsync(ERROR,{"id":t["id"],"phase":res.get("phase"),
                    "http_status":res.get("http_status"),"exception":res.get("exception")})
                print("PHASE16_RESCUE_ENDPOINT_FAILURE",t["id"],flush=True);break
            raw=str(res.get("answer") or "")
            decisions,status=verify_parse(raw,t["rescue_input"])
            rec={"id":t["id"],"ok":True,"rescue_decisions":decisions,
                "status":status,"raw_output":raw,"prompt_sha256":digest,
                "paid_remote_call":True,"model":MODEL}
            append_fsync(RESP,rec);INFLIGHT.unlink();saved[rec["id"]]=rec
            if len(saved)%8==0:
                print("PHASE16_RESCUE_PROGRESS",len(saved),"/",len(tasks),
                      "paid",paid,flush=True)
    print("PHASE16_RESCUE_END",len(saved),"/",len(tasks),"paid",paid,flush=True)

def freeze():
    rows=read_rows(RESP);tasks=prepare()
    assert len(rows)==len(tasks)==32
    assert [x["id"] for x in rows]==[x["id"] for x in tasks]
    if FROZEN.exists():raise FileExistsError(FROZEN)
    new_json(FROZEN,{"n":32,"responses_sha256":sha_file(RESP),
        "task_sha256":sha_file(OUTTASKS),
        "real_model_calls":sum(bool(x["paid_remote_call"]) for x in rows),
        "new_positive_edges":sum(sum(v is not None for v in x["rescue_decisions"].values())
                                 for x in rows),
        "invalid_rescue_json":sum(x["status"]=="invalid_verification_all_abstain" for x in rows),
        "no_gold_read":True,
        "source_candidates_original":True})
    print("PHASE16_RESCUE_FROZEN",len(rows),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    o=ap.parse_args()
    if o.prepare:print("PHASE16_RESCUE_PREPARED",len(prepare()),flush=True)
    elif o.freeze:freeze()
    else:execute()
