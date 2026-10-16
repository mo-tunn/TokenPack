"""Bounded Phase21 Qwen evidence verifier: one call per question, edge-wise
yes/no support over 4 original candidate sentences. Source text allowed, gold
answer/supports/relations labels never accessed or sent.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase21_reader_register import OUT
from submission.experiments.phase14_protocol import (
    EDGE_TOP_K,VERIFIER_SYSTEM,MAX_VERIFIER_OUTPUT_TOKENS)
from submission.experiments.phase21_reader_protocol import MAX_VERIFIER_CALLS
from submission.experiments.phase13_planner import (
    APP,infer_once,MODEL,BASE)
from submission.experiments.phase12_register import (
    sha_file,new_json)
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,append_fsync,atomic_write
)
POOLS=OUT/"scored_candidate_pools.jsonl"
INPUTS=OUT/"verifier_inputs.jsonl"
PLAN=OUT/"VERIFIER_PRE_REMOTE_FREEZE.json"
RESP=OUT/"verifier_responses.jsonl"
ERROR=OUT/"verifier_error.jsonl"
INFLIGHT=OUT/"verifier_inflight.json"
FROZEN=OUT/"VERIFIER_RESPONSES_FROZEN.json"
def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()

def prompts_for_row(case):
    edges=[e for b in case["branches"] for e in b["edges"]]
    assert len(edges)==len(case["candidate_pools"])
    blocks=[]
    for i,(edge,pool) in enumerate(zip(edges,case["candidate_pools"])):
        sub={"edge_id":f"e{i}","subject":edge["subject"],
             "relation":edge["relation"],"object":edge["object"],
             "retrieval_query":edge["query"],
             "candidates":[{"candidate_id":x["candidate_id"],
                            "document_title":x["title"],
                            "original_sentence_id":x["sentence_id"],
                            "original_sentence":x["sentence"]}
                            for x in pool]}
        assert 1<=len(pool)<=EDGE_TOP_K
        blocks.append(sub)
    return {
      "question":case["question"],
      "hypothesized_edges_with_source_candidates":blocks,
    }

def model_payload(obj):
    return {"model":MODEL,"messages":[
        {"role":"system","content":VERIFIER_SYSTEM},
        {"role":"user","content":json.dumps(obj,ensure_ascii=False,separators=(",",":"))}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,
        "seed":20261012,"max_tokens":MAX_VERIFIER_OUTPUT_TOKENS,
        "reasoning_effort":"none","stream":False}

def verify_parse(raw,request_obj):
    edges=request_obj["hypothesized_edges_with_source_candidates"]
    allowed={v["edge_id"]:{z["candidate_id"] for z in v["candidates"]}
             for v in edges}
    mapping={edge_id:None for edge_id in allowed}
    blob=str(raw or "").strip()
    if blob.startswith("```"):
        blob=re.sub(r"^\s*```(?:json)?","",blob).split("```")[0].strip()
    try:
        if not blob.startswith("{") and "{" in blob and "}" in blob:
            blob=blob[blob.find("{"):blob.rfind("}")+1]
        parsed=json.loads(blob)
        vals=parsed["decisions"]
        assert isinstance(vals,list) and len(vals)==len(edges)
        got=set()
        for item in vals:
            assert isinstance(item,dict)
            edge_id=item["edge_id"];chosen=item["candidate_id"]
            assert edge_id in allowed and edge_id not in got
            got.add(edge_id)
            assert chosen is None or (isinstance(chosen,str) and chosen in allowed[edge_id])
            mapping[edge_id]=chosen
        assert got==set(allowed)
        return mapping,"valid_model_verification"
    except (ValueError,KeyError,TypeError,AssertionError):
        return mapping,"invalid_verification_all_abstain"

def prepare():
    frozen=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(POOLS)==frozen["candidate_sha256"]
    rows=read_rows(POOLS)
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert [x["id"] for x in rows]==manifest["pilot_ids"]
    tasks=[{"id":r["id"],"verifier_input":prompts_for_row(r)} for r in rows]
    if INPUTS.exists():assert read_rows(INPUTS)==tasks
    else:
        with INPUTS.open("x",encoding="utf8") as f:
            for x in tasks:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    spec={"n":len(tasks),"candidate_file_sha256":sha_file(POOLS),
          "verifier_inputs_sha256":sha_file(INPUTS),
          "verifier_script_sha256":sha_file(Path(__file__)),
          "system_prompt_sha256":sha(VERIFIER_SYSTEM),
          "verifier_protocol_sha256":sha_file(
              ROOT/"submission/experiments/phase14_protocol.py"),
          "only_goldblind_original_candidate_text_uploaded":True,
          "gold_answer_and_labels_not_sent":True,
          "model":MODEL,"endpoint":BASE,
          "max_calls":MAX_VERIFIER_CALLS,
          "invalid_json_policy":"all-abstain, no question exclusion",
          "estimated_session_usd_guard":1.35}
    if PLAN.exists():assert json.loads(PLAN.read_text(encoding="utf8"))==spec
    else:new_json(PLAN,spec)
    return tasks

def execute(max_new=MAX_VERIFIER_CALLS):
    tasks=prepare()
    if ERROR.exists() or INFLIGHT.exists():raise RuntimeError("Unresolved remote incident")
    saved={}
    for x in read_rows(RESP):
        assert x["id"] not in saved and x.get("ok")
        saved[x["id"]]=x
    print("PHASE21_VERIFIER_START",len(saved),"/",len(tasks),flush=True)
    if len(saved)==len(tasks):return
    paid=0;started=time.monotonic()
    with APP.run():
        for task in tasks:
            if task["id"] in saved:continue
            elapsed=time.monotonic()-started
            if paid>=max_new or elapsed+120>=1800 or (elapsed+120)*2.60/3600>=1.35:
                print("PHASE21_VERIFIER_GUARD",len(saved),flush=True);break
            req=model_payload(task["verifier_input"])
            digest=sha(json.dumps(req,sort_keys=True,ensure_ascii=False))
            atomic_write(INFLIGHT,{"id":task["id"],"prompt_sha256":digest,
                "timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
            try:res=infer_once.remote(req)
            except BaseException as ex:
                append_fsync(ERROR,{"id":task["id"],"exception_type":type(ex).__name__})
                print("PHASE21_VERIFIER_EXCEPTION",type(ex).__name__,flush=True);break
            paid+=1
            if not res.get("ok"):
                append_fsync(ERROR,{"id":task["id"],"phase":res.get("phase"),
                    "http_status":res.get("http_status"),
                    "exception":res.get("exception")})
                print("PHASE21_VERIFIER_REMOTE_ERROR",task["id"],flush=True);break
            raw=str(res.get("answer") or "")
            decisions,status=verify_parse(raw,task["verifier_input"])
            row={"id":task["id"],"ok":True,
                 "decision_ids":decisions,"status":status,
                 "raw_response":raw,"prompt_sha256":digest,
                 "model":MODEL,"paid_remote_call":True,
                 "input_tokens":res.get("input_tokens"),
                 "output_tokens":res.get("output_tokens")}
            append_fsync(RESP,row);INFLIGHT.unlink()
            saved[row["id"]]=row
            if len(saved)%8==0:
                print("PHASE21_VERIFIER_PROGRESS",len(saved),"/",len(tasks),
                      "valid",sum(r["status"]=="valid_model_verification" for r in saved.values()),
                      "paid",paid,flush=True)
    print("PHASE21_VERIFIER_SESSION_END",len(saved),"/",len(tasks),flush=True)

def freeze():
    rows=read_rows(RESP)
    tasks=prepare()
    assert [r["id"] for r in rows]==[r["id"] for r in tasks] and len(rows)==len(tasks)
    assert all(r.get("ok") for r in rows)
    if FROZEN.exists():raise FileExistsError(FROZEN)
    valid=sum(r["status"]=="valid_model_verification" for r in rows)
    selected=sum(v is not None for r in rows for v in r["decision_ids"].values())
    total=sum(len(r["decision_ids"]) for r in rows)
    new_json(FROZEN,{"count":len(rows),"responses_sha256":sha_file(RESP),
          "pre_remote_sha256":sha_file(PLAN),
          "valid_json":valid,"abstain_cases":len(rows)-valid,
          "verified_edge_picks":selected,"total_edge_hypotheses":total,
          "model_judgments_not_proof":True,"gold_unread":True})
    print("PHASE21_VERIFIER_FROZEN",valid,len(rows),selected,total,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    group.add_argument("--freeze",action="store_true")
    args=ap.parse_args()
    if args.prepare:print("PHASE21_VERIFIER_PREPARED",len(prepare()),flush=True)
    elif args.freeze:freeze()
    else:execute()
