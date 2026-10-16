"""Isolated Phase13 Qwen27B relay. No local benchmark files uploaded.

Only difference from Phase7 tiny relay: graph JSON may be ~2000 chars
rather than an answer capped at 300 chars. TLS validation remains ON.
Secrets are only in Modal Secret and never printed.
"""
from __future__ import annotations
import modal
BASE="https://metekizilcik--tokenpack-qwen27b-longcontext-inference-server.modal.run"
MODEL="qwen3.8-27b-orcarouter-q4km"
APP=modal.App("tokenpack-phase13-qwen27b-graph-tiny-relay")

@APP.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    secrets=[modal.Secret.from_name("qwen-modal-secrets")],
    cpu=.125,memory=512,timeout=145,
)
def infer_once(payload:dict)->dict:
    import json,os,time,urllib.error,urllib.request
    key=os.environ.get("LLM_API_KEY","")
    if not key:return {"ok":False,"phase":"secret","error":"missing secret"}
    def http(path,data=None,timeout=12):
        req=urllib.request.Request(
            BASE+path,
            method="POST" if data is not None else "GET",
            data=json.dumps(data).encode("utf8") if data is not None else None,
            headers={"Authorization":"Bearer "+key,
                     "Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                return resp.status,json.loads(resp.read().decode("utf8"))
        except urllib.error.HTTPError as e:
            return e.code,{}
    for _ in range(8):
        try:status,_=http("/health",timeout=10)
        except Exception as ex:
            return {"ok":False,"phase":"health","exception":type(ex).__name__}
        if status==200:break
        if status not in (502,503,504):
            return {"ok":False,"phase":"health","http_status":status}
        time.sleep(3)
    else:return {"ok":False,"phase":"health","error":"not ready"}
    begin=time.monotonic()
    try:status,out=http("/v1/chat/completions",data=payload,timeout=83)
    except Exception as ex:
        return {"ok":False,"phase":"inference",
                "exception":type(ex).__name__}
    if status!=200:return {"ok":False,"phase":"inference","http_status":status}
    choices=out.get("choices") or []
    msg=(choices[0].get("message") or {}) if choices else {}
    usage=out.get("usage") or {}
    return {"ok":True,"answer":str(msg.get("content") or "").strip()[:4000],
            "input_tokens":usage.get("prompt_tokens"),
            "output_tokens":usage.get("completion_tokens"),
            "finish_reason":choices[0].get("finish_reason") if choices else None,
            "elapsed_s":round(time.monotonic()-begin,2)}
