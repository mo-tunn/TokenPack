"""TINY standalone Modal-only relay; automatically staged OUTSIDE TokenPack.

Important: importing the older submission/experiments relay causes Modal's
automatic upload scanner to hash the enormous project data/result files.
This source is copied to a temp directory BEFORE import and the Modal
container only receives this <5kB relay source, never local benchmark data.
The LLM_API_KEY lives exclusively in the user's existing Modal Secret.
"""
from __future__ import annotations

import modal

BASE="https://metekizilcik--tokenpack-qwen27b-longcontext-inference-server.modal.run"
MODEL="qwen3.8-27b-orcarouter-q4km"
APP=modal.App("tokenpack-phase7-qwen27b-tiny-relay")


@APP.function(
    image=modal.Image.debian_slim(python_version="3.11"),
    secrets=[modal.Secret.from_name("qwen-modal-secrets")],
    cpu=0.125,memory=512,timeout=145,
)
def infer_once(payload:dict)->dict:
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    key=os.environ.get("LLM_API_KEY","")
    if not key:
        return {"ok":False,"phase":"secret","error":"LLM_API_KEY missing"}

    def http(path,data=None,timeout=12):
        request=urllib.request.Request(
            BASE+path,
            method="POST" if data is not None else "GET",
            data=json.dumps(data).encode("utf8") if data is not None else None,
            headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
        )
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                return response.status,json.loads(response.read().decode("utf8"))
        except urllib.error.HTTPError as exc:
            # Never expose authorization or remote response contents to logs.
            return exc.code,{}

    started=time.monotonic()
    health_checks=0
    for _ in range(8):
        health_checks+=1
        try:
            status,_=http("/health",timeout=10)
        except Exception as e:
            return {"ok":False,"phase":"health","exception":type(e).__name__}
        if status==200:break
        if status not in (502,503,504):
            return {"ok":False,"phase":"health","http_status":status}
        time.sleep(3)
    else:
        return {"ok":False,"phase":"health","error":"not ready after 8 checks"}
    begin=time.monotonic()
    try:
        status,output=http("/v1/chat/completions",data=payload,timeout=83)
    except Exception as e:
        return {"ok":False,"phase":"inference",
                "exception":type(e).__name__,
                "elapsed_s":round(time.monotonic()-begin,2)}
    if status!=200:
        return {"ok":False,"phase":"inference","http_status":status,
                "elapsed_s":round(time.monotonic()-begin,2)}
    choices=output.get("choices") or []
    answer=(choices[0].get("message") or {}) if choices else {}
    usage=output.get("usage") or {}
    return {
        "ok":True,
        "answer":str(answer.get("content") or "").strip()[:300],
        "model_response":str(output.get("model") or ""),
        "output_tokens":usage.get("completion_tokens"),
        "input_tokens":usage.get("prompt_tokens"),
        "finish_reason":choices[0].get("finish_reason") if choices else None,
        "elapsed_s":round(time.monotonic()-begin,2),
        "gpu_health_wait_s":round(begin-started,2),
        "health_checks":health_checks,
    }
