"""Phase22 gold-BLIND deterministic hybrid composition.
All 64 fixed Qwen predictions are required. Gold dataset deliberately
NOT imported. Freeze route+prediction per ID BEFORE answer evaluation.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase22_reader_register import OUT
from submission.experiments.phase22_reader_protocol import FALLBACK_RULE
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
ANS=OUT/"reader_responses.jsonl"
TASKS=OUT/"reader_tasks.jsonl"
CONTEXT=OUT/"gold_blind_contexts.jsonl"
ROUTES=OUT/"HYBRID_GOLD_BLIND_PREDICTIONS.jsonl"
FROZEN=OUT/"HYBRID_PRE_GOLD_DECISIONS_FROZEN.json"
def main():
    assert not (OUT/"READER_POST_GOLD_RESULTS.json").exists(), "Gold already opened"
    assert not ROUTES.exists() and not FROZEN.exists()
    rows=read_rows(ANS)
    tasks=read_rows(TASKS)
    cases=read_rows(CONTEXT)
    assert len(rows)==len(tasks)==64
    assert [r["task_id"] for r in rows]==[t["task_id"] for t in tasks]
    assert all(x["ok"] for x in rows)
    assert not (OUT/"reader_error.jsonl").exists()
    assert not (OUT/"reader_inflight.json").exists()
    allowed=[c["id"] for c in cases]
    index={x["task_id"]:x for x in rows}
    assert len(allowed)==len(set(allowed))==32
    preds=[]
    for ident in allowed:
        direct=index[ident+"|verified-direct"]
        structured=index[ident+"|verified-evidence-first"]
        raw=structured.get("answer") or ""
        valid=structured.get("parse_status")=="valid_json" and bool(raw.strip())
        route="structured" if valid else "direct_fallback"
        prediction=raw if valid else direct["answer"]
        preds.append({"id":ident,"route":route,"prediction":prediction,
            "direct_prediction":direct["answer"],
            "structured_prediction":structured["answer"],
            "structured_parse_status":structured.get("parse_status"),
            "structured_empty":not bool(raw.strip()),
            "no_gold_in_routing":True})
    with ROUTES.open("x",encoding="utf8") as f:
        for rec in preds:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    manifest={
      "n":32,"hybrid_rule":FALLBACK_RULE,"reader_log_sha256":sha_file(ANS),
      "reader_tasks_sha256":sha_file(TASKS),
      "context_sha256":sha_file(CONTEXT),
      "hybrid_predictions_sha256":sha_file(ROUTES),
      "source_script_sha256":sha_file(Path(__file__)),
      "structured_used_n":sum(x["route"]=="structured" for x in preds),
      "direct_fallback_used_n":sum(x["route"]=="direct_fallback" for x in preds),
      "gold_never_imported_or_read":True,
      "no_remote_inference":True,
      "pilots_are_new_public_validation_IDs":True}
    new_json(FROZEN,manifest)
    print("PHASE22_HYBRID_PRE_GOLD_FROZEN",json.dumps({
        "n":manifest["n"],
        "structured":manifest["structured_used_n"],
        "fallback":manifest["direct_fallback_used_n"],
        "hash":manifest["hybrid_predictions_sha256"]}),flush=True)
if __name__=="__main__":main()
