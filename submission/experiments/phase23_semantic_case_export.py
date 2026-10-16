"""Local-only frozen casebook for Phase22 all-gold-present but hybrid EM0.
Never calls remote LLM, does not change Phase22 results, does not retrieve any
previously unopened validation ID, and does not touch QASPER sealed test.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase22_reader_register import OUT as PREV
from submission.experiments.phase12_register import DATA,sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
import pyarrow.parquet as pq
OUT=ROOT/"submission/results/phase23_semantic_failure_audit_20261012"
def n(x):return " ".join(str(x).casefold().split())
def main():
    if OUT.exists():raise FileExistsError("Preserve prior casebook")
    audit=json.loads((PREV/"HYBRID_POST_GOLD_ROUTE_EVIDENCE_AUDIT.json").read_text(encoding="utf8"))
    scored=json.loads((PREV/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    frozen=json.loads((PREV/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    hybrid=json.loads((PREV/"HYBRID_PRE_GOLD_DECISIONS_FROZEN.json").read_text(encoding="utf8"))
    assert scored["decision"]=="STOP_NO_VALIDATION"
    assert sha_file(PREV/"reader_responses.jsonl")==scored["responses_sha256"]==hybrid["reader_log_sha256"]
    assert sha_file(PREV/"gold_blind_contexts.jsonl")==frozen["sha256"]
    allcases=read_rows(PREV/"gold_blind_contexts.jsonl")
    log=read_rows(PREV/"reader_responses.jsonl")
    byid={case["id"]:case for case in allcases}
    predictions={x["task_id"]:x for x in log}
    wrong=[r for r in audit["per_case"]
           if r["full_gold_support"] and not r["hybrid_em"]]
    assert len(wrong)==8 and len({r["id"] for r in wrong})==8
    ids={r["id"] for r in wrong}
    gold={r["id"]:r for r in pq.ParquetFile(DATA).read(
        columns=["id","answer","type","supporting_facts"]).to_pylist()
        if r["id"] in ids}
    assert set(gold)==ids
    rows=[]
    for r in wrong:
        ident=r["id"]
        case=byid[ident]
        g=gold[ident]
        direct=predictions[ident+"|verified-direct"]
        struct=predictions[ident+"|verified-evidence-first"]
        source=case["contexts"]["verified-direct"]["text"]
        chosen=case["contexts"]["verified-direct"]["selected_sentence_indexes"]
        original=case["candidate_map"]
        goldkeys={(n(t),int(i)) for t,i in zip(
            g["supporting_facts"]["title"],g["supporting_facts"]["sent_id"])}
        selected_support=[original[i] for i in chosen
                if (n(original[i]["title"]),int(original[i]["sent_id"])) in goldkeys]
        assert {(n(x["title"]),int(x["sent_id"])) for x in selected_support}==goldkeys
        rows.append({
          "id":ident,"type":g["type"],"question":case["question"],
          "gold_answer":g["answer"],
          "gold_supporting_facts":list(zip(
            g["supporting_facts"]["title"],g["supporting_facts"]["sent_id"])),
          "gold_support_in_final_map":selected_support,
          "final_context":source,"final_context_tokens":case["contexts"]["verified-direct"]["tokens"],
          "control_answer":direct["answer"],"control_finish":direct.get("finish_reason"),
          "structured_answer":struct["answer"],
          "structured_raw_output":struct.get("raw_reader_output"),
          "structured_parse":struct["parse_status"],
          "structured_finish":struct.get("finish_reason"),
          "structured_steps":struct.get("source_citation_steps",[]),
          "structured_output_tokens":struct.get("output_tokens"),
          "hybrid_route":r["route"],"hybrid_EM":r["hybrid_em"],
          "all_annotated_gold_support_present":True,
          "case_is_exploratory_posthoc":True})
    meta={
      "source_results":"Phase22 public 2Wiki validation; all 8 gold-complete but hybrid EM0 cases",
      "source_files_sha256":{
        name:sha_file(PREV/name) for name in (
          "PRE_GOLD_PROTOCOL.json","CONTEXTS_FROZEN.json",
          "gold_blind_contexts.jsonl","reader_responses.jsonl",
          "HYBRID_GOLD_BLIND_PREDICTIONS.jsonl",
          "READER_POST_GOLD_RESULTS.json",
          "HYBRID_POST_GOLD_ROUTE_EVIDENCE_AUDIT.json")},
      "source_parquet_sha256":sha_file(DATA),
      "case_id_count":len(rows),
      "gold_already_unsealed_in_prior_phase":True,
      "no_remote_model_called":True,
      "qasper_official_test_not_touched":True,
      "phase22_archive_unchanged":True,
      "no_claim_independent_sample":True}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"CASEBOOK_PROVENANCE.json",meta)
    with (OUT/"POSTHOC_EIGHT_CASES.jsonl").open("x",encoding="utf8") as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    print("PHASE23_CASEBOOK_FROZEN",json.dumps({
        "n":len(rows),
        "source_sha256":sha_file(OUT/"POSTHOC_EIGHT_CASES.jsonl"),
        "types":{ty:sum(x["type"]==ty for x in rows)
                 for ty in sorted({x["type"] for x in rows})},
        "all_gold_support_present":True,
        "all_structured_nonempty":all(bool(x["structured_answer"]) for x in rows),
        "model_calls":0},ensure_ascii=False),flush=True)
    for i,x in enumerate(rows):
        print("CASE",i+1,json.dumps({
            k:x[k] for k in (
                "id","type","question","gold_answer","gold_supporting_facts",
                "control_answer","structured_answer","structured_steps",
                "structured_finish","final_context","hybrid_route")
        },ensure_ascii=False),flush=True)
if __name__=="__main__":main()
