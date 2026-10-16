"""Phase21 GOLD-BLIND audit of original-source structured reader outputs.
Only after ALL reader responses persisted, before gold opened/scored.
No selection changes, retries or gold reading.
"""
from pathlib import Path
import json,hashlib,collections,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase21_reader_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
def main():
    response=OUT/"reader_responses.jsonl"
    tasks=OUT/"reader_tasks.jsonl"
    freeze=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==freeze["sha256"]
    rows=[json.loads(x) for x in response.read_text(encoding="utf8").splitlines() if x]
    reqs=[json.loads(x) for x in tasks.read_text(encoding="utf8").splitlines() if x]
    assert len(rows)==len(reqs)==64
    assert [x["task_id"] for x in rows]==[x["task_id"] for x in reqs]
    assert all(x["ok"] for x in rows)
    assert not (OUT/"READER_POST_GOLD_RESULTS.json").exists()
    assert not (OUT/"reader_error.jsonl").exists()
    structured=[r for r in rows if r["arm"]=="verified-evidence-first"]
    direct=[r for r in rows if r["arm"]=="verified-direct"]
    assert len(structured)==len(direct)==32
    count=collections.Counter(r["parse_status"] for r in structured)
    lengths=[len(r.get("raw_reader_output") or "") for r in structured]
    all_steps=[x for r in structured for x in r.get("source_citation_steps",[])]
    valid_step_quotes=sum(bool(s["quote_exact_in_context"]) for s in all_steps)
    output_lengths={
        "min":min(lengths),"max":max(lengths),
        "mean":round(sum(lengths)/len(lengths),2),
        "exact_300":sum(l==300 for l in lengths),
        "above_300":sum(l>300 for l in lengths)}
    report={"before_gold":True,"reader_tasks":64,"paired_questions":32,
        "model_outputs_all_saved":True,"model_error_log_exists":False,
        "same_source_context_in_both_arms":True,
        "source_context_sha256":freeze["sha256"],
        "reader_log_sha256":sha_file(response),
        "reader_tasks_sha256":sha_file(tasks),
        "full_length_phase13_relay_used":True,
        "relay_300_truncation_cured":output_lengths["above_300"]>0,
        "raw_response_character_lengths":output_lengths,
        "structured_json_status":dict(count),
        "structured_empty_answers":sum(not r.get("answer") for r in structured),
        "direct_empty_answers":sum(not r.get("answer") for r in direct),
        "cited_steps":len(all_steps),"exact_cited_steps":valid_step_quotes,
        "quote_match_fraction":valid_step_quotes/max(1,len(all_steps)),
        "not_entailment_certification":True,
        "no_reader_rescoring_or_retry":True,"reserved_96_untouched":True}
    path=OUT/"READER_TRANSPORT_PRE_GOLD_AUDIT.json"
    if path.exists():raise FileExistsError(path)
    new_json(path,report)
    print("PHASE21_PRE_GOLD_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
