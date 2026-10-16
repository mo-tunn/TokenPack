"""Post-gold Phase21 error audit; no new inference, no retuning, no test.
Answers must have been frozen and scored first. Public validation ONLY.
"""
from __future__ import annotations
import json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase21_reader_register import OUT
from submission.experiments.phase12_register import DATA,sha_file,new_json
import pyarrow.parquet as pq
def norm(s):return " ".join(str(s).lower().split())
def main():
    score_path=OUT/"READER_POST_GOLD_RESULTS.json"
    scored=json.loads(score_path.read_text(encoding="utf8"))
    f=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    rawpath=OUT/"reader_responses.jsonl"
    assert scored["decision"]=="STOP_NO_VALIDATION" and scored["n"]==32
    assert sha_file(rawpath)==scored["responses_sha256"]
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==f["sha256"]
    ctx=[json.loads(x) for x in (OUT/"gold_blind_contexts.jsonl").read_text(encoding="utf8").splitlines() if x]
    raw=[json.loads(x) for x in rawpath.read_text(encoding="utf8").splitlines() if x]
    assert len(ctx)==32 and len(raw)==64
    pairs=collections.defaultdict(dict)
    for x in raw:pairs[x["id"]][x["arm"]]=x
    if set(pairs)!={r["id"] for r in ctx}:raise ValueError("Different cases")
    metrics={x["id"]:x for x in scored["per_case"]}
    assert len(metrics)==32
    golds={x["id"]:x for x in pq.ParquetFile(DATA).read(
      columns=["id","type","supporting_facts"]).to_pylist() if x["id"] in pairs}
    assert set(golds)==set(pairs)
    cases=[];counts=collections.Counter()
    for c in ctx:
        ident=c["id"]
        assert c["contexts"]["verified-direct"]["text"]==c["contexts"]["verified-evidence-first"]["text"]
        selected=c["contexts"]["verified-direct"]["selected_sentence_indexes"]
        pairs_gold={(norm(a),int(b)) for a,b in zip(
            golds[ident]["supporting_facts"]["title"],
            golds[ident]["supporting_facts"]["sent_id"])}
        selected_gold={(norm(c["candidate_map"][i]["title"]),int(c["candidate_map"][i]["sent_id"])) for i in selected}
        gold_hit=len(pairs_gold&selected_gold)
        allgold=pairs_gold<=selected_gold
        cur=pairs[ident]
        old=metrics[ident]["verified-direct"]
        new=metrics[ident]["verified-evidence-first"]
        abstained=cur["verified-evidence-first"]["answer"]==""
        model_steps=cur["verified-evidence-first"].get("source_citation_steps",[])
        exact=sum(bool(x.get("quote_exact_in_context")) for x in model_steps)
        row={"id":ident,"question_type":golds[ident]["type"],
            "gold_support_total":len(pairs_gold),
            "gold_support_present":gold_hit,
            "all_gold_support_present":allgold,
            "gold_support_recall":gold_hit/len(pairs_gold),
            "control_em":old["EM"],"new_em":new["EM"],
            "control_f1":old["F1"],"new_f1":new["F1"],
            "structured_empty":abstained,
            "structured_valid_json":cur["verified-evidence-first"]["parse_status"]=="valid_json",
            "cited_step_count":len(model_steps),"exact_quoted_step_count":exact,
            "baseline_answer_nonempty":bool(cur["verified-direct"]["answer"]),
            "context_tokens":c["contexts"]["verified-direct"]["tokens"]}
        cases.append(row)
        counts.update({
            "total":1,"all_support":int(allgold),
            "control_correct":int(old["EM"]),
            "new_correct":int(new["EM"]),
            "empty_structured":int(abstained),
            "empty_control_correct":int(abstained and old["EM"]),
            "empty_gold_complete":int(abstained and allgold),
            "empty_gold_complete_control_correct":int(abstained and allgold and old["EM"]),
            "nonempty_structured_wrong":int(not abstained and not new["EM"]),
            "nonempty_structured_correct":int(not abstained and bool(new["EM"])),
            "new_only_correct":int(new["EM"] and not old["EM"]),
            "old_only_correct":int(old["EM"] and not new["EM"]),
            "both_wrong":int(not old["EM"] and not new["EM"]),
            "both_correct":int(old["EM"] and new["EM"]),
            "full_support_control_wrong":int(allgold and not old["EM"]),
            "full_support_new_wrong":int(allgold and not new["EM"]),
            "steps_total":len(model_steps),"exact_quoted_steps":exact})
    result={"study":"Faz21 independent new-ID pilot, posthoc local taxonomy",
        "not_confirmatory":True,"gold_not_used_to_select_method":True,
        "no_new_model_calls":True,"reader_input_context_shared":True,
        "n":32,"counts":dict(counts),
        "by_question_type":{},
        "source_reader_sha256":sha_file(rawpath),
        "source_context_sha256":f["sha256"],
        "negative_pilot_gate":scored["decision"],
        "per_case":cases}
    for ty in sorted(set(r["question_type"] for r in cases)):
        rows=[r for r in cases if r["question_type"]==ty]
        result["by_question_type"][ty]={
            "n":len(rows),"control_em_correct":sum(r["control_em"] for r in rows),
            "new_em_correct":sum(r["new_em"] for r in rows),
            "full_gold_support":sum(r["all_gold_support_present"] for r in rows),
            "structured_empty":sum(r["structured_empty"] for r in rows)}
    out=OUT/"READER_POSTHOC_EVIDENCE_ABSTENTION_AUDIT.json"
    if out.exists():raise FileExistsError(out)
    new_json(out,result)
    print("PHASE21_POSTHOC_EVIDENCE_AUDIT",json.dumps({
       "counts":result["counts"],"types":result["by_question_type"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
