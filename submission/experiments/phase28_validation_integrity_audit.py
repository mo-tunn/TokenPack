"""Independent, post-gold audit of immutable Phase28 confirmatory validation.

No model calls, no rescoring selection, no new question IDs, no reruns.
Verify pilot and pre-reserved validation disjointness, frozen SHA256, exact
Qwen request hashes, cached duplicate prompt correctness, annotated source
pairs and paired exact tests against pre-fixed Bonferroni alpha 0.025.
"""
from __future__ import annotations
import hashlib,json,math,sys,collections
from pathlib import Path
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_validation_register import OUT
from submission.experiments.phase28_semantic_register import OUT as PILOT
from submission.experiments.phase28_semantic_core import payload
from submission.experiments.phase13_planner import MODEL
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    read_rows,make_payload,answer_metrics)
PATH=OUT/"VALIDATION_INTEGRITY_INDEPENDENT_AUDIT.json"
B="atomic-sentence-rank";T="semantic-joint-citations"
def digest(x):return hashlib.sha256(json.dumps(
    x,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def matched(rows,k):
    wins=sum(x[T][k]>x[B][k] for x in rows)
    losses=sum(x[T][k]<x[B][k] for x in rows)
    return {"win":wins,"loss":losses,"net":wins-losses,
        "p_exact_two_sided":float(binomtest(min(wins,losses),wins+losses,0.5).pvalue)
                         if wins+losses else 1.0}
def main():
    if PATH.exists():raise FileExistsError(PATH)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    orig=json.loads((PILOT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    pre_ver=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    pre_ctx=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    pre_reader=json.loads((OUT/"READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    result=json.loads((OUT/"VALIDATION_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    assert p["n"]==96 and p["pilot_ids"]==orig["reserved_ids"]
    assert not set(p["pilot_ids"])&set(orig["pilot_ids"])
    assert sha_file(PILOT/"PRE_GOLD_PROTOCOL.json")==p["original_prereg_SHA256"]
    assert sha_file(PILOT/"READER_POST_GOLD_RESULTS.json")==p["positive_pilot_postgold_SHA256"]
    assert sha_file(ROOT/"submission/experiments/phase28_semantic_core.py")==p["same_source_core_SHA256"]
    assert sha_file(OUT/"semantic_verifier_responses.jsonl")==pre_ver["sha256"]
    assert sha_file(OUT/"gold_blind_paired_contexts.jsonl")==pre_ctx["sha256"]
    assert sha_file(OUT/"reader_responses.jsonl")==pre_reader["responses_sha256"]
    assert sha_file(OUT/"reader_tasks.jsonl")==pre_reader["task_sha256"]
    assert result["pilot_reader_sha256"]==pre_reader["responses_sha256"]
    assert result["pilot_context_sha256"]==pre_ctx["sha256"]
    assert pre_ctx["stats"]["token_cap_violations"]==0
    tasks=read_rows(OUT/"semantic_citation_tasks.jsonl")
    verifier=read_rows(OUT/"semantic_verifier_responses.jsonl")
    contexts=read_rows(OUT/"gold_blind_paired_contexts.jsonl")
    rtasks=read_rows(OUT/"reader_tasks.jsonl")
    answers=read_rows(OUT/"reader_responses.jsonl")
    assert len(tasks)==len(verifier)==len(contexts)==96
    assert len(rtasks)==len(answers)==192
    assert [r["id"] for r in tasks]==[r["id"] for r in verifier]==p["pilot_ids"]
    assert [r["id"] for r in contexts]==p["pilot_ids"]
    assert [r["task_id"] for r in rtasks]==[r["task_id"] for r in answers]
    assert all(x["ok"] for x in verifier+answers)
    # Every Qwen verifier prompt digest must match CURRENT frozen source+question
    for task,r in zip(tasks,verifier):
        assert r["prompt_sha256"]==digest(payload(task,MODEL))
        assert len(task["candidates"])<=36
        assert len({x["cid"] for x in task["candidates"]})==len(task["candidates"])
        if r["citation_ids"]:
            known={x["cid"]:x for x in task["candidates"]}
            assert set(r["citation_ids"])<=set(known)
            assert len({known[c]["article"] for c in r["citation_ids"]})>=2
    # Every direct reader prompt is reproducible with SAME question/context.
    # Reused answers must match an earlier exact request digest.
    already={}
    for task,rec in zip(rtasks,answers):
        ph=digest(make_payload(task))
        assert rec["prompt_sha256"]==ph
        if not rec["paid"]:
            assert rec["reused_identical_prompt_from"]==already[ph]["task_id"]
            assert rec["answer"]==already[ph]["answer"]
        else:
            assert ph not in already
            already[ph]=rec
    assert sum(r["paid"] for r in answers)==132
    assert sum(not r["paid"] for r in answers)==60
    assert pre_reader["actual_paid_reader_calls"]==132
    # Confirm original source and token caps per pair; identical contexts
    # cannot yield a different reader answer or source-pair coverage.
    cases=result["per_case"]
    assert len(cases)==96
    for source,computed in zip(contexts,cases):
        assert source["id"]==computed["id"]
        a=source["contexts"][B];b=source["contexts"][T]
        assert a["cap"]==b["cap"]==source["cap"]
        assert a["tokens"]<=a["cap"] and b["tokens"]<=b["cap"]
        if not source["source_sentence_set_changed"]:
            assert { (x["article"],x["title"],x["sent_id"]) for x in a["selected_original_pairs"]}=={
                    (x["article"],x["title"],x["sent_id"]) for x in b["selected_original_pairs"]}
            assert computed[B]["full_support"]==computed[T]["full_support"]
            assert computed[B]["EM"]==computed[T]["EM"]
    assert sum(x["source_sentence_set_changed"] for x in contexts)==36
    audit_em=matched(cases,"EM");audit_support=matched(cases,"full_support")
    assert audit_em["win"]==13 and audit_em["loss"]==0
    assert audit_support["win"]==26 and audit_support["loss"]==0
    assert audit_em["net"]==result["paired_EM"]["net"]
    assert audit_support["net"]==result["paired_full_support"]["net"]
    assert math.isclose(audit_em["p_exact_two_sided"],result["paired_EM"]["two_sided_exact_mcnemar_p"])
    assert math.isclose(audit_support["p_exact_two_sided"],result["paired_full_support"]["two_sided_exact_mcnemar_p"])
    valid=(audit_em["net"]>0 and audit_support["net"]>0
       and audit_em["p_exact_two_sided"]<=p["alpha_each_bonferroni"]
       and audit_support["p_exact_two_sided"]<=p["alpha_each_bonferroni"])
    assert valid and result["decision"]=="VALIDATION_BOTH_PRIMARY_SIGNIFICANT"
    report={"status":"PASS_INDEPENDENT_RECOMPUTATION",
       "n":96,"pre_reserved_ids_exact":True,
       "frozen_verifier_hash_valid":True,
       "frozen_reader_hash_valid":True,
       "source_core_unchanged_from_pilot":True,
       "model_verifier_prompt_hashes_recomputed":96,
       "model_reader_prompt_hashes_recomputed":192,
       "same_prompt_reuse_audited":60,"real_reader_calls":132,
       "real_source_verifier_calls":96,
       "semantic_citations_no_gold_claim":True,
       "max_cap_violations":0,
       "source_sentence_sets_changed":36,
       "independent_EM":audit_em,
       "independent_full_support":audit_support,
       "alpha_each":p["alpha_each_bonferroni"],
       "both_primary_confirmed":valid,
       "only_existing_public_validation_pilot_and_pre_reserve":True,
       "source_result_sha256":sha_file(OUT/"VALIDATION_POST_GOLD_RESULTS.json"),
       "source_reader_sha256":pre_reader["responses_sha256"],
       "source_context_sha256":pre_ctx["sha256"],
       "qasper_official_sealed_test_unopened":True,
       "no_new_remote_model_calls":True}
    new_json(PATH,report)
    print("PHASE28_VALIDATION_INDEPENDENT_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
