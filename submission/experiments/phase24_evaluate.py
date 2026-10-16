"""Faz24 paired direct-vs-strict-answer-span contraction evaluation.

Gold ONLY loaded after all 64 direct reader and all eligible contractor outputs
were separately frozen, and final gold-blind predictions were SHA-256 frozen.
No remote calls. No new validation access. Two independent blind semantic
raters are NOT provided, so no confirmed semantics claim can be made.
"""
from __future__ import annotations
import json,sys,collections,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase24_answer_span_register import OUT
from submission.experiments.phase12_register import DATA,sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import (
    answer_metrics,read_rows)
FROZEN=OUT/"SPAN_FINAL_PREDICTIONS_FROZEN.json"
PRED=OUT/"FROZEN_GOLD_BLIND_FINAL_PREDICTIONS.jsonl"
RESULT=OUT/"READER_POST_GOLD_RESULTS.json"
def mean(vals):return sum(vals)/len(vals)
def paired(rows,key):
    direct=np.array([r["baseline"][key] for r in rows],dtype=float)
    new=np.array([r["span_contract"][key] for r in rows],dtype=float)
    diff=new-direct
    rng=np.random.default_rng(20261012)
    boot=[float(np.mean(diff[rng.integers(0,len(diff),len(diff))])) for _ in range(10000)]
    wins=sum(r["span_contract"]["EM"]>r["baseline"]["EM"] for r in rows)
    losses=sum(r["span_contract"]["EM"]<r["baseline"]["EM"] for r in rows)
    exact_p=(float(binomtest(min(wins,losses),wins+losses,0.5,
                      alternative="two-sided").pvalue)
             if wins+losses else 1.0)
    return {"difference":float(np.mean(diff)),"bootstrap_CI95":list(
          np.quantile(boot,[.025,.975])),
          "wins":wins,"losses":losses,"net":wins-losses,
          "exact_McNemar_two_sided_p":exact_p}
def main():
    assert not RESULT.exists()
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    fr=json.loads(FROZEN.read_text(encoding="utf8"))
    context=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert p["no_hypothesis_change_after_gold"] and fr["no_gold_read_or_uploaded"]
    assert sha_file(PRED)==fr["pilot_final_predictions_sha256"]
    assert sha_file(OUT/"direct_reader_responses.jsonl")==fr["baseline_sha256"]
    assert context["stats"]["cap_violations"]==0
    data=read_rows(PRED)
    assert [x["id"] for x in data]==p["pilot_ids"] and len(data)==64
    golden={r["id"]:r["answer"] for r in pq.ParquetFile(DATA).read(
        columns=["id","answer"]).to_pylist() if r["id"] in set(p["pilot_ids"])}
    assert len(golden)==64
    results=[]
    for r in data:
        base_em,base_f1=answer_metrics(r["baseline_prediction"],golden[r["id"]])
        new_em,new_f1=answer_metrics(r["contracted_prediction"],golden[r["id"]])
        results.append({"id":r["id"],"contractor_status":r["contractor_status"],
           "changed":r["changed"],"eligible":r["contractor_called"],
           "baseline":{"EM":base_em,"F1":base_f1},
           "span_contract":{"EM":new_em,"F1":new_f1}})
    em=paired(results,"EM");f1=paired(results,"F1")
    passed_numeric=(
         em["net"]>=p["pilot_success_gate"]["min_net_new_correct"]
         and em["exact_McNemar_two_sided_p"]<=
             p["pilot_success_gate"]["paired_two_sided_exact_mcnemar_max_p"]
         and context["stats"]["cap_violations"]==0)
    # There are zero independently qualified semantic reviewers available here.
    # This cannot be silently reported as confirmed success.
    result={
       "decision":"STOP_NO_VALIDATION" if not passed_numeric
                  else "PENDING_INDEPENDENT_BLIND_SEMANTIC_FIDELITY_REVIEW",
       "passed_numeric_EM_gate":passed_numeric,
       "validation_permission":False,
       "n":64,
       "arms":{
           arm:{"correct":sum(r[arm]["EM"] for r in results),
                "EM":mean([r[arm]["EM"] for r in results]),
                "F1":mean([r[arm]["F1"] for r in results])}
           for arm in ("baseline","span_contract")},
       "primary_em":em,"secondary_f1":f1,
       "changed_n":fr["changed_predictions_n"],
       "eligible_n":fr["eligible_n"],
       "changed_only_descriptive":{ "n":sum(r["changed"] for r in results),
           "new_only_correct":sum(r["changed"] and r["span_contract"]["EM"]==1
                      and r["baseline"]["EM"]==0 for r in results),
           "old_only_correct":sum(r["changed"] and r["span_contract"]["EM"]==0
                      and r["baseline"]["EM"]==1 for r in results)},
       "semantic_fidelity_review":"NOT PERFORMED; no independent blinded raters",
       "no_confirmation_claim_possible_without_blind_semantics":True,
       "pilot_is_public_validation_not_hidden_test":True,
       "new_gold_read_only_after_freeze":True,
       "source_prediction_sha256":sha_file(PRED),
       "source_frozen_metadata_sha256":sha_file(FROZEN),
       "source_context_sha256":context["sha256"],
       "source_dataset_sha256":sha_file(DATA),
       "validation_192_IDs_untouched":True,
       "production_code_unchanged":True,
       "qasper_official_test_untouched":True,
       "per_case":results}
    new_json(RESULT,result)
    print("PHASE24_POST_GOLD",json.dumps({
        "n":64,"baseline":result["arms"]["baseline"],
        "new":result["arms"]["span_contract"],
        "em":em,"f1":f1,"changed":fr["changed_predictions_n"],
        "decision":result["decision"],"validation_permission":False},
        ensure_ascii=False),flush=True)
if __name__=="__main__":main()
