"""Faz28 paired original gold support & reader EM evaluation, post-reader only.

Never compute/inspect gold until 192 Qwen reader prediction records (some
identical-prompt reused, with explicit provenance) hash-frozen and verified.
64? sample=96, 192 paired row outputs. Two hard preregistered pilot gates.
"""
from __future__ import annotations
import collections,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_validation_register import OUT
from submission.experiments.phase6_hotpot_eval import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows,answer_metrics
BASE="atomic-sentence-rank"
NEW="semantic-joint-citations"
RESULT=OUT/"VALIDATION_POST_GOLD_RESULTS.json"
def norm(s):return " ".join(str(s).casefold().split())
def fact(t,i):return (norm(t),int(i))
def paired(values,metric,seed):
    diffs=np.array([float(r[NEW][metric])-float(r[BASE][metric]) for r in values])
    rng=np.random.default_rng(seed)
    sample=np.array([np.mean(diffs[rng.integers(0,len(diffs),len(diffs))])
                     for _ in range(10000)])
    win=sum(r[NEW][metric]>r[BASE][metric] for r in values)
    lose=sum(r[NEW][metric]<r[BASE][metric] for r in values)
    return {"mean_delta":float(np.mean(diffs)),
      "bootstrap95":[float(z) for z in np.quantile(sample,[.025,.975])],
      "win":win,"loss":lose,"net":win-lose,
      "two_sided_exact_mcnemar_p":float(
          binomtest(min(win,lose),win+lose,0.5).pvalue) if win+lose else 1.0}
def main():
    if RESULT.exists():raise FileExistsError(RESULT)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    ctx=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    rf=json.loads((OUT/"READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    gs=json.loads((OUT/"PRE_GOLD_STRUCTURAL_DECISION.json").read_text(encoding="utf8"))
    assert gs["decision"]=="VALIDATION_FULL_READER_EVALUATION_REQUIRED"
    assert ctx["stats"]["token_cap_violations"]==0
    assert sha_file(DATA)==p["data_sha256"]
    assert sha_file(OUT/"gold_blind_paired_contexts.jsonl")==ctx["sha256"]
    assert sha_file(OUT/"reader_responses.jsonl")==rf["responses_sha256"]
    contexts=read_rows(OUT/"gold_blind_paired_contexts.jsonl")
    ans=read_rows(OUT/"reader_responses.jsonl")
    assert len(contexts)==96 and len(ans)==192
    assert [x["id"] for x in contexts]==p["pilot_ids"]
    assert [r["task_id"] for r in ans]==[
          c["id"]+"|"+arm for c in contexts for arm in (BASE,NEW)]
    assert not (OUT/"reader_error.jsonl").exists()
    assert not (OUT/"reader_inflight.json").exists()
    index={x["task_id"]:x for x in ans}
    pilot=set(p["pilot_ids"])
    gold={r["id"]:r for r in pq.ParquetFile(DATA).read(columns=[
       "id","answer","type","supporting_facts.title",
       "supporting_facts.sent_id"]).to_pylist() if r["id"] in pilot}
    assert set(gold)==pilot
    cases=[];stats=collections.Counter();types={}
    for c in contexts:
        ident=c["id"]
        g=gold[ident]
        target={fact(t,i) for t,i in zip(
             g["supporting_facts"]["title"],g["supporting_facts"]["sent_id"])}
        assert target
        pair={"id":ident,"type":g["type"],
              "source_sentence_set_changed":c["source_sentence_set_changed"],
              "needed_support_original_sentence_n":len(target)}
        for arm in (BASE,NEW):
            final={fact(x["title"],x["sent_id"])
                   for x in c["contexts"][arm]["selected_original_pairs"]}
            full=target<=final
            overlap=len(target&final)
            pred=index[ident+"|"+arm]["answer"]
            em,f1=answer_metrics(pred,g["answer"])
            pair[arm]={"full_support":int(full),
                "support_recall":overlap/len(target),
                "gold_sentence_hits":overlap,
                "EM":em,"F1":f1,
                "prediction_empty":not bool(pred),
                "actual_tokens":c["contexts"][arm]["tokens"],
                "document_count":c["contexts"][arm]["source_article_count"]}
            stats[arm+"_full_support"]+=int(full)
            stats[arm+"_EM"]+=int(em)
            stats[arm+"_selected_gold_sentences"]+=overlap
            stats[arm+"_tokens"]+=c["contexts"][arm]["tokens"]
        if not c["source_sentence_set_changed"]:
            assert pair[BASE]["full_support"]==pair[NEW]["full_support"]
            assert pair[BASE]["support_recall"]==pair[NEW]["support_recall"]
            assert index[ident+"|"+BASE]["answer"]==index[ident+"|"+NEW]["answer"]
        cases.append(pair)
        types.setdefault(g["type"],[]).append(pair)
    assert len(cases)==96
    em=paired(cases,"EM",20261013)
    f1=paired(cases,"F1",20261014)
    supp=paired(cases,"full_support",20261015)
    recall=paired(cases,"support_recall",20261016)
    pass_gate=(
        supp["net"]>0 and em["net"]>0
        and supp["two_sided_exact_mcnemar_p"]<=p["alpha_each_bonferroni"]
        and em["two_sided_exact_mcnemar_p"]<=p["alpha_each_bonferroni"]
        and ctx["stats"]["token_cap_violations"]==0)
    result={
      "decision":"VALIDATION_BOTH_PRIMARY_SIGNIFICANT" if pass_gate else "VALIDATION_NOT_CONFIRMED",
      "n":96,
      "same_question_matched_reader":True,
      "gold_first_opened_after_reader_freeze":True,
      "not_independent_hidden_test":True,
      "all_sources_original":True,
      "full_support_is_annotation_overlap_not_semantic_entailment":True,
      "paired_EM":em,"paired_F1":f1,
      "paired_full_support":supp,"paired_support_recall":recall,
      "arms":{arm:{"EM_correct":stats[arm+"_EM"],
         "EM_fraction":stats[arm+"_EM"]/96,
         "full_gold_support_questions":stats[arm+"_full_support"],
         "full_gold_support_fraction":stats[arm+"_full_support"]/96,
         "mean_gold_support_recall":sum(x[arm]["support_recall"] for x in cases)/96,
         "mean_F1":sum(x[arm]["F1"] for x in cases)/96,
         "mean_tokens":stats[arm+"_tokens"]/96}
         for arm in (BASE,NEW)},
      "question_type":{key:{"n":len(rows),
          "baseline_EM":sum(r[BASE]["EM"] for r in rows),
          "treatment_EM":sum(r[NEW]["EM"] for r in rows),
          "baseline_full_support":sum(r[BASE]["full_support"] for r in rows),
          "treatment_full_support":sum(r[NEW]["full_support"] for r in rows)}
          for key,rows in types.items()},
      "actual_paid_verifier_calls":96,
      "actual_paid_reader_calls":rf["actual_paid_reader_calls"],
      "same_prompt_reader_reuse":rf["identical_prompt_reuse_n"],
      "structural_context_changes":gs["differing_sentence_sets"],
      "validation_confirmatory_gates":{
          "alpha_each_bonferroni":p["alpha_each_bonferroni"],
          "both_positive_significant":pass_gate,
          "no_pilot_gate_reuse":True},
      "pilot_context_sha256":ctx["sha256"],
      "pilot_reader_sha256":rf["responses_sha256"],
      "phase28_pre_reserved_96_used_for_confirmation":True,
      "no_additional_unregistered_validation_ids_accessed":True,
      "no_gold_sent_remote":True,
      "qasper_official_test_untouched":True,
      "per_case":cases}
    new_json(RESULT,result)
    print("PHASE28_POST_GOLD",json.dumps({
       "n":96,"decision":result["decision"],"arms":result["arms"],
       "EM":em,"full_support":supp,"F1":f1,
       "paid_reader":rf["actual_paid_reader_calls"],
       "reused":rf["identical_prompt_reuse_n"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
