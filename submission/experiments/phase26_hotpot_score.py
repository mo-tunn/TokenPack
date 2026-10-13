"""Phase26 HotpotQA original-support-sentence evidence sufficiency taxonomy.

Gold labels only accessed once all 96 remote Qwen reader outputs are present,
hash-frozen and verified. No model calls; no tuning, no validation subset.
"""
from __future__ import annotations
import collections,json,math,sys,unicodedata
from pathlib import Path
from scipy.stats import binomtest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase26_hotpot_register import OUT
from submission.experiments.phase6_hotpot_eval import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows,answer_metrics
RESULT=OUT/"POST_GOLD_SENTENCE_SUFFICIENCY.json"
def norm(x):
    return " ".join(unicodedata.normalize("NFKC",str(x)).casefold().split())
def key(t,i):return (norm(t),int(i))
def ci(k,n):
    if not n:return None
    p=binomtest(k,n).proportion_ci(.95,method="wilson")
    return [float(p.low),float(p.high)]
def main():
    assert not RESULT.exists()
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    ctx_meta=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    reader_meta=json.loads((OUT/"HOTPOT_READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    source=OUT/"gold_blind_contexts.jsonl"
    responses=OUT/"hotpot_reader_responses.jsonl"
    assert sha_file(DATA)==p["dataset_sha256"]
    assert sha_file(source)==ctx_meta["sha256"]
    assert sha_file(responses)==reader_meta["responses_sha256"]
    assert ctx_meta["stats"]["cap_violations"]==0
    contexts=read_rows(source);answers=read_rows(responses)
    assert [r["id"] for r in contexts]==[r["id"] for r in answers]==p["pilot_ids"]
    assert len(answers)==len(contexts)==96
    assert not (OUT/"hotpot_reader_error.jsonl").exists()
    assert not (OUT/"hotpot_reader_inflight.json").exists()
    # GOLD physically loaded only here, after hash verification.
    golden={r["id"]:r for r in pq.ParquetFile(DATA).read(columns=[
          "id","answer","type","supporting_facts.title",
          "supporting_facts.sent_id"]).to_pylist() if r["id"] in set(p["pilot_ids"])}
    assert len(golden)==96
    totals=collections.Counter()
    cases=[];types=collections.defaultdict(list)
    for source_row,response in zip(contexts,answers):
        ident=source_row["id"]
        gold=golden[ident]
        actual_support={key(t,i) for t,i in zip(
             gold["supporting_facts"]["title"],gold["supporting_facts"]["sent_id"])}
        assert len(actual_support)>=1
        candidates={key(x["title"],x["sentence_id"])
                    for x in source_row["candidate_original_title_sentence_pairs"]}
        selected={key(x["title"],x["sentence_id"])
                  for x in source_row["selected_original_sentences"]}
        assert selected<=candidates
        available=actual_support<=candidates
        complete=actual_support<=selected
        assert not complete or available
        srate=len(actual_support&selected)/len(actual_support)
        em,f1=answer_metrics(response["answer"],gold["answer"])
        if not available:cat="A_support_missing_from_original_source_candidates"
        elif not complete:cat="B_support_available_but_not_fully_selected"
        elif not em:cat="C_exact_support_complete_reader_EM_wrong"
        else:cat="D_exact_support_complete_reader_EM_correct"
        gold_string=norm(gold["answer"]) in norm(source_row["context"])
        rec={"id":ident,"question_type":gold["type"],"required_support_fact_n":len(actual_support),
             "source_support_fact_n":len(actual_support&candidates),
             "selected_support_fact_n":len(actual_support&selected),
             "all_gold_support_available_original_source":available,
             "all_gold_support_in_final":complete,
             "support_recall":srate,
             "selected_gold_title_count":len({x[0] for x in actual_support&selected}),
             "reader_EM":em,"reader_F1":f1,
             "source_actual_tokens":source_row["actual_tokens"],
             "source_budget_tokens":source_row["cap"],
             "gold_answer_string_in_selected_context":gold_string,
             "error_stage":cat,"no_gold_in_remote_reader_request":True}
        cases.append(rec)
        types[gold["type"]].append(rec)
        totals[cat]+=1
        totals["n"]+=1
        totals["reader_EM_correct"]+=int(em)
        totals["gold_full_source"]+=int(available)
        totals["gold_full_final"]+=int(complete)
        totals["gold_any_final"]+=int(bool(actual_support&selected))
        totals["gold_string_in_final"]+=int(gold_string)
        totals["gold_full_final_reader_wrong"]+=int(complete and not em)
        totals["gold_full_final_reader_correct"]+=int(complete and em)
        totals["gold_incomplete_final_reader_correct"]+=int(not complete and em)
        totals["gold_incomplete_final_reader_wrong"]+=int(not complete and not em)
        totals["gold_sentence_required_total"]+=len(actual_support)
        totals["gold_sentence_selected_total"]+=len(actual_support&selected)
    assert totals["n"]==96 and sum(totals[x] for x in (
       "A_support_missing_from_original_source_candidates",
       "B_support_available_but_not_fully_selected",
       "C_exact_support_complete_reader_EM_wrong",
       "D_exact_support_complete_reader_EM_correct"))==96
    result={
      "n":96,
      "decision":"DIAGNOSTIC_COMPLETE_NO_METHOD_PROMOTION",
      "source_dataset":"HotpotQA distractor PUBLIC validation, new disjoint IDs",
      "gold_only_opened_after_remote_and_context_SHA256_freeze":True,
      "no_gold_transmitted_to_remote":True,
      "frozen_context_sha256":ctx_meta["sha256"],
      "frozen_remote_sha256":reader_meta["responses_sha256"],
      "all_previous_id_exclusions_confirmed":p["total_unique_excluded_ids"],
      "counts":dict(totals),
      "rates":{
          "reader_exact_match":totals["reader_EM_correct"]/96,
          "reader_answer_f1_mean":sum(c["reader_F1"] for c in cases)/96,
          "all_source_gold_support_accessible":totals["gold_full_source"]/96,
          "all_exact_gold_support_final":totals["gold_full_final"]/96,
          "mean_exact_support_sentence_recall":sum(c["support_recall"] for c in cases)/96,
          "gold_annotation_complete_but_reader_wrong_fraction_all":totals["gold_full_final_reader_wrong"]/96,
          "EM_Wilson95":ci(totals["reader_EM_correct"],96),
          "all_support_final_Wilson95":ci(totals["gold_full_final"],96),
          "reader_EM_given_gold_full_final":
              totals["gold_full_final_reader_correct"]/totals["gold_full_final"]
              if totals["gold_full_final"] else None,
          "reader_EM_given_gold_incomplete_final":
              totals["gold_incomplete_final_reader_correct"]/
              (96-totals["gold_full_final"]) if totals["gold_full_final"]<96 else None
      },
      "by_question_type":{
          t:{"n":len(items),"reader_EM_correct":sum(x["reader_EM"] for x in items),
            "all_gold_final_n":sum(x["all_gold_support_in_final"] for x in items),
            "mean_gold_recall":sum(x["support_recall"] for x in items)/len(items)}
          for t,items in types.items()},
      "precision_caveat":"Original gold supporting_facts/title+sent_id overlap is annotation coverage, not semantic entailment or necessity, and other evidence can exist.",
      "answer_correct_without_full_gold_not_leak":True,
      "no_method_comparison_this_phase":True,
      "validation_reserved_192_never_read":True,
      "qasper_official_test_untouched":True,
      "per_case":cases}
    new_json(RESULT,result)
    print("PHASE26_POSTGOLD",json.dumps({
       "decision":result["decision"],"counts":dict(totals),
       "rates":result["rates"],"types":result["by_question_type"]},
       ensure_ascii=False),flush=True)
if __name__=="__main__":main()
