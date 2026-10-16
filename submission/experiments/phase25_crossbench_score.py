"""Phase25 post-response-freeze MuSiQue lexical answer-calibration audit.

Not a model optimization: no LLM calls, no benchmark answers or aliases
read before all 64 answers are SHA-256 frozen. Contiguous substring
means LEXICAL POTENTIAL false-negative only (NOT semantic equivalence).
"""
from __future__ import annotations
import collections,hashlib,json,re,sys,unicodedata,math
from pathlib import Path
from scipy.stats import binomtest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase25_crossbench_register import OUT
from submission.experiments.phase9_musique_contexts import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import answer_metrics,read_rows
RESULT=OUT/"POST_GOLD_CROSSBENCH_GRANULARITY.json"
def canonical(x):
    s=unicodedata.normalize("NFKC",str(x or "")).casefold()
    return " ".join(s.split())
def contains_whole(gold,pred):
    g=canonical(gold)
    p=canonical(pred)
    if not g or not p:return False
    pattern=r"(?<!\w)"+re.escape(g)+r"(?!\w)"
    return re.search(pattern,p,flags=re.UNICODE) is not None
def question_type(question):
    q=canonical(question)
    if any(p in q for p in ("what country","which country","in which country",
                             "what nation","which nation")):
        return "country_requested"
    if any(p in q for p in ("which city","what city","which town","what town")):
        return "city_or_town_requested"
    if q.startswith("where") or "where was" in q or "where did" in q:
        return "location_unspecified_level"
    if q.startswith("when") or "what year" in q or "which year" in q or "what date" in q:
        return "time_date_requested"
    if q.startswith("who") or "whose " in q:
        return "person_or_group_requested"
    return "unspecified_what_which_other"
def ci_wilson(k,n):
    if not n:return [0,0]
    t=binomtest(k,n)
    c=t.proportion_ci(confidence_level=.95,method="wilson")
    return [float(c.low),float(c.high)]
def main():
    assert not RESULT.exists()
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    f=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    r=json.loads((OUT/"CROSSBENCH_READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert p["no_gold_read_before_reader_hash_freeze"]
    assert sha_file(DATA)==p["data_sha256"]
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==f["context_file_sha256"]
    assert sha_file(OUT/"crossbench_reader_responses.jsonl")==r["responses_sha256"]
    assert f["totals"]["final_cap_violations"]==0
    answers=read_rows(OUT/"crossbench_reader_responses.jsonl")
    context=read_rows(OUT/"gold_blind_contexts.jsonl")
    assert len(answers)==len(context)==64
    assert [a["id"] for a in answers]==[a["id"] for a in context]==p["pilot_ids"]
    assert not (OUT/"crossbench_reader_error.jsonl").exists()
    assert not (OUT/"crossbench_reader_inflight.json").exists()
    names=set(p["pilot_ids"])
    gold={row["id"]:row for row in pq.ParquetFile(DATA).read(
        columns=["id","answer","answer_aliases","answerable"]).to_pylist()
        if row["id"] in names}
    assert len(gold)==64
    counts=collections.Counter()
    cases=[]
    bytype=collections.defaultdict(list)
    for a,ctx in zip(answers,context):
        id=a["id"]
        g=gold[id]
        primary=str(g["answer"] or "")
        aliases=[str(x) for x in (g["answer_aliases"] or []) if str(x).strip()]
        reference=[primary]+[x for x in aliases if canonical(x)!=canonical(primary)]
        base_em,base_f1=answer_metrics(a["answer"],primary)
        alias_em=max(answer_metrics(a["answer"],x)[0] for x in reference)
        alias_f1=max(answer_metrics(a["answer"],x)[1] for x in reference)
        substring_any=any(contains_whole(x,a["answer"]) for x in reference)
        substring_gold=contains_whole(primary,a["answer"])
        lengthen=bool(substring_any and canonical(a["answer"])
                      not in {canonical(x) for x in reference})
        # Four mutually exclusive categories.
        if base_em:category="strict_correct"
        elif alias_em:category="alias_exact_recovery"
        elif lengthen:category="gold_or_alias_substring_in_longer_prediction"
        else:category="other_em_failure"
        counts[category]+=1
        counts["n"]+=1
        counts["alias_count_total"]+=len(aliases)
        counts["literal_EM"]+=int(base_em)
        counts["alias_EM"]+=int(alias_em)
        counts["whole_gold_substring_n"]+=int(substring_gold)
        counts["any_alias_or_gold_substring_n"]+=int(substring_any)
        counts["strict_wrong_with_gold_or_alias_whole_substring"]+=int(not base_em and lengthen)
        counts["strict_wrong_alias_exact"]+=int(not base_em and alias_em)
        counts["empty_answers"]+=int(not a["answer"])
        counts["sum_pred_character_count"]+=len(a["answer"])
        qtype=question_type(ctx["question"])
        rec={
           "id":id,"question":ctx["question"],"question_type_predefined":qtype,
           "gold_answer":primary,"accepted_gold_aliases":aliases,
           "prediction":a["answer"],"strict_EM":base_em,"strict_F1":base_f1,
           "alias_aware_EM":alias_em,"alias_aware_F1":alias_f1,
           "gold_whole_word_substring_in_prediction":substring_gold,
           "any_gold_or_alias_whole_word_substring":substring_any,
           "strict_wrong_but_longer_contains_reference":int(not base_em and lengthen),
           "category":category,"answerable_gold":g["answerable"],
           "actual_source_tokens":ctx["actual_tokens"],
           "lexical_containment_is_not_semantic_equivalence":True
        }
        cases.append(rec)
        bytype[qtype].append(rec)
    assert counts["n"]==64 and sum(counts[x] for x in (
         "strict_correct","alias_exact_recovery",
         "gold_or_alias_substring_in_longer_prediction","other_em_failure"))==64
    combo=counts["strict_wrong_with_gold_or_alias_whole_substring"]
    alias_lift=counts["alias_EM"]-counts["literal_EM"]
    rule=p["specificity_candidate_gate"]
    gate_pass=combo>=6 and alias_lift>=4
    stats={
      "n":64,"metrics":{
          "literal_EM":counts["literal_EM"]/64,
          "alias_aware_EM":counts["alias_EM"]/64,
          "strict_answer_F1":sum(r["strict_F1"] for r in cases)/64,
          "alias_aware_F1":sum(r["alias_aware_F1"] for r in cases)/64,
          "strict_EM_count":counts["literal_EM"],
          "alias_EM_count":counts["alias_EM"],
          "alias_only_EM_recoveries":alias_lift,
          "gold_or_alias_whole_span_in_wrong_EM_n":combo,
          "gold_or_alias_whole_span_in_wrong_EM_fraction_all64":combo/64,
          "candidate_substring_wilson95":ci_wilson(combo,64),
          "average_pred_chars":counts["sum_pred_character_count"]/64,
          "empty_pred_n":counts["empty_answers"],
          "strict_wrong_EM_n":64-counts["literal_EM"]
      },
      "categories":{x:counts[x] for x in (
          "strict_correct","alias_exact_recovery",
          "gold_or_alias_substring_in_longer_prediction","other_em_failure")},
      "answer_type_strata":{k:{
          "n":len(v),"strict_EM":sum(r["strict_EM"] for r in v),
          "alias_EM":sum(r["alias_aware_EM"] for r in v),
          "whole_reference_span_but_EM_wrong":sum(
              r["strict_wrong_but_longer_contains_reference"] for r in v)
          } for k,v in bytype.items()},
      "predeclared_descriptive_crossbench_signal_met":gate_pass,
      "predeclared_gate_details":{
         "needed_substring_cases":6,"observed_substring_cases":combo,
         "needed_alias_EM_lift":4,"observed_alias_EM_lift":alias_lift},
      "decision":"CROSSBENCH_LEXICAL_SIGNAL_PRESENT_NOT_SEMANTICALLY_CERTIFIED"
          if gate_pass else "CROSSBENCH_LEXICAL_SIGNAL_NOT_REPLICATED",
      "reserved_validation_IDs_untouched":128,
      "not_evidence_of_meaning_equivalence":True,
      "no_semantic_interrater_or_blind_gold_free_review":True,
      "not_an_accuracy_improving_new_method":True,
      "public_validation_not_hidden_test":True,
      "no_new_model_calls_after_gold":True,
      "source_response_sha256":r["responses_sha256"],
      "source_context_sha256":f["context_file_sha256"],
      "gold_opened_only_after_hash_freeze":True,
      "cases":cases
    }
    new_json(RESULT,stats)
    print("PHASE25_CROSSBENCH_SCORING",json.dumps({
        "metrics":stats["metrics"],"categories":stats["categories"],
        "strata":stats["answer_type_strata"],
        "decision":stats["decision"],
        "reserved_untouched":True},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
