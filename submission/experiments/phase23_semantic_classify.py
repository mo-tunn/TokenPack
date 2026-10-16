"""Local gold-aware, single-reviewer semantic taxonomy of Faz22 errors.

Researcher labels are interpretive, NOT blind inter-rater annotations or
proof of semantic equivalence. Objective gold-substring statistic separate.
No Qwen calls, gold remains local, no edits of prior phases.
"""
from __future__ import annotations
import collections,json,sys,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase23_semantic_case_export import OUT as OUT23
from submission.experiments.phase22_reader_register import OUT as OUT22
from submission.experiments.phase12_register import DATA,sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows,answer_metrics
import pyarrow.parquet as pq

# Human-reviewed posthoc primary coding: never feed these to the new model.
# Category 1: EM false-negative *candidate*, source supports a longer name
# where the entire gold answer appears inside the prediction.
LABELS={
"b5b63b160bdb11eba7f7acde48001122":{
  "code":"ANSWER_SPAN_OVER_SPECIFIED","confidence":"high",
  "why":"Tisch is explicitly contained in the longer cited institution name; entity/institution identity appears preserved.",
  "source_quote":"Bowler graduated from the Tisch School of the Arts of New York University"},
"4b37342c0bde11eba7f7acde48001122":{
  "code":"GEOGRAPHIC_HIERARCHY_OR_ANNOTATION_UNDERDETERMINED","confidence":"low",
  "why":"The supporting sentence gives death place Rabodomiarana but gold answer is Kingdom of Imerina. Source also calls him prime minister of this kingdom, which is not proof of death there. No geographic containment relation is supplied. Do not declare the gold answer factually wrong.",
  "source_quote":"died on 18 October 1852 in Rabodomiarana"},
"32e50df00bb011ebab90acde48001122":{
  "code":"ANSWER_SPAN_OVER_SPECIFIED","confidence":"high",
  "why":"Gold Louis II occurs within the cited full historical title Louis II, Landgrave of Thuringia.",
  "source_quote":"second son of Louis II, Landgrave of Thuringia"},
"e67c313b08a511ebbd7dac1f6bf848b6":{
  "code":"AMBIGUOUS_ENTITY_TITLE_WRONG_REFERENT","confidence":"high",
  "why":"There are distinct sources The Cat's Meow and The Cat's Meow (1924 film). The generic title source states Germany among co-producers; the model reinterprets question as 1924 film and answers No.",
  "source_quote":"international co-production between the United States, the United Kingdom, and Germany"},
"f77f95da0bd911eba7f7acde48001122":{
  "code":"ANSWER_SPAN_OVER_SPECIFIED","confidence":"high",
  "why":"The cited death place is Arcadia, California and gold Arcadia is a contiguous named place within it.",
  "source_quote":"died in Arcadia, California"},
"1222be100baf11ebab90acde48001122":{
  "code":"ANSWER_SPAN_OVER_SPECIFIED","confidence":"high",
  "why":"Cited Valide Sultan Emine Hatun and gold Emine Hatun refer to the same titled person in the source.",
  "source_quote":"His mother was Valide Sultan Emine Hatun"},
"845645da0bda11eba7f7acde48001122":{
  "code":"TARGET_RELATION_NOT_EXPLICITLY_ENTAILED","confidence":"high",
  "why":"The cited source says Victor Civita relocated to Brazil and established Editora Abril in Sao Paulo; this does NOT directly say he died in Sao Paulo. Model substitutes Brazil without a death-location relation.",
  "source_quote":"In 1949 Civita relocated his family to Brazil"},
"6ae79dd40bde11eba7f7acde48001122":{
  "code":"ANSWER_SPAN_OVER_SPECIFIED","confidence":"medium",
  "why":"The source states birth in Casarano, Apulia, Italy. Gold Italy is contained as the country, while model returned the full locality. It is an answer-granularity difference, not evidence of a wrong country.",
  "source_quote":"4 March 1920 in Casarano, Apulia, Italy"},
}
def normal(x):
    return unicodedata.normalize("NFKC",str(x)).casefold().strip()
def main():
    cases=read_rows(OUT23/"POSTHOC_EIGHT_CASES.jsonl")
    assert len(cases)==8 and set(x["id"] for x in cases)==set(LABELS)
    results=[];cnt=collections.Counter()
    for case in cases:
        label=LABELS[case["id"]]
        assert label["source_quote"] in case["final_context"]
        norm_gold=normal(case["gold_answer"])
        norm_pred=normal(case["structured_answer"])
        substring=bool(norm_gold and norm_gold in norm_pred)
        if label["code"]=="ANSWER_SPAN_OVER_SPECIFIED":
            assert substring,case["id"]
        else:assert not substring,case["id"]
        cnt[label["code"]]+=1
        cnt["gold_contiguous_substring_of_structured_answer"]+=int(substring)
        cnt["all_gold_fact_pairs_in_context"]+=int(case["all_annotated_gold_support_present"])
        results.append({
            "id":case["id"],"question":case["question"],"benchmark_type":case["type"],
            "gold_answer":case["gold_answer"],
            "structured_answer":case["structured_answer"],
            "direct_answer":case["control_answer"],
            "label":label["code"],"reviewer_confidence":label["confidence"],
            "reviewer_rationale":label["why"],"original_source_quote":label["source_quote"],
            "objective_gold_answer_substring_of_model_output":substring,
            "gold_annotation_support_sentences_all_present":True,
            "reviewer_is_not_gold_blind":True,
            "reviewer_independent_second_annotation":False,
            "label_is_not_causal_proof":True})
    raw=read_rows(OUT22/"reader_responses.jsonl")
    ids={r["id"] for r in raw}
    assert len(ids)==32
    freeze=json.loads((OUT22/"HYBRID_PRE_GOLD_DECISIONS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(OUT22/"reader_responses.jsonl")==freeze["reader_log_sha256"]
    pred=read_rows(OUT22/"HYBRID_GOLD_BLIND_PREDICTIONS.jsonl")
    assert len(pred)==32
    gold={r["id"]:r["answer"] for r in pq.ParquetFile(DATA).read(columns=[
        "id","answer"]).to_pylist() if r["id"] in ids}
    assert set(gold)==ids
    whole=collections.Counter()
    for r in pred:
        g=normal(gold[r["id"]])
        h=normal(r["prediction"])
        em,_=answer_metrics(r["prediction"],gold[r["id"]])
        if not em:
            whole["all32_hybrid_EM_wrong"]+=1
            whole["all32_hybrid_EM_wrong_gold_substring_in_prediction"]+=int(
                bool(g and h and g in h))
    manifest={
      "n_full_gold_but_hybrid_wrong":8,
      "taxonomy":dict(cnt),
      "all32_objective_substring_statistics":dict(whole),
      "manual_review_posthoc_not_blind":True,
      "source_contexts_original":True,
      "gold_label_not_reinterpreted_as_corrected_answer":True,
      "phase22_outputs_not_rescored":True,
      "casebook_sha256":sha_file(OUT23/"POSTHOC_EIGHT_CASES.jsonl"),
      "phase22_reader_sha256":sha_file(OUT22/"reader_responses.jsonl"),
      "no_new_model_calls":True,
      "no_new_validation_IDs_touched":True,
      "not_semantic_accuracy_certification":True,
      "single_reviewer_no_independent_agreement":True,
      "uncertainty_case_notes":"Geographic hierarchy and named-entity-level answers lack independent semantic annotation."}
    dest=OUT23/"MANUAL_SEMANTIC_TAXONOMY.json"
    if dest.exists():raise FileExistsError(dest)
    new_json(dest,{"summary":manifest,"cases":results})
    print("PHASE23_MANUAL_TAXONOMY",json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
