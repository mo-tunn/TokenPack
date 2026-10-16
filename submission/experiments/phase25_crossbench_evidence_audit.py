"""Post-gold MuSiQue 64-case retrieval/reader diagnostic only.

Does NOT change original EM, context selection, inference, sample or
preregistered descriptive decision. Gold answer and support paragraphs are
read locally only after 64 reader outputs were frozen and scored.
"""
from __future__ import annotations
import collections,json,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase25_crossbench_register import OUT
from submission.experiments.phase25_crossbench_score import contains_whole
from submission.experiments.phase9_musique_contexts import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
RESULT=OUT/"POST_GOLD_EVIDENCE_DIAGNOSTIC.json"
def main():
    if RESULT.exists():raise FileExistsError(RESULT)
    scores=json.loads((OUT/"POST_GOLD_CROSSBENCH_GRANULARITY.json").read_text(encoding="utf8"))
    prereg=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    context_meta=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    reader_meta=json.loads((OUT/"CROSSBENCH_READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==context_meta["context_file_sha256"]
    assert sha_file(OUT/"crossbench_reader_responses.jsonl")==reader_meta["responses_sha256"]
    assert len(scores["cases"])==64 and scores["reserved_validation_IDs_untouched"]==128
    ids=set(prereg["pilot_ids"])
    # Explicitly GOLD-LOCAL columns, after freeze only. No model calls.
    data={r["id"]:r for r in pq.ParquetFile(DATA).read(columns=[
       "id","paragraphs.list.element.idx",
       "paragraphs.list.element.is_supporting"]).to_pylist()
       if r["id"] in ids}
    assert len(data)==64
    contexts={x["id"]:x for x in read_rows(OUT/"gold_blind_contexts.jsonl")}
    totals=collections.Counter()
    rows=[]
    for x in scores["cases"]:
        ident=x["id"]
        r=data[ident]
        needed={int(p["idx"]) for p in r["paragraphs"] if p["is_supporting"]}
        assert needed
        selected=set(contexts[ident]["selection_metadata"]["selected_source_idxs"])
        full=needed<=selected
        gold_string=contains_whole(x["gold_answer"],contexts[ident]["final_text"])
        alias_string=any(contains_whole(a,contexts[ident]["final_text"])
                         for a in [x["gold_answer"]]+x["accepted_gold_aliases"])
        strict_wrong=not x["strict_EM"]
        if strict_wrong and not full:
            cat="A_missing_one_or_more_support_paragraphs"
        elif strict_wrong and not alias_string:
            cat="B_support_paragraphs_touched_but_answer_string_absent"
        elif strict_wrong:
            cat="C_support_paragraphs_touched_and_answer_string_present_but_reader_EM_wrong"
        else:cat="D_strict_EM_correct"
        totals[cat]+=1
        totals["n"]+=1
        totals["support_paragraph_all_present"]+=int(full)
        totals["at_least_one_support_paragraph_present"]+=int(bool(needed&selected))
        totals["gold_string_in_compressed"]+=int(gold_string)
        totals["gold_or_alias_string_in_compressed"]+=int(alias_string)
        totals["full_support_and_answer_string_in_source_but_reader_wrong"]+=int(
             strict_wrong and full and alias_string)
        totals["strict_wrong_answer_alias_recovered"]+=int(
             strict_wrong and x["alias_aware_EM"])
        rows.append({
            "id":ident,"strict_EM":x["strict_EM"],
            "alias_aware_EM":x["alias_aware_EM"],
            "support_paragraphs_annotated":sorted(needed),
            "support_paragraphs_selected_at_least_one_sentence":sorted(needed&selected),
            "all_support_paragraphs_touched":full,
            "gold_or_alias_literal_in_rendered_context":alias_string,
            "gold_primary_literal_in_rendered_context":gold_string,
            "stage_category":cat,
            "caveat":"At least one sentence in supporting paragraph is NOT necessarily sufficient evidence",
            "answer_type":x["question_type_predefined"],
        })
    assert totals["n"]==64
    candidates=[x for x in scores["cases"] if x["category"] in (
       "alias_exact_recovery","gold_or_alias_substring_in_longer_prediction")]
    report={
       "n":64,"public_validation_not_hidden_test":True,
       "post_gold_only":True,"no_additional_model_calls":True,
       "original_scores_unchanged":True,
       "no_validation_reserved_cases_opened":True,
       "source_scores_sha256":sha_file(OUT/"POST_GOLD_CROSSBENCH_GRANULARITY.json"),
       "source_context_sha256":context_meta["context_file_sha256"],
       "counts":dict(totals),
       "lexical_candidate_examples":[{
           "id":x["id"],"question":x["question"],"gold":x["gold_answer"],
           "aliases":x["accepted_gold_aliases"],
           "model_prediction":x["prediction"],
           "category":x["category"],
           "selected_source_support_paragraphs_complete":next(
               r["all_support_paragraphs_touched"] for r in rows if r["id"]==x["id"])
         } for x in candidates],
       "per_case":rows}
    new_json(RESULT,report)
    print("PHASE25_POSTGOLD_SUPPORT_AUDIT",json.dumps({
        "counts":report["counts"],
        "lexical_examples":report["lexical_candidate_examples"]},ensure_ascii=False),
        flush=True)
if __name__=="__main__":main()
