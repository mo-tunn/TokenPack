"""Posthoc gold-local failure audit of the ALREADY FINISHED 96 Qwen27B cases.

No inference, models, network, parameter search, or data changes.
This is diagnostic development data, NOT a new blind test.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/"submission/results/phase7_hotpotqa_reader256_20261012/gold_blind_contexts.jsonl"
ANS=ROOT/"submission/results/phase7_modal_qwen27b_hotpotqa_matched96_20261012/answer_em_f1_report.json"
DATA=ROOT/"submission/data/phase6_hotpotqa_validation_20261011/distractor/validation-00000-of-00001.parquet"
OUT=ROOT/"submission/results/phase8_failure_diagnostic_20261012"
ARMS=("full-source","manual-25","msmarco-25")


def norm(s):
    return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(s)).casefold()).strip()


def inspect_evidence(row,context):
    article=dict(zip(row["context"]["title"],row["context"]["sentences"],strict=True))
    tagged=list(dict.fromkeys(zip(row["supporting_facts"]["title"],
                                  row["supporting_facts"]["sent_id"],strict=True)))
    text=norm(context)
    facts=[]
    for title,index in tagged:
        sentence=norm(article[title][int(index)])
        facts.append({"title":title,"id":int(index),
                      "present":bool(sentence and sentence in text)})
    titles=set(x["title"] for x in facts)
    found={x["title"] for x in facts if x["present"]}
    answer=norm(row["answer"])
    factoid=answer not in ("yes","no") and bool(answer)
    return {
        "gold_fact_count":len(facts),
        "gold_title_count":len(titles),
        "all_support_sentences":int(all(x["present"] for x in facts)),
        "support_sentence_recall":sum(x["present"] for x in facts)/len(facts),
        "all_support_titles":int(titles<=found),
        "missing_gold_titles":sorted(titles-found),
        "answer_string_eligible":factoid,
        "answer_string_present":int(answer in text) if factoid else None,
    }


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Never overwrite a prior diagnosis")
    finished=json.loads(ANS.read_text(encoding="utf8"))
    ids={x["case_id"] for x in finished["scored_rows"]}
    if len(ids)!=96:raise AssertionError("Need all 96 reader cases")
    contexts={x["id"]:x for x in
              (json.loads(line) for line in PREP.read_text(encoding="utf8").splitlines())
              if x["id"] in ids}
    raw={x["id"]:x for x in pq.read_table(DATA).to_pylist()
         if x["id"] in ids}
    assert set(raw)==set(contexts)==ids
    preds=defaultdict(dict)
    for x in finished["scored_rows"]:
        preds[x["case_id"]][x["method"]]=x
    rows=[]
    for cid,by_arm in preds.items():
        question=raw[cid]
        result={
            "id":cid,"question":question["question"],
            "level":question.get("level"),"type":question.get("type"),
            "gold_answer":question["answer"],
            "gold_support_titles":list(dict.fromkeys(
                question["supporting_facts"]["title"])),
        }
        for arm in ARMS:
            rec=contexts[cid]["contexts"][arm]
            result[arm]={
                **inspect_evidence(question,rec["text"]),
                "reader_exact_match":int(by_arm[arm]["exact_match"]),
                "reader_f1":float(by_arm[arm]["token_f1"]),
                "reader_prediction":by_arm[arm]["model_answer"],
                "realized_tokens":rec["tokens"],
            }
        rows.append(result)
    groups={
        "all96":rows,
        "full_source_correct":[x for x in rows if x["full-source"]["reader_exact_match"]],
        "msmarco_full_correct_lost":[x for x in rows
                                    if x["full-source"]["reader_exact_match"]
                                    and not x["msmarco-25"]["reader_exact_match"]],
        "manual_full_correct_lost":[x for x in rows
                                    if x["full-source"]["reader_exact_match"]
                                    and not x["manual-25"]["reader_exact_match"]],
        "msmarco_rescues_manual":[x for x in rows if
                                  x["msmarco-25"]["reader_exact_match"]
                                  and not x["manual-25"]["reader_exact_match"]],
        "manual_rescues_msmarco":[x for x in rows if
                                  x["manual-25"]["reader_exact_match"]
                                  and not x["msmarco-25"]["reader_exact_match"]],
    }
    def stat(rows,arm):
        aa=[x[arm] for x in rows]
        return {
            "n":len(rows),
            "full_support_retained":sum(x["all_support_sentences"] for x in aa),
            "all_support_titles_retained":sum(x["all_support_titles"] for x in aa),
            "any_gold_sentence_absent":sum(x["support_sentence_recall"]<1 for x in aa),
            "no_gold_sentence":sum(x["support_sentence_recall"]==0 for x in aa),
            "mean_support_sentence_recall":round(
                sum(x["support_sentence_recall"] for x in aa)/max(1,len(aa)),6),
            "reader_correct":sum(x["reader_exact_match"] for x in aa),
            "factoid_answer_text_present":sum(bool(x["answer_string_present"])
                                                for x in aa),
            "mean_realized_tokens":round(
                sum(x["realized_tokens"] for x in aa)/max(1,len(aa)),1),
        }
    summary={
        "study":"Phase8 posthoc diagnosis of 96 completed Modal Qwen27B questions",
        "only_existing_reader_and_context_outputs":True,
        "observational_only_not_a_gold_blind_algorithm":True,
        "groups":{
            group:{"case_count":len(subset),
                   "manual-25":stat(subset,"manual-25"),
                   "msmarco-25":stat(subset,"msmarco-25")}
            for group,subset in groups.items()},
        "msmarco_full_source_correct_loss_cases":[
            {
                "id":x["id"],"question":x["question"],
                "gold_answer":x["gold_answer"],
                "reader_answer":x["msmarco-25"]["reader_prediction"],
                "evidence_all_present":bool(x["msmarco-25"]["all_support_sentences"]),
                "missing_gold_titles":x["msmarco-25"]["missing_gold_titles"],
                "evidence_fraction":x["msmarco-25"]["support_sentence_recall"],
                "manual_reader_correct":bool(x["manual-25"]["reader_exact_match"]),
            } for x in groups["msmarco_full_correct_lost"]],
    }
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/"annotated_cases.jsonl").write_text(
        "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),
        encoding="utf8")
    (OUT/"failure_breakdown.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FAILURE_AUDIT",json.dumps(summary["groups"],
                                    ensure_ascii=False),flush=True)
    print("ANALYSIS_SAVED",OUT,flush=True)


if __name__=="__main__":
    main()
