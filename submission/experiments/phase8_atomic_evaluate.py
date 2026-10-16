"""Gold-local scoring AFTER SHA256 frozen Phase8 output is complete.

No model inference, no API, no selection changes. Exploratory evaluation
of 152 Phase7-prepared development questions (NOT a new unseen benchmark
relative to earlier HotpotQA analyses).
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase8_atomic_protocol import (
    EVAL_COUNT,EVAL_START,EVAL_STOP,ARMS,READER_ADVANCE_THRESHOLD_PP,
)
from submission.experiments.phase6_hotpot_eval import DATA,PROVENANCE

OUT=ROOT/"submission/results/phase8_atomic_goldblind152_20261012"


def normalized(s):
    import re
    import unicodedata
    return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(s)).casefold()).strip()


def measure(row,text):
    articles={str(k):[str(s) for s in sentences] for k,sentences
              in zip(row["context"]["title"],row["context"]["sentences"],
                     strict=True)}
    facts=list(dict.fromkeys(
        (str(t),int(i)) for t,i in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],strict=True)))
    if not facts:raise AssertionError("HotpotQA question without annotated evidence")
    source=all((t in articles and 0<=i<len(articles[t])
                and bool(normalized(articles[t][i]))) for t,i in facts)
    if not source:raise AssertionError("Missing gold annotation in source")
    output=normalized(text)
    presence=[normalized(articles[t][i]) in output for t,i in facts]
    found_titles={t for flag,(t,i) in zip(presence,facts) if flag}
    titles={t for t,i in facts}
    ans=normalized(row["answer"])
    factoid=bool(ans) and ans not in ("yes","no")
    return {
        "all_support":int(all(presence)),
        "support_sentence_recall":sum(presence)/len(presence),
        "all_gold_source_titles":int(titles<=found_titles),
        "answer_string_eligible":factoid,
        "answer_string_present":int(ans in output) if factoid else None,
        "gold_sentence_count":len(facts),
    }


def paired_bootstrap(delta,replicates=10000):
    a=np.asarray(delta,dtype=float)
    gen=np.random.default_rng(20261012)
    arr=a[gen.integers(0,len(a),size=(replicates,len(a)))].mean(axis=1)
    return [round(float(x),6) for x in np.quantile(arr,[.025,.975])]


def run():
    frozen=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    source=OUT/"gold_blind_atomic_contexts.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=frozen["contexts_sha256"]:
        raise AssertionError("Gold-blind contexts changed after freeze")
    rows=[json.loads(x) for x in source.read_text(encoding="utf8").splitlines()
          if x.strip()]
    if len(rows)!=EVAL_COUNT:
        raise AssertionError("Incomplete gold-blind contexts")
    prov=json.loads(PROVENANCE.read_text(encoding="utf8"))
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=prov["sha256"]:
        raise AssertionError("Benchmark bytes changed")
    ids={row["id"] for row in rows}
    gold={x["id"]:x for x in pq.read_table(DATA).to_pylist()
          if x["id"] in ids}
    if set(gold)!=ids:raise AssertionError("Inconsistent ID set")
    scored=[]
    for rec in rows:
        if set(rec["contexts"])!=set(ARMS):
            raise AssertionError("Missing method")
        question=gold[rec["id"]]
        for arm in ARMS:
            ctx=rec["contexts"][arm]
            scored.append({
                "case_id":rec["id"],"method":arm,
                "budget":ctx["budget"],"used_tokens":ctx["tokens"],
                **measure(question,ctx["text"]),
            })
    cases=defaultdict(dict)
    for record in scored:
        if record["used_tokens"]>record["budget"]:
            raise AssertionError("Real token budget breach")
        cases[record["case_id"]][record["method"]]=record
    if len(cases)!=152 or any(set(x)!=set(ARMS) for x in cases.values()):
        raise AssertionError("Incomplete pairing")
    arm_stats={}
    for arm in ARMS:
        aa=[r[arm] for r in cases.values()]
        factoid=[x for x in aa if x["answer_string_eligible"]]
        arm_stats[arm]={
            "n":len(aa),
            "all_support_rate":round(statistics.mean(x["all_support"] for x in aa),6),
            "mean_support_sentence_recall":round(
                statistics.mean(x["support_sentence_recall"] for x in aa),6),
            "gold_title_coverage_rate":round(
                statistics.mean(x["all_gold_source_titles"] for x in aa),6),
            "factoid_answer_text_rate":round(
                statistics.mean(x["answer_string_present"] for x in factoid),6),
            "n_factoid":len(factoid),
            "mean_used_tokens":round(statistics.mean(x["used_tokens"] for x in aa),2),
            "mean_budget_tokens":round(statistics.mean(x["budget"] for x in aa),2),
            "budget_violations":0,
        }
    comparisons={}
    for method in ("atomic-sentence-rank","atomic-bridge-first"):
        key=method+" minus msmarco-25"
        samples=[(x[method],x["msmarco-25"]) for x in cases.values()]
        comparisons[key]={}
        for metric in ("all_support","support_sentence_recall",
                       "all_gold_source_titles"):
            values=[float(a[metric])-float(b[metric]) for a,b in samples]
            comparisons[key][metric]={
                "mean_difference":round(statistics.mean(values),6),
                "paired_question_bootstrap_ci95":paired_bootstrap(values),
                "wins":sum(x>0 for x in values),
                "losses":sum(x<0 for x in values),
                "ties":sum(x==0 for x in values),
            }
    result={
        "experiment":"Phase8 sentence-level two-hop evidence packing, development ablation",
        "n":len(cases),"sample_indices":[EVAL_START,EVAL_STOP],
        "previous_104_Qwen_reader_questions_excluded":True,
        "no_remote_Qwen_reader_calls":True,
        "gold_blind_context_hash_checked":True,
        "gold_was_not_loaded_until_after_contexts_saved":True,
        "different_compact_source_header_format_than_baseline":True,
        "same_real_cl100k_base_output_ceiling":True,
        "status":"development only; HotpotQA annotations encountered in earlier Phase6",
        "arms":arm_stats,"contrasts":comparisons,
        "readiness_threshold_pp":READER_ADVANCE_THRESHOLD_PP,
        "readiness_rule":"Advance only if strict all-support rate improves >=8.0pp over frozen MS MARCO 300, no budget breaches",
        "readiness_met":{
            k:100*comparisons[k]["all_support"]["mean_difference"]>=
            READER_ADVANCE_THRESHOLD_PP for k in comparisons
        },
    }
    result_path=OUT/"PHASE8_GOLD_LOCAL_RESULTS.json"
    if result_path.exists():raise FileExistsError("Refuse to overwrite previous results")
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",
                           encoding="utf8")
    print("PHASE8_RESULT",json.dumps({
        "arm_stats":arm_stats,
        "comparisons":comparisons,
        "readiness_met":result["readiness_met"],
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()
