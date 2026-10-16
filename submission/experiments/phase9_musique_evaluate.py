"""Post-freeze evaluation of new benchmark; MuSiQue support PARAGRAPH labels.

MuSiQue gold paragraph index, not sentence evidence. Count any selected
original sentence from paragraph as a hit. This is a WEAK structural
coverage metric, explicitly not proof of sufficient evidence or QA.
Gold labels are read here, only after output bytes are frozen and checked.
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

from submission.experiments.phase9_musique_contexts import (
    DATA,PROV,OUTPUT,
)
from submission.experiments.phase9_musique_protocol import (
    N_QUESTIONS,METHODS,DATA_SHA256,
)


def normalize(s):
    import re
    import unicodedata
    return " ".join(re.findall(
        r"[a-z0-9]+",unicodedata.normalize("NFKC",str(s)).casefold()))


def paired_ci(values,seed=20261012,rep=10000):
    rng=np.random.default_rng(seed)
    a=np.asarray(values,dtype=float)
    bootstrap=a[rng.integers(len(a),size=(rep,len(a)))].mean(axis=1)
    return [round(float(v),6) for v in np.quantile(bootstrap,[.025,.975])]


def run():
    resultpath=OUTPUT/"EXTERNAL_MUSIQUE_RESULTS.json"
    if resultpath.exists():raise FileExistsError("Do not rewrite original evaluation")
    frozen=json.loads((OUTPUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    contexts=OUTPUT/"gold_blind_contexts.jsonl"
    if hashlib.sha256(contexts.read_bytes()).hexdigest()!=frozen["case_file_sha256"]:
        raise AssertionError("Gold-blind candidate decisions were modified")
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=DATA_SHA256:
        raise AssertionError("Benchmark digest changed")
    rows=[json.loads(x) for x in contexts.read_text(encoding="utf8").splitlines()
          if x.strip()]
    if len(rows)!=N_QUESTIONS:raise AssertionError("Missing questions")
    sample_ids={x["id"] for x in rows}
    if len(sample_ids)!=N_QUESTIONS:
        raise AssertionError("Nonunique question ID")
    gold={x["id"]:x for x in pq.ParquetFile(DATA).read(
        columns=["id","paragraphs","answer","answer_aliases",
                 "answerable"]).to_pylist() if x["id"] in sample_ids}
    if set(gold)!=sample_ids:raise AssertionError("Wrong gold labels")
    records=[]
    for case in rows:
        label=gold[case["id"]]
        supporting={int(p["idx"]) for p in label["paragraphs"]
                    if p["is_supporting"]}
        available={int(p["idx"]) for p in label["paragraphs"]
                   if str(p["paragraph_text"]).strip()}
        if not supporting<=available:
            raise AssertionError("Source supporting paragraph is absent")
        for arm in METHODS:
            ctx=case["contexts"][arm]
            chosen=set(int(i) for i in ctx["selected_source_idxs"])
            if not chosen<=available:
                raise AssertionError("Source provenance index invalid")
            if ctx["tokens"]>case["budget"]:
                raise AssertionError("Token budget breached")
            covered=len(chosen&supporting)
            valid_support=bool(supporting)
            factoid=bool(normalize(label["answer"]))
            output_text=normalize(ctx["text"])
            matches=([label["answer"]]+(label.get("answer_aliases") or []))
            ans_present=any(
                bool(normalize(v)) and normalize(v) in output_text for v in matches)
            records.append({
                "case_id":case["id"],"method":arm,
                "hop_category":case["id"].split("__")[0],
                "answerable":bool(label["answerable"]),
                "supporting_paragraphs":len(supporting),
                "support_paragraph_all_present":int(valid_support and
                                                    supporting<=chosen),
                "support_paragraph_recall":covered/len(supporting)
                    if valid_support else 0.,
                "support_paragraph_any_present":int(bool(supporting&chosen)),
                "answer_or_alias_text_present":int(ans_present) if factoid else None,
                "actual_tokens":ctx["tokens"],
                "budget":case["budget"],
                "selected_paragraphs":len(chosen),
            })
    cases=defaultdict(dict)
    for rec in records:
        cases[rec["case_id"]][rec["method"]]=rec
    if len(cases)!=N_QUESTIONS or any(set(x)!=set(METHODS) for x in cases.values()):
        raise AssertionError("Non-paired evaluation")
    arms={}
    for arm in METHODS:
        x=[c[arm] for c in cases.values()]
        arms[arm]={
            "n":len(x),
            "all_supporting_paragraphs_covered":sum(
                a["support_paragraph_all_present"] for a in x),
            "all_supporting_paragraph_rate":round(statistics.mean(
                a["support_paragraph_all_present"] for a in x),6),
            "mean_supporting_paragraph_recall":round(statistics.mean(
                a["support_paragraph_recall"] for a in x),6),
            "answer_or_alias_text_present":round(statistics.mean(
                a["answer_or_alias_text_present"] for a in x
                if a["answer_or_alias_text_present"] is not None),6),
            "mean_used_tokens":round(statistics.mean(
                a["actual_tokens"] for a in x),2),
            "mean_budget":round(statistics.mean(
                a["budget"] for a in x),2),
            "mean_paragraphs_selected":round(statistics.mean(
                a["selected_paragraphs"] for a in x),2),
            "budget_violations":0,
        }
    contrasts={}
    pairs=[(x["crossdoc-bridge"],x["atomic-rank"]) for x in cases.values()]
    for metric in (
        "support_paragraph_all_present","support_paragraph_recall",
        "answer_or_alias_text_present",
    ):
        delta=[float(a[metric])-float(b[metric]) for a,b in pairs
               if a[metric] is not None and b[metric] is not None]
        contrasts[metric]={
            "difference":round(statistics.mean(delta),6),
            "question_bootstrap_ci95":paired_ci(delta),
            "wins":sum(x>0 for x in delta),
            "losses":sum(x<0 for x in delta),
            "ties":sum(x==0 for x in delta),
        }
    byhop={}
    for category in sorted({x["hop_category"] for x in records}):
        grouped=[x for x in pairs if x[0]["hop_category"]==category]
        if not grouped:continue
        byhop[category]={
            "n":len(grouped),
            "atomic_all_support":round(statistics.mean(
                b["support_paragraph_all_present"] for a,b in grouped),6),
            "bridge_all_support":round(statistics.mean(
                a["support_paragraph_all_present"] for a,b in grouped),6),
        }
    summary={
        "experiment":"Pristine 256 MuSiQue validation ID-only sample, gold evaluated after render freeze",
        "n":N_QUESTIONS,"method_order":METHODS,
        "data_sha256":DATA_SHA256,
        "gold_seen_only_after_frozen_context_hash":True,
        "is_not_hidden_test":True,
        "metric":"Gold supporting paragraph index has AT LEAST one selected original sentence; not full paragraph preservation",
        "arms":arms,"contrasts_bridge_minus_atomic":contrasts,
        "hop_breakdown":byhop,
        "question_records":records,
        "limitations":[
            "Supporting paragraphs hit by one source sentence do not prove answer sufficiency.",
            "MuSiQue labels may include unanswerable questions; all IDs retained.",
            "Model and fixed hyperparameters never tuned on MuSiQue.",
            "Same max cl100k rendered budget, realized tokens may differ.",
            "Distinct benchmark but public validation, not held-out hidden test.",
        ],
    }
    resultpath.write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("MUSIQUE_GOLD_RESULTS",json.dumps({
        "arms":arms,"contrasts":contrasts,"byhop":byhop,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    run()
