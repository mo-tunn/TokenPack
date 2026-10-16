"""Faz15 post-freeze gold evaluation of exact support sentences and relation types.

Only invoked after reader outputs complete; gold labels never enter any
planner, ranker, selector, or Modal inference.
"""
from __future__ import annotations
import collections,itertools,json,math,re,statistics,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase15_register import OUT
from submission.experiments.phase15_protocol import METHODS
from submission.experiments.phase12_register import (
    DATA,verify_data,sha_file,new_json
)
from submission.experiments.phase12_contexts import candidates
from submission.experiments.phase9_crossdoc_core import render_atomic
from tokenpack.tokenization import TokenCounter

def canonical(s):
    return " ".join(str(s or "").casefold().split())
def normalized(s):
    return " ".join(re.findall(r"\w+",str(s or "").casefold(),flags=re.UNICODE))
def mentions(s,ctx):
    key=normalized(s)
    if not key:return False
    return bool(re.search(r"(?<!\w)"+re.escape(key)+r"(?!\w)",normalized(ctx)))
def ci(vals):
    arr=np.asarray(vals,dtype=float)
    rng=np.random.default_rng(20261012)
    m=arr[rng.integers(0,len(arr),(10000,len(arr)))].mean(axis=1)
    return [float(x) for x in np.quantile(m,[.025,.975])]

def main():
    prereg=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    context_file=OUT/"gold_blind_contexts.jsonl"
    assert sha_file(context_file)==frozen["sha256"]
    reader_file=OUT/"READER_POST_GOLD_RESULTS.json"
    reader=json.loads(reader_file.read_text(encoding="utf8"))
    assert reader["n"]==32
    cases=[json.loads(s) for s in context_file.open(encoding="utf8") if s.strip()]
    assert [x["id"] for x in cases]==prereg["pilot_ids"]
    verify_data()
    ids=set(prereg["pilot_ids"])
    gold={x["id"]:x for x in pq.ParquetFile(DATA).read(
        columns=["id","type","supporting_facts","evidences","context"]).to_pylist()
        if x["id"] in ids}
    assert set(gold)==ids
    counter=TokenCounter()
    assert counter._encoding is not None
    metrics=[]
    oracle=[]
    for case in cases:
        row=gold[case["id"]]
        support={(canonical(a),int(b)) for a,b in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"])}
        assert support
        cmap=[(canonical(a["title"]),int(a["sent_id"])) for a in case["candidate_map"]]
        assert support<=set(cmap),"Unexpected unsupported annotated sentence"
        atoms,full,blocks=candidates({"context":row["context"]})
        assert [(canonical(a["title"]),int(a["sent_id"])) for a in atoms]==cmap
        support_candidates=[[i for i,key in enumerate(cmap) if key==s]
            for s in sorted(support)]
        combinations=math.prod(len(x) for x in support_candidates)
        assert combinations<=15000
        exact_min=min(counter.count(render_atomic(combo,atoms))
            for combo in itertools.product(*support_candidates))
        oracle.append({"id":case["id"],"type":row["type"],
            "required_gold_sentences":len(support),
            "oracle_min_tokens":exact_min,"cap":case["budget"],
            "gold_all_fit_25pct":int(exact_min<=case["budget"])})
        ev=[t for t in (row.get("evidences") or []) if len(t)==3]
        src={canonical(t[0]) for t in ev}
        dst={canonical(t[2]) for t in ev}
        intermediate=sorted(src&dst-{""})
        for arm in METHODS:
            ctx=case["contexts"][arm]
            assert ctx["tokens"]==counter.count(ctx["text"])<=ctx["budget"]==case["budget"]
            chosen={cmap[i] for i in ctx["selected_sentence_indexes"]}
            captured=len(chosen&support)
            goldtitles={x[0] for x in support}
            chosentitles={x[0] for x in chosen}
            has_mid=[mentions(x,ctx["text"]) for x in intermediate]
            has_edge=[mentions(x[0],ctx["text"]) and mentions(x[2],ctx["text"]) for x in ev]
            metrics.append({"id":case["id"],"method":arm,"type":row["type"],
                "gold_sentence_count":len(support),
                "all_gold_sentences":int(support<=chosen),
                "gold_sentence_recall":captured/len(support),
                "all_gold_titles":int(goldtitles<=chosentitles),
                "selected_sentence_count":ctx["selected_sentence_count"],
                # Recompute source count from actual SELECTED indices. The
                # legacy Phase11 flat-edge selector uses defaultdict(int)
                # during trial-utility reads, which can create empty keys and
                # inflate its reported selected_paragraph_count metadata.
                "selected_paragraph_count":len({
                    case["candidate_map"][i]["article"]
                    for i in ctx["selected_sentence_indexes"]}),
                "actual_cl100k_tokens":ctx["tokens"],"cap":ctx["budget"],
                "bridge_intermediate_entity_count":len(intermediate),
                "all_intermediate_entity_names_mentioned":int(all(has_mid)) if has_mid else None,
                "triple_endpoint_mention_fraction":sum(has_edge)/len(has_edge) if has_edge else None,
            })
    by=collections.defaultdict(dict)
    for r in metrics:by[r["id"]][r["method"]]=r
    assert len(by)==32 and all(set(x)==set(METHODS) for x in by.values())
    counts={}
    for arm in METHODS:
        x=[v[arm] for v in by.values()]
        counts[arm]={"n":len(x),
            "all_support_sentences_n":sum(t["all_gold_sentences"] for t in x),
            "all_support_sentences_rate":statistics.mean(t["all_gold_sentences"] for t in x),
            "mean_support_sentence_recall":statistics.mean(t["gold_sentence_recall"] for t in x),
            "all_gold_titles_n":sum(t["all_gold_titles"] for t in x),
            "mean_real_tokens":statistics.mean(t["actual_cl100k_tokens"] for t in x),
            "mean_cap":statistics.mean(t["cap"] for t in x),
            "mean_sentences":statistics.mean(t["selected_sentence_count"] for t in x),
            "mean_paragraphs":statistics.mean(t["selected_paragraph_count"] for t in x),
            "cap_violations":0,
            "all_intermediate_name_mentioned_n":sum(
                t["all_intermediate_entity_names_mentioned"] for t in x
                if t["all_intermediate_entity_names_mentioned"] is not None),
            "intermediate_name_label_available":sum(
                t["all_intermediate_entity_names_mentioned"] is not None for t in x)}
    comparisons={}
    for a,b in (("conditional-rescue","top4-verified"),):
        x=list(by.values())
        binary=[t[a]["all_gold_sentences"]-t[b]["all_gold_sentences"] for t in x]
        recall=[t[a]["gold_sentence_recall"]-t[b]["gold_sentence_recall"] for t in x]
        wins=sum(d==1 for d in binary);losses=sum(d==-1 for d in binary)
        comparisons[a+"__minus__"+b]={
            "all_sentence_coverage_difference":statistics.mean(binary),
            "all_sentence_coverage_CI95":ci(binary),
            "only_first_covers_all":wins,"only_second_covers_all":losses,
            "exact_mcnemar_p":binomtest(min(wins,losses),wins+losses,.5).pvalue
                if wins+losses else 1.0,
            "recall_difference":statistics.mean(recall),
            "recall_CI95":ci(recall)}
    typ={}
    for kind in sorted(set(x["type"] for x in metrics)):
        xs=[v for v in by.values() if v[METHODS[0]]["type"]==kind]
        typ[kind]={"n":len(xs),
            "gold_sentence_n":xs[0][METHODS[0]]["gold_sentence_count"],
            "all_support_by_arm":{
                a:sum(t[a]["all_gold_sentences"] for t in xs) for a in METHODS},
            "all_gold_titles_by_arm":{
                a:sum(t[a]["all_gold_titles"] for t in xs) for a in METHODS}}
    oracle_by_type={}
    for k in sorted(set(x["type"] for x in oracle)):
        x=[a for a in oracle if a["type"]==k]
        oracle_by_type[k]={"n":len(x),
            "gold_aware_feasible_n":sum(a["gold_all_fit_25pct"] for a in x)}
    result={
        "dataset":"2WikiMultiHopQA public validation, Phase15 preregistered 32 IDs",
        "gold_read_only_after_complete_reader":True,
        "inference_gold_leakage":False,
        "primary_exact_support_sentence_index":"title+original sentence index",
        "secondary_relation_metrics_are_weak_lexical_proxies":True,
        "n":len(cases),"arms":counts,"paired":comparisons,
        "by_type_exploratory":typ,
        "gold_aware_non_deployable_oracle":{
            "total_fit":sum(a["gold_all_fit_25pct"] for a in oracle),
            "n":len(oracle),"by_type":oracle_by_type,
            "oracle_not_in_model_selection":True,
            "combinations_exhaustively_scored":True},
        "reader":{k:v for k,v in reader.items() if k!="per_case"},
        "gold_support_rows":metrics,
        "oracle_rows":oracle,
        "reserved_96_validation_untouched":True,
        "qasper_test_unused":True,"production_unchanged":True,
        "limitations":["Only 32 new publicly labelled validation IDs",
            "The graph is a question-only hypothesis, not a verified KG",
            "Source title/entity mentions do not prove entailment",
            "Potential pretrained world knowledge in planner/reader",
            "Same max budget but actual counts not guaranteed identical",
            "No claim of generalization without untouched validation"]}
    # Never overwrite the first archived posthoc output. The corrected
    # version fixes metadata counts only, not contexts, scores, or gold.
    file=OUT/"GOLD_SUPPORT_RELATION_POSTHOC.json"
    if file.exists():raise FileExistsError(file)
    new_json(file,result)
    print("PHASE15_GOLD_DIAGNOSTICS",json.dumps({
        "arms":counts,"paired":comparisons,"oracle":result["gold_aware_non_deployable_oracle"],
        "by_type":typ},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
