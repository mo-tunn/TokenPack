"""Post-freeze gold scoring for Phase29 MuSiQue; never runs before reader freeze."""
from __future__ import annotations
import argparse,collections,json,os,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase7_modal27b_matched8 import read_rows,answer_metrics
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
DATA=ROOT/"submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"
BASE="atomic-sentence-rank";NEW="semantic-joint-citations"

def new_json(path,obj):
    with Path(path).open("x",encoding="utf8") as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
def paired_binary(rows,metric,seed):
    diffs=np.array([float(r[NEW][metric])-float(r[BASE][metric]) for r in rows])
    rng=np.random.default_rng(seed)
    boot=np.array([np.mean(diffs[rng.integers(0,len(diffs),len(diffs))]) for _ in range(10000)])
    win=sum(r[NEW][metric]>r[BASE][metric] for r in rows)
    lose=sum(r[NEW][metric]<r[BASE][metric] for r in rows)
    return {"mean_delta":float(np.mean(diffs)),
      "bootstrap95":[float(x) for x in np.quantile(boot,[.025,.975])],
      "win":win,"loss":lose,"net":win-lose,
      "two_sided_exact_mcnemar_p":float(binomtest(min(win,lose),win+lose,.5).pvalue) if win+lose else 1.0}
def paired_continuous(rows,metric,seed):
    diffs=np.array([float(r[NEW][metric])-float(r[BASE][metric]) for r in rows])
    rng=np.random.default_rng(seed)
    boot=np.array([np.mean(diffs[rng.integers(0,len(diffs),len(diffs))]) for _ in range(10000)])
    return {"mean_delta":float(np.mean(diffs)),
      "bootstrap95":[float(x) for x in np.quantile(boot,[.025,.975])],
      "win":sum(r[NEW][metric]>r[BASE][metric] for r in rows),
      "loss":sum(r[NEW][metric]<r[BASE][metric] for r in rows),
      "net":sum(r[NEW][metric]>r[BASE][metric] for r in rows)-sum(r[NEW][metric]<r[BASE][metric] for r in rows)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["pilot","validation"],required=True);a=ap.parse_args()
    sub=OUT/a.split;result_path=sub/("PILOT_POST_GOLD_RESULTS.json" if a.split=="pilot" else "VALIDATION_POST_GOLD_RESULTS.json")
    if result_path.exists():raise FileExistsError(result_path)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    c=json.loads((sub/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    rf=json.loads((sub/"READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    gate=json.loads((sub/"PRE_GOLD_STRUCTURAL_DECISION.json").read_text(encoding="utf8"))
    assert gate["decision"]=="READER_COMPARISON_PERMITTED_NOT_YET_SCORED"
    assert c["stats"]["token_cap_violations"]==0
    assert sha_file(DATA)==p["data_sha256"]
    assert sha_file(sub/"gold_blind_paired_contexts.jsonl")==c["sha256"]
    assert sha_file(sub/"reader_responses.jsonl")==rf["responses_sha256"]
    contexts=read_rows(sub/"gold_blind_paired_contexts.jsonl")
    ans=read_rows(sub/"reader_responses.jsonl")
    ids=p["pilot_ids"] if a.split=="pilot" else p["reserved_ids"]
    assert len(contexts)==len(ids) and len(ans)==2*len(ids)
    assert [x["id"] for x in contexts]==ids
    assert [r["task_id"] for r in ans]==[x["id"]+"|"+arm for x in contexts for arm in (BASE,NEW)]
    assert not (sub/"reader_error.jsonl").exists() and not (sub/"reader_inflight.json").exists()
    index={x["task_id"]:x for x in ans}
    # Gold fields are opened only now, after every paired reader output is hash-frozen.
    gold_cols=["id","answer","paragraphs.list.element.idx","paragraphs.list.element.is_supporting"]
    gold_rows=pq.ParquetFile(DATA).read(columns=gold_cols).to_pylist()
    wanted=set(ids)
    gold={r["id"]:r for r in gold_rows if r["id"] in wanted}
    assert set(gold)==wanted
    cases=[];stats=collections.Counter()
    for c0 in contexts:
        ident=c0["id"];g=gold[ident]
        target={int(x["idx"]) for x in g["paragraphs"] if bool(x["is_supporting"])}
        assert target, "MuSiQue row has no annotated supporting paragraph"
        pair={"id":ident,"gold_supporting_paragraph_n":len(target),
              "source_sentence_set_changed":c0["source_sentence_set_changed"],
              "source_paragraph_set_changed":c0["source_paragraph_set_changed"]}
        for arm in (BASE,NEW):
            selected=c0["contexts"][arm]["selected_original_pairs"]
            final={int(x["paragraph_idx"]) for x in selected}
            hits=len(target&final);full=int(target<=final)
            pred=index[ident+"|"+arm]["answer"]
            em,f1=answer_metrics(pred,g["answer"])
            pair[arm]={"full_support":full,"support_recall":hits/len(target),
                "gold_support_paragraph_hits":hits,"EM":em,"F1":f1,
                "prediction_empty":not bool(pred),"actual_tokens":c0["contexts"][arm]["tokens"],
                "source_paragraph_count":c0["contexts"][arm]["source_article_count"]}
            stats[arm+"_full_support"]+=full;stats[arm+"_EM"]+=em
            stats[arm+"_support_hits"]+=hits;stats[arm+"_tokens"]+=c0["contexts"][arm]["tokens"]
        if not c0["source_sentence_set_changed"]:
            assert pair[BASE]["full_support"]==pair[NEW]["full_support"]
            assert pair[BASE]["support_recall"]==pair[NEW]["support_recall"]
            assert index[ident+"|"+BASE]["answer"]==index[ident+"|"+NEW]["answer"]
        cases.append(pair)
    n=len(cases);offset=0 if a.split=="pilot" else 100
    em=paired_binary(cases,"EM",20261013+offset)
    supp=paired_binary(cases,"full_support",20261015+offset)
    f1=paired_continuous(cases,"F1",20261014+offset)
    recall=paired_continuous(cases,"support_recall",20261016+offset)
    pass_pilot=(supp["net"]>=p["pilot_gates"]["post_reader_net_full_support_paragraph_coverage_gain_minimum"]
        and em["net"]>=p["pilot_gates"]["post_reader_net_EM_correct_gain_minimum"]
        and c["stats"]["token_cap_violations"]==0)
    pass_validation=(supp["net"]>0 and em["net"]>0
        and supp["two_sided_exact_mcnemar_p"]<=p["confirmation_gate"]["exact_two_sided_mcnemar_p_each_max"]
        and em["two_sided_exact_mcnemar_p"]<=p["confirmation_gate"]["exact_two_sided_mcnemar_p_each_max"]
        and c["stats"]["token_cap_violations"]==0)
    decision=("GO_TO_VALIDATION" if pass_pilot else "STOP_NO_VALIDATION") if a.split=="pilot" else (
        "VALIDATION_BOTH_PRIMARY_SIGNIFICANT" if pass_validation else "VALIDATION_NOT_CONFIRMED")
    result={"split":a.split,"decision":decision,"n":n,"same_question_matched_reader":True,
      "gold_first_opened_after_reader_freeze":True,"not_independent_hidden_test":True,
      "all_selected_content_from_original_source_sentences":True,
      "full_support_definition":"Every gold is_supporting paragraph idx has at least one selected source sentence; paragraph coverage is not sentence-level entailment.",
      "paired_EM":em,"paired_F1":f1,"paired_full_support_paragraph_coverage":supp,
      "paired_support_paragraph_recall":recall,
      "arms":{arm:{"EM_correct":stats[arm+"_EM"],"EM_fraction":stats[arm+"_EM"]/n,
        "full_gold_support_paragraph_questions":stats[arm+"_full_support"],
        "full_gold_support_paragraph_fraction":stats[arm+"_full_support"]/n,
        "mean_gold_support_paragraph_recall":sum(x[arm]["support_recall"] for x in cases)/n,
        "mean_F1":sum(x[arm]["F1"] for x in cases)/n,
        "mean_tokens":stats[arm+"_tokens"]/n} for arm in (BASE,NEW)},
      "actual_paid_verifier_calls":json.loads((sub/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))["paid_calls"],
      "actual_paid_reader_calls":rf["actual_paid_reader_calls"],
      "same_prompt_reader_reuse":rf["identical_prompt_reuse_n"],
      "source_sentence_sets_changed":c["stats"]["source_sentence_sets_changed"],
      "pilot_gate":{"min_net_full_support":p["pilot_gates"]["post_reader_net_full_support_paragraph_coverage_gain_minimum"],
                    "min_net_EM":p["pilot_gates"]["post_reader_net_EM_correct_gain_minimum"],
                    "passed":pass_pilot} if a.split=="pilot" else None,
      "confirmatory_gate":{"alpha_each":p["confirmation_gate"]["exact_two_sided_mcnemar_p_each_max"],
                    "both_positive_significant":pass_validation} if a.split=="validation" else None,
      "pilot_context_sha256":c["sha256"],"reader_sha256":rf["responses_sha256"],
      "reserved_96_were_pre_registered":True,"no_gold_sent_remote":True,
      "qasper_official_test_untouched":True,"production_unchanged":True,"per_case":cases}
    new_json(result_path,result)
    print("PHASE29_POST_GOLD",json.dumps({"split":a.split,"n":n,"decision":decision,
      "arms":result["arms"],"EM":em,"full_support":supp,"F1":f1,
      "paid_verifier":result["actual_paid_verifier_calls"],"paid_reader":rf["actual_paid_reader_calls"],
      "reused":rf["identical_prompt_reuse_n"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
