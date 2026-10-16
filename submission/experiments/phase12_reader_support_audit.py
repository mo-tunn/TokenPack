"""Phase12 posthoc only: reader correctness vs exact gold supporting sentence coverage.
No inference, model changes, or tuning. Reader subset fixed to first 32 by ID hash.
"""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import OUT,new_json
def run():
    support=json.loads((OUT/"RELATION_SUPPORT_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    reader=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    oracle=json.loads((OUT/"GOLD_AWARE_FEASIBILITY_POSTHOC.json").read_text(encoding="utf8"))
    bys=collections.defaultdict(dict)
    for r in support["per_case_evidence_metrics"]:bys[r["id"]][r["method"]]=r
    byoracle={r["id"]:r for r in oracle["per_case"]}
    bytype=collections.defaultdict(lambda:collections.defaultdict(int))
    perarm={}
    for arm in ("crossdoc-bridge","hop-planner"):
        confusion=collections.Counter()
        for x in reader["per_case"]:
            id=x["id"];value=x[arm]["EM"]
            exact=bys[id][arm]["supporting_sentences_all"]
            confusion[(exact,value)]+=1
            bytype[bys[id][arm]["type"]][arm+"_n"]+=1
            bytype[bys[id][arm]["type"]][arm+"_correct"]+=value
            bytype[bys[id][arm]["type"]][arm+"_exact_support"]+=exact
        perarm[arm]={
            "reader_and_all_support_correct":confusion[(1,1)],
            "reader_correct_but_support_missing":confusion[(0,1)],
            "reader_wrong_despite_all_support":confusion[(1,0)],
            "reader_wrong_and_support_missing":confusion[(0,0)],
        }
    result={
        "posthoc_exploratory_only":True,
        "n_reader":len(reader["per_case"]),
        "per_arm_concordance":perarm,
        "reader_by_question_type_descriptive":{k:dict(v) for k,v in bytype.items()},
        "gold_oracle_feasible_cases_total":oracle["total_oracle_fit"],
        "gold_oracle_fit_bridge_comparison":oracle["by_type"]["bridge_comparison"]["oracle_all_gold_support_fit"],
        "gold_oracle_not_deployable":True,
        "source_extraction_confident":True,
    }
    target=OUT/"READER_SUPPORT_CONCORDANCE_POSTHOC.json"
    if target.exists():raise FileExistsError(target)
    new_json(target,result)
    print("PHASE12_READER_SUPPORT_POSTHOC",json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":run()
