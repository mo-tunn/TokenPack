"""Faz13 gold-aware POSTHOC only: graph branch coverage and reader discordance.

Do not tune or rerun on these labels. This diagnostic never invokes models.
"""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase13_register import OUT
from submission.experiments.phase12_register import new_json

def run():
    ctx={r["id"]:r for r in (
        json.loads(x) for x in (OUT/"gold_blind_contexts.jsonl").open(encoding="utf8") if x.strip())}
    scored=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    gold=json.loads((OUT/"GOLD_SUPPORT_RELATION_POSTHOC_CORRECTED.json").read_text(encoding="utf8"))
    evidence=collections.defaultdict(dict)
    for r in gold["gold_support_rows"]:evidence[r["id"]][r["method"]]=r
    summaries=collections.defaultdict(lambda:collections.Counter())
    readers=[]
    for r in scored["per_case"]:
        id=r["id"];kind=evidence[id]["graph-path"]["type"]
        v=ctx[id]
        branch_count=len(v["graph_branches"])
        g=v["contexts"]["graph-path"]
        summaries[kind]["n"]+=1
        summaries[kind]["multi_branch"]+=int(branch_count==2)
        summaries[kind]["graph_paired_seed"]+=g["paired_branch_seeds"]
        summaries[kind]["graph_all_gold_titles"]+=evidence[id]["graph-path"]["all_gold_titles"]
        summaries[kind]["graph_all_gold_sentences"]+=evidence[id]["graph-path"]["all_gold_sentences"]
        summaries[kind]["flat_all_gold_sentences"]+=evidence[id]["flat-edge"]["all_gold_sentences"]
        ge=r["graph-path"]["EM"];fe=r["flat-edge"]["EM"]
        if ge!=fe:
            readers.append({"id":id,"type":kind,
                "winner":"graph-path" if ge>fe else "flat-edge",
                "graph_branches":branch_count,
                "graph_paired_seeds":g["paired_branch_seeds"],
                "graph_all_support":evidence[id]["graph-path"]["all_gold_sentences"],
                "flat_all_support":evidence[id]["flat-edge"]["all_gold_sentences"],
                "graph_support_recall":evidence[id]["graph-path"]["gold_sentence_recall"],
                "flat_support_recall":evidence[id]["flat-edge"]["gold_sentence_recall"]})
    result={"exploratory_posthoc":True,
        "sample":32,
        "by_question_type":{k:dict(v) for k,v in summaries.items()},
        "reader_graph_vs_flat_discordances":readers,
        "graph_linked_seed_is_lexical_not_verified_relational_edge":True,
        "labels_never_used_in_inference":True}
    target=OUT/"GRAPH_PATH_FAILURE_AUDIT_POSTHOC.json"
    if target.exists():raise FileExistsError(target)
    new_json(target,result)
    print("PHASE13_FAILURE_AUDIT",json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":run()
