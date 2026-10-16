"""Phase12 POSTHOC gold-aware feasibility oracle, not deployable.
For small duplicate-title alternatives, exhaustive set of support sentence occurrences;
otherwise choose shortest heuristic and mark non-exhaustive.
"""
import collections,itertools,json,math,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase12_register import OUT,DATA,verify_data,sha_file,new_json
from submission.experiments.phase12_contexts import candidates
from submission.experiments.phase9_crossdoc_core import render_atomic
from tokenpack.tokenization import TokenCounter

def canon(s):return " ".join(str(s).casefold().split())

def run():
    c=OUT/"gold_blind_contexts.jsonl"
    frozen=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(c)==frozen["context_file_sha256"]
    original=[json.loads(x) for x in c.open(encoding="utf8") if x.strip()]
    keys={x["id"] for x in original}
    pf=verify_data()
    gs={r["id"]:r for r in pf.read(columns=["id","context","type","supporting_facts"]).to_pylist()
        if r["id"] in keys}
    assert set(gs)==keys
    count=TokenCounter()
    assert count._encoding is not None
    details=[]
    for row in original:
        g=gs[row["id"]]
        atoms,_,_=candidates({"context":g["context"]})
        assert [(a["title"],a["sent_id"]) for a in atoms]==[
            (p["title"],p["sent_id"]) for p in row["candidate_map"]]
        support={(canon(t),int(j)) for t,j in zip(
            g["supporting_facts"]["title"],g["supporting_facts"]["sent_id"])}
        opts=[[i for i,a in enumerate(atoms) if (canon(a["title"]),a["sent_id"])==key]
              for key in sorted(support)]
        assert all(opts)
        prod=math.prod(map(len,opts))
        if prod<=15000:
            choices=itertools.product(*opts)
            exhaustive=True
        else:
            choices=[tuple(min(x,key=lambda i:count.count(render_atomic([i],atoms))) for x in opts)]
            exhaustive=False
        best=min(count.count(render_atomic(chosen,atoms)) for chosen in choices)
        gold_feasible=int(best<=row["budget"])
        line={"id":row["id"],"type":g["type"],
              "gold_support_n":len(support),"gold_oracle_min_tokens":best,
              "gold_oracle_fits_actual_25cap":gold_feasible,
              "exhaustive":exhaustive,
              "budget":row["budget"]}
        for arm in ("crossdoc-bridge","hop-planner"):
            chosen=row["contexts"][arm]
            selected={(
              canon(row["candidate_map"][i]["title"]),
              int(row["candidate_map"][i]["sent_id"])) for i in chosen["selected_sentence_indexes"]}
            title_selected={title for title,_ in selected}
            line[arm+"_title_all"]=int({t for t,_ in support}<=title_selected)
            line[arm+"_sentence_all"]=int(support<=selected)
        details.append(line)
    by={}
    for typ in sorted({d["type"] for d in details}):
        x=[d for d in details if d["type"]==typ]
        by[typ]={"n":len(x),"mean_required_support_sentences":
            sum(d["gold_support_n"] for d in x)/len(x),
            "oracle_all_gold_support_fit":sum(d["gold_oracle_fits_actual_25cap"] for d in x),
            "mean_oracle_min_tokens":sum(d["gold_oracle_min_tokens"] for d in x)/len(x),
            "bridge_title_all":sum(d["crossdoc-bridge_title_all"] for d in x),
            "hop_title_all":sum(d["hop-planner_title_all"] for d in x),
            "bridge_sentence_all":sum(d["crossdoc-bridge_sentence_all"] for d in x),
            "hop_sentence_all":sum(d["hop-planner_sentence_all"] for d in x)}
    result={"gold_aware_oracle_is_not_deployable":True,
       "interpretation":"Minimum true original labeled sentences plus rendered source headings, after seeing gold; feasibility not actual blind retrieval.",
       "exhaustive_count":sum(x["exhaustive"] for x in details),"n":len(details),
       "total_oracle_fit":sum(x["gold_oracle_fits_actual_25cap"] for x in details),
       "by_type":by,"per_case":details}
    target=OUT/"GOLD_AWARE_FEASIBILITY_POSTHOC.json"
    if target.exists():raise FileExistsError(target)
    new_json(target,result)
    print("PHASE12_GOLD_ORACLE",json.dumps({"n":len(details),"exhaustive":result["exhaustive_count"],
        "total_feasible":result["total_oracle_fit"],"by_type":by},ensure_ascii=False),flush=True)

if __name__=="__main__":run()
