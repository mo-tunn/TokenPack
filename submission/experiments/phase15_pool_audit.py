"""Phase15 post-freeze gold-local candidate accessibility and selection audit.
Never invoked before all reader predictions have been saved and scored.
This is retrospective diagnostic, never fed to source selection or verifier.
"""
from __future__ import annotations
import collections,json,statistics,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase15_register import OUT
from submission.experiments.phase12_register import DATA,verify_data,sha_file,new_json
def key(title,i):return (" ".join(str(title or "").casefold().split()),int(i))
def main():
    score=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    assert score["n"]==32
    froze=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    v1f=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    v2f=json.loads((OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    p=OUT/"scored_candidate_pools.jsonl";v1=OUT/"verifier_responses.jsonl";v2=OUT/"rescue_verifier_responses.jsonl"
    assert sha_file(p)==froze["candidate_sha256"]
    assert sha_file(v1)==v1f["responses_sha256"]
    assert sha_file(v2)==v2f["responses_sha256"]
    cases=[json.loads(x) for x in p.open(encoding="utf8") if x.strip()]
    initial=[json.loads(x) for x in v1.open(encoding="utf8") if x.strip()]
    rescue=[json.loads(x) for x in v2.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in cases]==[x["id"] for x in initial]==[x["id"] for x in rescue]
    ids={x["id"] for x in cases}
    verify_data()
    labels={r["id"]:r for r in pq.ParquetFile(DATA).read(columns=[
        "id","supporting_facts","type"]).to_pylist() if r["id"] in ids}
    assert set(labels)==ids
    per=[]
    for c,i,r in zip(cases,initial,rescue):
        support={key(a,b) for a,b in zip(
            labels[c["id"]]["supporting_facts"]["title"],
            labels[c["id"]]["supporting_facts"]["sent_id"])}
        old={key(a["title"],a["sentence_id"]) for pool in c["candidate_pools"] for a in pool}
        active_rescue_edges={int(x[1:]) for x in r["rescue_decisions"]}
        assert all(i["decision_ids"][f"e{x}"] is None for x in active_rescue_edges)
        expanded=old|{
            key(a["title"],a["sentence_id"])
            for e,pool in enumerate(c["expanded_pools"])
            if e in active_rescue_edges
            for a in pool}
        positives=[]
        for e,pool in enumerate(c["candidate_pools"]):
            pid=i["decision_ids"][f"e{e}"]
            if pid is not None:
                hit=[a for a in pool if a["candidate_id"]==pid]
                assert len(hit)==1
                positives.append(("top4",key(hit[0]["title"],hit[0]["sentence_id"])))
        for edge_id,pid in r["rescue_decisions"].items():
            if pid is None:continue
            e=int(edge_id[1:])
            hit=[a for a in c["expanded_pools"][e] if a["candidate_id"]==pid]
            assert len(hit)==1
            positives.append(("rescue",key(hit[0]["title"],hit[0]["sentence_id"])))
        per.append({
            "id":c["id"],"type":labels[c["id"]]["type"],
            "gold_sentences":len(support),
            "initial_top4_gold_hits":len(old&support),
            "expanded_operational_gold_hits":len(expanded&support),
            "initial_all_gold_available":int(support<=old),
            "expanded_all_gold_available":int(support<=expanded),
            "newly_accessible_gold_sentences":len((expanded-old)&support),
            "initial_abstained_edges":len(active_rescue_edges),
            "initial_verified_picks":sum(x[0]=="top4" for x in positives),
            "rescue_verified_picks":sum(x[0]=="rescue" for x in positives),
            "initial_positive_gold_annotation_overlap":sum(x[0]=="top4" and x[1] in support for x in positives),
            "rescue_positive_gold_annotation_overlap":sum(x[0]=="rescue" and x[1] in support for x in positives),
        })
    total_gold=sum(r["gold_sentences"] for r in per)
    result={"n":len(per),
        "initial_top4_gold_sentence_recall":sum(r["initial_top4_gold_hits"] for r in per)/total_gold,
        "operational_rescued_pool_gold_sentence_recall":sum(r["expanded_operational_gold_hits"] for r in per)/total_gold,
        "initial_all_gold_available":sum(r["initial_all_gold_available"] for r in per),
        "operational_expanded_all_gold_available":sum(r["expanded_all_gold_available"] for r in per),
        "gold_sentences_newly_accessible_by_rescue":sum(r["newly_accessible_gold_sentences"] for r in per),
        "first_round_abstained_edges":sum(r["initial_abstained_edges"] for r in per),
        "first_round_verified_positive":sum(r["initial_verified_picks"] for r in per),
        "second_round_verified_positive":sum(r["rescue_verified_picks"] for r in per),
        "first_round_annotation_overlap":sum(r["initial_positive_gold_annotation_overlap"] for r in per),
        "second_round_annotation_overlap":sum(r["rescue_positive_gold_annotation_overlap"] for r in per),
        "rescue_called_only_when_abstain":True,
        "gold_opened_only_after_reader_scored":True,
        "gold_support_annotation_overlap_is_not_semantic_precision":True,
        "by_type":{k:{
           "n":len([r for r in per if r["type"]==k]),
           "initial_all_gold":sum(r["initial_all_gold_available"] for r in per if r["type"]==k),
           "rescued_all_gold":sum(r["expanded_all_gold_available"] for r in per if r["type"]==k),
           "new_gold_access":sum(r["newly_accessible_gold_sentences"] for r in per if r["type"]==k),
        } for k in sorted({r["type"] for r in per})},
        "per_case":per}
    f=OUT/"CANDIDATE_ACCESS_POST_GOLD_AUDIT.json"
    if f.exists():raise FileExistsError(f)
    new_json(f,result)
    print("PHASE15_POOL_POSTGOLD",json.dumps({k:v for k,v in result.items() if k!="per_case"},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
