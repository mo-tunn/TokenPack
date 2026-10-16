"""Post-reader-only audit of Qwen verification selections against annotated
gold support sentence indices. Gold overlap != semantic verification accuracy:
unannotated alternative sentences can still be valid evidence.
"""
from __future__ import annotations
import collections,json,statistics,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase14_register import OUT
from submission.experiments.phase12_register import (
    DATA,verify_data,sha_file,new_json)
def norm(s):return " ".join(str(s or "").casefold().split())

def run():
    reader=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    assert reader["n"]==32 and reader["decision"] in ("STOP_NO_VALIDATION","GO_VALIDATION")
    candidate_sha=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    verifier_sha=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    cfile=OUT/"scored_candidate_pools.jsonl"
    vfile=OUT/"verifier_responses.jsonl"
    assert sha_file(cfile)==candidate_sha["candidate_sha256"]
    assert sha_file(vfile)==verifier_sha["responses_sha256"]
    candidates=[json.loads(x) for x in cfile.open(encoding="utf8") if x.strip()]
    verified=[json.loads(x) for x in vfile.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in candidates]==[x["id"] for x in verified]
    verify_data()
    ids={x["id"] for x in candidates}
    gold={r["id"]:r for r in pq.ParquetFile(DATA).read(
        columns=["id","type","supporting_facts"]).to_pylist()
        if r["id"] in ids}
    assert set(gold)==ids
    records=[];bytype=collections.defaultdict(list)
    all_verified=0;all_selected_overlap=0;all_pools=0;all_gold_pools=0
    for c,v in zip(candidates,verified):
        support={(norm(t),int(j)) for t,j in zip(
            gold[c["id"]]["supporting_facts"]["title"],
            gold[c["id"]]["supporting_facts"]["sent_id"])}
        pool_keys=set()
        taken=[]
        for e,pool in enumerate(c["candidate_pools"]):
            for atom in pool:
                pool_keys.add((norm(atom["title"]),int(atom["sentence_id"])))
            choice=v["decision_ids"].get(f"e{e}")
            if choice is None:continue
            matched=[x for x in pool if x["candidate_id"]==choice]
            assert len(matched)==1
            z=matched[0];taken.append((norm(z["title"]),int(z["sentence_id"])))
        covered=len(pool_keys&support)
        match=sum(x in support for x in taken)
        row={"id":c["id"],"type":gold[c["id"]]["type"],
            "total_gold_support_sentences":len(support),
            "gold_support_accessible_in_any_top4":covered,
            "all_gold_support_in_candidate_pools":int(support<=pool_keys),
            "verified_edge_positive_count":len(taken),
            "verified_selected_annotations_overlap_n":match,
            "verified_pick_annotation_overlap_fraction":match/len(taken) if taken else None,
            "verifier_missing_edge_count":len(c["candidate_pools"])-len(taken)}
        records.append(row);bytype[row["type"]].append(row)
        all_verified+=len(taken)
        all_selected_overlap+=match
        all_pools+=len(support)
        all_gold_pools+=covered
    info={"n":len(records),
        "model_verified_positive_edge_count":all_verified,
        "model_verified_picks_matching_gold_support_sentence_count":all_selected_overlap,
        "verified_picks_matching_gold_annotation_fraction":all_selected_overlap/all_verified if all_verified else None,
        "gold_support_sentence_accessible_in_top4_pools_fraction":all_gold_pools/all_pools,
        "cases_all_support_sentences_in_top4_pools":sum(r["all_gold_support_in_candidate_pools"] for r in records),
        "by_question_type":{k:{
            "n":len(xs),
            "positive_verified_edges":sum(x["verified_edge_positive_count"] for x in xs),
            "verified_overlapping_annotated_support":sum(x["verified_selected_annotations_overlap_n"] for x in xs),
            "all_gold_accessible_in_any_pool":sum(x["all_gold_support_in_candidate_pools"] for x in xs),
        } for k,xs in bytype.items()},
        "per_case":records,
        "caveat":"Gold-support annotation overlap is NOT the verifier's semantic precision: there may be valid unannotated evidence; planned edge could target a different fact.",
        "gold_read_after_reader_freeze":True,
        "no_model_reevaluation":True,
        "phase14_reader_gate_decision":reader["decision"]}
    target=OUT/"VERIFIER_ANNOTATION_OVERLAP_POSTHOC.json"
    if target.exists():raise FileExistsError(target)
    new_json(target,info)
    print("PHASE14_VERIFIER_AUDIT",json.dumps({k:v for k,v in info.items()
        if k!="per_case"},ensure_ascii=False),flush=True)
if __name__=="__main__":run()
