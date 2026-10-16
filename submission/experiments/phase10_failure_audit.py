"""Phase10 posthoc pilot-only failure audit; no reranking or deployment changes."""
import json
from pathlib import Path
import pyarrow.parquet as pq
from submission.experiments.phase10_musique_pilot import OUT,DATA,DATA_SHA256,sha_file

def main():
    reader=json.loads((OUT/"pilot_READER_SCORED_POST_GOLD.json").read_text(encoding="utf8"))
    contexts={r["id"]:r for r in (
        json.loads(x) for x in (OUT/"pilot_gold_blind_contexts.jsonl").open(encoding="utf8") if x.strip())}
    by={}
    for item in reader["per_case_metrics"]:
        by.setdefault(item["case_id"],{})[item["arm"]]=item
    losses=[i for i,v in by.items() if v["crossdoc-bridge"]["EM"]==1 and v["bridge-neighbor"]["EM"]==0]
    wins=[i for i,v in by.items() if v["crossdoc-bridge"]["EM"]==0 and v["bridge-neighbor"]["EM"]==1]
    assert len(losses)==2 and not wins
    gold={x["id"]:x for x in pq.ParquetFile(DATA).read(columns=["id","paragraphs"]).to_pylist()
          if x["id"] in losses}
    audit=[]
    for case_id in losses:
        g={p["idx"] for p in gold[case_id]["paragraphs"] if p["is_supporting"]}
        old=set(contexts[case_id]["contexts"]["crossdoc-bridge"]["selected_source_idxs"])
        new=set(contexts[case_id]["contexts"]["bridge-neighbor"]["selected_source_idxs"])
        audit.append({"case_id":case_id,
            "support_paragraph_count":len(g),
            "bridge_support_hit":len(old&g),
            "neighbor_support_hit":len(new&g),
            "bridge_all_support":g<=old,
            "neighbor_all_support":g<=new,
            "bridge_tokens":contexts[case_id]["contexts"]["crossdoc-bridge"]["tokens"],
            "neighbor_tokens":contexts[case_id]["contexts"]["bridge-neighbor"]["tokens"]})
    result={"status":"Exploratory ONLY on used 32 pilot IDs",
            "reader_old_only_correct":audit,"new_only_correct_count":0,
            "causal_attribution_not_established":True}
    f=OUT/"PILOT_READER_LOSS_AUDIT.json"
    if f.exists():raise FileExistsError(f)
    f.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("PHASE10_FAILURE_AUDIT",json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
