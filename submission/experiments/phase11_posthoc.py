"""Posthoc Phase11 MuSiQue supporting-paragraph proxy. Gold only after reader freeze."""
import json,statistics,sys
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase11_register import OUT,DATA,digest,save_new
from submission.experiments.phase9_musique_protocol import DATA_SHA256

def main():
    meta=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"pilot_CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    file=OUT/"pilot_gold_blind_contexts.jsonl"
    assert digest(file)==frozen["contexts_sha256"]
    reader=json.loads((OUT/"pilot_reader_SCORED_POST_GOLD.json").read_text(encoding="utf8"))
    assert reader["n"]==32 and reader["decision"]=="STOP_NO_VALIDATION"
    assert digest(DATA)==DATA_SHA256
    rows=[json.loads(line) for line in file.open(encoding="utf8") if line.strip()]
    assert [r["id"] for r in rows]==meta["pilot_ids"]
    ids=set(meta["pilot_ids"])
    gold={r["id"]:r for r in pq.ParquetFile(DATA).read(
        columns=["id","paragraphs"]).to_pylist() if r["id"] in ids}
    assert set(gold)==ids
    arms=("crossdoc-bridge","hop-planner")
    stats={a:defaultdict(list) for a in arms}
    support_per_case={}
    for row in rows:
        required={int(p["idx"]) for p in gold[row["id"]]["paragraphs"] if p["is_supporting"]}
        assert required
        support_per_case[row["id"]]={}
        for arm in arms:
            c=row["contexts"][arm]
            assert c["budget"]==row["budget"] and c["tokens"]<=row["budget"]
            chosen=set(c["selected_source_idxs"])
            metrics={"all_support":int(required<=chosen),
                "support_fraction":len(required&chosen)/len(required),
                "real_tokens":c["tokens"],"budget":row["budget"],
                "selected_paragraphs":c["selected_paragraph_count"],
                "selected_sentences":c["selected_sentence_count"]}
            for k,v in metrics.items():stats[arm][k].append(v)
            support_per_case[row["id"]]=support_per_case.get(row["id"],{})
            support_per_case[row["id"]][arm]=metrics
    sums={arm:{"n":len(rows),
        "support_all_n":sum(stats[arm]["all_support"]),
        "mean_support_recall":statistics.mean(stats[arm]["support_fraction"]),
        "mean_real_tokens":statistics.mean(stats[arm]["real_tokens"]),
        "mean_max_budget":statistics.mean(stats[arm]["budget"]),
        "mean_selected_paragraphs":statistics.mean(stats[arm]["selected_paragraphs"]),
        "mean_selected_sentences":statistics.mean(stats[arm]["selected_sentences"]),
        "token_budget_violations":0} for arm in arms}
    paired=defaultdict(dict)
    for x in reader["sample_metrics"]:paired[x["case_id"]][x["arm"]]=x
    discordants=[]
    for id,v in paired.items():
        if v["hop-planner"]["EM"]!=v["crossdoc-bridge"]["EM"]:
            discordants.append({"id":id,
                "winner":"hop-planner" if v["hop-planner"]["EM"] else "crossdoc-bridge",
                "gold_support_proxy":support_per_case[id]})
    plan=json.loads((OUT/"pilot_planner_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    result={"public_validation_not_hidden_test":True,
        "retrieval":sums,"discordants":discordants,
        "reader":{"counts":reader["counts"],"paired":reader["paired"],
                  "decision":reader["decision"]},
        "planning":{"valid_json":plan["valid_json"],"fallbacks":plan["fallbacks"],
                    "real_modal_calls":32},
        "reading":{"real_modal_calls":reader["remote_paid_calls"],
                   "reused_identical_prompt":reader["same_payload_reused"],
                   "empty_successful_outputs":reader["empty_successful_model_outputs"]},
        "real_modal_calls_total":32+reader["remote_paid_calls"],
        "actual_invoice_not_audited":True,
        "remaining_96_not_evaluated":True,
        "phase9_excluded":256,"phase10_excluded":128,
        "qasper_test_reused":False,"production_changed":False,
        "weak_proxy_not_answer_sufficiency":True}
    out=OUT/"PILOT_POSTHOC_DIAGNOSTICS.json"
    if out.exists():raise FileExistsError(out)
    save_new(out,result)
    print("PHASE11_POSTHOC",json.dumps({
        "retrieval":sums,"discordants":discordants,
        "calls":result["real_modal_calls_total"],
        "decision":reader["decision"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
