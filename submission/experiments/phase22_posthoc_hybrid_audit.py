"""Phase22 post-freeze hybrid route and reader evidence audit.
Never change predictions or preregistered gating after seeing gold.
"""
from __future__ import annotations
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase22_reader_register import OUT
from submission.experiments.phase12_register import DATA,sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import answer_metrics
import pyarrow.parquet as pq
def norm(x):return " ".join(str(x).casefold().split())
def main():
    score=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    routes_meta=json.loads((OUT/"HYBRID_PRE_GOLD_DECISIONS_FROZEN.json").read_text(encoding="utf8"))
    contexts_meta=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    routes_path=OUT/"HYBRID_GOLD_BLIND_PREDICTIONS.jsonl"
    log_path=OUT/"reader_responses.jsonl"
    contexts_path=OUT/"gold_blind_contexts.jsonl"
    assert score["decision"]=="STOP_NO_VALIDATION"
    assert sha_file(log_path)==score["responses_sha256"]==routes_meta["reader_log_sha256"]
    assert sha_file(routes_path)==routes_meta["hybrid_predictions_sha256"]
    assert sha_file(contexts_path)==contexts_meta["sha256"]
    rows=lambda p:[json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s]
    frozen=rows(routes_path)
    cases=rows(contexts_path)
    answers=rows(log_path)
    assert len(frozen)==len(cases)==32 and len(answers)==64
    byans={r["task_id"]:r for r in answers}
    per_score={r["id"]:r for r in score["per_case"]}
    ids={r["id"] for r in cases}
    golds={r["id"]:r for r in pq.ParquetFile(DATA).read(
        columns=["id","answer","supporting_facts","type"]).to_pylist() if r["id"] in ids}
    assert set(golds)==ids
    counts=collections.Counter()
    per=[]
    for item,ctx in zip(frozen,cases):
        ident=item["id"]
        assert ident==ctx["id"] and ident in per_score
        raw_direct=byans[ident+"|verified-direct"]
        raw_struct=byans[ident+"|verified-evidence-first"]
        expect_struct=(raw_struct.get("parse_status")=="valid_json" and bool(
             (raw_struct.get("answer") or "").strip()))
        route="structured" if expect_struct else "direct_fallback"
        expected=raw_struct["answer"] if expect_struct else raw_direct["answer"]
        assert item["route"]==route and item["prediction"]==expected
        assert item["direct_prediction"]==raw_direct["answer"]
        assert item["structured_prediction"]==raw_struct["answer"]
        em,f1=answer_metrics(item["prediction"],golds[ident]["answer"])
        actual=per_score[ident]["verified-hybrid-empty-fallback"]
        assert abs(em-actual["EM"])<1e-12 and abs(f1-actual["F1"])<1e-12
        assert ctx["contexts"]["verified-direct"]["text"]==ctx["contexts"]["verified-evidence-first"]["text"]
        cand=ctx["candidate_map"]
        support={(norm(t),int(i)) for t,i in zip(golds[ident]["supporting_facts"]["title"],
                                                golds[ident]["supporting_facts"]["sent_id"])}
        chosen=ctx["contexts"]["verified-direct"]["selected_sentence_indexes"]
        final={(norm(cand[i]["title"]),int(cand[i]["sent_id"])) for i in chosen}
        full=support<=final
        direct=per_score[ident]["verified-direct"]["EM"]
        struct=per_score[ident]["verified-evidence-first"]["EM"]
        counts.update({
            "question_n":1,"hybrid_correct":int(em),"direct_correct":int(direct),
            "structured_correct":int(struct),
            "full_gold_support_in_context":int(full),
            "structured_used":int(expect_struct),
            "fallback_used":int(not expect_struct),
            "fallback_direct_correct":int(not expect_struct and direct),
            "fallback_direct_wrong":int(not expect_struct and not direct),
            "hybrid_only_correct":int(em and not direct),
            "direct_only_correct":int(direct and not em),
            "struct_only_correct_direct_wrong":int(struct and not direct),
            "struct_only_wrong_direct_correct":int(direct and not struct),
            "full_support_hybrid_wrong":int(full and not em)})
        per.append({"id":ident,"type":golds[ident]["type"],"route":route,
            "direct_em":direct,"structured_em":struct,"hybrid_em":em,
            "direct_f1":per_score[ident]["verified-direct"]["F1"],
            "structured_f1":per_score[ident]["verified-evidence-first"]["F1"],
            "hybrid_f1":f1,
            "full_gold_support":int(full),"gold_sentence_recall":len(support&final)/len(support),
            "structured_citation_exact":sum(
                 bool(x["quote_exact_in_context"]) for x in raw_struct.get("source_citation_steps",[])),
            "structured_citation_count":len(raw_struct.get("source_citation_steps",[])),
            "source_actual_tokens":ctx["contexts"]["verified-direct"]["tokens"]})
    assert counts["hybrid_correct"]==score["arms"]["verified-hybrid-empty-fallback"]["correct"]
    assert counts["hybrid_only_correct"]==1 and counts["direct_only_correct"]==0
    assert counts["fallback_used"]==routes_meta["direct_fallback_used_n"]==8
    audit={"phase":22,"post_gold_reader_audit":True,
      "source_data_public_validation_not_official_test":True,
      "no_tuning_or_new_remote_calls":True,
      "hybrid_determined_and_frozen_before_gold":True,
      "reader_log_sha256":sha_file(log_path),
      "hybrid_frozen_sha256":sha_file(routes_path),
      "context_frozen_sha256":sha_file(contexts_path),
      "counts":dict(counts),
      "per_case":per}
    dest=OUT/"HYBRID_POST_GOLD_ROUTE_EVIDENCE_AUDIT.json"
    if dest.exists():raise FileExistsError(dest)
    new_json(dest,audit)
    print("PHASE22_HYBRID_POSTGOLD",json.dumps(audit["counts"],ensure_ascii=False),flush=True)
if __name__=="__main__":main()
