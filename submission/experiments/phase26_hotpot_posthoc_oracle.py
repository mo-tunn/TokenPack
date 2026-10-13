"""Gold-aware (NONDEPLOYABLE) theoretical evidence capacity for Faz26.

AFTER 96 reader outputs & score frozen. Does not change selected context,
models, scores or pilot decision. Determine whether all annotated gold
source sentences alone fit true 25% cap using original exact renderer.
"""
from __future__ import annotations
import collections,json,sys
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase26_hotpot_register import OUT
from submission.experiments.phase6_hotpot_eval import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
from submission.experiments.phase8_atomic_contexts import sentence_candidates,render
from tokenpack.tokenization import TokenCounter
OUTFILE=OUT/"POST_GOLD_NONDEPLOYABLE_ORACLE_CAPACITY.json"
def main():
    if OUTFILE.exists():raise FileExistsError(OUTFILE)
    diag=json.loads((OUT/"POST_GOLD_SENTENCE_SUFFICIENCY.json").read_text(encoding="utf8"))
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    preds=json.loads((OUT/"HOTPOT_READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==frozen["sha256"]
    assert sha_file(OUT/"hotpot_reader_responses.jsonl")==preds["responses_sha256"]
    assert diag["n"]==96
    tok=TokenCounter();assert tok._encoding is not None
    ids=set(manifest["pilot_ids"])
    # Gold is read only here, after reader freeze.
    goldrows={r["id"]:r for r in pq.ParquetFile(DATA).read(columns=[
        "id","context.title","context.sentences",
        "supporting_facts.title","supporting_facts.sent_id"]).to_pylist()
        if r["id"] in ids}
    contexts={r["id"]:r for r in read_rows(OUT/"gold_blind_contexts.jsonl")}
    prev={r["id"]:r for r in diag["per_case"]}
    assert set(goldrows)==set(contexts)==set(prev)==ids
    count=collections.Counter()
    records=[]
    for ident in manifest["pilot_ids"]:
        row=goldrows[ident];_,atoms=sentence_candidates(row)
        needed={(str(t).casefold().strip(),int(i))
              for t,i in zip(row["supporting_facts"]["title"],
                              row["supporting_facts"]["sent_id"])}
        bestidx={}
        for k,a in enumerate(atoms):
            key=(a["title"].casefold().strip(),int(a["sent_id"]))
            if key in needed and key not in bestidx:bestidx[key]=k
        assert set(bestidx)==needed
        chosen=list(bestidx.values())
        rendered=render(chosen,atoms)
        needed_tokens=tok.count(rendered)
        cap=contexts[ident]["cap"]
        fit=needed_tokens<=cap
        full=prev[ident]["all_gold_support_in_final"]
        correct=prev[ident]["reader_EM"]
        count["n"]+=1
        count["gold_oracle_all_fit"]+=int(fit)
        count["actual_gold_all_present"]+=int(full)
        count["gold_fits_but_not_selected"]+=int(fit and not full)
        count["gold_does_not_fit"]+=int(not fit)
        count["gold_fits_not_selected_reader_wrong"]+=int(
             fit and not full and not correct)
        count["gold_fits_not_selected_reader_correct"]+=int(
             fit and not full and bool(correct))
        count["gold_not_fit_reader_wrong"]+=int(not fit and not correct)
        count["gold_not_fit_reader_correct"]+=int(not fit and bool(correct))
        records.append({"id":ident,"needed_gold_token_count":needed_tokens,
            "hard_budget":cap,"gold_bundle_theoretically_fits":fit,
            "gold_fully_selected":full,"reader_EM":correct,
            "oracle_is_not_deployable":True})
    assert count["n"]==96
    report={"n":96,"nondeployable_gold_aware_oracle":True,
      "only_posthoc_after_reader_hash_freeze":True,
      "same_original_sentence_renderer":True,
      "no_gold_in_any_model":True,
      "no_new_modal_or_reranker_calls":True,
      "counts":dict(count),"mean_gold_only_bundle_tokens":sum(
          x["needed_gold_token_count"] for x in records)/96,
      "mean_real_budget":sum(x["hard_budget"] for x in records)/96,
      "source_score_file_sha256":sha_file(OUT/"POST_GOLD_SENTENCE_SUFFICIENCY.json"),
      "source_context_sha256":frozen["sha256"],
      "no_validation_ids_read":True,
      "caveat":"This is an unrealistic GOLD label oracle; support annotated sentence presence does NOT guarantee complete semantic entailment or reader accuracy.",
      "per_case":records}
    new_json(OUTFILE,report)
    print("PHASE26_POSTHOC_ORACLE",json.dumps({
        "counts":report["counts"],
        "mean_gold_only_bundle_tokens":report["mean_gold_only_bundle_tokens"],
        "mean_real_budget":report["mean_real_budget"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
