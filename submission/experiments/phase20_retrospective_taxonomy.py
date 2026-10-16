"""Faz20 part A: reproducible, EXPLORATORY error taxonomy of archived
Phase14/15/16/18/19 2Wiki public-validation pilots.

Reads already-revealed posthoc gold support labels; never sends gold to any
model. No new Qwen calls, no previous archive edits, no QASPER sealed test.
These five nonoverlapping samples are NOT a same-question method ranking.
"""
from __future__ import annotations
import collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import DATA,verify_data,sha_file,new_json
from submission.experiments.phase14_register import OUT as P14
from submission.experiments.phase15_register import OUT as P15
from submission.experiments.phase16_register import OUT as P16
from submission.experiments.phase18_register import OUT as P18
from submission.experiments.phase19_register import OUT as P19
OUT=ROOT/"submission/results/phase20_error_taxonomy_20261012"
STUDIES={
 14:(P14,"edge-verified"),
 15:(P15,"conditional-rescue"),
 16:(P16,"branch-lock"),
 18:(P18,"branch-lock"),
 19:(P19,"branch-lock")
}
def norm(x):return " ".join(str(x).casefold().split())
def factkey(title,sent):return (norm(title),int(sent))
def rows(path):
    return [json.loads(line) for line in path.open(encoding="utf8") if line.strip()]
def summarize(items):
    c=collections.Counter(r["error_stage"] for r in items)
    fields={"n":len(items),"stage_counts":dict(sorted(c.items()))}
    for k in ("reader_correct","all_gold_final","all_gold_any_pool",
              "all_gold_top4_pool","has_bridge_type"):
        fields[k]=sum(int(r[k]) for r in items)
    fields["avg_gold_recall_final"]=(
        sum(r["gold_recall_final"] for r in items)/len(items) if items else 0)
    fields["avg_candidate_recall"]=(
        sum(r["gold_recall_any_pool"] for r in items)/len(items) if items else 0)
    fields["mean_actual_tokens"]=(
        sum(r["actual_tokens"] for r in items)/len(items) if items else 0)
    return fields
def main():
    if OUT.exists():raise FileExistsError("Immutable Faz20 taxonomy archive exists")
    verify_data()
    import pyarrow.parquet as pq
    projections=pq.ParquetFile(DATA).read(columns=[
        "id","type","supporting_facts"]).to_pylist()
    gold={r["id"]:r for r in projections}
    manifest={"research_phase":"20A retrospective fault taxonomy; never confirmatory",
        "benchmark":"2WikiMultiHopQA PUBLIC validation",
        "dataset_sha256":sha_file(DATA),
        "analysis_script_sha256":sha_file(Path(__file__)),
        "sources":{},"selection_rule":"Use recorded new/best-intended arm per phase, not method ranking across different questions",
        "study_arm":{str(n):method for n,(_,method) in STUDIES.items()},
        "no_new_inference":True,"old_reports_not_modified":True,
        "gold_local_posthoc":True,"no_qasper_test":True}
    allrows=[];by_phase={};ids=set();comparisons={}
    for phase,(p,selected_arm) in STUDIES.items():
        paths={
         "plan":p/"PRE_GOLD_PROTOCOL.json",
         "contexts":p/"gold_blind_contexts.jsonl",
         "answers":p/"READER_POST_GOLD_RESULTS.json",
         "gold_diagnostic":p/"GOLD_SUPPORT_RELATION_POSTHOC.json",
         "candidates":p/"scored_candidate_pools.jsonl"}
        assert all(f.exists() for f in paths.values())
        manifest["sources"][str(phase)]={name:sha_file(f) for name,f in paths.items()}
        protocol=json.loads(paths["plan"].read_text(encoding="utf8"))
        ctx=rows(paths["contexts"])
        can=rows(paths["candidates"])
        rescue_targets={}
        if (p/"rescue_verifier_inputs.jsonl").exists():
            for x in rows(p/"rescue_verifier_inputs.jsonl"):
                rescue_targets[x["id"]]={
                    int(edge["edge_id"][1:])
                    for edge in x["rescue_input"]["hypothesized_edges_with_source_candidates"]}
        answers=json.loads(paths["answers"].read_text(encoding="utf8"))
        diag=json.loads(paths["gold_diagnostic"].read_text(encoding="utf8"))
        assert len(ctx)==len(can)==answers["n"]==32
        stored=protocol.get("pilot_ids")
        assert [r["id"] for r in ctx]==[r["id"] for r in can]==stored
        assert not ids.intersection(stored),"Phase samples overlap"
        ids.update(stored)
        byid={r["id"]:r for r in answers["per_case"]}
        local={r["id"]:r for r in diag["gold_support_rows"]
               if r["method"]==selected_arm}
        assert len(local)==32 and len(byid)==32
        comparisons[str(phase)]={
         "same_sample_comparison_ONLY":True,
         "arms":{name:{"EM":info["EM"],"correct":info["correct"],
                        "F1":info["F1"]}
                 for name,info in answers["arms"].items()},
         "selected_arm":selected_arm,
         "phase_reader_decision":answers["decision"]
        }
        for item,case in zip(ctx,can):
            id=item["id"]
            g=gold[id]
            supp={factkey(title,sid) for title,sid in zip(
                g["supporting_facts"]["title"],
                g["supporting_facts"]["sent_id"])}
            assert len(supp)>0
            initial={factkey(c["title"],c["sentence_id"])
                     for pool in case["candidate_pools"] for c in pool}
            expanded=set(initial)
            operational_expansion=[
                pool for j,pool in enumerate(case.get("expanded_pools",[]))
                if j in rescue_targets.get(id,set())]
            for pool in operational_expansion:
                expanded.update(factkey(c["title"],c["sentence_id"]) for c in pool)
            selected_ctx=item["contexts"][selected_arm]
            chosen_idx=selected_ctx["selected_sentence_indexes"]
            final={factkey(item["candidate_map"][j]["title"],
                           item["candidate_map"][j]["sent_id"])
                   for j in chosen_idx}
            assert final
            hits=len(supp&final)
            ann=local[id]
            assert int(ann["all_gold_sentences"])==int(supp<=final)
            assert abs(ann["gold_sentence_recall"]-hits/len(supp))<1e-10
            em=int(byid[id][selected_arm]["EM"])
            pool_contains_all=supp<=expanded
            final_all=supp<=final
            if not pool_contains_all:
                stage="A_missing_gold_from_all_candidate_pools"
            elif not final_all:
                stage="B_gold_accessible_but_not_fully_packed"
            elif em==0:
                stage="C_full_gold_present_reader_incorrect"
            else:
                stage="D_full_gold_present_reader_correct"
            # Some gold can be selected via bulk background ranking even
            # when absent from top4 pools. Label stage as a diagnostic of
            # declared edge-candidate pools, not a total source retrieval bound.
            if not pool_contains_all and final_all:
                stage="E_full_gold_selected_outside_edge_pools"
            allrows.append({
                "id":id,"phase":phase,"method":selected_arm,
                "question_type":g["type"],"has_bridge_type":g["type"]=="bridge_comparison",
                "required_gold_sentence_n":len(supp),
                "top4_gold_sentence_n":len(supp&initial),
                "all_edge_pool_gold_sentence_n":len(supp&expanded),
                "all_gold_top4_pool":int(supp<=initial),
                "all_gold_any_pool":int(supp<=expanded),
                "gold_recall_any_pool":len(supp&expanded)/len(supp),
                "gold_final_sentence_n":hits,
                "all_gold_final":int(final_all),
                "gold_recall_final":hits/len(supp),
                "actual_tokens":selected_ctx["tokens"],
                "true_25pct_cap":selected_ctx["budget"],
                "final_sentence_count":len(chosen_idx),
                "final_distinct_source_docs":len({
                   item["candidate_map"][j]["article"] for j in chosen_idx}),
                "reader_correct":em,"reader_F1":byid[id][selected_arm]["F1"],
                "error_stage":stage,
                "stage_note":"Candidate pool may omit valid original source sentences selected by independent filler."
            })
        by_phase[str(phase)]=summarize([r for r in allrows if r["phase"]==phase])
    assert len(allrows)==160 and len(ids)==160
    # Avoid mistakenly summarizing the source of gold-aware feasibility
    # as a method observation. Quote frozen per-phase oracle rates separately.
    for n,(p,_) in STUDIES.items():
        d=json.loads((p/"GOLD_SUPPORT_RELATION_POSTHOC.json").read_text(encoding="utf8"))
        by_phase[str(n)]["posthoc_gold_oracle_fit"]=d[
            "gold_aware_non_deployable_oracle"]["total_fit"]
    def safe_summary(seq):
        return summarize(seq)
    summary={
       "n":160,
       "unique_questions":len(ids),
       "five_nonoverlapping_public_validation_pilots":True,
       "summary":safe_summary(allrows),
       "by_phase":by_phase,
       "by_question_type":{t:safe_summary([r for r in allrows if r["question_type"]==t])
                          for t in sorted({r["question_type"] for r in allrows})},
       "by_reader_correctness":{
         "correct":safe_summary([r for r in allrows if r["reader_correct"]]),
         "incorrect":safe_summary([r for r in allrows if not r["reader_correct"]])},
       "within_phase_reader_comparisons":comparisons,
       "cross_phase_accuracy_is_not_head_to_head":True,
       "error_stage_interpretation":{
         "A":"At least one annotated gold sentence absent from union of top4+all proposed rerank candidates; not necessarily absent from source",
         "B":"All annotated gold sentences appear in candidate pools but final chosen context lacks at least one",
         "C":"Every annotated gold sentence is final yet reader EM incorrect (annotation coverage not sufficient for reasoning correctness)",
         "D":"Every annotated gold sentence is final and answer correct",
         "E":"Final has every annotated gold sentence although edge candidate pool misses some (filler recovered it)."
       },
       "gold_reused_only_to_diagnose_not_select":True,
       "reader_correct_among_gold_complete_cases":sum(r["reader_correct"] for r in allrows if r["all_gold_final"]),
       "reader_error_despite_gold_complete":sum(not r["reader_correct"] for r in allrows if r["all_gold_final"]),
       "no_new_paid_requests":True,
       "archive_input_hashes":manifest["sources"]}
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"DATA_INPUTS_AND_PROVENANCE.json",manifest)
    with (OUT/"RETROSPECTIVE_CASE_TAXONOMY.jsonl").open("x",encoding="utf8") as f:
        for row in allrows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    new_json(OUT/"RETROSPECTIVE_TAXONOMY_SUMMARY.json",summary)
    print("PHASE20_TAXONOMY",json.dumps({
        "n":summary["n"],"summary":summary["summary"],
        "by_phase":{k:{"n":v["n"],"stages":v["stage_counts"],
             "gold_full":v["all_gold_final"],
             "EM":comparisons[k]["arms"][STUDIES[int(k)][1]]["correct"]}
             for k,v in by_phase.items()},
        "by_type":{k:{"n":v["n"],"stages":v["stage_counts"]}
             for k,v in summary["by_question_type"].items()}},
        ensure_ascii=False),flush=True)
if __name__=="__main__":main()
