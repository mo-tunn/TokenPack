"""Phase19 GOLD-AWARE audit ONLY after all 64 reader outputs frozen/scored.

Do not tune Phase19 on audited gold labels. No remote inference.
"""
from __future__ import annotations
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase19_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
def main():
    reader=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    diag=json.loads((OUT/"GOLD_SUPPORT_RELATION_POSTHOC.json").read_text(encoding="utf8"))
    freeze=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    gate=json.loads((OUT/"STRUCTURAL_PRE_READER_GATE.json").read_text(encoding="utf8"))
    assert reader["decision"]=="STOP_NO_VALIDATION" and reader["n"]==32
    assert gate["decision"]=="GO_READER_PILOT_ONLY"
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==freeze["sha256"]
    ctx={r["id"]:r for r in (
        json.loads(s) for s in (OUT/"gold_blind_contexts.jsonl").open(
            encoding="utf8") if s.strip())}
    support={}
    for entry in diag["gold_support_rows"]:
        support.setdefault(entry["id"],{})[entry["method"]]=entry
    discordance=[]
    for r in reader["per_case"]:
        id=r["id"]
        new=r["soft-bridge-matched"]["EM"]
        old=r["branch-lock"]["EM"]
        if new==old:continue
        row=ctx[id];a=support[id]
        discordance.append({
            "id":id,"winner":"soft-bridge-matched" if new>old else "branch-lock",
            "baseline_EM":old,"new_EM":new,
            "control_all_support":a["branch-lock"]["all_gold_sentences"],
            "new_all_support":a["soft-bridge-matched"]["all_gold_sentences"],
            "control_support_recall":a["branch-lock"]["gold_sentence_recall"],
            "new_support_recall":a["soft-bridge-matched"]["gold_sentence_recall"],
            "control_real_tokens":row["contexts"]["branch-lock"]["tokens"],
            "new_real_tokens":row["contexts"]["soft-bridge-matched"]["tokens"],
            "control_source_docs":row["contexts"]["branch-lock"]["selected_paragraph_count"],
            "new_source_docs":row["contexts"]["soft-bridge-matched"]["selected_paragraph_count"]})
    t0=diag["arms"]["branch-lock"]["mean_real_tokens"]
    t1=diag["arms"]["soft-bridge-matched"]["mean_real_tokens"]
    report={
        "n":32,"reader_discordance_count":len(discordance),
        "new_only_correct":sum(x["winner"]=="soft-bridge-matched" for x in discordance),
        "control_only_correct":sum(x["winner"]=="branch-lock" for x in discordance),
        "mean_actual_tokens_baseline":t0,
        "mean_actual_tokens_new":t1,
        "mean_actual_token_difference":t1-t0,
        "mean_token_ratio_new_to_control":t1/t0,
        "max_absolute_token_delta":freeze["stats"]["max_abs_token_delta"],
        "per_case_token_parity_violations":freeze["stats"]["parity_errors"],
        "number_different_contexts":freeze["stats"]["different_rendered_contexts"],
        "mean_distinct_source_docs_control":diag["arms"]["branch-lock"]["mean_paragraphs"],
        "mean_distinct_source_docs_new":diag["arms"]["soft-bridge-matched"]["mean_paragraphs"],
        "complete_gold_support_control":diag["arms"]["branch-lock"]["all_support_sentences_n"],
        "complete_gold_support_new":diag["arms"]["soft-bridge-matched"]["all_support_sentences_n"],
        "oracle_gold_fit_cases_not_deployable":diag["gold_aware_non_deployable_oracle"]["total_fit"],
        "not_causal_proof":True,
        "posthoc_only_never_used_for_selection":True,
        "per_reader_discordance":discordance}
    path=OUT/"READER_FAILURE_POSTHOC_AUDIT.json"
    if path.exists():raise FileExistsError(path)
    new_json(path,report)
    print("PHASE19_POSTHOC_FAILURE_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
