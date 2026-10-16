"""Faz18 immutable posthoc reader-discordance and token confound audit.
No model calls. Posthoc only after freeze and reader scoring; not for tuning.
"""
import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase18_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
def main():
    reader=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    diag=json.loads((OUT/"GOLD_SUPPORT_RELATION_POSTHOC.json").read_text(encoding="utf8"))
    freeze=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert reader["decision"]=="STOP_NO_VALIDATION" and reader["n"]==32
    f=OUT/"gold_blind_contexts.jsonl"
    assert sha_file(f)==freeze["sha256"]
    contexts={x["id"]:x for x in (json.loads(line) for line in f.open(encoding="utf8") if line.strip())}
    metric={}
    for x in diag["gold_support_rows"]:
        metric.setdefault(x["id"],{})[x["method"]]=x
    by=[]
    for case in reader["per_case"]:
        ident=case["id"]
        baseline=case["branch-lock"]["EM"]
        new=case["citation-closed"]["EM"]
        if baseline==new:continue
        ctx=contexts[ident]["contexts"]
        g=metric[ident]
        by.append({"id":ident,
            "winner":"branch-lock" if baseline>new else "citation-closed",
            "baseline_EM":baseline,"new_EM":new,
            "baseline_all_support":g["branch-lock"]["all_gold_sentences"],
            "new_all_support":g["citation-closed"]["all_gold_sentences"],
            "baseline_support_recall":g["branch-lock"]["gold_sentence_recall"],
            "new_support_recall":g["citation-closed"]["gold_sentence_recall"],
            "baseline_tokens":ctx["branch-lock"]["tokens"],
            "new_tokens":ctx["citation-closed"]["tokens"],
            "baseline_document_count":ctx["branch-lock"]["selected_paragraph_count"],
            "new_document_count":ctx["citation-closed"]["selected_paragraph_count"]})
    result={
        "reader_discordant_n":len(by),
        "baseline_only_correct":sum(v["winner"]=="branch-lock" for v in by),
        "new_only_correct":sum(v["winner"]=="citation-closed" for v in by),
        "mean_token_delta_over_32":diag["arms"]["citation-closed"]["mean_real_tokens"]-diag["arms"]["branch-lock"]["mean_real_tokens"],
        "actual_token_ratio_new_over_control":diag["arms"]["citation-closed"]["mean_real_tokens"]/diag["arms"]["branch-lock"]["mean_real_tokens"],
        "mean_document_delta":diag["arms"]["citation-closed"]["mean_paragraphs"]-diag["arms"]["branch-lock"]["mean_paragraphs"],
        "all_support_control":diag["arms"]["branch-lock"]["all_support_sentences_n"],
        "all_support_new":diag["arms"]["citation-closed"]["all_support_sentences_n"],
        "gold_aware_oracle_fit":diag["gold_aware_non_deployable_oracle"]["total_fit"],
        "posthoc_only":True,"not_causal_proof":True,
        "per_discordance":by
    }
    out=OUT/"READER_FAILURE_POSTHOC_AUDIT.json"
    if out.exists():raise FileExistsError(out)
    new_json(out,result)
    print("PHASE18_READER_POSTHOC_AUDIT",json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
