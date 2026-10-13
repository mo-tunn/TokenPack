"""Apply precommitted gold-blind Qwen multi-hop citations to ORIGINAL sentences.

Two arms share scores/source/hard 25% cap. Reader gold unopened.
"""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase28_semantic_register import OUT
from submission.experiments.phase28_semantic_core import select
from submission.experiments.phase27_joint_chain_core import baseline_indices
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
from tokenpack.tokenization import TokenCounter
OUTPUT=OUT/"gold_blind_paired_contexts.jsonl"
PRE=OUT/"CONTEXT_ASSEMBLY_PRE_GOLD_FREEZE.json"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
ARMS=("atomic-sentence-rank","semantic-joint-citations")
def main():
    if any(x.exists() for x in (OUTPUT,PRE,FROZEN)):raise FileExistsError("Immutable")
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    candidatef=json.loads((OUT/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    verifierf=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(OUT/"scored_original_candidates.jsonl")==candidatef["source_sha256"]
    assert sha_file(OUT/"semantic_citation_tasks.jsonl")==candidatef["task_sha256"]
    assert sha_file(OUT/"semantic_verifier_responses.jsonl")==verifierf["sha256"]
    cases=read_rows(OUT/"scored_original_candidates.jsonl")
    tasks=read_rows(OUT/"semantic_citation_tasks.jsonl")
    answers=read_rows(OUT/"semantic_verifier_responses.jsonl")
    assert len(cases)==len(tasks)==len(answers)==48
    assert [x["id"] for x in cases]==[x["id"] for x in tasks]==[x["id"] for x in answers]==protocol["pilot_ids"]
    counter=TokenCounter();assert counter._encoding is not None
    new_json(PRE,{"n":48,
        "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "candidate_sha256":candidatef["source_sha256"],
        "verifier_sha256":verifierf["sha256"],
        "semantic_core_sha256":sha_file(ROOT/"submission/experiments/phase28_semantic_core.py"),
        "assembly_code_sha256":sha_file(Path(__file__)),
        "gold_not_opened":True,"same_budget_same_scoring":True})
    stats={"n":0,"contexts_changed":0,"source_sentence_sets_changed":0,
       "citation_verification_positive":0,"anchors_fitted":0,
       "fallback_count":0,"baseline_selected_sentences":0,
       "treatment_selected_sentences":0,"mean_baseline_tokens":0,
       "mean_treatment_tokens":0,"mean_cap":0,
       "token_cap_violations":0,
       "fallback_reasons":{}}
    with OUTPUT.open("x",encoding="utf8") as f:
      for case,task,answer in zip(cases,tasks,answers):
        atoms=case["atoms"];scores=case["scores"];cap=case["cap"]
        base,bi=baseline_indices(atoms,scores,cap,counter)
        treat,ti,status=select(task,atoms,scores,answer["raw_output"],counter)
        assert base["budget"]==treat["budget"]==cap
        assert counter.count(base["text"])==base["tokens"]<=cap
        assert counter.count(treat["text"])==treat["tokens"]<=cap
        selected={}
        for name,details,indexes in (
            (ARMS[0],base,bi),(ARMS[1],treat,ti)):
            selected[name]={"text":details["text"],"tokens":details["tokens"],
               "cap":cap,"source_article_count":details["source_articles"],
               "selected_sentence_count":len(indexes),
               "selected_original_pairs":[{"title":atoms[i]["title"],
                 "sent_id":atoms[i]["sent_id"],"article":atoms[i]["article"]}
                 for i in indexes]}
        changed={(x["title"],x["sent_id"],x["article"])
                    for x in selected[ARMS[0]]["selected_original_pairs"]}!={
                  (x["title"],x["sent_id"],x["article"])
                    for x in selected[ARMS[1]]["selected_original_pairs"]}
        record={"id":case["id"],"question":case["question"],
          "source_total":case["source_token_total"],"cap":cap,
          "contexts":selected,"semantic_citation_parse_status":answer["parse_status"],
          "semantic_seed_status":status,
          "source_sentence_set_changed":changed,
          "source_has_no_gold_annotations":True}
        f.write(json.dumps(record,ensure_ascii=False)+"\n")
        f.flush();os.fsync(f.fileno())
        stats["n"]+=1
        stats["contexts_changed"]+=int(base["text"]!=treat["text"])
        stats["source_sentence_sets_changed"]+=int(changed)
        stats["citation_verification_positive"]+=int(answer["parse_status"]=="valid_cross_article_citations")
        stats["anchors_fitted"]+=int(status=="verified_citation_anchors_selected")
        stats["fallback_count"]+=int(status!="verified_citation_anchors_selected")
        if status!="verified_citation_anchors_selected":
            stats["fallback_reasons"][status]=stats["fallback_reasons"].get(status,0)+1
        stats["baseline_selected_sentences"]+=len(bi)
        stats["treatment_selected_sentences"]+=len(ti)
        stats["mean_baseline_tokens"]+=base["tokens"]/48
        stats["mean_treatment_tokens"]+=treat["tokens"]/48
        stats["mean_cap"]+=cap/48
    new_json(FROZEN,{"n":48,"sha256":sha_file(OUTPUT),
      "pre_gold_assembly_sha256":sha_file(PRE),
      "gold_unopened":True,"same_hard_caps":True,"stats":stats})
    print("PHASE28_CONTEXTS_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
