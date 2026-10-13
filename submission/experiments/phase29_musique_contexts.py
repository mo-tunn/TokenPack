"""Assemble and hash-freeze Phase29 paired contexts without loading gold labels."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from submission.experiments.phase28_semantic_core import select
from submission.experiments.phase27_joint_chain_core import baseline_indices
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase7_modal27b_matched8 import read_rows
from tokenpack.tokenization import TokenCounter
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
ARMS=("atomic-sentence-rank","semantic-joint-citations")

def new_json(path,obj):
    with Path(path).open("x",encoding="utf8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["pilot","validation"],required=True);a=ap.parse_args()
    sub=OUT/a.split; source=sub/"scored_original_candidates.jsonl";tasks_path=sub/"semantic_citation_tasks.jsonl"
    verifier_path=sub/"semantic_verifier_responses.jsonl"; verifier_freeze=sub/"VERIFIER_RESPONSES_FROZEN.json"
    output=sub/"gold_blind_paired_contexts.jsonl"; pre=sub/"CONTEXT_ASSEMBLY_PRE_GOLD_FREEZE.json"; frozen=sub/"CONTEXTS_FROZEN.json"
    if any(x.exists() for x in (output,pre,frozen)): raise FileExistsError("Immutable context artifacts already exist.")
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    cf=json.loads((sub/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    vf=json.loads(verifier_freeze.read_text(encoding="utf8"))
    assert sha_file(source)==cf["source_sha256"] and sha_file(tasks_path)==cf["task_sha256"]
    assert sha_file(verifier_path)==vf["sha256"]
    cases=read_rows(source);tasks=read_rows(tasks_path);answers=read_rows(verifier_path)
    ids=p["pilot_ids"] if a.split=="pilot" else p["reserved_ids"]
    assert len(cases)==len(tasks)==len(answers)==len(ids)
    assert [x["id"] for x in cases]==[x["id"] for x in tasks]==[x["id"] for x in answers]==ids
    counter=TokenCounter();assert counter._encoding is not None
    new_json(pre,{"split":a.split,"n":len(ids),"protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "candidate_sha256":cf["source_sha256"],"verifier_sha256":vf["sha256"],
        "semantic_core_sha256":sha_file(ROOT/"submission/experiments/phase28_semantic_core.py"),
        "assembly_code_sha256":sha_file(Path(__file__)),"gold_not_opened":True,
        "same_budget_same_scoring":True})
    stats={"n":0,"contexts_changed":0,"source_sentence_sets_changed":0,
           "valid_citation_judgements":0,"anchors_fitted":0,"fallback_count":0,
           "baseline_selected_sentences":0,"treatment_selected_sentences":0,
           "mean_baseline_tokens":0.0,"mean_treatment_tokens":0.0,"mean_cap":0.0,
           "token_cap_violations":0,"fallback_reasons":{}}
    with output.open("x",encoding="utf8") as f:
        for case,task,answer in zip(cases,tasks,answers):
            atoms=case["atoms"];scores=case["scores"];cap=case["cap"]
            base,bi=baseline_indices(atoms,scores,cap,counter)
            treat,ti,status=select(task,atoms,scores,answer["raw_output"],counter)
            assert base["budget"]==treat["budget"]==cap
            assert counter.count(base["text"])==base["tokens"]<=cap
            assert counter.count(treat["text"])==treat["tokens"]<=cap
            selected={}
            for name,details,indexes in ((ARMS[0],base,bi),(ARMS[1],treat,ti)):
                selected[name]={"text":details["text"],"tokens":details["tokens"],
                    "cap":cap,"source_article_count":details["source_articles"],
                    "selected_sentence_count":len(indexes),
                    "selected_original_pairs":[{"article":atoms[i]["article"],
                        "paragraph_idx":int(atoms[i]["original_paragraph_idx"]),
                        "title":atoms[i]["title"],"sent_id":int(atoms[i]["sent_id"])}
                        for i in indexes]}
            set_base={(x["article"],x["paragraph_idx"],x["sent_id"]) for x in selected[ARMS[0]]["selected_original_pairs"]}
            set_treat={(x["article"],x["paragraph_idx"],x["sent_id"]) for x in selected[ARMS[1]]["selected_original_pairs"]}
            changed=set_base!=set_treat
            record={"id":case["id"],"question":case["question"],
                "source_token_total":case["source_token_total"],"cap":cap,
                "contexts":selected,"semantic_citation_parse_status":answer["parse_status"],
                "semantic_seed_status":status,"source_sentence_set_changed":changed,
                "source_paragraph_set_changed":{x[1] for x in set_base}!={x[1] for x in set_treat},
                "source_has_no_gold_annotations":True}
            f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
            stats["n"]+=1;stats["contexts_changed"]+=int(base["text"]!=treat["text"])
            stats["source_sentence_sets_changed"]+=int(changed)
            stats["valid_citation_judgements"]+=int(answer["parse_status"]=="valid_cross_article_citations")
            stats["anchors_fitted"]+=int(status=="verified_citation_anchors_selected")
            stats["fallback_count"]+=int(status!="verified_citation_anchors_selected")
            if status!="verified_citation_anchors_selected":
                stats["fallback_reasons"][status]=stats["fallback_reasons"].get(status,0)+1
            stats["baseline_selected_sentences"]+=len(bi);stats["treatment_selected_sentences"]+=len(ti)
            stats["mean_baseline_tokens"]+=base["tokens"]/len(ids)
            stats["mean_treatment_tokens"]+=treat["tokens"]/len(ids);stats["mean_cap"]+=cap/len(ids)
            stats["token_cap_violations"]+=int(base["tokens"]>cap or treat["tokens"]>cap)
    new_json(frozen,{"split":a.split,"n":len(ids),"sha256":sha_file(output),
        "pre_gold_assembly_sha256":sha_file(pre),"gold_unopened":True,
        "same_hard_caps":True,"stats":stats})
    print("PHASE29_CONTEXTS_FROZEN",a.split,json.dumps(stats,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
