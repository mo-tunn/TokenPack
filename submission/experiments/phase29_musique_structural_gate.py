"""Gold-blind structural futility gate for Phase29 paired contexts."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase7_modal27b_matched8 import read_rows
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["pilot","validation"],required=True);a=ap.parse_args()
    sub=OUT/a.split;path=sub/"PRE_GOLD_STRUCTURAL_DECISION.json"
    if path.exists():raise FileExistsError(path)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    c=json.loads((sub/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    source=sub/"gold_blind_paired_contexts.jsonl"
    assert sha_file(source)==c["sha256"]
    rows=read_rows(source);ids=p["pilot_ids"] if a.split=="pilot" else p["reserved_ids"]
    assert len(rows)==len(ids) and [x["id"] for x in rows]==ids
    changes=[]
    for r in rows:
        base={(x["article"],x["paragraph_idx"],x["sent_id"]) for x in r["contexts"]["atomic-sentence-rank"]["selected_original_pairs"]}
        new={(x["article"],x["paragraph_idx"],x["sent_id"]) for x in r["contexts"]["semantic-joint-citations"]["selected_original_pairs"]}
        assert r["contexts"]["atomic-sentence-rank"]["cap"]==r["contexts"]["semantic-joint-citations"]["cap"]==r["cap"]
        assert r["contexts"]["atomic-sentence-rank"]["tokens"]<=r["cap"]
        assert r["contexts"]["semantic-joint-citations"]["tokens"]<=r["cap"]
        if base!=new:changes.append(r["id"])
    assert len(changes)==c["stats"]["source_sentence_sets_changed"]
    assert c["stats"]["token_cap_violations"]==0
    needed=5;impossible=len(changes)<needed
    assert not (sub/"reader_responses.jsonl").exists()
    info={"split":a.split,
      "decision":"STOP_PRE_GOLD_STRUCTURAL_FUTILITY" if impossible else "READER_COMPARISON_PERMITTED_NOT_YET_SCORED",
      "modified_original_source_sets":len(changes),
      "maximum_possible_net_full_support_paragraph_coverage_gain":len(changes),
      "required_support_net_gain":needed,"structural_support_gate_impossible":impossible,
      "modified_case_ids":changes,"n":len(rows),
      "reserved_confirmation_ids_untouched":a.split=="pilot",
      "gold_unopened":True,"no_reader_calls_yet":True,
      "context_source_sha256":c["sha256"]}
    with path.open("x",encoding="utf8") as f:json.dump(info,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print("PHASE29_STRUCTURAL_GATE",json.dumps({k:info[k] for k in ["split","decision","modified_original_source_sets","required_support_net_gain","gold_unopened"]}),flush=True)
if __name__=="__main__":main()
