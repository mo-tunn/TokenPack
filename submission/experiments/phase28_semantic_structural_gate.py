"""Pre-gold structural bound check for Phase28 before reader inference.

If K modified original-source sentence sets < mechanistic +5 NET gate,
impossibility proven regardless of unknown gold labels, so stop.
Else permit next paired reader stage; no gold or answers accessed.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase28_semantic_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
PATH=OUT/"PRE_GOLD_STRUCTURAL_DECISION.json"
def main():
    if PATH.exists():raise FileExistsError(PATH)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    c=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    source=OUT/"gold_blind_paired_contexts.jsonl"
    assert sha_file(source)==c["sha256"]
    rows=read_rows(source)
    assert len(rows)==48 and [x["id"] for x in rows]==p["pilot_ids"]
    changes=[]
    for r in rows:
        a={ (s["article"],s["title"],s["sent_id"])
           for s in r["contexts"]["atomic-sentence-rank"]["selected_original_pairs"]}
        b={ (s["article"],s["title"],s["sent_id"])
           for s in r["contexts"]["semantic-joint-citations"]["selected_original_pairs"]}
        assert r["contexts"]["atomic-sentence-rank"]["cap"]==r["contexts"]["semantic-joint-citations"]["cap"]
        if a!=b:changes.append(r["id"])
    assert len(changes)==c["stats"]["source_sentence_sets_changed"]
    assert c["stats"]["token_cap_violations"]==0
    needed=p["mechanistic_go_gate_net_full_support"]
    assert needed==5
    impossible=len(changes)<needed
    assert not (OUT/"reader_responses.jsonl").exists()
    info={"decision":"STOP_PRE_GOLD_STRUCTURAL_FUTILITY" if impossible
                  else "READER_COMPARISON_PERMITTED_NOT_YET_SCORED",
       "modified_original_source_sets":len(changes),
       "maximum_possible_net_full_gold_support_gain":len(changes),
       "required_support_net_gain":needed,
       "structural_support_gate_impossible":impossible,
       "modified_case_ids":changes,
       "pilot_n":48,"reserved_96_untouched":True,
       "gold_unopened":True,"no_reader_calls_yet":True,
       "context_source_sha256":c["sha256"],
       "no_question_selected_on_gold":True}
    new_json(PATH,info)
    print("PHASE28_STRUCTURAL_GATE",json.dumps({
        "decision":info["decision"],"changed":len(changes),
        "gate":needed,"gold_opened":False}),flush=True)
if __name__=="__main__":main()
