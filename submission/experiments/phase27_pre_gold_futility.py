"""Phase27 PROSPECTIVE early futility certificate BEFORE any gold or Qwen.

Logic: if gold-blind final contexts differ in D questions, any deterministic
annotated full-support decision depends only on final sentences, so maximum
paired NET difference in full-support count <= D. The predeclared minimum
gain is +5; if D=4 it is provably unattainable for any annotations.
No pilot or reserve gold is read. No reader is invoked.
"""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase27_joint_chain_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase7_modal27b_matched8 import read_rows
DEST=OUT/"PRE_GOLD_FUTILITY_STOP.json"
def main():
    if DEST.exists():raise FileExistsError(DEST)
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    f=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    source=OUT/"gold_blind_paired_contexts.jsonl"
    assert sha_file(source)==f["sha256"]
    cases=read_rows(source)
    assert [r["id"] for r in cases]==protocol["pilot_ids"]
    assert len(cases)==48
    changed=[]
    for row in cases:
        base=row["contexts"]["atomic-sentence-rank"]
        new=row["contexts"]["joint-title-link-seed"]
        assert base["cap"]==new["cap"]==row["cap"]
        assert base["tokens"]<=row["cap"] and new["tokens"]<=row["cap"]
        if {(x["title"],x["sent_id"],x["article"])
             for x in base["selected_original_title_sentence"]}!={
            (x["title"],x["sent_id"],x["article"])
             for x in new["selected_original_title_sentence"]}:
            changed.append(row["id"])
    assert len(changed)==f["stats"]["contexts_changed"]==4
    assert f["stats"]["cap_violations"]==0
    threshold=protocol["support_net_gate"]
    assert threshold==5 and len(changed)<threshold
    # Every unchanged selection has identical exact original support-pair
    # coverage, regardless of the unknown labels. At most 4 decisions
    # can change -> best possible support net win <= 4 < +5.
    assert not (OUT/"reader_responses.jsonl").exists()
    assert not (OUT/"reader_tasks.jsonl").exists()
    assert not (OUT/"READER_RESPONSES_FROZEN.json").exists()
    assert not (OUT/"READER_POST_GOLD_RESULTS.json").exists()
    record={
       "decision":"STOP_PRE_GOLD_STRUCTURAL_FUTILITY",
       "n_pilot":48,"reserved_n_untouched":96,
       "baseline_contexts_same_selection":48-len(changed),
       "contexts_with_modified_original_sentence_sets":len(changed),
       "modified_ids":changed,
       "maximum_possible_net_all_gold_support_gain":len(changed),
       "minimum_required_net_all_gold_support_gain":threshold,
       "mechanistic_gate_mathematically_impossible":True,
       "co_primary_reader_EM_gate_not_executed":True,
       "no_reader_model_calls":True,
       "gold_supporting_fact_labels_never_opened":True,
       "gold_answers_never_opened":True,
       "did_not_modify_or_retry_selection":True,
       "exact_original_sentence_set_comparison":True,
       "structural_report_pre_gold_sha256":f["sha256"],
       "preregister_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
       "no_production_or_qasper_changes":True,
       "interpretation":"This specific title-mention pair seed cannot pass the preset +5 NET full annotated support gate. No claims about answer EM or gold support are made."
    }
    new_json(DEST,record)
    print("PHASE27_FUTILITY",json.dumps({
      "status":record["decision"],
      "contexts_differ":len(changed),
      "net_support_gain_upper_bound":len(changed),
      "support_gate":threshold,"reader_calls":0,"gold_opened":False}),flush=True)
if __name__=="__main__":main()
