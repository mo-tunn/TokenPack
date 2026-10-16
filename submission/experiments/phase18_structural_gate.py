"""Faz18 preregistered reader-before-gold structural effect-size gate."""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase18_register import OUT
from submission.experiments.phase18_protocol import (
    STRUCTURAL_DIFFERENT_MIN,GATE_NET_EM)
from submission.experiments.phase18_reader import prepare,make_payload
from submission.experiments.phase12_register import sha_file,new_json
def digest(s):
    return hashlib.sha256(json.dumps(s,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def main():
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    f=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert f["stats"]["token_violations"]==0
    assert sha_file(OUT/"gold_blind_contexts.jsonl")==f["sha256"]
    assert p["structural_gate"]["at_least_N_distinct_final_reader_prompts"]==STRUCTURAL_DIFFERENT_MIN
    assert p["pilot_go_net"]==GATE_NET_EM
    pairs=defaultdict(dict)
    for t in prepare():
        pairs[t["id"]][t["arm"]]=digest(make_payload(t))
    assert len(pairs)==32 and all(set(x)=={"branch-lock","citation-closed"} for x in pairs.values())
    diff=[id for id,x in pairs.items() if x["branch-lock"]!=x["citation-closed"]]
    success=len(diff)>=STRUCTURAL_DIFFERENT_MIN and f["stats"]["token_violations"]==0
    # With exact-prompt re-use, EM improvement cannot exceed number of distinct prompts.
    exact_futility=len(diff)<GATE_NET_EM
    if not success:decision="STOP_BEFORE_READER_STRUCTURAL_GATE_FAILED"
    else:decision="GO_READER_PILOT_ONLY"
    info={"decision":decision,"n":len(pairs),"distinct_reader_payload_pairs":len(diff),
        "identical_pairs":len(pairs)-len(diff),
        "distinct_ids":diff,"structural_min_required":STRUCTURAL_DIFFERENT_MIN,
        "theoretical_max_net_EM_correct":len(diff),
        "reader_EM_gate_net":GATE_NET_EM,
        "reader_gate_already_impossible":exact_futility,
        "token_cap_violations":f["stats"]["token_violations"],
        "contexts_sha256":f["sha256"],
        "reader_tasks_sha256":sha_file(OUT/"reader_tasks.jsonl"),
        "reader_answers_exist":(OUT/"reader_responses.jsonl").exists(),
        "gold_unread":True,"validation_96_untouched":True}
    assert not info["reader_answers_exist"]
    fpath=OUT/"STRUCTURAL_PRE_READER_GATE.json"
    if fpath.exists():raise FileExistsError(fpath)
    new_json(fpath,info)
    print("PHASE18_STRUCTURAL_GATE",json.dumps({
        "decision":decision,"different":len(diff),
        "required":STRUCTURAL_DIFFERENT_MIN,
        "max_net":len(diff)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
