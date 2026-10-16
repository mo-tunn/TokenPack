"""Formal Phase17 early-futility stop before any reader or gold is opened.

By the frozen reader executor design, exact duplicate prompts for the same
question reuse the previous response without a second call. Therefore the
max possible net EM difference is the number of DISTINCT matched prompts.
A +3 gate cannot pass when that number <3, irrespective of gold answers.
"""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase17_register import OUT
from submission.experiments.phase17_reader import make_payload,prepare,PLAN,TASKS,SCORED,ANS
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase17_protocol import GATE_NET_EM
def h(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def main():
    m=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    frozen=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert frozen["stats"]["token_cap_violations"]==0
    contexts=OUT/"gold_blind_contexts.jsonl"
    assert sha_file(contexts)==frozen["sha256"]
    assert not SCORED.exists() and not ANS.exists()
    tasks=prepare()
    assert len(tasks)==64
    grouped=defaultdict(dict)
    for t in tasks:
        grouped[t["id"]][t["arm"]]=h(make_payload(t))
    assert len(grouped)==32
    assert all(set(x)=={"branch-lock","coherence-gated"} for x in grouped.values())
    distinct=[id for id,s in grouped.items() if s["branch-lock"]!=s["coherence-gated"]]
    same=[id for id,s in grouped.items() if s["branch-lock"]==s["coherence-gated"]]
    assert len(distinct)==frozen["stats"]["contexts_changed"]
    assert len(distinct)==1 and len(same)==31
    # Identical request hashes *cannot* generate paired differences under
    # the frozen reader executor's exact-prompt reuse policy.
    best_case_net_EM_advantage=len(distinct)
    assert GATE_NET_EM==m["pilot_go_net"]==3
    decision="STOP_FUTILITY_GATE_MATHEMATICALLY_UNREACHABLE_BEFORE_READER"
    # Parent 96 IDs are reserved but never processed in this phase.
    assert len(m["validation_ids"])==96
    plan=json.loads((OUT/"PLANNER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    initial=json.loads((OUT/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    rescue=json.loads((OUT/"RESCUE_VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    coherence=json.loads((OUT/"COHERENCE_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert plan["count"]==initial["count"]==rescue["n"]==coherence["n"]==32
    calls=32+32+rescue["real_model_calls"]+32
    out={
      "decision":decision,
      "n_question_pairs":len(grouped),
      "arm_a":"branch-lock","arm_b":"coherence-gated",
      "same_reader_payload_SHA256_pairs":len(same),
      "distinct_reader_payload_SHA256_pairs":len(distinct),
      "distinct_question_ids":distinct,
      "max_possible_net_EM_correct_coherence_minus_branch":best_case_net_EM_advantage,
      "minimum_preregistered_net_EM_to_proceed":GATE_NET_EM,
      "gate_mathematically_possible":best_case_net_EM_advantage>=GATE_NET_EM,
      "proof":"For 31 paired questions, sha256(sorted-json(make_payload)) is byte-identical and the frozen reader reuses exact duplicate prompts; their EM differences must equal 0. The sole distinct question can contribute at most +1 correct; upper bound +1<+3.",
      "read_gold_to_decide_stop":False,
      "reader_model_calls":0,"reader_tasks_created":len(tasks),
      "reader_answers_recorded":0,
      "reader_gold_answer_EM_NOT_MEASURED":True,
      "final_contexts_sha256":frozen["sha256"],
      "reader_tasks_sha256":sha_file(TASKS),
      "reader_pre_remote_protocol_sha256":sha_file(PLAN),
      "estimated_paid_modal_planning_calls":plan["count"],
      "paid_modal_initial_verification_calls":initial["count"],
      "paid_modal_conditional_rescue_calls":rescue["real_model_calls"],
      "paid_modal_coherence_calls":coherence["n"],
      "real_remote_calls_before_reader":calls,
      "real_invoice_not_audited":True,
      "token_cap_violations":0,
      "untouched_validation_ID_count":len(m["validation_ids"]),
      "all_gold_columns_unread":True,
      "source_contexts_frozen":True,
      "qasper_official_test_untouched":True,"production_untouched":True,
      "not_a_reader_accuracy_improvement_claim":True
    }
    path=OUT/"FUTILITY_STOP_PRE_GOLD_AUDIT.json"
    if path.exists():raise FileExistsError(path)
    new_json(path,out)
    print("PHASE17_FORMAL_FUTILITY_STOP",json.dumps({
        "same":len(same),"different":len(distinct),
        "max_net":best_case_net_EM_advantage,"gate":GATE_NET_EM,
        "reader_remote_calls":0,"cost_calls":calls,
        "decision":decision}),flush=True)
if __name__=="__main__":main()
