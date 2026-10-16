"""Preregister Phase13 2Wiki graph-vs-flat comparison using only Phase12 RESERVED IDs.

Validation IDs are chosen before planning and never touched if pilot gate fails.
No gold columns are read. Existing frozen artifacts remain untouched.
"""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import (
    DATA,verify_data,OUT as PREVIOUS,sha_file,new_json,
)
OUT=ROOT/"submission/results/phase13_2wiki_relation_graph_20261012"
PILOT_N=32
VALIDATION_N=96
BUDGET_RATIO=.25
METHODS=("crossdoc-bridge","flat-edge","graph-path")
PILOT_GATE_EM_NET=3
MAX_REMOTE_PLANNER=32
MAX_REMOTE_READER=96

GRAPH_SYSTEM=(
    "Design a minimal evidence RELATION GRAPH for multi-hop question answering. "
    "Do NOT answer the question or assert the final answer. "
    "Return ONLY compact JSON of exactly this shape: "
    '{"branches":[{"edges":[{"subject":"entity","relation":"relation sought",'
    '"object":"unknown or entity","query":"short evidence search query"},'
    '{"subject":"entity","relation":"relation sought",'
    '"object":"unknown or entity","query":"short evidence search query"}]}]}. '
    "Use one branch of 1-2 edges for sequential relation questions. "
    "Use TWO branches of 1-2 edges each when comparing properties or paths "
    "of two entities. Each query <=18 words and useful for locating an ORIGINAL "
    "supporting sentence. Avoid duplicate edges, do not guess final facts, "
    "do not add labels, commentary or markdown. No gold or documents available."
)

def main():
    if OUT.exists():raise FileExistsError("Phase13 already registered")
    prev=json.loads((PREVIOUS/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert prev["phase"]=="12" and len(prev["ids"])==64
    assert len(prev["reserved_ids"])==128
    prevf=PREVIOUS/"gold_blind_contexts.jsonl"
    frozen=json.loads((PREVIOUS/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(prevf)==frozen["context_file_sha256"]
    assert not set(prev["ids"])&set(prev["reserved_ids"])
    verify_data()
    ids=prev["reserved_ids"][:PILOT_N]
    hold=prev["reserved_ids"][PILOT_N:PILOT_N+VALIDATION_N]
    assert len(ids)==PILOT_N and len(hold)==VALIDATION_N
    assert set(ids).isdisjoint(hold)
    files=["phase13_graph_core.py","phase13_graph_modal_relay_source.py",
           "phase9_crossdoc_core.py","phase11_contexts.py",
           "phase12_contexts.py","phase7_modal27b_matched8.py"]
    base=ROOT/"submission/experiments"
    manifest={
        "study":"Phase13 gold-blind question-derived relation graph vs flat edge-query ranking",
        "registered_before_graph_inference_and_gold":True,
        "dataset":"2WikiMultiHopQA public validation (NOT hidden independent test)",
        "parent_phase12_protocol_sha256":sha_file(PREVIOUS/"PRE_GOLD_PROTOCOL.json"),
        "parent_phase12_goldblind_context_sha256":frozen["context_file_sha256"],
        "data_sha256":prev["parquet_sha256"],
        "sampling":"exact 128 preregistered UNUSED Phase12 reserved IDs; first 32 pilot, next 96 validation",
        "pilot_ids":ids,"validation_ids":hold,
        "pilot_id_sha256":hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "validation_id_sha256":hashlib.sha256("\n".join(hold).encode()).hexdigest(),
        "context_token_max_ratio":BUDGET_RATIO,
        "methods":METHODS,
        "planner_system":GRAPH_SYSTEM,
        "planner_max_new_tokens":520,
        "planner_temperature":0,
        "planner_input_only_question":True,
        "graph_parse":"phase13_graph_core.normalized_edges: at most two branches, two edges each; fallback original question",
        "fair_ablation":"flat-edge and graph-path share identical Qwen edge queries and MS MARCO edge scores; differ in edge-pair linking strategy",
        "crossdoc_bridge":"Phase9 existing zero-shot method on same 2Wiki source",
        "flat_edge":"exact Phase11 unstructured reranked_selection on flattened edge queries",
        "graph_path":"graph core connected pair seeding by lexical inter-document links, budget-conscious greedy",
        "gold_primary":"exact all annotated (title,sentence-index) supporting pairs in rendered context",
        "gold_secondary":"support sentence recall, by 2Wiki question type, gold-aware exact fit posthoc only",
        "reader":"same Qwen27B zero temp short answer prompt, 32 cases * 3 arms",
        "reader_primary":"paired normalized EM graph-path minus flat-edge",
        "reader_gate":"at least +3 net correct graph-path vs flat-edge on fixed 32, zero budget errors; otherwise 96 reserved IDs untouched",
        "reader_gate_net":PILOT_GATE_EM_NET,
        "reader_max_paid_calls":MAX_REMOTE_READER,
        "planner_max_paid_calls":MAX_REMOTE_PLANNER,
        "model":"MS MARCO zero-shot MiniLM and Qwen27B Q4KM",
        "cloud_secret":"qwen-modal-secrets; gold not uploaded; TLS checks enabled",
        "per_session_estimated_usd_guard":1.35,
        "estimated_hour_usd":2.60,
        "real_invoice_not_a_hard_cap":True,
        "training_test_data_not_accessed":True,
        "phase3d_qasper_test_never_reused":True,"production_changed":False,
        "source_sha256":{p:sha_file(base/p) for p in files},
    }
    OUT.mkdir(parents=True,exist_ok=False)
    new_json(OUT/"PRE_GOLD_PROTOCOL.json",manifest)
    print("PHASE13_PRE_GOLD_REGISTERED",
        json.dumps({"pilot_n":len(ids),"reserved_validation":len(hold),
                    "all_disjoint_phase12_used":True,
                    "pilot_id_sha256":manifest["pilot_id_sha256"]}),flush=True)
if __name__=="__main__":main()
