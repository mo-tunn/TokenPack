"""Faz13 fixed pilot contexts: bridge, flat-edge, graph-path, same scores and budget.
Zero gold columns accessed; GPU MS MARCO weights pinned and verified.
"""
from __future__ import annotations
import json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2";os.environ["MKL_NUM_THREADS"]="2"
from submission.experiments.phase13_register import OUT,BUDGET_RATIO,METHODS
from submission.experiments.phase12_register import DATA,verify_data,sha_file,new_json
from submission.experiments.phase12_contexts import candidates
from submission.experiments.phase9_musique_contexts import inference
from submission.experiments.phase9_crossdoc_core import choose
from submission.experiments.phase11_contexts import reranked_selection
from submission.experiments.phase13_graph_core import graph_select
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
import pyarrow.parquet as pq

def main():
    prereg=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    planner=json.loads((OUT/"PLANNER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    planfile=OUT/"planner_remote.jsonl"
    assert sha_file(planfile)==planner["response_sha256"]
    plans=[json.loads(x) for x in planfile.open(encoding="utf8") if x.strip()]
    ids=prereg["pilot_ids"]
    assert [r["id"] for r in plans]==ids
    if (OUT/"gold_blind_contexts.jsonl").exists():
        raise FileExistsError("Previous partial/frozen contexts must be preserved")
    record={"source_data_sha256":prereg["data_sha256"],
        "pre_gold_protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "planner_sha256":sha_file(planfile),
        "context_script_sha256":sha_file(Path(__file__)),
        "graph_code_sha256":sha_file(ROOT/"submission/experiments/phase13_graph_core.py"),
        "flat_code_sha256":sha_file(ROOT/"submission/experiments/phase11_contexts.py"),
        "bridge_code_sha256":sha_file(ROOT/"submission/experiments/phase9_crossdoc_core.py"),
        "reranker_sha256":WEIGHTS_SHA256,"reranker_revision":REVISION,
        "only_columns_read":["id","question","context.title","context.sentences"],
        "actual_tokenization":"cl100k_base exact rendered text",
        "gold_columns_read":False,
        "methods":list(METHODS)}
    manifest_path=OUT/"CONTEXTS_PRE_GOLD_FREEZE.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf8"))==record
    else:new_json(manifest_path,record)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    assert torch.cuda.is_available()
    counter=TokenCounter()
    assert counter._encoding is not None
    verify_local_model()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    pf=verify_data()
    table=pf.read(columns=["id","question","context.title","context.sentences"])
    assert set(table.column_names)=={"id","question","context"}
    assert "answer:" not in repr(table.schema)
    assert "supporting_facts" not in repr(table.schema)
    assert "evidences" not in repr(table.schema)
    rows={x["id"]:x for x in table.to_pylist() if x["id"] in set(ids)}
    assert set(rows)==set(ids)
    start=time.perf_counter()
    stats={"cases":0,"budget_violations":0,"different_flat_vs_graph":0,
        "different_bridge_vs_graph":0,"graph_seed_pairs":0,
        "fallback_plans":0}
    output=OUT/"gold_blind_contexts.jsonl"
    with output.open("x",encoding="utf8") as writer:
        for plan in plans:
            r=rows[plan["id"]]
            atoms,full,blocks=candidates(r)
            assert atoms and r["question"]
            original_scores=inference(model,tok,r["question"],
                [a["input_text"] for a in atoms])
            branches=plan["branches"]
            edge_meta=[e for b in branches for e in b["edges"]]
            edge_scores=[inference(model,tok,e["query"],
                [a["input_text"] for a in atoms]) for e in edge_meta]
            assert len(edge_scores)==len(edge_meta)
            source_token_sum=sum(counter.count(b) for b in blocks)
            budget=max(1,math.floor(BUDGET_RATIO*source_token_sum))
            control=choose(atoms,original_scores,budget,counter,graph=True)
            flat=reranked_selection(atoms,original_scores,edge_scores,budget,counter)
            graph=graph_select(atoms,original_scores,edge_scores,branches,budget,counter)
            arms={"crossdoc-bridge":control,"flat-edge":flat,"graph-path":graph}
            assert set(arms)==set(METHODS)
            for ctx in arms.values():
                assert ctx["budget"]==budget
                assert ctx["tokens"]==counter.count(ctx["text"])<=budget
            result={
                "id":plan["id"],"question":r["question"],
                "budget":budget,"source_token_sum":source_token_sum,
                "source_context":full,"candidate_sentence_count":len(atoms),
                "candidate_map":[{"title":a["title"],"sent_id":a["sent_id"],
                    "article":a["article"]} for a in atoms],
                "graph_branches":branches,"graph_parse_status":plan["parse_status"],
                "contexts":arms}
            writer.write(json.dumps(result,ensure_ascii=False)+"\n")
            writer.flush();os.fsync(writer.fileno())
            stats["cases"]+=1
            stats["different_flat_vs_graph"]+=int(flat["text"]!=graph["text"])
            stats["different_bridge_vs_graph"]+=int(control["text"]!=graph["text"])
            stats["graph_seed_pairs"]+=graph["paired_branch_seeds"]
            stats["fallback_plans"]+=int(plan["parse_status"]!="valid_structured_graph")
            if stats["cases"]%8==0:
                print("PHASE13_GOLDBLIND_PROGRESS",stats["cases"],len(ids),
                      "seconds",round(time.perf_counter()-start,1),flush=True)
    final={"context_file_sha256":sha_file(output),
        "pre_gold_manifest_sha256":sha_file(manifest_path),
        "samples":len(ids),"stats":stats,"gold_opened":False,
        "wall_seconds":round(time.perf_counter()-start,1)}
    new_json(OUT/"CONTEXTS_FROZEN.json",final)
    print("PHASE13_CONTEXTS_FROZEN",json.dumps(final),flush=True)

if __name__=="__main__":main()
