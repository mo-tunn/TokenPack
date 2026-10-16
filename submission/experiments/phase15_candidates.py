"""Gold-blind frozen candidate pools for batched source-grounded Qwen verification.

All source sentences are original 2Wiki provided context, never supporting flags.
Writes scored originals and per-edge top4 candidates. No gold, no Modal calls.
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
from submission.experiments.phase15_register import OUT
from submission.experiments.phase15_protocol import (
    EDGE_TOP_K, EXPANSION_TOP_K, TOKEN_CAP_RATIO as BUDGET_RATIO,
    EDGE_WEIGHT, REFORM_WEIGHT, ORIGINAL_WEIGHT)
from submission.experiments.phase9_crossdoc_core import rank_rescale
from submission.experiments.phase12_register import (
    DATA,verify_data,sha_file,new_json)
from submission.experiments.phase12_contexts import candidates
from submission.experiments.phase9_musique_contexts import inference
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
import pyarrow.parquet as pq

def main():
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    plans=json.loads((OUT/"PLANNER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    planfile=OUT/"planner_responses.jsonl"
    assert sha_file(planfile)==plans["responses_sha256"]
    responses=[json.loads(x) for x in planfile.open(encoding="utf8") if x.strip()]
    assert [x["id"] for x in responses]==protocol["pilot_ids"]
    output=OUT/"scored_candidate_pools.jsonl"
    freeze=OUT/"CANDIDATES_FROZEN.json"
    if output.exists() or freeze.exists():raise FileExistsError("No overwrite")
    meta={"phase15_protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "source_sha256":sha_file(Path(__file__)),
        "data_sha256":sha_file(DATA),
        "plan_sha256":sha_file(planfile),
        "weights_sha256":WEIGHTS_SHA256,"weights_revision":REVISION,
        "candidate_top_k":EDGE_TOP_K,"expansion_top_k":EXPANSION_TOP_K,
        "gold_columns_read":False,
        "columns":["id","question","context.title","context.sentences"]}
    new_json(OUT/"CANDIDATES_PRE_GOLD_PROTOCOL.json",meta)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    assert torch.cuda.is_available()
    counter=TokenCounter()
    assert counter._encoding is not None
    verify_local_model()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    table=verify_data().read(columns=["id","question","context.title","context.sentences"])
    assert set(table.column_names)=={"id","question","context"}
    assert "supporting_facts" not in repr(table.schema)
    rows={x["id"]:x for x in table.to_pylist()
        if x["id"] in set(protocol["pilot_ids"])}
    assert set(rows)==set(protocol["pilot_ids"])
    t0=time.monotonic();total_candidates=0
    with output.open("x",encoding="utf8") as f:
        for idx,plan in enumerate(responses,1):
            row=rows[plan["id"]]
            atoms,full,blocks=candidates(row)
            original=inference(model,tok,row["question"],
                 [a["input_text"] for a in atoms])
            edges=[e for b in plan["branches"] for e in b["edges"]]
            edge_scores=[inference(model,tok,e["query"],
                [a["input_text"] for a in atoms]) for e in edges]
            cap=max(1,math.floor(BUDGET_RATIO*sum(counter.count(v) for v in blocks)))
            pools=[];expansions=[]
            for e,scores in enumerate(edge_scores):
                order=sorted(range(len(atoms)),
                    key=lambda i:(-scores[i],atoms[i]["article"],
                                  atoms[i]["sent_id"],i))[:EDGE_TOP_K]
                pool=[{"candidate_id":f"e{e}_c{j}","atom_index":i,
                       "title":atoms[i]["title"],
                       "sentence_id":atoms[i]["sent_id"],
                       "sentence":atoms[i]["sentence"]}
                      for j,i in enumerate(order)]
                pools.append(pool)
                total_candidates+=len(pool)
                # A second gold-blind search with edge-specific structured
                # query, plus the original edge-query and question scores.
                # Preserve the first four candidates; append a distinct set.
                src=edges[e]
                reform=" ".join(str(src.get(k) or "") for k in
                    ("subject","relation","object") if
                    str(src.get(k) or "").casefold() not in ("?","unknown"))
                if not reform.strip(): reform=src["query"]
                reform_scores=inference(model,tok,reform,
                    [a["input_text"] for a in atoms])
                original_norm=rank_rescale(original)
                first_norm=rank_rescale(scores)
                reform_norm=rank_rescale(reform_scores)
                combine=[
                    EDGE_WEIGHT*first_norm[i]+REFORM_WEIGHT*reform_norm[i]
                    +ORIGINAL_WEIGHT*original_norm[i]
                    for i in range(len(atoms))]
                excluded=set(order)
                available=sorted(
                    (i for i in range(len(atoms)) if i not in excluded),
                    key=lambda i:(-combine[i],atoms[i]["article"],
                         atoms[i]["sent_id"],i))
                chosen=[];used_docs={atoms[i]["article"] for i in order}
                # Round one prefer documents absent from the old top4,
                # then unseen documents relative to new picks.
                for i in available:
                    if atoms[i]["article"] not in used_docs:
                        chosen.append(i);used_docs.add(atoms[i]["article"])
                        if len(chosen)==EXPANSION_TOP_K:break
                if len(chosen)<EXPANSION_TOP_K:
                    for i in available:
                        if i not in chosen:
                            chosen.append(i)
                            if len(chosen)==EXPANSION_TOP_K:break
                expansion=[{"candidate_id":f"e{e}_x{j}","atom_index":i,
                    "title":atoms[i]["title"],"sentence_id":atoms[i]["sent_id"],
                    "sentence":atoms[i]["sentence"]}
                    for j,i in enumerate(chosen)]
                assert not ({p["atom_index"] for p in expansion}
                            & {p["atom_index"] for p in pool})
                expansions.append(expansion)
            data={"id":plan["id"],"question":row["question"],
               "source_context":full,"source_token_sum":sum(counter.count(v) for v in blocks),
               "budget":cap,"atoms":atoms,"branches":plan["branches"],
               "edge_queries":[e["query"] for e in edges],
               "original_scores":[float(v) for v in original],
               "edge_scores":[[float(v) for v in s] for s in edge_scores],
               "candidate_pools":pools,
               "expanded_pools":expansions}
            f.write(json.dumps(data,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            if idx%8==0:
                print("PHASE15_CANDIDATE_PROGRESS",idx,len(responses),
                    "elapsed_s",round(time.monotonic()-t0,1),flush=True)
    new_json(freeze,{"count":len(responses),"candidate_count":total_candidates,
        "candidate_sha256":sha_file(output),
        "pre_gold_metadata_sha256":sha_file(OUT/"CANDIDATES_PRE_GOLD_PROTOCOL.json"),
        "gold_read":False,"wall_seconds":round(time.monotonic()-t0,1)})
    print("PHASE15_CANDIDATES_FROZEN",len(responses),total_candidates,flush=True)
if __name__=="__main__":main()
