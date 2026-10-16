"""Phase11 local GPU multi-query sentence selection; explicit gold-blind parquet projection."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
import pyarrow.parquet as pq
from submission.experiments.phase11_register import OUT,DATA,PROV,digest,save_new
from submission.experiments.phase11_musique_protocol import *
from submission.experiments.phase9_musique_protocol import DATA_REV,DATA_SHA256
from submission.experiments.phase9_musique_contexts import candidates,inference
from submission.experiments.phase9_crossdoc_core import choose,render_atomic,rank_rescale,bridge_link
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter

def load_source(ids):
    assert digest(DATA)==DATA_SHA256
    assert json.loads(PROV.read_text(encoding="utf8"))["rev"]==DATA_REV
    columns=["id","question","paragraphs.list.element.idx",
             "paragraphs.list.element.title",
             "paragraphs.list.element.paragraph_text"]
    table=pq.ParquetFile(DATA).read(columns=columns)
    assert set(table.column_names)=={"id","question","paragraphs"}
    assert "is_supporting" not in repr(table.schema)
    assert "answer:" not in repr(table.schema)
    chosen={r["id"]:r for r in table.to_pylist() if r["id"] in set(ids)}
    assert set(chosen)==set(ids)
    return chosen

def reranked_selection(atoms,base,sub_scores,budget,counter):
    """Fixed 3-query diversification + subsequent cost-aware greedy.
    Score blend and anchor policy frozen in phase11_musique_protocol.
    """
    if not atoms or len(atoms)!=len(base):raise ValueError("Wrong candidate list")
    direct=rank_rescale(base)
    other=[rank_rescale(x) for x in sub_scores]
    assert all(len(x)==len(atoms) for x in other) and len(other)>=1
    rel=[ORIGINAL_WEIGHT*direct[i]+QUERY_WEIGHT*max(s[i] for s in other)
         for i in range(len(atoms))]
    bydoc=defaultdict(int)
    chosen=[];seen=set()
    checks=0
    def add(i,lim):
        nonlocal checks
        if i in seen:return False
        checks+=1
        if counter.count(render_atomic([*chosen,i],atoms))>lim:return False
        chosen.append(i);seen.add(i);bydoc[atoms[i]["article"]]+=1
        return True
    anchor_cap=max(1,math.floor(budget*ANCHOR_BUDGET_SHARE))
    for sub in other:
        order=sorted(range(len(atoms)),key=lambda i:(-sub[i],-direct[i],
                   atoms[i]["article"],atoms[i]["sent_id"],i))
        for i in order:
            if bydoc.get(atoms[i]["article"],0)>0:continue
            if add(i,anchor_cap):break
    remaining=set(range(len(atoms)))-seen
    while remaining:
        def util(i):
            edge=max((bridge_link(atoms[i],atoms[j]) for j in chosen
                     if atoms[i]["article"]!=atoms[j]["article"]),default=0.0)
            return (rel[i]+CROSSDOC_LINK_BONUS*edge
                    -DOC_REPEAT_PENALTY*min(3,bydoc[atoms[i]["article"]]))
        order=sorted(remaining,key=lambda i:(-util(i),atoms[i]["article"],
                                               atoms[i]["sent_id"],i))
        selected=None
        for i in order:
            if add(i,budget):selected=i;break
        if selected is None:break
        remaining.remove(selected)
    text=render_atomic(chosen,atoms)
    used=counter.count(text)
    assert used<=budget
    return {"text":text,"tokens":used,"budget":budget,
        "selected_sentence_indexes":sorted(chosen),
        "selected_source_idxs":sorted({atoms[i]["original_paragraph_idx"] for i in chosen}),
        "selected_paragraph_count":len(bydoc),
        "selected_sentence_count":len(chosen),"anchor_cap":anchor_cap,
        "render_checks":checks}

def main():
    manifest=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    ids=manifest["pilot_ids"]
    plans=OUT/"pilot_planner_remote.jsonl"
    freeze=json.loads((OUT/"pilot_planner_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert digest(plans)==freeze["raw_plans_sha256"]
    prep=[json.loads(x) for x in plans.open(encoding="utf8") if x.strip()]
    assert [r["id"] for r in prep]==ids
    assert not (OUT/"pilot_gold_blind_contexts.jsonl").exists()
    assert not (OUT/"pilot_CONTEXTS_FROZEN.json").exists()
    record={"data_sha256":DATA_SHA256,"planner_response_sha256":digest(plans),
            "plan_sha256":digest(OUT/"PRE_GOLD_PROTOCOL.json"),
            "model_weights_sha256":WEIGHTS_SHA256,"model_revision":REVISION,
            "script_sha256":digest(Path(__file__)),
            "selector_sha256":digest(ROOT/"submission/experiments/phase9_crossdoc_core.py"),
            "phase11_parameter_sha256":digest(ROOT/"submission/experiments/phase11_musique_protocol.py"),
            "methods":METHODS,
            "gold_visible_to_inference":False,
            "budget_note":"Exact final cl100k rendered cap, equal cap not equal realized tokens"}
    planfile=OUT/"pilot_CONTEXTS_PRE_GOLD_FREEZE.json"
    if planfile.exists():
        assert json.loads(planfile.read_text(encoding="utf8"))==record
    else:save_new(planfile,record)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    assert torch.cuda.is_available(),"Local CUDA unavailable"
    token_count=TokenCounter()
    assert token_count._encoding is not None
    verify_local_model()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    source=load_source(ids)
    start=time.perf_counter()
    n=0
    totals={"budget_violations":0,"different_contexts":0,"fallback_plans":0}
    target=OUT/"pilot_gold_blind_contexts.jsonl"
    with target.open("x",encoding="utf8") as output:
        for t in prep:
            row=source[t["id"]]
            atoms,full=candidates(row)
            assert atoms
            scores=inference(model,tok,row["question"],[x["input_text"] for x in atoms])
            subs=t["queries"][:MAX_PLANNED_QUERIES]
            assert subs
            extra=[inference(model,tok,q,[x["input_text"] for x in atoms]) for q in subs]
            source_sum=sum(token_count.count(p["title"]+"\n"+p["paragraph_text"])
                for p in row["paragraphs"] if p["paragraph_text"].strip())
            cap=max(1,math.floor(BUDGET_RATIO*source_sum))
            old=choose(atoms,scores,cap,token_count,graph=True)
            new=reranked_selection(atoms,scores,extra,cap,token_count)
            assert old["budget"]==new["budget"]==cap
            assert token_count.count(old["text"])==old["tokens"]<=cap
            assert token_count.count(new["text"])==new["tokens"]<=cap
            data={"id":t["id"],"question":row["question"],"source_token_sum":source_sum,
                "source_context":full,"budget":cap,"candidate_sentences":len(atoms),
                "planner_queries":subs,"planner_parse_status":t["parse_status"],
                "contexts":{"crossdoc-bridge":old,"hop-planner":new}}
            output.write(json.dumps(data,ensure_ascii=False)+"\n")
            output.flush();os.fsync(output.fileno())
            n+=1
            totals["different_contexts"]+=int(old["text"]!=new["text"])
            totals["fallback_plans"]+=int(t["parse_status"]!="valid_json_queries")
            if n%8==0 or n==len(ids):
                print("PHASE11_GOLDBLIND_PROGRESS",n,len(ids),
                     "elapsed_s",round(time.perf_counter()-start,1),flush=True)
    final={"contexts_sha256":digest(target),"pre_context_manifest_sha256":digest(planfile),
        "sample_n":n,"stats":totals,
        "wall_s":round(time.perf_counter()-start,1),"gold_read":False,
        "production_changed":False}
    save_new(OUT/"pilot_CONTEXTS_FROZEN.json",final)
    print("PHASE11_CONTEXTS_FROZEN",json.dumps(final),flush=True)

if __name__=="__main__":main()
