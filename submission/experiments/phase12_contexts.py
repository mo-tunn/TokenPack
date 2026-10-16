"""Gold-blind 2WikiMultiHopQA sentence selection at actual cl100k cap.
Retains Phase9 bridge selector and exact unmodified Phase11 hop selector.
"""
from __future__ import annotations
import hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2";os.environ["MKL_NUM_THREADS"]="2"
from submission.experiments.phase12_register import OUT,DATA,verify_data,sha_file,new_json
from submission.experiments.phase9_musique_contexts import inference
from submission.experiments.phase9_crossdoc_core import choose
from submission.experiments.phase11_contexts import reranked_selection
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
import pyarrow.parquet as pq

def candidates(row):
    ctx=row["context"]
    assert set(ctx)=={"title","sentences"}
    assert len(ctx["title"])==len(ctx["sentences"]) and len(ctx["title"])>0
    atoms=[];source=[];source_sum=[]
    for article,(title,sents) in enumerate(zip(ctx["title"],ctx["sentences"])):
        if not sents:continue
        title=str(title)
        sents=[str(s).strip() for s in sents]
        source.append(title+"\n"+" ".join(sents))
        source_sum.append(title+"\n"+" ".join(sents))
        for sid,s in enumerate(sents):
            if not s:continue
            atoms.append({"article":article,"original_paragraph_idx":article,
                "sent_id":sid,"title":title,"sentence":s,
                "input_text":title+"\n"+s})
    assert atoms
    return atoms,"\n\n".join(source),source_sum

def main():
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    plansfile=OUT/"planner_remote.jsonl"
    frozen=json.loads((OUT/"PLANNER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    assert sha_file(plansfile)==frozen["responses_sha256"]
    plans=[json.loads(x) for x in plansfile.open(encoding="utf8") if x.strip()]
    assert [p["id"] for p in plans]==protocol["ids"]
    record={"dataset_sha256":protocol["parquet_sha256"],
        "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "planner_sha256":sha_file(plansfile),
        "context_code_sha256":sha_file(Path(__file__)),
        "phase11_selector_code_sha256":sha_file(ROOT/"submission/experiments/phase11_contexts.py"),
        "phase9_bridge_code_sha256":sha_file(ROOT/"submission/experiments/phase9_crossdoc_core.py"),
        "model_sha256":WEIGHTS_SHA256,"model_revision":REVISION,
        "gold_columns_read":False,
        "dataset_projection":["id","question","context.title","context.sentences"]}
    prereg=OUT/"CONTEXTS_PRE_GOLD_PROTOCOL.json"
    if prereg.exists():assert json.loads(prereg.read_text(encoding="utf8"))==record
    else:new_json(prereg,record)
    if (OUT/"gold_blind_contexts.jsonl").exists():
        raise FileExistsError("Preserve existing frozen/partial contexts")
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    assert torch.cuda.is_available()
    counter=TokenCounter()
    assert counter._encoding is not None
    verify_local_model()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(FOLDER,local_files_only=True).to("cuda").eval()
    pf=verify_data()
    table=pf.read(columns=["id","question","context.title","context.sentences"])
    assert set(table.column_names)=={"id","question","context"}
    assert "answer:" not in repr(table.schema)
    assert "supporting_facts" not in repr(table.schema) and "evidences" not in repr(table.schema)
    ids=set(protocol["ids"])
    rows={r["id"]:r for r in table.to_pylist() if r["id"] in ids}
    assert set(rows)==ids
    start=time.perf_counter()
    n=0;differ=0;violations=0
    output=OUT/"gold_blind_contexts.jsonl"
    with output.open("x",encoding="utf8") as writer:
        for plan in plans:
            r=rows[plan["id"]]
            atoms,full,source_blocks=candidates(r)
            score=inference(model,tok,r["question"],[a["input_text"] for a in atoms])
            extra=[inference(model,tok,q,[a["input_text"] for a in atoms]) for q in plan["queries"][:3]]
            budget=max(1,math.floor(.25*sum(counter.count(p) for p in source_blocks)))
            old=choose(atoms,score,budget,counter,graph=True)
            new=reranked_selection(atoms,score,extra,budget,counter)
            for c in (old,new):
                assert c["tokens"]==counter.count(c["text"])<=budget
            mapping=[{"title":a["title"],"sent_id":a["sent_id"],
                "article":a["article"]} for a in atoms]
            result={"id":plan["id"],"question":r["question"],
                "source_context":full,"source_token_sum":sum(counter.count(p) for p in source_blocks),
                "budget":budget,"candidates":len(atoms),"candidate_map":mapping,
                "planner_queries":plan["queries"],"planner_status":plan["status"],
                "contexts":{"crossdoc-bridge":old,"hop-planner":new}}
            writer.write(json.dumps(result,ensure_ascii=False)+"\n");writer.flush();os.fsync(writer.fileno())
            n+=1;differ+=int(old["text"]!=new["text"])
            if n%8==0:
                print("PHASE12_GOLDBLIND_PROGRESS",n,len(plans),
                    "elapsed_s",round(time.perf_counter()-start,1),flush=True)
    final={"cases":n,"real_cl100k_cap_violations":violations,
           "different_contexts":differ,
           "context_file_sha256":sha_file(output),
           "pre_gold_protocol_sha256":sha_file(prereg),
           "inference_read_gold":False,
           "wall_seconds":round(time.perf_counter()-start,1)}
    new_json(OUT/"CONTEXTS_FROZEN.json",final)
    print("PHASE12_CONTEXTS_FROZEN",json.dumps(final),flush=True)

if __name__=="__main__":main()
