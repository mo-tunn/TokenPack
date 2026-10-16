"""Gold-blind local MuSiQue contexts: pinned Phase9 bridge selector, no new policy.

Only id/question/paragraph idx/title/paragraph_text projected from parquet.
Source and original sentence boundaries preserved. cl100k final cap.
"""
from __future__ import annotations
import hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2";os.environ["MKL_NUM_THREADS"]="2"
import pyarrow.parquet as pq
from submission.experiments.phase25_crossbench_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase9_musique_contexts import (
    DATA,PROV,candidates,inference)
from submission.experiments.phase9_crossdoc_core import choose
from submission.experiments.phase9_musique_protocol import (
    DATA_SHA256,DATA_REV,BUDGET_RATIO)
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
OUTFILE=OUT/"gold_blind_contexts.jsonl"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(DATA)==protocol["data_sha256"]==DATA_SHA256
    assert not OUTFILE.exists() and not FROZEN.exists()
    assert BUDGET_RATIO==.25
    fields=["id","question","paragraphs.list.element.idx",
          "paragraphs.list.element.title","paragraphs.list.element.paragraph_text"]
    table=pq.ParquetFile(DATA).read(columns=fields)
    assert set(table.column_names)=={"id","question","paragraphs"}
    for forbidden in ("is_supporting","answer","answer_aliases","question_decomposition"):
        assert forbidden not in repr(table.schema)
    ids=set(protocol["pilot_ids"])
    data={row["id"]:row for row in table.to_pylist() if row["id"] in ids}
    assert set(data)==ids
    assert torch.cuda.is_available()
    counter=TokenCounter()
    assert counter._encoding is not None
    verify_local_model()
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    pre={"dataset_sha256":sha_file(DATA),
       "sampling_sha256":protocol["pilot_sha256"],
       "code_sha256":sha_file(Path(__file__)),
       "phase9_core_sha256":sha_file(
          ROOT/"submission/experiments/phase9_crossdoc_core.py"),
       "phase9_contexts_code_sha256":sha_file(
          ROOT/"submission/experiments/phase9_musique_contexts.py"),
       "reranker_weights_sha256":WEIGHTS_SHA256,
       "parquet_columns":fields,
       "gold_physically_not_read":True,"source_facts_original":True}
    new_json(OUT/"CONTEXT_PRE_GOLD_FREEZE.json",pre)
    t0=time.monotonic()
    totals={"n":0,"source_sentences":0,"bridge_seeds_activated":0,
            "final_cap_violations":0,"mean_actual_tokens":0,
            "mean_hard_cap_tokens":0,"min_final_tokens":None}
    with OUTFILE.open("x",encoding="utf8") as output:
        for j,ident in enumerate(protocol["pilot_ids"],1):
            row=data[ident]
            atoms,_=candidates(row)
            assert atoms
            scores=inference(model,tokenizer,row["question"],
                   [x["input_text"] for x in atoms])
            full=[str(p["title"])+"\n"+str(p["paragraph_text"])
                  for p in row["paragraphs"]
                  if str(p["paragraph_text"]).strip()]
            src_total=sum(counter.count(x) for x in full)
            cap=max(1,math.floor(src_total*BUDGET_RATIO))
            selected=choose(atoms,scores,cap,counter,graph=True)
            assert selected["budget"]==cap
            actual=counter.count(selected["text"])
            assert actual==selected["tokens"]<=cap
            assert selected["text"]
            rec={"id":ident,"question":row["question"],
                 "source_token_total":src_total,"budget":cap,
                 "final_text":selected["text"],"actual_tokens":actual,
                 "selected_original_source_sentences":{
                    "index":selected["selected_sentence_indexes"],
                    "map":[{"article":a["article"],"title":a["title"],
                            "original_paragraph_idx":a["original_paragraph_idx"],
                            "sent_id":a["sent_id"],"sentence":a["sentence"]}
                           for a in atoms]},
                 "selection_metadata":{key:value for key,value in selected.items()
                     if key not in ("text","selected_sentence_indexes")}}
            output.write(json.dumps(rec,ensure_ascii=False)+"\n")
            output.flush();os.fsync(output.fileno())
            totals["n"]+=1
            totals["source_sentences"]+=len(atoms)
            totals["bridge_seeds_activated"]+=int(selected["seed_graph_activated"])
            totals["mean_actual_tokens"]+=actual/64
            totals["mean_hard_cap_tokens"]+=cap/64
            totals["min_final_tokens"]=actual if totals["min_final_tokens"] is None else min(
                 totals["min_final_tokens"],actual)
            if j%8==0:
                print("PHASE25_CONTEXT_PROGRESS",j,64,
                     "elapsed_s",round(time.monotonic()-t0,1),flush=True)
    new_json(FROZEN,{"n":64,"context_file_sha256":sha_file(OUTFILE),
       "preregistered_protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
       "source_pre_gold_metadata_sha256":sha_file(OUT/"CONTEXT_PRE_GOLD_FREEZE.json"),
       "totals":totals,"gold_unopened":True,
       "duration_seconds":round(time.monotonic()-t0,1)})
    print("PHASE25_CONTEXTS_FROZEN",json.dumps(totals,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
