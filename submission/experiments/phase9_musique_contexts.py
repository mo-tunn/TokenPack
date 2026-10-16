"""MuSiQue gold-blind fixed sample and 2-arm local reranking experiment.

Parquet leaf column projection omits gold is_supporting and answer fields.
No previous experiment is changed; no paid endpoint or model downloads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

from submission.experiments.phase9_musique_protocol import (
    DATA_REV,DATA_SHA256,EXPECTED_DATASET_ROWS,SAMPLE_SEED,N_QUESTIONS,
    BUDGET_RATIO,METHODS,MAX_SCORE_INPUT_TOKENS,RANK_BATCH_SIZE,
)
from submission.experiments.phase9_crossdoc_core import choose
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model,
)
from tokenpack.tokenization import TokenCounter

DATA=ROOT/"submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"
PROV=DATA.parent.parent/"PROVENANCE.json"
OUTPUT=ROOT/"submission/results/phase9_musique_bridge_new256_20261012"


def sample_ids(ids,n=N_QUESTIONS):
    if len(ids)!=len(set(ids)):
        raise AssertionError("Duplicate MuSiQue ID")
    if len(ids)<n:raise ValueError("Not enough MuSiQue cases")
    return sorted(ids,key=lambda x:(
        hashlib.sha256((SAMPLE_SEED+"|"+x).encode("utf8")).hexdigest(),x))[:n]


def split_sentences(text):
    parts=re.split(r"(?<=[.!?])\s+(?=[\"'\[(]?[A-Z0-9])",str(text))
    return [x.strip() for x in parts if x.strip()]


def candidates(record):
    atoms=[]
    source=[]
    for article,para in enumerate(record["paragraphs"]):
        if set(para)!={"idx","paragraph_text","title"}:
            raise AssertionError("Gold/answer field may have leaked")
        text=str(para["paragraph_text"]).strip()
        title=str(para["title"])
        if not text:continue
        source.append(title+"\n"+text)
        for idx,sentence in enumerate(split_sentences(text)):
            atoms.append({
                "article":article,
                "original_paragraph_idx":int(para["idx"]),
                "sent_id":idx,"title":title,
                "sentence":sentence,
                "input_text":title+"\n"+sentence,
            })
    return atoms,"\n\n".join(source)


@torch.inference_mode()
def inference(model,tokenizer,question,texts):
    res=[]
    for i in range(0,len(texts),RANK_BATCH_SIZE):
        part=texts[i:i+RANK_BATCH_SIZE]
        toks=tokenizer([question]*len(part),part,padding=True,
                       truncation="only_second",
                       max_length=MAX_SCORE_INPUT_TOKENS,
                       return_tensors="pt")
        toks={k:v.to("cuda") for k,v in toks.items()}
        res.extend(model(**toks).logits.squeeze(-1).float().cpu().tolist())
    if len(res)!=len(texts):
        raise AssertionError("Incomplete cross-encoder scoring")
    return res


def main():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Results directory already exists")
    proven=json.loads(PROV.read_text(encoding="utf8"))
    if proven["rev"]!=DATA_REV or proven["sha256"]!=DATA_SHA256:
        raise AssertionError("Unpinned benchmark file")
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=DATA_SHA256:
        raise AssertionError("MuSiQue parquet has changed")
    source_columns=[
        "id","question",
        "paragraphs.list.element.idx",
        "paragraphs.list.element.title",
        "paragraphs.list.element.paragraph_text",
    ]
    pf=pq.ParquetFile(DATA)
    if pf.metadata.num_rows!=EXPECTED_DATASET_ROWS:
        raise AssertionError("Unexpected benchmark rows")
    table=pf.read(columns=source_columns)
    if set(table.column_names)!={"id","question","paragraphs"}:
        raise AssertionError("Unexpected projected columns")
    if "is_supporting" in repr(table.schema) or "answer:" in repr(table.schema):
        raise AssertionError("Gold fields leaked into inference schema")
    rows=table.to_pylist()
    sample=sample_ids([x["id"] for x in rows])
    sample_set=set(sample)
    chosen={x["id"]:x for x in rows if x["id"] in sample_set}
    assert len(chosen)==N_QUESTIONS
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("Real cl100k token counter unavailable")
    verify_local_model()
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    manifest={
        "benchmark":"MuSiQue public validation, new to TokenPack experiments",
        "repo":proven["repo"],"data_commit":proven["rev"],
        "data_sha256":DATA_SHA256,
        "sample_seed":SAMPLE_SEED,
        "sample_ids_sha256":hashlib.sha256(
            "\n".join(sorted(sample)).encode("utf8")).hexdigest(),
        "sample_n":N_QUESTIONS,
        "models":"Pretrained MS MARCO MiniLM, same weights both arms",
        "reranker_revision":REVISION,"reranker_sha256":WEIGHTS_SHA256,
        "real_budget_ratio":BUDGET_RATIO,
        "methods":METHODS,
        "gold_fields_physically_not_read":True,
        "answer_or_supporting_labels_in_model_inputs":False,
        "QASPER_test_reused":False,
        "production_changed":False,
    }
    (OUTPUT/"PRE_GOLD_PROTOCOL.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    start=time.perf_counter()
    totals={"source_sentences":0,"cases":0,"graph_seed_activated":0,
            "different_graph_render":0,"budget_violations":0}
    filepath=OUTPUT/"gold_blind_contexts.jsonl"
    with filepath.open("x",encoding="utf8") as output:
        for j,id in enumerate(sample,1):
            row=chosen[id]
            atom,source=candidates(row)
            if not atom:raise AssertionError("No source candidates")
            score=inference(model,tokenizer,row["question"],
                            [a["input_text"] for a in atom])
            source_token_sum=sum(counter.count(
                p["title"]+"\n"+p["paragraph_text"])
                for p in row["paragraphs"] if p["paragraph_text"].strip())
            budget=max(1,math.floor(BUDGET_RATIO*source_token_sum))
            baseline=choose(atom,score,budget,counter,graph=False)
            graph=choose(atom,score,budget,counter,graph=True)
            assert baseline["budget"]==graph["budget"]==budget
            assert counter.count(baseline["text"])<=budget
            assert counter.count(graph["text"])<=budget
            record={
                "id":id,"question":row["question"],
                "source_context":source,
                "source_token_count":counter.count(source),
                "source_token_sum":source_token_sum,
                "budget":budget,
                "candidate_sentences":len(atom),
                "contexts":{
                    "atomic-rank":baseline,
                    "crossdoc-bridge":graph,
                },
            }
            output.write(json.dumps(record,ensure_ascii=False)+"\n")
            output.flush()
            totals["cases"]+=1
            totals["source_sentences"]+=len(atom)
            totals["graph_seed_activated"]+=int(graph["seed_graph_activated"])
            totals["different_graph_render"]+=int(
                baseline["text"]!=graph["text"])
            if j%32==0 or j==len(sample):
                print("MUSIQUE_GOLDBLIND_PROGRESS",j,len(sample),
                      "seconds",round(time.perf_counter()-start,1),
                      flush=True)
    final={
        **manifest,"selected_cases":len(sample),
        "case_file_sha256":hashlib.sha256(filepath.read_bytes()).hexdigest(),
        "audit":totals,
        "wall_seconds":round(time.perf_counter()-start,2),
    }
    (OUTPUT/"CONTEXTS_FROZEN.json").write_text(
        json.dumps(final,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("MUSIQUE_CONTEXTS_FROZEN",json.dumps(
        {"audit":totals,"wall_seconds":final["wall_seconds"]},
        ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
