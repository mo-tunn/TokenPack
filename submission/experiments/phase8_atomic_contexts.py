"""Gold-blind extraction of atomic HotpotQA evidence contexts on 152 new IDs.

Never reads any answer/supporting_fact field. Do not use this script to
optimize against labeled observations. The selection method uses fixed
predeclared parameters and true final cl100k_base accounting.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["OMP_NUM_THREADS"]="2"

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

from submission.experiments.phase8_atomic_protocol import (
    EVAL_START,EVAL_STOP,EVAL_COUNT,BUDGET_RATIO,ARMS,
    SCORER_MAX_INPUT_TOKENS,DOCUMENT_MAX_INPUT_TOKENS,BATCH_SIZE,
)
from submission.experiments.phase7_prepare_contexts import OUTPUT as PHASE7
from submission.experiments.phase6_hotpot_eval import DATA,PROVENANCE
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model,
)
from tokenpack.tokenization import TokenCounter

OUT=ROOT/"submission/results/phase8_atomic_goldblind152_20261012"


def sentence_candidates(row):
    """Only title, sentence list from provided source, never supporting facts."""
    c=row["context"]
    if len(c["title"])!=len(c["sentences"]):
        raise AssertionError("Mismatch source article titles and sentences")
    all_sentences=[]
    articles=[]
    for j,(title,sentences) in enumerate(
            zip(c["title"],c["sentences"],strict=True)):
        if not sentences:continue
        title=str(title)
        full=[str(x) for x in sentences if str(x).strip()]
        if not full:continue
        # Distinct article instances are preserved even for duplicate titles.
        articles.append({"source_index":j,"title":title,
                         "text":title+"\n"+" ".join(full)})
        for i,s in enumerate(sentences):
            if not str(s).strip():continue
            all_sentences.append({
                "article":j,"title":title,"sent_id":i,
                "sentence":str(s),
                "input_text":title+"\n"+str(s),
            })
    return articles,all_sentences


def render(selected,atoms):
    """Compact, source-ordered title grouping; includes whole sentences only."""
    grouped={}
    for idx in sorted(selected,key=lambda i:(atoms[i]["article"],
                                               atoms[i]["sent_id"])):
        article=atoms[idx]["article"]
        grouped.setdefault(article,[]).append((atoms[idx]["title"],
                                                atoms[idx]["sentence"]))
    parts=[]
    for _,pairs in grouped.items():
        parts.append("[Source: "+pairs[0][0]+"]\n"+
                     " ".join(s for _,s in pairs))
    return "\n\n".join(parts).strip()+("\n" if parts else "")


def choose_atoms(atoms,scores,doc_scores,limit,counter,*,bridge):
    """Fully gold-blind, exact final-render cap at EVERY decision.

    Baseline: score-ranked best single source sentences.
    Bridge: select one highest-scoring sentence from each of the two
    highest scoring distinct source articles before filling with ranked
    sentences. Two titles must fit; if not, fallback to baseline.
    """
    if len(atoms)!=len(scores) or not atoms:
        raise AssertionError("Candidate scoring mismatch")
    rank=sorted(range(len(atoms)),
                key=lambda i:(-scores[i],atoms[i]["article"],
                              atoms[i]["sent_id"]))
    chosen=[]
    seen=set()
    used=0
    def try_add(index):
        nonlocal used
        if index in seen:return False
        text=render([*chosen,index],atoms)
        real=counter.count(text)
        if real>limit:return False
        chosen.append(index)
        seen.add(index)
        used=real
        return True

    if bridge:
        # The ranker is PRETRAINED and no gold annotation enters this rank.
        distinct=sorted(doc_scores,
                        key=lambda i:(-doc_scores[i],i))
        if len(distinct)>=2:
            seed_indices=[]
            for article in distinct[:2]:
                best=next((i for i in rank if atoms[i]["article"]==article),
                          None)
                if best is not None:seed_indices.append(best)
            if len(seed_indices)==2 and counter.count(
                    render(seed_indices,atoms))<=limit:
                for i in seed_indices:
                    assert try_add(i)
    for i in rank:
        try_add(i)
    final=render(chosen,atoms)
    if counter.count(final)>limit:
        raise AssertionError("Final sentence render exceeds budget")
    return {
        "text":final,"tokens":counter.count(final),"budget":limit,
        "source_articles":len({atoms[i]["article"] for i in chosen}),
        "sentences":len(chosen),
        "candidate_count":len(atoms),
    }


@torch.inference_mode()
def encode_scores(model,tokenizer,question,texts,*,max_length):
    result=[]
    for start in range(0,len(texts),BATCH_SIZE):
        chunk=texts[start:start+BATCH_SIZE]
        tokens=tokenizer([question]*len(chunk),chunk,
                         padding=True,truncation="only_second",
                         max_length=max_length,return_tensors="pt")
        tokens={k:v.to("cuda") for k,v in tokens.items()}
        result.extend(model(**tokens).logits.squeeze(-1).float().cpu().tolist())
    if len(result)!=len(texts) or not all(math.isfinite(x) for x in result):
        raise AssertionError("Invalid model scores")
    return result


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Do not overwrite a prior Phase8 experiment")
    provenance=json.loads(PROVENANCE.read_text(encoding="utf8"))
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=provenance["sha256"]:
        raise AssertionError("Official HotpotQA data changed")
    saved=PHASE7/"gold_blind_contexts.jsonl"
    freeze=json.loads((PHASE7/"CONTEXTS_COMPLETE.json").read_text(encoding="utf8"))
    if hashlib.sha256(saved.read_bytes()).hexdigest()!=freeze["contexts_sha256"]:
        raise AssertionError("Phase7 original contexts changed")
    previous=[json.loads(s) for s in saved.read_text(encoding="utf8").splitlines()
              if s.strip()]
    if len(previous)!=256:raise AssertionError("Need frozen Phase7 256 sample")
    chosen=previous[EVAL_START:EVAL_STOP]
    assert len(chosen)==EVAL_COUNT
    ids={z["id"] for z in chosen}
    first_ids={z["id"] for z in previous[:EVAL_START]}
    if ids&first_ids:
        raise AssertionError("Phase8 overlaps earlier Qwen27B evaluation")
    # Read NO answer or supporting_fact fields into this process.
    raw={
        r["id"]:r for r in pq.read_table(
            DATA,columns=["id","question","context"]).to_pylist()
        if r["id"] in ids
    }
    if len(raw)!=EVAL_COUNT:
        raise AssertionError("Dataset question missing")
    verify_local_model()
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    counter=TokenCounter()
    if counter._encoding is None:raise RuntimeError("Real tokenizer absent")
    OUT.mkdir(parents=True,exist_ok=False)
    manifest={
        "experiment":"Phase8 atomic/bridge evidence selector",
        "source_ids":f"Frozen Phase7 source index [{EVAL_START},{EVAL_STOP})",
        "n":EVAL_COUNT,"methods":ARMS,
        "model_id":"cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_revision":REVISION,"weight_sha256":WEIGHTS_SHA256,
        "ratio":BUDGET_RATIO,
        "input_gold_fields_loaded":False,
        "initial_104_Qwen_answer_cases_excluded":True,
        "full_original_contexts_SHA256":freeze["contexts_sha256"],
        "benchmark_sha256":provenance["sha256"],
        "formatting_changed_vs_original_chunk_renderer":True,
        "reader_answer_accuracy_not_yet_measured":True,
        "candidate_level":"literal entire original article sentence, with title",
    }
    (OUT/"GOLD_BLIND_PROTOCOL.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    start=time.perf_counter()
    details=Counter()
    output=OUT/"gold_blind_atomic_contexts.jsonl"
    with output.open("x",encoding="utf8") as file:
        for j,item in enumerate(chosen,1):
            row=raw[item["id"]]
            if row["question"]!=item["question"]:
                raise AssertionError("Question source mismatch")
            baseline=item["contexts"]["msmarco-25"]
            limit=baseline["budget"]
            assert limit==item["contexts"]["manual-25"]["budget"]
            articles,atoms=sentence_candidates(row)
            scores=encode_scores(
                model,tokenizer,row["question"],
                [x["input_text"] for x in atoms],
                max_length=SCORER_MAX_INPUT_TOKENS)
            document_scores_list=encode_scores(
                model,tokenizer,row["question"],
                [a["text"] for a in articles],
                max_length=DOCUMENT_MAX_INPUT_TOKENS)
            doc_scores={
                x["source_index"]:score
                for x,score in zip(articles,document_scores_list,strict=True)}
            plain=choose_atoms(atoms,scores,doc_scores,limit,counter,bridge=False)
            bridge=choose_atoms(atoms,scores,doc_scores,limit,counter,bridge=True)
            cases={
                "msmarco-25":baseline,
                "atomic-sentence-rank":plain,
                "atomic-bridge-first":bridge,
            }
            if any(v["tokens"]>limit for v in cases.values()):
                raise AssertionError("True output budget violated")
            result={
                "id":item["id"],"question":row["question"],
                "source_tokens":item["source_tokens"],
                "contexts":cases,
                "source_context_count":len(articles),
                "candidate_sentences":len(atoms),
            }
            file.write(json.dumps(result,ensure_ascii=False)+"\n")
            file.flush()
            details["questions"]+=1
            details["source_sentences"]+=len(atoms)
            details["plain_selected_sentences"]+=plain["sentences"]
            details["bridge_selected_sentences"]+=bridge["sentences"]
            details["bridge_changes"]+=int(plain["text"]!=bridge["text"])
            if j%25==0 or j==len(chosen):
                print("ATOMIC_PROGRESS",j,len(chosen),
                      "elapsed_s",round(time.perf_counter()-start,1),
                      flush=True)
    (OUT/"CONTEXTS_FROZEN.json").write_text(
        json.dumps({
            **manifest,"audit":dict(details),
            "contexts_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "seconds":round(time.perf_counter()-start,2),
        },indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("ATOMIC_FROZEN",dict(details),flush=True)


if __name__=="__main__":
    run()
