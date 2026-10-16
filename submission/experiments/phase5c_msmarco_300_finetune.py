"""Faz 5C train-only MS MARCO cross-encoder domain adaptation on QASPER.

Offline GPU. Only official QASPER TRAIN (888 paper group), 300-token native
chunks. One fixed hyperparameter configuration, pairwise logistic ranking
objective (positive gold evidence vs hard negative, within same paper).
No validation/test access, no model downloading, no production changes.

After training, evaluate saved checkpoint separately in phase5c_300_eval.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_DATA,DEFAULT_CACHE,groups_with_ngrams,alt_relevance_labels,
    parse_input,
)
from submission.experiments.phase3b_native_qasper import (
    native_questions_from_qasper_row,native_blocks_with_section_hints,
)
from submission.experiments.phase4a_qasper_multires_oracle import (
    SCALES,make_cached,
)
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,MODEL_ID,REVISION,WEIGHTS_SHA256,verify_local_model,
)
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import make_embedder
from tokenpack.scoring import score_chunks
from tokenpack.tokenization import TokenCounter

SEED=20261011
PAIRS_PER_POSITIVE=2


def construct_train_triplets(rows,embedder,counter,*,paper_limit=888,
                             max_triplets=4000,seed=SEED):
    """Positive/negative candidates share the query's train paper only."""
    if not 1<=paper_limit<=888:
        raise ValueError("Train paper limit out of range")
    audit=Counter()
    randomizer=random.Random(seed)
    triplets=[]
    for pi,paper in enumerate(rows[:paper_limit],1):
        pid=str(paper["id"])
        native,_=native_questions_from_qasper_row(paper)
        eligible=[(q,groups_with_ngrams(q.evidence_groups)) for q in native]
        eligible=[(q,g) for q,g in eligible if g]
        if not eligible:continue
        blocks=native_blocks_with_section_hints(paper,counter)
        if not blocks:continue
        cache=make_cached(
            ROOT/".tokenpack/cache/phase4a_multires_20261011",
            DEFAULT_CACHE,pid)
        block_vectors=cache.get_or_embed([b.text for b in blocks],embedder)
        chunks=StructureAwareChunker(
            **SCALES[300],token_counter=counter,
            block_embeddings=block_vectors,semantic_threshold=.35).chunk(blocks)
        vectors=cache.get_or_embed([c.text for c in chunks],embedder)
        queries=cache.get_or_embed([q.question for q,_ in eligible],embedder)
        for (q,gold),vector in zip(eligible,queries,strict=True):
            scored=score_chunks(vector,chunks,vectors,scoring="evidence-hybrid",
                                query_text=q.question)
            labels=alt_relevance_labels(scored,gold)
            positive=sorted(
                [i for i,label in enumerate(labels) if label>0],
                key=lambda i:(-labels[i],-scored[i].value,i))
            negative=sorted(
                [i for i,label in enumerate(labels) if label==0],
                key=lambda i:(-scored[i].value,i))
            if not positive or not negative:
                audit["no_positive_or_negative"]+=1
                continue
            audit["eligible_questions"]+=1
            for chosen_pos in positive[:min(2,len(positive))]:
                for neg in negative[:PAIRS_PER_POSITIVE]:
                    triplets.append((q.question,
                                     scored[chosen_pos].chunk.text,
                                     scored[neg].chunk.text))
        audit["train_papers"]+=1
        if pi%120==0 or pi==paper_limit:
            print("TRIPLET_SOURCE",pi,paper_limit,
                  "train_pairs",len(triplets),flush=True)
    # Fix deterministic sampling without ever consulting validation labels.
    randomizer.shuffle(triplets)
    audit["all_available_triplets"]=len(triplets)
    triplets=triplets[:max_triplets]
    audit["selected_triplets"]=len(triplets)
    return triplets,dict(audit)


def train_pairwise(model,tokenizer,triplets,*,device,batch_size,
                   gradient_accumulation,epochs,learning_rate):
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,
                                weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    history=[]
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for step in range(0,len(triplets),batch_size):
            pairs=triplets[step:step+batch_size]
            n=len(pairs)
            q=[a[0] for a in pairs]
            inp=tokenizer(q+q,[a[1] for a in pairs]+[a[2] for a in pairs],
                          truncation="only_second",padding=True,
                          max_length=512,return_tensors="pt")
            inp={k:v.to(device) for k,v in inp.items()}
            with torch.amp.autocast("cuda",enabled=device=="cuda"):
                logits=model(**inp).logits.squeeze(-1)
                loss=torch.nn.functional.softplus(
                    -(logits[:n]-logits[n:])).mean()
                scaled_loss=loss/gradient_accumulation
            scaler.scale(scaled_loss).backward()
            batch_num=step//batch_size+1
            total_batches=(len(triplets)+batch_size-1)//batch_size
            if batch_num%gradient_accumulation==0 or batch_num==total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if batch_num%125==0:
                print("PAIRWISE_TRAIN",epoch+1,epochs,batch_num,total_batches,
                      "rolling_loss",round(float(np.mean(losses[-125:])),4),
                      flush=True)
        history.append({
            "epoch":epoch+1,
            "mean_softplus_pair_loss":round(float(np.mean(losses)),6),
            "batches":total_batches,
        })
        print("EPOCH_FINISHED",history[-1],flush=True)
    return history


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--train-papers",type=int,default=888)
    parser.add_argument("--max-triplets",type=int,default=4000)
    parser.add_argument("--batch-size",type=int,default=2)
    parser.add_argument("--gradient-accumulation",type=int,default=4)
    parser.add_argument("--learning-rate",type=float,default=8e-6)
    parser.add_argument("--epochs",type=int,default=1)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Will not overwrite prior model or experiment")
    verify_local_model()
    split,counts=parse_input(DEFAULT_DATA)
    if counts!={"train":888,"validation":281}:
        raise AssertionError("Official QASPER split mismatch")
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device!="cuda":
        raise RuntimeError("This is an explicit GPU training experiment")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    start=time.perf_counter()
    counter=TokenCounter()
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    examples,stats=construct_train_triplets(
        split["train"],embedder,counter,paper_limit=args.train_papers,
        max_triplets=args.max_triplets)
    if len(examples)<50:
        raise ValueError("Not enough QASPER train positive/negative pairs")
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to(device)
    print("PAIRWISE_PRETRAINED",MODEL_ID,
          "pairs",len(examples),"TRAIN_ONLY",True,flush=True)
    history=train_pairwise(
        model,tokenizer,examples,device=device,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,learning_rate=args.learning_rate)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"model").mkdir(parents=True,exist_ok=False)
    model.save_pretrained(args.output_dir/"model")
    tokenizer.save_pretrained(args.output_dir/"model")
    result={
        "experiment":"Phase5C QASPER TRAIN ONLY MS MARCO native300 pairwise fine-tuning",
        "pretrained_model":MODEL_ID,
        "pretrained_revision":REVISION,
        "pretrained_model_sha256":WEIGHTS_SHA256,
        "training":stats,
        "seed":SEED,"config":{
            "epochs":args.epochs,"max_triplets":args.max_triplets,
            "batch_size":args.batch_size,
            "gradient_accumulation":args.gradient_accumulation,
            "lr":args.learning_rate,
            "loss":"softplus(-(positive_logit-negative_logit))",
            "max_wordpiece_input_length":512,
            "source_chunk_target_tokens":300,
        },
        "loss_history":history,
        "validation_used":False,"test_used":False,
        "network_or_new_model_download_during_training":False,
        "gpu":device,
        "elapsed_seconds":round(time.perf_counter()-start,2),
    }
    (args.output_dir/"TRAIN_CHECKPOINT.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("SAVED_PAIRWISE_MODEL",args.output_dir/"model",flush=True)


if __name__=="__main__":
    main()
