"""Phase 5A: local cached MiniLM cross-encoder reranker, opt-in GPU experiment.

Paper-disjoint official QASPER train for fine-tune, validation for exploratory
evaluation only; NEVER read held-out official QASPER test.

No data/model network download. The local sentence-transformers MiniLM
encoder backbone is initialized as BertForSequenceClassification with
new, trainable scalar head. Train positive/negative within-paper QASPER
question-chunk pairs on CPU or CUDA. At inference all candidate chunks
receive a joint question-chunk cross-encoder score, and the frozen Phase 3C
render-aware token-budget selector builds output contexts.

All gold is used only for TRAIN pair labels, never inference features.
Production source and previous experiment files are unchanged.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter,defaultdict
import hashlib
import json
import math
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
from torch.utils.data import DataLoader,TensorDataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from submission.experiments.phase3b_full_qasper_train import (
    DEFAULT_CACHE,DEFAULT_DATA,parse_input,prepare_split_batched,
)
from submission.experiments.phase3b_learned_scoring import (
    make_learned_candidates,
)
from submission.experiments.phase3c_qasper_validation import (
    choose,compile_gold_groups,fast_group_coverage,
)
from submission.experiments.phase2_qasper_cluster_posthoc import paper_cluster_ci
from tokenpack.embeddings import make_embedder
from tokenpack.tokenization import TokenCounter

BASE=ROOT/"submission/results/phase3c_qasper_full889_optimized_20261011/paired_contexts.csv"
BACKBONE="sentence-transformers/all-MiniLM-L6-v2"
RATIOS=(.25,.50)
METHODS=("manual-render","lambdamart-render","crossencoder-render")


def training_pairs(questions,*,seed:int,negative_per_question:int=4):
    """Prioritize hard, gold-negative within-paper candidates. TRAIN only."""
    random_state=random.Random(seed)
    result=[]
    audit=Counter()
    for q in questions:
        relevances=np.asarray(q.labels)
        positive=[i for i,val in enumerate(relevances) if val>0]
        negative=[i for i,val in enumerate(relevances) if val<=0]
        if not positive or not negative:
            audit["questions_without_contrast"]+=1
            continue
        positive.sort(key=lambda i:(-int(relevances[i]),-q.scored[i].value,i))
        hard_negative=sorted(negative,key=lambda i:(-q.scored[i].value,i))
        chosen_positives=positive[:min(2,len(positive))]
        chosen_negative=hard_negative[:negative_per_question]
        rest=[i for i in negative if i not in set(chosen_negative)]
        extra=random_state.sample(rest,k=min(1,len(rest)))
        chosen_negative+=extra
        for index in chosen_positives:
            result.append((q.text,q.scored[index].chunk.text,1.0))
        for index in chosen_negative:
            result.append((q.text,q.scored[index].chunk.text,0.0))
        audit["questions_with_contrast"]+=1
        audit["positives"]+=len(chosen_positives)
        audit["negatives"]+=len(chosen_negative)
    random_state.shuffle(result)
    return result,dict(audit)


def tokenize_pairs(tokenizer,questions,contexts,max_length):
    return tokenizer(
        questions,contexts,truncation="only_second",max_length=max_length,
        padding="max_length",return_tensors="pt")


def fine_tune(train,output_dir,*,epochs,batch_size,learning_rate,
              max_length,seed,device,max_pairs):
    torch.manual_seed(seed)
    if device=="cuda":torch.cuda.manual_seed_all(seed)
    tokenizer=AutoTokenizer.from_pretrained(BACKBONE,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        BACKBONE,local_files_only=True,num_labels=1)
    model.to(device)
    pairs,audit=training_pairs(train,seed=seed)
    if max_pairs and len(pairs)>max_pairs:
        pairs=pairs[:max_pairs]
    if len(pairs)<20:
        raise ValueError("Too few labeled positive/negative pairs")
    data=tokenize_pairs(
        tokenizer,[x[0] for x in pairs],[x[1] for x in pairs],max_length)
    ds=TensorDataset(data["input_ids"],data["attention_mask"],
                     data["token_type_ids"],torch.tensor(
                         [x[2] for x in pairs],dtype=torch.float32))
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.01)
    amp=device=="cuda"
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    history=[]
    for epoch in range(epochs):
        model.train()
        losses=[]
        for k,(ids,mask,type_ids,y) in enumerate(loader):
            ids,mask,type_ids,y=(
                ids.to(device),mask.to(device),type_ids.to(device),y.to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=amp):
                logits=model(input_ids=ids,attention_mask=mask,
                             token_type_ids=type_ids).logits.squeeze(-1)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if (k+1)%50==0:
                print(f"GPU TRAIN epoch {epoch+1}/{epochs}, "
                      f"step {k+1}/{len(loader)}, loss={np.mean(losses[-50:]):.4f}",
                      flush=True)
        history.append({
            "epoch":epoch+1,"mean_train_bce_loss":round(float(np.mean(losses)),6),
            "num_batches":len(loader)})
        print("FINISHED_TRAIN_EPOCH",history[-1],flush=True)
    output_dir.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(output_dir/"model")
    tokenizer.save_pretrained(output_dir/"model")
    checkpoint={
        "model":BACKBONE,
        "device":str(device),"max_length":max_length,
        "training_pairs":len(pairs),
        "pair_audit":audit,"history":history,
        "trained_only_on_qasper_train":True,"internet_used":False,
    }
    (output_dir/"TRAIN_CHECKPOINT.json").write_text(
        json.dumps(checkpoint,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("MODEL_CHECKPOINT",str(output_dir/"model"),flush=True)
    return model,tokenizer,checkpoint


@torch.inference_mode()
def cross_scores(model,tokenizer,q,max_length,device,batch_size):
    model.eval()
    texts=[x.chunk.text for x in q.scored]
    scores=[]
    for start in range(0,len(texts),batch_size):
        batch=texts[start:start+batch_size]
        data=tokenizer(
            [q.text]*len(batch),batch,max_length=max_length,
            truncation="only_second",padding=True,return_tensors="pt")
        data={k:v.to(device) for k,v in data.items()}
        logits=model(**data).logits.squeeze(-1)
        scores.extend(logits.float().cpu().tolist())
    if len(scores)!=len(q.scored):
        raise RuntimeError("Score count mismatch")
    return scores


def old_baselines():
    old={}
    with BASE.open(encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["method"] in ("manual-render-value","lightgbm-render-value"):
                key=(row["case_id"],float(row["budget_ratio"]),row["method"])
                old[key]=row
    return old


def results_stats(records,*,bootstrap_replicates):
    result=[]
    for ratio in RATIOS:
        grouped=defaultdict(dict)
        for r in records:
            if r["ratio"]==ratio:
                grouped[r["case_id"]][r["method"]]=r
        if not grouped or any(set(x)!=set(METHODS) for x in grouped.values()):
            raise AssertionError("Incomplete paired cross-encoder evaluation")
        arms={}
        for name in METHODS:
            values=[x[name] for x in grouped.values()]
            arms[name]={
                "n":len(values),
                "gold_5gram_mean":round(float(np.mean(
                    [v["gold_5gram_recall"] for v in values])),6),
                "mean_realized_tokens":round(float(np.mean(
                    [v["actual_tokens"] for v in values])),2),
                "budget_violations":sum(
                    v["actual_tokens"]>v["budget"] for v in values),
            }
        diffs={}
        for new,old in (("crossencoder-render","lambdamart-render"),
                        ("crossencoder-render","manual-render")):
            pair={
                cid:{
                    "control":{"paper_id":v[old]["paper_id"],"delta":0.},
                    "new":{"paper_id":v[new]["paper_id"],
                           "delta":v[new]["gold_5gram_recall"]-
                                   v[old]["gold_5gram_recall"]},
                } for cid,v in grouped.items()
            }
            changes=[v["new"]["delta"] for v in pair.values()]
            diffs[f"{new} minus {old}"]={
                "mean_difference":round(float(np.mean(changes)),6),
                "wins":sum(x>1e-9 for x in changes),
                "losses":sum(x< -1e-9 for x in changes),
                "ties":sum(abs(x)<=1e-9 for x in changes),
                "bootstrap":paper_cluster_ci(
                    pair,"delta","new",replicates=bootstrap_replicates),
            }
        result.append({"ratio":ratio,"n":len(grouped),"arms":arms,
                       "contrasts":diffs})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--train-papers",type=int,default=80)
    p.add_argument("--validation-papers",type=int,default=30)
    p.add_argument("--epochs",type=int,default=1)
    p.add_argument("--train-batch",type=int,default=4)
    p.add_argument("--inference-batch",type=int,default=12)
    p.add_argument("--max-train-pairs",type=int,default=2400)
    p.add_argument("--max-length",type=int,default=320)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--seed",type=int,default=20261011)
    args=p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Will not overwrite any previous run")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU but CUDA is unavailable")
    started=time.perf_counter()
    split,papers=parse_input(DEFAULT_DATA)
    if papers!={"train":888,"validation":281}:
        raise ValueError("Pinned native QASPER split mismatch")
    tokenizer_counter=TokenCounter()
    embedder=make_embedder(
        BACKBONE,local_files_only=True)
    training,train_meta=prepare_split_batched(
        "train",split["train"],embedder,tokenizer_counter,
        DEFAULT_CACHE,args.train_papers)
    model,tokenizer,checkpoint=fine_tune(
        training,args.output_dir,epochs=args.epochs,batch_size=args.train_batch,
        learning_rate=args.learning_rate,max_length=args.max_length,
        seed=args.seed,device=args.device,max_pairs=args.max_train_pairs)
    train_papers={q.paper_id for q in training}
    training=[]
    validation,val_meta=prepare_split_batched(
        "validation",split["validation"],embedder,tokenizer_counter,
        DEFAULT_CACHE,args.validation_papers)
    if train_papers.intersection(q.paper_id for q in validation):
        raise AssertionError("Train/validation paper leakage")
    prior=old_baselines()
    rows=[]
    selected=[]
    for k,q in enumerate(validation,1):
        logits=cross_scores(model,tokenizer,q,args.max_length,
                            args.device,args.inference_batch)
        learned=make_learned_candidates(q.scored,logits)
        plans=[]
        for ratio in RATIOS:
            budget=max(1,math.floor(q.source_tokens*ratio))
            picked=choose(learned,query=q.text,budget=budget,
                          counter=tokenizer_counter,policy="render-value")
            plans.append((ratio,budget,picked))
        # gold only AFTER selection of all output contexts:
        gold=compile_gold_groups(q.evidence_groups)
        for ratio,budget,picked in plans:
            _,five,_=fast_group_coverage(picked["context"],gold)
            rows.append({
                "case_id":q.case_id,"paper_id":q.paper_id,
                "ratio":ratio,"method":"crossencoder-render",
                "budget":budget,"actual_tokens":picked["used_tokens"],
                "gold_5gram_recall":round(five,8),
            })
            selected.append({
                "case_id":q.case_id,"ratio":ratio,
                "selected_ids":[x.chunk.id for x in picked["selected"]],
            })
            for method,historical_name in (
                ("manual-render","manual-render-value"),
                ("lambdamart-render","lightgbm-render-value"),
            ):
                old=prior[(q.case_id,ratio,historical_name)]
                rows.append({
                    "case_id":q.case_id,"paper_id":q.paper_id,
                    "ratio":ratio,"method":method,
                    "budget":budget,
                    "actual_tokens":int(old["rendered_tokens"]),
                    "gold_5gram_recall":float(old["gold_5gram_recall"]),
                })
        if k%15==0 or k==len(validation):
            print(f"CROSSENCODER VALIDATED {k}/{len(validation)} "
                  f"questions, elapsed={time.perf_counter()-started:.1f}s",
                  flush=True)
    if any(r["actual_tokens"]>r["budget"] for r in rows):
        raise AssertionError("Rendered token budget breach")
    outcome={
        "experiment":"Phase5A GPU cross-encoder pilot against frozen Phase3B/3C",
        "training_papers":len(train_papers),
        "training_questions":train_meta["questions_evaluated"],
        "validation_papers":len({q.paper_id for q in validation}),
        "validation_questions":len(validation),
        "train_pair_checkpoint":checkpoint,
        "max_length":args.max_length,
        "max_train_pairs":args.max_train_pairs,
        "same_650_chunk_pool_and_render_aware_selector":True,
        "gold_labels_train_only":True,
        "test_split_used":False,
        "new_model_downloaded":False,
        "validation_was_previously_inspected":True,
        "limitations":[
            "Pilot, not independent validation: same QASPER development split seen before.",
            "MiniLM weights pretrained as bi-encoder; new classifier head fine-tuned only on sampled QASPER question-chunk pairs.",
            "Input length truncates second sequence at max_length, possibly missing evidence near end of 650-token chunk.",
            "Evidence lexical overlap used as supervision, not human answer correctness.",
            "Comparisons are identical maximum output budgets, not identical realized token counts.",
            "No QASPER held-out test, paid API, generative LLM, network/model download.",
        ],
        "results":results_stats(rows,bootstrap_replicates=1500),
        "elapsed_seconds":round(time.perf_counter()-started,2),
    }
    (args.output_dir/"summary.json").write_text(
        json.dumps(outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with (args.output_dir/"paired_contexts.csv").open(
            "w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with (args.output_dir/"selected_chunks.jsonl").open(
            "w",encoding="utf8") as f:
        for row in selected:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("PILOT_RESULT "+json.dumps(
        {"output":str(args.output_dir),"results":outcome["results"],
         "duration":outcome["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
