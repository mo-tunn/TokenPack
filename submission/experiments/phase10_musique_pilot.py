"""Phase 10 MuSiQue: immutable ID preregistration, gold-blind 32 pilot / 96 validation contexts.

Do not score from gold or call paid endpoints here. Do not touch prior experiment output.
Usage: python -m submission.experiments.phase10_musique_pilot --register
       python -m submission.experiments.phase10_musique_pilot --build pilot
       python -m submission.experiments.phase10_musique_pilot --build validation
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
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
from submission.experiments.phase9_musique_protocol import (
    DATA_SHA256,DATA_REV,EXPECTED_DATASET_ROWS,MAX_SCORE_INPUT_TOKENS,RANK_BATCH_SIZE,
    DOC_DIVERSITY_DISCOUNT,LINKED_DOC_BONUS)
from submission.experiments.phase9_musique_contexts import DATA,PROV,candidates,inference
from submission.experiments.phase9_crossdoc_core import (
    choose,render_atomic,rank_rescale,bridge_link)
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter

OUT=ROOT/"submission/results/phase10_musique_neighbor_pilot_20261012"
PREV=ROOT/"submission/results/phase9_musique_bridge_new256_20261012"
SEED="tokenpack-phase10-musique-20261012-v1"
SIZES={"pilot":32,"validation":96}
ANCHOR_FRACTION=0.74
BUDGET_RATIO=0.25
METHODS=("atomic-rank","crossdoc-bridge","bridge-neighbor")


def sha_file(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def atomic_write_new(path,data):
    with path.open("x",encoding="utf8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def source_rows():
    info=json.loads(PROV.read_text(encoding="utf8"))
    assert info["rev"]==DATA_REV and info["sha256"]==DATA_SHA256
    assert sha_file(DATA)==DATA_SHA256, "MuSiQue parquet checksum mismatch"
    pf=pq.ParquetFile(DATA)
    assert pf.metadata.num_rows==EXPECTED_DATASET_ROWS
    projected=pf.read(columns=[
        "id","question","paragraphs.list.element.idx",
        "paragraphs.list.element.title",
        "paragraphs.list.element.paragraph_text"])
    assert set(projected.column_names)=={"id","question","paragraphs"}
    assert "is_supporting" not in repr(projected.schema) and "answer:" not in repr(projected.schema)
    rows=projected.to_pylist()
    ids=[x["id"] for x in rows]
    assert len(ids)==len(set(ids))==EXPECTED_DATASET_ROWS
    return rows


def register(rows):
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Phase10 output already exists; verify, do not overwrite")
    frozen=json.loads((PREV/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    prev_file=PREV/"gold_blind_contexts.jsonl"
    assert sha_file(prev_file)==frozen["case_file_sha256"]
    prior_ids={json.loads(line)["id"] for line in prev_file.open(encoding="utf8") if line.strip()}
    assert len(prior_ids)==256
    ids={r["id"] for r in rows}
    assert prior_ids<ids
    order=sorted(ids-prior_ids,key=lambda x:(hashlib.sha256((SEED+"|"+x).encode()).hexdigest(),x))
    samples={"pilot":order[:32],"validation":order[32:128]}
    assert len(set(samples["pilot"])|set(samples["validation"]))==128
    script_sha=sha_file(Path(__file__))
    from submission.experiments import phase9_crossdoc_core,phase9_musique_contexts,phase9_musique_protocol
    p10={
        "study":"Phase10 neighbor extension, new disjoint MuSiQue public validation IDs",
        "source_data_sha256":DATA_SHA256,"source_data_revision":DATA_REV,
        "prior_phase9_context_sha256":frozen["case_file_sha256"],
        "previous_phase9_n":len(prior_ids),
        "sampling_seed":SEED,"sampling_rule":"SHA256(seed+'|'+ID) on Phase9 complement only",
        "samples":samples,
        "sample_sha256":{k:hashlib.sha256("\n".join(v).encode()).hexdigest() for k,v in samples.items()},
        "token_budget_ratio":BUDGET_RATIO,"anchor_fraction":ANCHOR_FRACTION,
        "neighbor_expansion":"bridge anchor policy at floor(0.74*full budget); neighbors +/-1 in same source paragraph in descending anchor MS MARCO rank, tie earlier sentence first; then gold-blind bridge utility greedy until exact final cap",
        "methods":METHODS,"model_revision":REVISION,"model_weights_sha256":WEIGHTS_SHA256,
        "scoring_max_tokens":MAX_SCORE_INPUT_TOKENS,
        "code_sha256":{
            "phase10":script_sha,
            "phase9_crossdoc_core":sha_file(Path(phase9_crossdoc_core.__file__)),
            "phase9_musique_contexts":sha_file(Path(phase9_musique_contexts.__file__)),
            "phase9_musique_protocol":sha_file(Path(phase9_musique_protocol.__file__))
        },
        "pilot_reader_arms":["crossdoc-bridge","bridge-neighbor"],
        "pilot_n":32,"validation_n":96,
        "gate":"pilot neighbor minus bridge >= +3 net correct EM out of 32 AND 0 budget violations",
        "primary_validation_metric":"paired reader normalized EM neighbor minus crossdoc bridge",
        "preregistered_before_gold":True,"inference_gold_fields_read":False,
        "validation_will_not_be_run_if_pilot_fails":True,
        "qasper_official_test_reused":False,"production_changed":False,
        "benchmark_is_public_validation_not_hidden_test":True,
    }
    OUT.mkdir(parents=True,exist_ok=False)
    atomic_write_new(OUT/"PRE_GOLD_PROTOCOL.json",p10)
    print("PHASE10_REGISTERED",json.dumps({"pilot":32,"validation":96,
        "phase9_disjoint":True,"script_sha256":script_sha}),flush=True)


def expand_neighbors(atoms,scores,budget,counter):
    """Fixed gold-blind reranker anchors, local +/-1 context, linked-utility fill."""
    if budget<2: return choose(atoms,scores,budget,counter,graph=True)
    norm=rank_rescale(scores)
    anchor_budget=max(1,math.floor(ANCHOR_FRACTION*budget))
    anchors=choose(atoms,scores,anchor_budget,counter,graph=True)
    selected=list(anchors["selected_sentence_indexes"])
    seen=set(selected)
    bydoc={}
    for i in selected:
        d=atoms[i]["article"]
        bydoc[d]=bydoc.get(d,0)+1
    sent_lookup={(x["article"],x["sent_id"]):i for i,x in enumerate(atoms)}
    attempts=0
    added_neighbors=0
    def fit(i):
        nonlocal attempts
        if i in seen:return False
        attempts+=1
        if counter.count(render_atomic([*selected,i],atoms))>budget:return False
        selected.append(i);seen.add(i)
        d=atoms[i]["article"]; bydoc[d]=bydoc.get(d,0)+1
        return True
    anchor_order=sorted(selected,key=lambda i:(-norm[i],atoms[i]["article"],atoms[i]["sent_id"],i))
    for anchor in anchor_order:
        for off in (-1,1):
            j=sent_lookup.get((atoms[anchor]["article"],atoms[anchor]["sent_id"]+off))
            if j is not None and fit(j):
                added_neighbors+=1
    remaining=set(range(len(atoms)))-seen
    while remaining:
        def util(i):
            n=bydoc.get(atoms[i]["article"],0)
            edge=max((bridge_link(atoms[i],atoms[j]) for j in selected
                      if atoms[i]["article"]!=atoms[j]["article"]),default=0.0)
            return norm[i]+LINKED_DOC_BONUS*edge-DOC_DIVERSITY_DISCOUNT*min(3,n)
        queue=sorted(remaining,key=lambda i:(-util(i),atoms[i]["article"],atoms[i]["sent_id"],i))
        added=None
        for i in queue:
            if fit(i):
                added=i;break
        if added is None:break
        remaining.remove(added)
    text=render_atomic(selected,atoms)
    used=counter.count(text)
    assert used<=budget
    return {
        "text":text,"tokens":used,"budget":budget,
        "selected_sentence_indexes":sorted(selected),
        "selected_source_idxs":sorted({atoms[i]["original_paragraph_idx"] for i in selected}),
        "selected_paragraph_count":len(bydoc),
        "selected_sentence_count":len(selected),
        "anchor_sentence_count":anchors["selected_sentence_count"],
        "neighbor_sentences_added":added_neighbors,
        "render_checks":attempts,
    }


def build(which,rows):
    record=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert record["code_sha256"]["phase10"]==sha_file(Path(__file__)),"Frozen implementation changed"
    ids=record["samples"][which]
    assert len(ids)==SIZES[which]
    output=OUT/(which+"_gold_blind_contexts.jsonl")
    freeze=OUT/(which+"_CONTEXTS_FROZEN.json")
    if output.exists() or freeze.exists():
        raise FileExistsError("Already started; never overwrite partial/frozen output")
    chosen={x["id"]:x for x in rows if x["id"] in set(ids)}
    assert set(chosen)==set(ids)
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    counter=TokenCounter()
    assert counter._encoding is not None
    assert torch.cuda.is_available(),"GPU required for local pretrained reranker"
    verify_local_model()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(FOLDER,local_files_only=True).to("cuda").eval()
    start=time.perf_counter()
    stats={"cases":0,"budget_violations":0,"neighbor_added":0,"all_distinct_contexts":0}
    with output.open("x",encoding="utf8") as f:
        for n,id in enumerate(ids,1):
            row=chosen[id]
            atom,source=candidates(row)
            assert atom
            raw=inference(model,tok,row["question"],[a["input_text"] for a in atom])
            source_sum=sum(counter.count(p["title"]+"\n"+p["paragraph_text"])
                for p in row["paragraphs"] if p["paragraph_text"].strip())
            budget=max(1,math.floor(BUDGET_RATIO*source_sum))
            contexts={
                "atomic-rank":choose(atom,raw,budget,counter,graph=False),
                "crossdoc-bridge":choose(atom,raw,budget,counter,graph=True),
                "bridge-neighbor":expand_neighbors(atom,raw,budget,counter),
            }
            for c in contexts.values():
                assert c["tokens"]==counter.count(c["text"]) and c["tokens"]<=budget
            record={"id":id,"question":row["question"],"source_context":source,
                "source_token_sum":source_sum,"budget":budget,
                "candidate_sentences":len(atom),"contexts":contexts}
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
            f.flush();os.fsync(f.fileno())
            stats["cases"]+=1
            stats["neighbor_added"]+=contexts["bridge-neighbor"]["neighbor_sentences_added"]
            stats["all_distinct_contexts"]+=int(len({x["text"] for x in contexts.values()})==3)
            if n%8==0 or n==len(ids):
                print("PHASE10_CONTEXT_PROGRESS",which,n,len(ids),
                    "elapsed_s",round(time.perf_counter()-start,1),flush=True)
    frozen={"which":which,"context_file_sha256":sha_file(output),
        "manifest_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "source_data_sha256":DATA_SHA256,"count":len(ids),"stats":stats,
        "wall_s":round(time.perf_counter()-start,2),
        "gold_fields_read":False,"production_changed":False}
    atomic_write_new(freeze,frozen)
    print("PHASE10_FROZEN",json.dumps(frozen),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--register",action="store_true")
    group.add_argument("--build",choices=["pilot","validation"])
    args=p.parse_args()
    if args.register:register(source_rows())
    else:build(args.build,source_rows())
