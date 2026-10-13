"""Gold-blind Phase29 MuSiQue candidate scoring; no answer/support labels loaded."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"; os.environ["OMP_NUM_THREADS"]="2"
import pyarrow.parquet as pq
from submission.experiments.phase9_musique_protocol import DATA_SHA256, DATA_REV
from submission.experiments.phase5b_pretrained_rerank_eval import FOLDER, verify_local_model
from submission.experiments.phase8_atomic_contexts import encode_scores
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from submission.experiments.phase28_semantic_core import make_task
from submission.experiments.phase12_register import sha_file
from tokenpack.tokenization import TokenCounter

OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
DATA=ROOT/"submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"
SCORED_NAME="scored_original_candidates.jsonl"
TASKS_NAME="semantic_citation_tasks.jsonl"
FROZEN_NAME="CANDIDATES_FROZEN.json"

def read_rows(path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]

def split_sentences(text):
    # Same deterministic segmentation used by Phase9 MuSiQue; no gold input.
    import re
    parts=re.split(r"(?<=[.!?])\s+(?=[\"'\[(]?[A-Z0-9])",str(text))
    return [x.strip() for x in parts if x.strip()]

def source_atoms(record):
    atoms=[]; paragraphs=[]
    for article,para in enumerate(record["paragraphs"]):
        if set(para)!={"idx","title","paragraph_text"}:
            raise AssertionError("Unexpected field in gold-blind paragraph projection.")
        text=str(para["paragraph_text"]).strip()
        title=str(para["title"])
        if not text: continue
        paragraphs.append({"article":article,"idx":int(para["idx"]),
                           "title":title,"text":text})
        for sid,sentence in enumerate(split_sentences(text)):
            atoms.append({"article":article,"original_paragraph_idx":int(para["idx"]),
                          "title":title,"sent_id":sid,"sentence":sentence,
                          "input_text":title+"\n"+sentence})
    return paragraphs,atoms

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",choices=["pilot","validation"],required=True)
    args=ap.parse_args()
    sub=OUT/args.split
    sub.mkdir(parents=True,exist_ok=True)
    names=[sub/SCORED_NAME,sub/TASKS_NAME,sub/FROZEN_NAME]
    if any(x.exists() for x in names): raise FileExistsError("Immutable candidate artifacts already exist.")
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(DATA)==p["data_sha256"]==DATA_SHA256
    assert p["dataset_revision"]==DATA_REV
    ids=p["pilot_ids"] if args.split=="pilot" else p["reserved_ids"]
    columns=["id","question","paragraphs.list.element.idx",
             "paragraphs.list.element.title","paragraphs.list.element.paragraph_text"]
    table=pq.ParquetFile(DATA).read(columns=columns)
    assert set(table.column_names)=={"id","question","paragraphs"}
    schema=str(table.schema.remove_metadata())
    assert not any(k in schema for k in ("is_supporting","answer_aliases","question_decomposition","answerable","answer:"))
    allowed=set(ids)
    rows={r["id"]:r for r in table.to_pylist() if r["id"] in allowed}
    assert set(rows)==allowed and len(rows)==len(ids)

    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    verify_local_model()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(FOLDER,local_files_only=True).to("cuda").eval()
    counter=TokenCounter()
    assert counter._encoding is not None
    stats={"n":len(ids),"source_sentences":0,"candidate_sentences_sent_to_verifier":0,
           "mean_cap":0.0,"mean_source_tokens":0.0,"token_cap_violations":0}
    scored_path=sub/SCORED_NAME; tasks_path=sub/TASKS_NAME
    start=time.monotonic()
    with scored_path.open("x",encoding="utf8") as sf, tasks_path.open("x",encoding="utf8") as tf:
        for j,ident in enumerate(ids,1):
            row=rows[ident]
            paragraphs,atoms=source_atoms(row)
            assert atoms and len(atoms)>=2
            total=sum(counter.count(x["title"]+"\n"+x["text"]) for x in paragraphs)
            cap=max(1,math.floor(total*.25))
            scores=[float(x) for x in encode_scores(
                model,tok,row["question"],[a["input_text"] for a in atoms],
                max_length=SCORER_MAX_INPUT_TOKENS)]
            assert len(scores)==len(atoms) and all(math.isfinite(x) for x in scores)
            case={"id":ident,"question":row["question"],"source_token_total":total,
                  "cap":cap,
                  "paragraphs":[{"article":x["article"],"idx":x["idx"],"title":x["title"]}
                                for x in paragraphs],
                  "atoms":[{"article":a["article"],"original_paragraph_idx":a["original_paragraph_idx"],
                            "title":a["title"],"sent_id":a["sent_id"],"sentence":a["sentence"]}
                           for a in atoms],
                  "scores":scores,"gold_unopened":True}
            task=make_task(ident,row["question"],atoms,scores,cap,total)
            assert 2<=len(task["candidates"])<=36
            sf.write(json.dumps(case,ensure_ascii=False)+"\n"); sf.flush(); os.fsync(sf.fileno())
            tf.write(json.dumps(task,ensure_ascii=False)+"\n"); tf.flush(); os.fsync(tf.fileno())
            stats["source_sentences"]+=len(atoms)
            stats["candidate_sentences_sent_to_verifier"]+=len(task["candidates"])
            stats["mean_cap"]+=cap/len(ids)
            stats["mean_source_tokens"]+=total/len(ids)
            stats["token_cap_violations"]+=int(cap<1)
            if j%12==0 or j==len(ids):
                print("PHASE29_CANDIDATES_PROGRESS",args.split,j,len(ids),
                      "elapsed_s",round(time.monotonic()-start,1),flush=True)
    freeze={"split":args.split,"n":len(ids),
            "source_sha256":sha_file(scored_path),"task_sha256":sha_file(tasks_path),
            "prereg_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
            "candidate_code_sha256":sha_file(Path(__file__)),
            "source_projection_gold_unopened":True,"stats":stats}
    with (sub/FROZEN_NAME).open("x",encoding="utf8") as f:
        json.dump(freeze,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print("PHASE29_CANDIDATES_FROZEN",json.dumps(freeze["stats"]),flush=True)

if __name__=="__main__": main()
