"""Rank original HotpotQA sentences locally and freeze Qwen candidate tasks.

Never loads gold annotations. Same unmodified Faz8 pretrained MiniLM.
One Qwen semantic judgement later per task. Rendered budget source cap.
"""
from __future__ import annotations
import json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false";os.environ["OMP_NUM_THREADS"]="2"
import pyarrow.parquet as pq
from submission.experiments.phase28_validation_register import OUT
from submission.experiments.phase6_hotpot_eval import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase8_atomic_contexts import sentence_candidates,encode_scores
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from submission.experiments.phase28_semantic_core import make_task
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
SCORED=OUT/"scored_original_candidates.jsonl"
TASKS=OUT/"semantic_citation_tasks.jsonl"
FROZEN=OUT/"CANDIDATES_FROZEN.json"
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    if any(p.exists() for p in (SCORED,TASKS,FROZEN)):raise FileExistsError("Immutable")
    proto=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(DATA)==proto["data_sha256"]
    table=pq.ParquetFile(DATA).read(columns=[
        "id","question","context.title","context.sentences"])
    assert set(table.column_names)=={"id","question","context"}
    assert not any(x in str(table.schema.remove_metadata())
       for x in ("supporting_facts","answer","type"))
    ids=set(proto["pilot_ids"])
    data={r["id"]:r for r in table.to_pylist() if r["id"] in ids}
    assert set(data)==ids
    verify_local_model()
    tk=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    token=TokenCounter();assert token._encoding is not None
    totals={"n":0,"original_candidate_sentences":0,"qwen_candidate_sentences":0,
            "mean_source_cap":0,"max_qwen_candidates":0}
    t0=time.monotonic()
    with SCORED.open("x",encoding="utf8") as f, TASKS.open("x",encoding="utf8") as taskf:
      for j,ident in enumerate(proto["pilot_ids"],1):
        row=data[ident]
        articles,atoms=sentence_candidates(row)
        total=sum(token.count(a["text"]) for a in articles)
        cap=max(1,math.floor(total*.25))
        scores=encode_scores(model,tk,row["question"],
             [a["input_text"] for a in atoms],max_length=SCORER_MAX_INPUT_TOKENS)
        scores=[float(s) for s in scores]
        case={"id":ident,"question":row["question"],
          "source_token_total":total,"cap":cap,
          "atoms":[{"article":a["article"],"title":a["title"],"sent_id":a["sent_id"],
                    "sentence":a["sentence"]} for a in atoms],
          "scores":scores,"gold_unopened":True}
        task=make_task(ident,row["question"],atoms,scores,cap,total)
        assert len(atoms)==len(scores) and 2<=len(task["candidates"])<=36
        f.write(json.dumps(case,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
        taskf.write(json.dumps(task,ensure_ascii=False)+"\n")
        taskf.flush();os.fsync(taskf.fileno())
        totals["n"]+=1
        totals["original_candidate_sentences"]+=len(atoms)
        totals["qwen_candidate_sentences"]+=len(task["candidates"])
        totals["mean_source_cap"]+=cap/96
        totals["max_qwen_candidates"]=max(totals["max_qwen_candidates"],len(task["candidates"]))
        if j%12==0:
          print("PHASE28_LOCAL_CANDIDATES",j,96,"sec",round(time.monotonic()-t0,1),flush=True)
    new_json(FROZEN,{"n":96,
      "source_sha256":sha_file(SCORED),"task_sha256":sha_file(TASKS),
      "pre_register_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
      "candidate_code_sha256":sha_file(Path(__file__)),
      "original_gold_annotations_unopened":True,"stats":totals})
    print("PHASE28_CANDIDATES_FROZEN",json.dumps(totals),flush=True)
if __name__=="__main__":main()
