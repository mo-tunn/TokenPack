"""Gold-blind, checkpointable Qwen reader generation on frozen Phase7 contexts.

Identical pretrained reader, system prompt, greedy decoding and max 32 output
tokens across all arms. Gold HotpotQA answer or supporting facts are NEVER
loaded. Generation JSONL is append-only; resumptions skip prior case+arm IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

from submission.experiments.phase7_answer_protocol import (
    ARMS,SAMPLE_COUNT,READER_ID,READER_REVISION,
    READER_MAX_NEW_TOKENS,READER_MAX_INPUT_TOKENS,
    READER_PROMPT,READER_DO_SAMPLE,
)
from submission.experiments.phase7_prepare_contexts import OUTPUT

FOLDER=(Path.home()/".cache/huggingface/hub"/
        "models--Qwen--Qwen2.5-0.5B-Instruct"/"snapshots"/READER_REVISION)


def make_chat(question,context,tokenizer):
    messages=[
        {"role":"system","content":READER_PROMPT},
        {"role":"user","content":f"Question: {question}\n\nContext:\n{context}"},
    ]
    tokens=tokenizer.apply_chat_template(
        messages,tokenize=True,add_generation_prompt=True)
    original_len=len(tokens)
    if original_len>READER_MAX_INPUT_TOKENS:
        # Preserve question, instructions and assistant generation header.
        # Only shorten the evidence tail (never gold-driven).
        text_tokens=tokenizer.encode(context,add_special_tokens=False)
        drop=original_len-READER_MAX_INPUT_TOKENS+32
        if drop>=len(text_tokens):
            raise ValueError("Question/prompt exceeds input limit")
        messages[1]["content"]=(
            f"Question: {question}\n\nContext:\n"+
            tokenizer.decode(text_tokens[:-drop],skip_special_tokens=True))
        tokens=tokenizer.apply_chat_template(
            messages,tokenize=True,add_generation_prompt=True)
        if len(tokens)>READER_MAX_INPUT_TOKENS:
            raise AssertionError("Reader input hard cap not enforced")
    return tokens,original_len


def generate(args):
    contexts=OUTPUT/"gold_blind_contexts.jsonl"
    completed=json.loads((OUTPUT/"CONTEXTS_COMPLETE.json").read_text(encoding="utf8"))
    if hashlib.sha256(contexts.read_bytes()).hexdigest()!=completed["contexts_sha256"]:
        raise AssertionError("Reader contexts changed since gold-blind freeze")
    rows=[json.loads(s) for s in contexts.read_text(encoding="utf8").splitlines()]
    if len(rows)!=SAMPLE_COUNT:raise AssertionError("Frozen sample is incomplete")
    model_file=FOLDER/"model.safetensors"
    if not model_file.is_file():raise FileNotFoundError(model_file)
    digest=hashlib.sha256(model_file.read_bytes()).hexdigest()
    tok=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(
        FOLDER,local_files_only=True,torch_dtype=torch.float16).to("cuda").eval()
    print("READER_READY",READER_ID,"pinned",READER_REVISION,
          "gpu",torch.cuda.get_device_name(0),"weight_sha256",digest,
          "memory_mb",round(torch.cuda.max_memory_allocated()/1048576),
          flush=True)
    output=OUTPUT/"reader_predictions.jsonl"
    done=set()
    if output.exists():
        for line in output.read_text(encoding="utf8").splitlines():
            if line:
                rec=json.loads(line)
                key=(rec["id"],rec["arm"])
                if key in done:raise AssertionError("Duplicate reader prediction")
                done.add(key)
    start=time.perf_counter()
    generated=0
    with output.open("a",encoding="utf8") as f:
        for n,row in enumerate(rows,1):
            if set(row["contexts"])!=set(ARMS):
                raise AssertionError("Frozen input arms missing")
            for arm in ARMS:
                if (row["id"],arm) in done:continue
                tokens,raw_len=make_chat(
                    row["question"],row["contexts"][arm]["text"],tok)
                input_ids=torch.tensor([tokens],dtype=torch.long,device="cuda")
                mask=torch.ones_like(input_ids)
                with torch.inference_mode():
                    output_ids=model.generate(
                        input_ids=input_ids,attention_mask=mask,
                        do_sample=READER_DO_SAMPLE,
                        max_new_tokens=READER_MAX_NEW_TOKENS,
                        pad_token_id=tok.eos_token_id,
                        use_cache=True,
                    )
                new_tokens=output_ids[0,input_ids.shape[1]:]
                result=tok.decode(new_tokens,skip_special_tokens=True).strip()
                rec={
                    "id":row["id"],"arm":arm,"predicted_answer":result,
                    "input_token_count":input_ids.shape[1],
                    "raw_input_token_count":raw_len,
                    "output_token_count":len(new_tokens),
                    "reader_model_revision":READER_REVISION,
                    "gold_seen":False,
                }
                f.write(json.dumps(rec,ensure_ascii=False)+"\n")
                f.flush()
                generated+=1
            if n%16==0 or n==len(rows):
                print("READER_PROGRESS",n,len(rows),
                      "predicted",len(done)+generated,
                      "elapsed",round(time.perf_counter()-start,1),
                      "gpu_memory_mb",round(torch.cuda.max_memory_allocated()/1048576),
                      flush=True)
            if args.max_cases and n>=args.max_cases:
                break
    all_lines=[json.loads(s) for s in output.read_text(encoding="utf8").splitlines()]
    if len(all_lines)==SAMPLE_COUNT*len(ARMS):
        result={
            "predictions":len(all_lines),"questions":SAMPLE_COUNT,
            "reader_model_id":READER_ID,"revision":READER_REVISION,
            "weight_sha256":digest,"greedy":True,
            "max_output_tokens":READER_MAX_NEW_TOKENS,
            "input_limit":READER_MAX_INPUT_TOKENS,
            "fixed_prompt":READER_PROMPT,
            "output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "gold_never_read_during_generation":True,
            "runtime_seconds_this_invocation":round(time.perf_counter()-start,2),
        }
        (OUTPUT/"READER_GENERATION_COMPLETE.json").write_text(
            json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        print("READER_GENERATION_COMPLETE",result["predictions"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--max-cases",type=int,default=0,
                    help="Optional gold-blind technical checkpoint; resumable")
    generate(ap.parse_args())
