"""Build *gold-blind* frozen reader contexts for 256 HotpotQA dev questions.

No gold answers/support annotations read. Reuses Phase6 pretrained models,
native input data, frozen chunking and render-aware selector, with identical
question IDs across 5 contexts. Saves answer-free contexts before inference.
"""
from __future__ import annotations

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
os.environ["TOKENPACK_HF_OFFLINE"]="1"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["TOKENIZERS_PARALLELISM"]="false"

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

from submission.experiments.phase7_answer_protocol import (
    SOURCE_QUESTION_IDS,SAMPLE_COUNT,SAMPLE_SALT,
    READER_PROMPT,BUDGET_RATIOS,ARMS,
)
from submission.experiments.phase6_hotpot_eval import (
    DATA,PROVENANCE,build_blocks,
)
from submission.experiments.phase6_hotpot_protocol import (
    CHUNK_CONFIG,SEMANTIC_THRESHOLD,
)
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,verify_local_model,score,
)
from submission.experiments.phase3b_learned_scoring import (
    make_learned_candidates,
)
from submission.experiments.phase3c_qasper_validation import choose
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache,make_embedder
from tokenpack.scoring import score_chunks
from tokenpack.tokenization import TokenCounter

OUTPUT=ROOT/"submission/results/phase7_hotpotqa_reader256_20261012"


def select_ids(ids,n=SAMPLE_COUNT):
    return sorted(ids,key=lambda k:(
        hashlib.sha256((SAMPLE_SALT+"|"+k).encode("utf8")).hexdigest(),k))[:n]


def prepare():
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Refuse to overwrite a previous Phase7 dataset")
    proven=json.loads(PROVENANCE.read_text(encoding="utf8"))
    if hashlib.sha256(DATA.read_bytes()).hexdigest()!=proven["sha256"]:
        raise AssertionError("HotpotQA validation parquet digest changed")
    allowed=(ROOT/SOURCE_QUESTION_IDS).read_text(encoding="utf8").splitlines()
    if len(allowed)!=512 or len(set(allowed))!=512:
        raise ValueError("Phase6 fixed ID list changed")
    chosen=select_ids(allowed)
    rows={x["id"]:x for x in pq.read_table(
        DATA,columns=["id","question","context"]).to_pylist()
        if x["id"] in set(chosen)}
    if len(rows)!=SAMPLE_COUNT:raise AssertionError("Some ID missing")
    counter=TokenCounter()
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    verify_local_model()
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    ranker=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    OUTPUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    manifest={
        "experiment":"phase7 reader contexts (gold blind)",
        "ids_selection":"pinned 512 Phase6 IDs, lowest salted id-only sha256",
        "num_questions":SAMPLE_COUNT,
        "sample_salt":SAMPLE_SALT,
        "ordered_ids_sha256":hashlib.sha256(
            "\n".join(chosen).encode("utf8")).hexdigest(),
        "methods":ARMS,
        "budgets":BUDGET_RATIOS,
        "gold_answers_or_evidence_accessed":False,
        "source_parquet_revision":proven["revision"],
        "source_sha256":proven["sha256"],
        "no_QASPER_test_used":True,
        "no_tuning_on_HotpotQA_gold":True,
    }
    (OUTPUT/"PREDECLARED_CONTEXTS.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    output=OUTPUT/"gold_blind_contexts.jsonl"
    count=0
    with output.open("w",encoding="utf8") as dest:
        for j,cid in enumerate(chosen,1):
            row=rows[cid]
            blocks,source=build_blocks(row,counter)
            src_tokens=sum(counter.count(b.text) for b in blocks)
            cache=EmbeddingCache(
                ROOT/".tokenpack/cache/phase6_hotpotqa_20261011"/
                (cid+".embeddings.json"))
            bvec=cache.get_or_embed([b.text for b in blocks],embedder)
            qvec=cache.get_or_embed([row["question"]],embedder)[0]
            ranks={}
            for scale in (650,300):
                chunks=StructureAwareChunker(
                    **CHUNK_CONFIG[scale],token_counter=counter,
                    block_embeddings=bvec,semantic_threshold=SEMANTIC_THRESHOLD).chunk(blocks)
                vecs=cache.get_or_embed([c.text for c in chunks],embedder)
                ranks[scale]=score_chunks(
                    qvec,chunks,vecs,scoring="evidence-hybrid",
                    query_text=row["question"])
            mscore=score(ranker,tokenizer,row["question"],ranks[300],
                         "cuda",512,12)
            cross=make_learned_candidates(ranks[300],mscore)
            contexts={"full-source":{"text":source,
                                     "tokens":counter.count(source),
                                     "budget":None}}
            for ratio,ratio_label in ((.25,"25"),(.5,"50")):
                budget=max(1,math.floor(src_tokens*ratio))
                for tag,scored in ((f"manual-{ratio_label}",ranks[650]),
                                   (f"msmarco-{ratio_label}",cross)):
                    out=choose(scored,query=row["question"],budget=budget,
                               counter=counter,policy="render-value")
                    if out["used_tokens"]>budget:
                        raise AssertionError("Real tokenizer budget breached")
                    contexts[tag]={"text":out["context"],
                                   "tokens":out["used_tokens"],
                                   "budget":budget}
            if set(contexts)!=set(ARMS):
                raise AssertionError("Missing experimental arm")
            dest.write(json.dumps({
                "id":cid,"question":row["question"],"source_tokens":src_tokens,
                "contexts":contexts,
            },ensure_ascii=False)+"\n")
            dest.flush()
            count+=1
            if j%32==0 or j==SAMPLE_COUNT:
                print("PHASE7 CONTEXTS",j,SAMPLE_COUNT,
                      "elapsed",round(time.perf_counter()-start,1),flush=True)
    (OUTPUT/"CONTEXTS_COMPLETE.json").write_text(
        json.dumps({**manifest,"written_cases":count,
                    "contexts_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                    "elapsed_seconds":round(time.perf_counter()-start,2)},
                   ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("CONTEXTS_FROZEN",output,flush=True)


if __name__=="__main__":
    prepare()
