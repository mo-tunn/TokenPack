"""Phase6 genuinely out-of-domain HotpotQA multi-hop supporting fact evaluation.

Fixed *before* looking at HotpotQA labels:
 - hotpotqa/hotpot_qa distractor/validation (not train/test),
 - lowest SHA256(question ID + pinned seed) 512 question IDs,
 - 650-token manual, 650-token frozen native QASPER LightGBM,
   300-token manual, 300-token MS MARCO pretrained zero-shot,
 - exact final-rendered token caps of 25% and 50% of provided context,
 - primary: all supporting sentences present, whitespace/case normalized,
 - secondary: per-fact recall, both distinct supporting titles present,
   extractive answer string present (exclude yes/no, not QA correctness).

This is evaluation in provided 10-paragraph "distractor" contexts, NOT
retrieval against full Wikipedia and NOT a generated answer accuracy test.
No model fit/tuning, no gold use until ALL arms have rendered their context.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENPACK_HF_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["OMP_NUM_THREADS"]="2"
os.environ["MKL_NUM_THREADS"]="2"
os.environ["OPENBLAS_NUM_THREADS"]="2"

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

from submission.experiments.phase6_hotpot_protocol import (
    N_CASES,SAMPLE_SEED,BUDGET_RATIOS,METHODS,CHUNK_CONFIG,
    BOOTSTRAP_REPLICATES,SEMANTIC_THRESHOLD,SELECTOR,
)
from submission.experiments.phase3b_learned_scoring import (
    features_for_question,make_learned_candidates,
)
from submission.experiments.phase3c_qasper_validation import choose
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,REVISION,MODEL_ID,WEIGHTS_SHA256,verify_local_model,score,
)
from tokenpack.chunking import StructureAwareChunker
from tokenpack.embeddings import make_embedder,EmbeddingCache
from tokenpack.models import TextBlock
from tokenpack.scoring import score_chunks
from tokenpack.tokenization import TokenCounter

DATA=ROOT/"submission/data/phase6_hotpotqa_validation_20261011/distractor/validation-00000-of-00001.parquet"
PROVENANCE=DATA.parent.parent/"PROVENANCE.json"
LGB_MODEL=ROOT/"submission/results/phase3b_qasper_full_train_validation_20261011/full_train_lambdamart.model.txt"


def hashed_sample(rows,n=N_CASES,seed=SAMPLE_SEED):
    # ID-only hash; answers and gold MUST NOT enter sample decision.
    unique={r["id"]:r for r in rows}
    if len(unique)!=len(rows):
        raise AssertionError("Duplicate original HotpotQA IDs")
    order=sorted(unique,key=lambda k:(
        hashlib.sha256((seed+"|"+k).encode("utf8")).hexdigest(),k))
    return [unique[k] for k in order[:min(n,len(order))]]


def norm_text(text):
    return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",str(text)).casefold()).strip()


def build_blocks(row,counter):
    """Convert only non-gold supplied context into source-provenant wiki blocks."""
    titles=row["context"]["title"]
    articles=row["context"]["sentences"]
    if len(titles)!=len(articles):
        raise ValueError("HotpotQA article/title mismatch")
    blocks=[]
    source_parts=[]
    for i,(title,sents) in enumerate(zip(titles,articles,strict=True)):
        clean=[str(sentence) for sentence in sents if str(sentence).strip()]
        if not clean:continue
        text=f"{title}\n"+" ".join(clean)
        # Each article has a distinct document_index, so no chunk
        # implicitly merges text from independent Wikipedia pages.
        blocks.append(TextBlock(
            text=text,source_path=f"hotpotqa/{row['id']}/article-{i}",
            document_index=i,paragraph_index=i,
            char_start=0,char_end=len(text),
            metadata={"content_type":"document",
                      "section_hint":str(title)},
        ))
        source_parts.append(text)
    source="\n\n".join(source_parts)
    return blocks,source


def measure_evidence(row,rendered):
    """Gold-only evaluation AFTER all method choices; never a scoring feature."""
    lookup={
        str(title):[str(s) for s in sentence_list]
        for title,sentence_list in zip(
            row["context"]["title"],row["context"]["sentences"],strict=True)
    }
    gold=list(zip(row["supporting_facts"]["title"],
                  row["supporting_facts"]["sent_id"],strict=True))
    gold=list(dict.fromkeys((str(t),int(i)) for t,i in gold))
    if not gold:
        raise AssertionError("No Hotpot supporting facts")
    context=norm_text(rendered)
    flags=[]
    sourceflags=[]
    per_title={}
    for title,i in gold:
        item=lookup.get(title,[])
        valid=0<=i<len(item) and bool(norm_text(item[i]))
        sentence=norm_text(item[i]) if valid else ""
        sourceflags.append(valid)
        present=bool(sentence and sentence in context)
        flags.append(present)
        per_title.setdefault(title,[]).append(present)
    answer=norm_text(row["answer"])
    answer_eligible=bool(answer) and answer not in ("yes","no")
    # This merely tests text presence, NOT whether the context permits
    # correct reasoning or whether a generative model would answer correctly.
    answer_present=bool(answer_eligible and answer in context)
    return {
        "gold_fact_count":len(gold),
        "source_support_complete":all(sourceflags),
        "source_support_fact_fraction":sum(sourceflags)/len(gold),
        "all_support_sentences":int(all(flags)),
        "support_sentence_recall":sum(flags)/len(gold),
        "all_support_titles":int(all(any(x) for x in per_title.values())),
        "support_title_count":len(per_title),
        "answer_string_eligible":answer_eligible,
        "answer_string_present":int(answer_present) if answer_eligible else None,
    }


def question_bootstrap(values,seed=20261011,n=BOOTSTRAP_REPLICATES):
    arr=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=np.asarray([
        arr[rng.integers(0,len(arr),size=len(arr))].mean()
        for _ in range(n)])
    return [round(float(x),6) for x in np.percentile(means,[2.5,97.5])]


def aggregate(records):
    result=[]
    for ratio in BUDGET_RATIOS:
        cases={}
        for r in records:
            if r["ratio"]==ratio:
                cases.setdefault(r["case_id"],{})[r["method"]]=r
        if any(set(m)!=set(METHODS) for m in cases.values()):
            raise AssertionError("Unpaired methods at fixed budget")
        arms={}
        for method in METHODS:
            subset=[data[method] for data in cases.values()]
            answer_values=[
                x["answer_string_present"] for x in subset
                if x["answer_string_eligible"]]
            arms[method]={
                "n":len(subset),
                "strict_all_support_rate":round(statistics.mean(
                    x["all_support_sentences"] for x in subset),6),
                "support_sentence_recall":round(statistics.mean(
                    x["support_sentence_recall"] for x in subset),6),
                "all_support_titles_rate":round(statistics.mean(
                    x["all_support_titles"] for x in subset),6),
                "extractive_answer_string_rate":round(
                    statistics.mean(answer_values),6) if answer_values else None,
                "eligible_extractive_answers":len(answer_values),
                "mean_realized_tokens":round(statistics.mean(
                    x["actual_tokens"] for x in subset),2),
                "mean_budget":round(statistics.mean(
                    x["budget"] for x in subset),2),
                "budget_violations":sum(
                    x["actual_tokens"]>x["budget"] for x in subset),
            }
        contrasts={}
        for newer,older in [
            ("msmarco-zero-shot-300","manual-650"),
            ("msmarco-zero-shot-300","lambdamart-650"),
            ("msmarco-zero-shot-300","manual-300"),
        ]:
            paired=[
                (case[newer],case[older])
                for case in cases.values()]
            def comparison(metric):
                diffs=[float(x[metric])-float(y[metric]) for x,y in paired]
                return {
                    "mean_difference":round(statistics.mean(diffs),6),
                    "question_bootstrap_ci95":question_bootstrap(diffs),
                    "wins":sum(x>0 for x in diffs),
                    "losses":sum(x<0 for x in diffs),
                    "ties":sum(x==0 for x in diffs),
                }
            contrasts[f"{newer} minus {older}"]={
                "strict_all_support":comparison("all_support_sentences"),
                "support_sentence_recall":comparison("support_sentence_recall"),
                "both_articles":comparison("all_support_titles"),
            }
        result.append({"ratio":ratio,"n":len(cases),
                       "arms":arms,"contrasts":contrasts})
    return result


def evaluate(args):
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Never overwrite other run")
    prov=json.loads(PROVENANCE.read_text(encoding="utf8"))
    hh=hashlib.sha256(DATA.read_bytes()).hexdigest()
    if hh!=prov["sha256"] or prov["revision"]!="1908d6afbbead072334abe2965f91bd2709910ab":
        raise ValueError("HotpotQA data hash/revision changed")
    verify_local_model()
    raw=pq.read_table(DATA).to_pylist()
    if len(raw)!=7405:raise AssertionError("Expected official distractor dev")
    selected=hashed_sample(raw,n=args.num_questions)
    if len(selected)!=args.num_questions:
        raise AssertionError("Missing requested fixed ID sample")
    if args.num_questions==N_CASES:
        sample_tag="pinned512"
    else:
        sample_tag="diagnostic_subset_not_confirmatory"
    counter=TokenCounter()
    if counter._encoding is None:
        raise RuntimeError("cl100k_base required")
    lgb_model=lgb.Booster(model_file=str(LGB_MODEL))
    tokenizer=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    crossencoder=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    embedder=make_embedder("sentence-transformers/all-MiniLM-L6-v2",
                          local_files_only=True)
    t0=time.perf_counter()
    records=[]
    audit={"questions":0,"blocks":0,"650_chunks":0,"300_chunks":0,
           "max_chunk_tokens":0,"source_gold_support_availability":0}
    for j,row in enumerate(selected,1):
        # Until choose(...) returns for every arm, access ONLY id, question,
        # context. Do not inspect answer, supporting_facts, level or type.
        blocks,source=build_blocks(row,counter)
        source_tokens=sum(counter.count(b.text) for b in blocks)
        if not source_tokens or not blocks:
            raise AssertionError("Unexpected empty supplied HotpotQA context")
        # Per case cache is isolated, opt-in research, cannot touch QASPER.
        cache=EmbeddingCache(
            ROOT/".tokenpack/cache/phase6_hotpotqa_20261011"/(row["id"]+".embeddings.json"))
        bvec=cache.get_or_embed([b.text for b in blocks],embedder)
        qvec=cache.get_or_embed([row["question"]],embedder)[0]
        scored_by_scale={}
        for scale in (650,300):
            chunks=StructureAwareChunker(
                **CHUNK_CONFIG[scale],token_counter=counter,
                block_embeddings=bvec,
                semantic_threshold=SEMANTIC_THRESHOLD).chunk(blocks)
            if not chunks:raise AssertionError("No evidence candidates")
            vecs=cache.get_or_embed([c.text for c in chunks],embedder)
            scored_by_scale[scale]=score_chunks(
                qvec,chunks,vecs,scoring="evidence-hybrid",
                query_text=row["question"])
            audit[f"{scale}_chunks"]+=len(chunks)
            audit["max_chunk_tokens"]=max(audit["max_chunk_tokens"],
                                           max(c.token_count for c in chunks))
        old=scored_by_scale[650]
        new=scored_by_scale[300]
        pred=lgb_model.predict(features_for_question(old,row["question"]),
                               num_threads=2)
        learned=make_learned_candidates(old,pred)
        rank=score(crossencoder,tokenizer,row["question"],new,
                   "cuda",512,args.batch_size)
        cross=make_learned_candidates(new,rank)
        arms={
            "manual-650":old,
            "lambdamart-650":learned,
            "manual-300":new,
            "msmarco-zero-shot-300":cross,
        }
        output=[]
        for ratio in BUDGET_RATIOS:
            budget=max(1,math.floor(source_tokens*ratio))
            for method in METHODS:
                choice=choose(
                    arms[method],query=row["question"],
                    budget=budget,counter=counter,policy=SELECTOR)
                if choice["used_tokens"]>budget:raise AssertionError("Token cap exceeded")
                output.append((ratio,budget,method,choice))
        # End of gold-blind inference. Only now inspect annotation labels.
        for ratio,budget,method,choice in output:
            metric=measure_evidence(row,choice["context"])
            record={
                "case_id":row["id"],"ratio":ratio,"method":method,
                "budget":budget,"actual_tokens":choice["used_tokens"],
                "source_tokens":source_tokens,"selected_chunks":len(choice["selected"]),
                **metric,
            }
            records.append(record)
            audit["source_gold_support_availability"]+=int(
                metric["source_support_complete"]) if (ratio,method)==(.25,"manual-650") else 0
        audit["questions"]+=1
        audit["blocks"]+=len(blocks)
        if j%40==0 or j==len(selected):
            print(f"PHASE6 {j}/{len(selected)} source questions "
                  f"elapsed={time.perf_counter()-t0:.1f}s",flush=True)
    summary={
        "experiment":"Phase6 pristine out-of-domain HotpotQA supporting-fact compression",
        "benchmark":"hotpotqa/hotpot_qa distractor validation",
        "benchmark_revision":prov["revision"],"benchmark_sha256":prov["sha256"],
        "sample_selection":"ID-only smallest SHA256 of pinned sample seed",
        "sample_seed":SAMPLE_SEED,"sample_size":len(selected),
        "sample_hash_sha256":hashlib.sha256(
            "\n".join(sorted(r["id"] for r in selected)).encode("utf8")).hexdigest(),
        "evaluation_status":sample_tag,
        "no_Hotpot_training_or_metric_tuning":True,
        "no_QASPER_test_reuse":True,
        "reference_retrieval_setting":"HotpotQA distractor supplied 10 candidate paragraphs; NOT fullwiki retrieval",
        "primary_metric":"all annotated support sentences present in final rendered text after NFKC/casefold/whitespace normalization",
        "support_sentence_alignment":"title and sent_id in supplied context",
        "answer_presence_not_answer_correctness":True,
        "model":"pinned MS MARCO MiniLM-L6-v2 zero-shot; QASPER full-train frozen LambdaMART comparator",
        "MS_MARCO_revision":REVISION,"MS_MARCO_SHA256":WEIGHTS_SHA256,
        "ratios":list(BUDGET_RATIOS),"methods":METHODS,
        "audit":audit,"runtime_seconds":round(time.perf_counter()-t0,2),
        "results":aggregate(records),
        "limitations":[
            "Public HotpotQA validation unseen by TokenPack prior experiments, but it is NOT an official hidden test set.",
            "Wikipedia multi-hop distractor contexts may have short passages and distinct token distribution compared with QASPER full papers.",
            "All-support strict is literal sentence inclusion; does not prove sufficient reasoning or generated answer accuracy.",
            "Extractive answer-string rate is only a weak textual proxy; yes/no answers excluded.",
            "All methods same true max rendered-token budget, realized tokens can differ.",
            "No tuning of thresholds/scorers based on HotpotQA gold; no HotpotQA training.",
            "Sentence-level support annotations may refer to source sentences that are missing or reformatted.",
        ],
    }
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    with (args.output_dir/"paired_contexts.csv").open("w",newline="",encoding="utf8") as f:
        cw=csv.DictWriter(f,fieldnames=list(records[0]))
        cw.writeheader();cw.writerows(records)
    (args.output_dir/"question_ids.txt").write_text(
        "\n".join(sorted(r["id"] for r in selected))+"\n",encoding="utf8")
    print("PHASE6_FINISHED",json.dumps({
        "audit":audit,"results":summary["results"],
        "output":str(args.output_dir),
    },ensure_ascii=False),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--num-questions",type=int,default=N_CASES)
    ap.add_argument("--batch-size",type=int,default=12)
    args=ap.parse_args()
    evaluate(args)


if __name__=="__main__":
    main()
