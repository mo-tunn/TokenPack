"""Faz26 fixed Faz8 atomic-sentence-rank on new HotpotQA distractor IDs.

Read gold-blind leaf projection only. Exact Phase8 greedy selection verified
byte-for-byte for each of 96 questions, while also storing original
(title, sentence_index) references for post-gold analysis.
"""
from __future__ import annotations
import json,hashlib,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false";os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["OMP_NUM_THREADS"]="2"
import pyarrow.parquet as pq
from submission.experiments.phase26_hotpot_register import OUT
from submission.experiments.phase6_hotpot_eval import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase8_atomic_contexts import (
    sentence_candidates,render,choose_atoms,encode_scores)
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
CONTEXT=OUT/"gold_blind_contexts.jsonl"
PRE=OUT/"CONTEXT_SELECTION_PRE_GOLD_FREEZE.json"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
def reproduce_ranked_indices(atoms,scores,cap,counter):
    ranking=sorted(range(len(atoms)),key=lambda i:(
         -scores[i],atoms[i]["article"],atoms[i]["sent_id"]))
    chosen=[];used=0
    for idx in ranking:
        trial=render([*chosen,idx],atoms)
        cost=counter.count(trial)
        if cost<=cap:
            chosen.append(idx);used=cost
    return chosen
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    if CONTEXT.exists() or PRE.exists() or FROZEN.exists():
        raise FileExistsError("Never rewrite context results")
    pre=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert pre["model_weights_sha256"]==WEIGHTS_SHA256
    assert sha_file(DATA)==pre["dataset_sha256"]
    ids=set(pre["pilot_ids"]);assert len(ids)==96
    columns=["id","question","context.title","context.sentences"]
    table=pq.ParquetFile(DATA).read(columns=columns)
    assert set(table.column_names)=={"id","question","context"}
    # HuggingFace dataset metadata embeds the FULL original feature schema;
    # inspect only physically projected leaf fields, not metadata text.
    schema=str(table.schema.remove_metadata())
    assert all(bad not in schema for bad in (
       "supporting_facts","answer","answer_aliases","type"))
    selected={r["id"]:r for r in table.to_pylist() if r["id"] in ids}
    assert set(selected)==ids
    verify_local_model()
    tk=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    ranker=AutoModelForSequenceClassification.from_pretrained(
        FOLDER,local_files_only=True).to("cuda").eval()
    tok=TokenCounter();assert tok._encoding is not None
    new_json(PRE,{"dataset_sha256":sha_file(DATA),
        "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
        "source_selector_sha256":sha_file(ROOT/"submission/experiments/phase8_atomic_contexts.py"),
        "new_context_code_sha256":sha_file(Path(__file__)),
        "model_weights_sha256":WEIGHTS_SHA256,
        "columns_read":columns,"gold_opened":False,
        "n":96,"same_byte_render_as_Faz8_atomic_sentence_rank":True})
    stats={"n":0,"source_sentences":0,"selected_sentences":0,
       "cap_violations":0,"exact_selector_render_mismatches":0,
       "mean_cap":0,"mean_final_tokens":0,
       "any_empty":0}
    t0=time.monotonic()
    with CONTEXT.open("x",encoding="utf8") as f:
      for j,ident in enumerate(pre["pilot_ids"],1):
        case=selected[ident]
        articles,atoms=sentence_candidates(case)
        assert len(atoms)>0 and len(articles)>0
        src=sum(tok.count(a["text"]) for a in articles)
        cap=max(1,int(math.floor(src*.25)))
        scores=encode_scores(ranker,tk,case["question"],
              [a["input_text"] for a in atoms],max_length=SCORER_MAX_INPUT_TOKENS)
        chosen=reproduce_ranked_indices(atoms,scores,cap,tok)
        expected=choose_atoms(atoms,scores,{},cap,tok,bridge=False)
        rendered=render(chosen,atoms)
        assert rendered==expected["text"]
        actual=tok.count(rendered)
        assert actual==expected["tokens"]<=cap
        indices=[{"title":atoms[k]["title"],"sentence_id":atoms[k]["sent_id"],
                 "article_index":atoms[k]["article"],
                 "original_source_sentence":atoms[k]["sentence"]} for k in chosen]
        rec={"id":ident,"question":case["question"],
             "context":rendered,"source_token_total":src,
             "cap":cap,"actual_tokens":actual,
             "selected_original_sentences":indices,
             "selected_sentence_count":len(chosen),
             "total_candidate_original_sentences":len(atoms),
             "original_source_titles":[a["title"] for a in articles],
             "candidate_original_title_sentence_pairs":[{
                 "title":a["title"],"sentence_id":a["sent_id"],
                 "article_index":a["article"]} for a in atoms],
             "source_has_only_original_sentences":True}
        f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        f.flush();os.fsync(f.fileno())
        stats["n"]+=1
        stats["source_sentences"]+=len(atoms)
        stats["selected_sentences"]+=len(chosen)
        stats["mean_cap"]+=cap/96
        stats["mean_final_tokens"]+=actual/96
        stats["any_empty"]+=int(not chosen)
        if j%16==0:print("PHASE26_CONTEXTS_PROGRESS",j,96,
            "wall_s",round(time.monotonic()-t0,1),flush=True)
    new_json(FROZEN,{"n":96,"sha256":sha_file(CONTEXT),
        "pre_gold_selection_sha256":sha_file(PRE),
        "stats":stats,"gold_unopened":True,
        "source_gold_parquet_columns_never_read":True})
    print("PHASE26_CONTEXTS_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
