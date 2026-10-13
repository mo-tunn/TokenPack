"""Produce frozen paired ORIGINAL HotpotQA sentence contexts, no gold.

Identical per-sentence MiniLM zero-shot scores and exact 25% render cap.
Baseline Faz8 unmodified choose_atoms; treatment fixed lexical linked pair.
"""
from __future__ import annotations
import json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false";os.environ["OMP_NUM_THREADS"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import pyarrow.parquet as pq
from submission.experiments.phase27_joint_chain_register import OUT
from submission.experiments.phase6_hotpot_eval import DATA
from submission.experiments.phase12_register import sha_file,new_json
from submission.experiments.phase8_atomic_contexts import (
    sentence_candidates,render,encode_scores)
from submission.experiments.phase8_atomic_protocol import SCORER_MAX_INPUT_TOKENS
from submission.experiments.phase27_joint_chain_core import (
    joint_chain_select,baseline_indices)
from submission.experiments.phase5b_pretrained_rerank_eval import (
    FOLDER,WEIGHTS_SHA256,verify_local_model)
from tokenpack.tokenization import TokenCounter
FILE=OUT/"gold_blind_paired_contexts.jsonl"
FROZEN=OUT/"CONTEXTS_FROZEN.json"
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    if FILE.exists() or FROZEN.exists():raise FileExistsError("Never overwrite")
    protocol=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    core=ROOT/"submission/experiments/phase27_joint_chain_core.py"
    assert sha_file(core)==protocol["source_code_sha256"]["phase27_joint_chain_core.py"]
    assert sha_file(DATA)==protocol["dataset_sha256"] and protocol["weights_sha256"]==WEIGHTS_SHA256
    table=pq.ParquetFile(DATA).read(columns=[
         "id","question","context.title","context.sentences"])
    assert set(table.column_names)=={"id","question","context"}
    schema=str(table.schema.remove_metadata())
    assert all(x not in schema for x in ("supporting_facts","answer","type"))
    ids=set(protocol["pilot_ids"])
    data={r["id"]:r for r in table.to_pylist() if r["id"] in ids}
    assert set(data)==ids
    verify_local_model()
    tk=AutoTokenizer.from_pretrained(FOLDER,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(
          FOLDER,local_files_only=True).to("cuda").eval()
    tok=TokenCounter();assert tok._encoding is not None
    new_json(OUT/"CONTEXT_SELECTION_PRE_GOLD_FREEZE.json",{
       "n":48,"dataset_sha256":sha_file(DATA),
       "pre_register_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
       "new_code_sha256":sha_file(Path(__file__)),
       "selector_core_sha256":sha_file(core),
       "model_weights_sha256":WEIGHTS_SHA256,
       "gold_not_read":True,
       "columns":["id","question","context.title","context.sentences"]})
    stats={"n":0,"source_sentence_candidates":0,
       "baseline_selected_sentences":0,"joint_selected_sentences":0,
       "matching_lexical_link_n":0,"contexts_changed":0,
       "cap_violations":0,"mean_baseline_tokens":0,
       "mean_joint_tokens":0,"mean_cap_tokens":0,
       "baseline_exact_faz8_mismatches":0}
    start=time.monotonic()
    with FILE.open("x",encoding="utf8") as f:
      for count,ident in enumerate(protocol["pilot_ids"],1):
        case=data[ident]
        articles,atoms=sentence_candidates(case)
        total=sum(tok.count(a["text"]) for a in articles)
        cap=max(1,math.floor(total*.25))
        scores=encode_scores(model,tk,case["question"],
           [a["input_text"] for a in atoms],max_length=SCORER_MAX_INPUT_TOKENS)
        base,bi=baseline_indices(atoms,scores,cap,tok)
        treat,ti,why=joint_chain_select(atoms,scores,cap,tok)
        if why=="no_fitting_original_title_link":ti=list(bi)
        assert treat["text"]==render(ti,atoms)
        assert tok.count(base["text"])==base["tokens"]<=cap
        assert tok.count(treat["text"])==treat["tokens"]<=cap
        assert base["budget"]==treat["budget"]==cap
        info={}
        for name,details,indexes in (("atomic-sentence-rank",base,bi),
              ("joint-title-link-seed",treat,ti)):
           info[name]={"text":details["text"],"tokens":details["tokens"],
             "cap":cap,
             "source_article_count":details["source_articles"],
             "selected_sentence_count":len(indexes),
             "selected_original_title_sentence":[
               {"title":atoms[i]["title"],"sent_id":atoms[i]["sent_id"],
                "article":atoms[i]["article"]} for i in indexes]}
        rec={"id":ident,"question":case["question"],"source_token_sum":total,
             "cap":cap,"candidate_source_sentence_count":len(atoms),
             "original_source_title_sentence":[
               {"title":a["title"],"sent_id":a["sent_id"],"article":a["article"]}
               for a in atoms],
             "contexts":info,"linked_pair_fitted":why=="fitting_cross_article_title_link",
             "selection_rule_reason":why,
             "no_gold_or_support_labels_used":True}
        f.write(json.dumps(rec,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
        stats["n"]+=1
        stats["source_sentence_candidates"]+=len(atoms)
        stats["baseline_selected_sentences"]+=len(bi)
        stats["joint_selected_sentences"]+=len(ti)
        stats["matching_lexical_link_n"]+=int(rec["linked_pair_fitted"])
        stats["contexts_changed"]+=int(base["text"]!=treat["text"])
        stats["mean_baseline_tokens"]+=base["tokens"]/48
        stats["mean_joint_tokens"]+=treat["tokens"]/48
        stats["mean_cap_tokens"]+=cap/48
        if count%12==0:
            print("PHASE27_CONTEXT_PROGRESS",count,48,
                  "sec",round(time.monotonic()-start,1),flush=True)
    new_json(FROZEN,{"n":48,"sha256":sha_file(FILE),
      "protocol_sha256":sha_file(OUT/"PRE_GOLD_PROTOCOL.json"),
      "stats":stats,"gold_unopened":True,"paired_hard_caps_exact":True})
    print("PHASE27_CONTEXT_FROZEN",json.dumps(stats),flush=True)
if __name__=="__main__":main()
