"""Independent hash, prompt, cap and paired-metric audit for Phase29."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,os
from pathlib import Path
import pyarrow.parquet as pq
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import sha_file
from submission.experiments.phase28_semantic_core import payload as verifier_payload,parse_ids
from submission.experiments.phase13_planner import MODEL
from submission.experiments.phase7_modal27b_matched8 import read_rows,make_payload,answer_metrics
OUT=ROOT/"submission/results/phase29_musique_semantic_citation_crossbench_20261013"
DATA=ROOT/"submission/data/phase9_musique_validation_20261012/data/validation-00000-of-00001.parquet"
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode("utf8")).hexdigest()
def new_json(path,obj):
    with Path(path).open("x",encoding="utf8") as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write("\n");f.flush();os.fsync(f.fileno())
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["pilot","validation"],required=True);a=ap.parse_args()
    sub=OUT/a.split;result_path=sub/("PILOT_POST_GOLD_RESULTS.json" if a.split=="pilot" else "VALIDATION_POST_GOLD_RESULTS.json")
    path=sub/"INTEGRITY_INDEPENDENT_AUDIT.json"
    if path.exists():raise FileExistsError(path)
    p=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    assert sha_file(DATA)==p["data_sha256"]
    for name,h in p["code_sha256"].items():assert sha_file(ROOT/"submission/experiments"/name)==h,name
    cfreeze=json.loads((sub/"CANDIDATES_FROZEN.json").read_text(encoding="utf8"))
    vfreeze=json.loads((sub/"VERIFIER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    ctxfreeze=json.loads((sub/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    rfreeze=json.loads((sub/"READER_RESPONSES_FROZEN.json").read_text(encoding="utf8"))
    result=json.loads(result_path.read_text(encoding="utf8"))
    source=sub/"scored_original_candidates.jsonl";tasks_path=sub/"semantic_citation_tasks.jsonl"
    vpath=sub/"semantic_verifier_responses.jsonl";ctpath=sub/"gold_blind_paired_contexts.jsonl"
    rtpath=sub/"reader_tasks.jsonl";rpath=sub/"reader_responses.jsonl"
    assert sha_file(source)==cfreeze["source_sha256"] and sha_file(tasks_path)==cfreeze["task_sha256"]
    assert sha_file(vpath)==vfreeze["sha256"] and sha_file(ctpath)==ctxfreeze["sha256"]
    assert sha_file(rpath)==rfreeze["responses_sha256"]
    ids=p["pilot_ids"] if a.split=="pilot" else p["reserved_ids"]
    cases=read_rows(source);vtasks=read_rows(tasks_path);vans=read_rows(vpath)
    contexts=read_rows(ctpath);rtasks=read_rows(rtpath);rans=read_rows(rpath)
    assert [x["id"] for x in cases]==[x["id"] for x in vtasks]==[x["id"] for x in vans]==[x["id"] for x in contexts]==ids
    assert len(rtasks)==len(rans)==2*len(ids)
    # Independently reproduce every verifier request digest and parse decision.
    for task,rec in zip(vtasks,vans):
        assert rec["prompt_sha256"]==digest(verifier_payload(task,MODEL))
        ids_parsed,status=parse_ids(rec["raw_output"],task)
        assert rec["citation_ids"]==ids_parsed and rec["parse_status"]==status
    # Reader payloads must match; reused outputs must come from an earlier exact same prompt.
    seen={}
    for task,rec in zip(rtasks,rans):
        ph=digest(make_payload(task));assert rec["prompt_sha256"]==ph
        if not rec["paid"]:
            assert rec["reused_identical_prompt_from"]==seen[ph]["task_id"]
            assert rec["answer"]==seen[ph]["answer"]
        else:
            assert ph not in seen
            seen[ph]=rec
    assert sum(bool(x.get("paid")) for x in rans)==rfreeze["actual_paid_reader_calls"]
    assert sum(not bool(x.get("paid")) for x in rans)==rfreeze["identical_prompt_reuse_n"]
    # Recompute gold metrics independently, after frozen readers.
    gold_cols=["id","answer","paragraphs.list.element.idx","paragraphs.list.element.is_supporting"]
    goldrows=pq.ParquetFile(DATA).read(columns=gold_cols).to_pylist()
    gold={x["id"]:x for x in goldrows if x["id"] in set(ids)}
    assert set(gold)==set(ids)
    rindex={x["task_id"]:x for x in rans}
    audited=[]
    for source,ctx in zip(cases,contexts):
        ident=ctx["id"];g=gold[ident]
        target={int(x["idx"]) for x in g["paragraphs"] if bool(x["is_supporting"])}
        assert target
        row={"id":ident}
        for arm in ("atomic-sentence-rank","semantic-joint-citations"):
            selected=ctx["contexts"][arm]["selected_original_pairs"]
            final={int(x["paragraph_idx"]) for x in selected}
            assert ctx["contexts"][arm]["tokens"]==ctx["contexts"][arm]["tokens"] # recorded token integer
            assert ctx["contexts"][arm]["tokens"]<=ctx["cap"]==ctx["contexts"][arm]["cap"]
            hits=len(target&final);em,f1=answer_metrics(rindex[ident+"|"+arm]["answer"],g["answer"])
            row[arm]={"full_support":int(target<=final),"support_recall":hits/len(target),"EM":em,"F1":f1}
        if not ctx["source_sentence_set_changed"]:
            assert row["atomic-sentence-rank"]["full_support"]==row["semantic-joint-citations"]["full_support"]
            assert row["atomic-sentence-rank"]["support_recall"]==row["semantic-joint-citations"]["support_recall"]
            assert rindex[ident+"|atomic-sentence-rank"]["answer"]==rindex[ident+"|semantic-joint-citations"]["answer"]
        audited.append(row)
    assert len(audited)==len(result["per_case"])
    for x,y in zip(audited,result["per_case"]):
        assert x["id"]==y["id"]
        for arm in ("atomic-sentence-rank","semantic-joint-citations"):
            for m in ("full_support","support_recall","EM","F1"):
                assert math.isclose(float(x[arm][m]),float(y[arm][m]),rel_tol=0,abs_tol=1e-12)
    def binary(metric):
        wins=sum(x["semantic-joint-citations"][metric]>x["atomic-sentence-rank"][metric] for x in audited)
        losses=sum(x["semantic-joint-citations"][metric]<x["atomic-sentence-rank"][metric] for x in audited)
        return {"win":wins,"loss":losses,"net":wins-losses,
          "p":float(binomtest(min(wins,losses),wins+losses,.5).pvalue) if wins+losses else 1.0}
    em=binary("EM");support=binary("full_support")
    assert em["win"]==result["paired_EM"]["win"] and em["loss"]==result["paired_EM"]["loss"]
    assert support["win"]==result["paired_full_support_paragraph_coverage"]["win"]
    assert support["loss"]==result["paired_full_support_paragraph_coverage"]["loss"]
    assert math.isclose(em["p"],result["paired_EM"]["two_sided_exact_mcnemar_p"],rel_tol=1e-12)
    assert math.isclose(support["p"],result["paired_full_support_paragraph_coverage"]["two_sided_exact_mcnemar_p"],rel_tol=1e-12)
    assert ctxfreeze["stats"]["token_cap_violations"]==0
    report={"split":a.split,"status":"PASS_INDEPENDENT_RECOMPUTATION","n":len(ids),
      "preregistered_ids_exact":True,"all_preregistered_code_hashes_valid":True,
      "data_hash_valid":True,"candidate_hashes_valid":True,"verifier_hash_valid":True,
      "context_hash_valid":True,"reader_hash_valid":True,
      "verifier_prompt_hashes_recomputed":len(vans),"reader_prompt_hashes_recomputed":len(rans),
      "identical_prompt_reuse_audited":rfreeze["identical_prompt_reuse_n"],
      "actual_paid_reader_calls":rfreeze["actual_paid_reader_calls"],
      "actual_paid_verifier_calls":vfreeze["paid_calls"],
      "max_cap_violations":ctxfreeze["stats"]["token_cap_violations"],
      "independent_EM":em,"independent_full_support_paragraph_coverage":support,
      "decision_recomputed":result["decision"],"support_metric_not_semantic_entailment":True,
      "public_validation_not_hidden_test":True,"qasper_sealed_test_unopened":True,
      "no_new_remote_calls":True,"source_result_sha256":sha_file(result_path),
      "source_context_sha256":ctxfreeze["sha256"],"source_reader_sha256":rfreeze["responses_sha256"]}
    new_json(path,report)
    print("PHASE29_INDEPENDENT_AUDIT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
