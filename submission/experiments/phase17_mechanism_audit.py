"""Phase17 gold-blind mechanism audit. Does NOT read any gold labels."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase17_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
FILES={
    "initial":OUT/"verifier_responses.jsonl",
    "rescue":OUT/"rescue_verifier_responses.jsonl",
    "coherence":OUT/"coherence_verifier_responses.jsonl",
    "ctx":OUT/"gold_blind_contexts.jsonl"}
def read(path):return [json.loads(s) for s in path.open(encoding="utf8") if s.strip()]
def main():
    a,b,c,d=[read(FILES[k]) for k in ("initial","rescue","coherence","ctx")]
    assert len(a)==len(b)==len(c)==len(d)==32
    assert [x["id"] for x in a]==[x["id"] for x in b]==[x["id"] for x in c]==[x["id"] for x in d]
    counts=collections.Counter()
    per=[]
    for initial,rescue,coherence,context in zip(a,b,c,d):
        prior=dict(initial["decision_ids"])
        for key,pick in rescue["rescue_decisions"].items():
            assert key in prior and prior[key] is None
            if pick is not None:prior[key]=pick
        curr=coherence["decision_ids"]
        assert set(prior)==set(curr)
        equal=sum(prior[key]==curr[key] for key in prior)
        both_pos_equal=sum(prior[key]==curr[key] and curr[key] is not None for key in prior)
        prior_pos_cross_null=sum(prior[key] is not None and curr[key] is None for key in prior)
        prior_null_cross_pos=sum(prior[key] is None and curr[key] is not None for key in prior)
        different_positive=sum(prior[key] is not None and curr[key] is not None and prior[key]!=curr[key] for key in prior)
        counts.update({
            "edges_total":len(curr),"decisions_identical":equal,
            "same_positive":both_pos_equal,"prior_positive_to_null":prior_pos_cross_null,
            "prior_null_to_positive":prior_null_cross_pos,
            "positive_changed_candidate":different_positive,
            "question_context_changed":int(
                context["contexts"]["branch-lock"]["text"]!=context["contexts"]["coherence-gated"]["text"]),
            "valid_json":int(coherence["parse_status"]=="valid_model_verification"),
        })
        per.append({"id":initial["id"],
            "n_edges":len(curr),"decision_identical":equal,
            "prior_positive_to_null":prior_pos_cross_null,
            "prior_null_to_positive":prior_null_cross_pos,
            "positive_changed_candidate":different_positive,
            "context_changed":int(context["contexts"]["branch-lock"]["text"]!=context["contexts"]["coherence-gated"]["text"])})
    result={"gold_unread":True,
        "question_count":32,"counts":dict(counts),
        "contexts_sha256":sha_file(FILES["ctx"]),
        "reader_primary_gate_impossible":True,
        "interpretation":"Only model-decision and selected-context overlap. This does not measure semantically correct relations, gold evidence, or reader EM.",
        "per_case":per}
    path=OUT/"GOLD_BLIND_MECHANISM_AUDIT.json"
    if path.exists():raise FileExistsError(path)
    new_json(path,result)
    print("PHASE17_GOLDBLIND_MECHANISM",json.dumps(result["counts"]),flush=True)
if __name__=="__main__":main()
