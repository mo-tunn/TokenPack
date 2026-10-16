"""Post-freeze exact sentence support and Wikidata chain-entity diagnostics.
2Wiki gold columns are NEVER loaded by ranker/planner/reader contexts.
All linked triple lexical measures are explicitly weak proxies.
"""
from __future__ import annotations
import collections,hashlib,json,re,statistics,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from submission.experiments.phase12_register import OUT,DATA,verify_data,sha_file,new_json
METHODS=("crossdoc-bridge","hop-planner")
def canon(s):
    return " ".join(str(s or "").casefold().split())
def normalize_mention(s):
    return " ".join(re.findall(r"\w+",str(s or "").casefold(),flags=re.UNICODE))
def mentions(s,text):
    key=normalize_mention(s)
    if not key:return False
    value=normalize_mention(text)
    return bool(re.search(r"(?<!\w)"+re.escape(key)+r"(?!\w)",value))
def paired_ci(vals):
    x=np.asarray(vals,dtype=float)
    rng=np.random.default_rng(20261012)
    draws=x[rng.integers(0,len(x),(10000,len(x)))].mean(axis=1)
    return [float(z) for z in np.quantile(draws,[.025,.975])]

def run():
    proto=json.loads((OUT/"PRE_GOLD_PROTOCOL.json").read_text(encoding="utf8"))
    freeze=json.loads((OUT/"CONTEXTS_FROZEN.json").read_text(encoding="utf8"))
    ctx=OUT/"gold_blind_contexts.jsonl"
    assert sha_file(ctx)==freeze["context_file_sha256"]
    reader=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    assert reader["n"]==32
    rows=[json.loads(x) for x in ctx.open(encoding="utf8") if x.strip()]
    assert len(rows)==64 and [r["id"] for r in rows]==proto["ids"]
    pf=verify_data()
    ids=set(proto["ids"])
    # FIRST access to all gold supporting-fact and structured-relation columns:
    gold={r["id"]:r for r in pf.read(
        columns=["id","type","supporting_facts","evidences"]).to_pylist()
        if r["id"] in ids}
    assert set(gold)==ids
    obs=[]
    missing_examples=0
    triples_summary={}
    for row in rows:
        label=gold[row["id"]]
        required=list(zip(label["supporting_facts"]["title"],label["supporting_facts"]["sent_id"]))
        required=set((canon(t),int(i)) for t,i in required)
        assert required
        atom_keys=[(canon(c["title"]),int(c["sent_id"])) for c in row["candidate_map"]]
        available=set(atom_keys)
        missing=len(required-available)
        missing_examples+=bool(missing)
        triples=[t for t in (label.get("evidences") or []) if len(t)==3]
        source_nodes={canon(t[0]) for t in triples}
        obj_nodes={canon(t[2]) for t in triples}
        intermediate=sorted((source_nodes&obj_nodes)-{""})
        for arm in METHODS:
            chosen=row["contexts"][arm]
            assert chosen["budget"]==row["budget"]
            assert chosen["tokens"]<=row["budget"]
            ix=set(chosen["selected_sentence_indexes"])
            assert all(0<=j<len(atom_keys) for j in ix)
            selected={atom_keys[j] for j in ix}
            hit=len(required&selected)
            # Support sentences strictly indexed from the original labeled wiki page and sentence.
            all_support=int(required<=selected)
            support_recall=hit/len(required)
            doc_titles={atom_keys[j][0] for j in ix}
            # Weak label-only chain proxies: mention nodes/endpoints in selected text.
            text=chosen["text"]
            mids=[mentions(s,text) for s in intermediate]
            edge_endpoints=[mentions(t[0],text) and mentions(t[2],text) for t in triples]
            obs.append({
                "id":row["id"],"method":arm,"type":label["type"],
                "required_support_sentence_n":len(required),
                "missing_support_sentence_n":missing,
                "supporting_sentences_all":all_support,
                "supporting_sentence_recall":support_recall,
                "supporting_sentence_hits":hit,
                "selected_source_titles":len(doc_titles),
                "real_cl100k_tokens":chosen["tokens"],
                "budget":row["budget"],
                "intermediate_entity_n":len(intermediate),
                "intermediate_entity_mentions_fraction":sum(mids)/len(mids) if mids else None,
                "all_intermediate_entities_mentioned":int(all(mids)) if mids else None,
                "gold_triple_count":len(triples),
                "triple_endpoint_mention_fraction":sum(edge_endpoints)/len(edge_endpoints) if edge_endpoints else None,
                "all_triple_endpoints_mentioned":int(all(edge_endpoints)) if edge_endpoints else None,
            })
    grouped=collections.defaultdict(dict)
    for x in obs:grouped[x["id"]][x["method"]]=x
    assert len(grouped)==64 and all(set(v)==set(METHODS) for v in grouped.values())
    averages={}
    for arm in METHODS:
        x=[s[arm] for s in grouped.values()]
        averages[arm]={"n":len(x),
            "all_supporting_sentences_correct":sum(t["supporting_sentences_all"] for t in x),
            "all_supporting_sentences_rate":statistics.mean(t["supporting_sentences_all"] for t in x),
            "mean_supporting_sentence_recall":statistics.mean(t["supporting_sentence_recall"] for t in x),
            "mean_used_tokens":statistics.mean(t["real_cl100k_tokens"] for t in x),
            "mean_max_budget":statistics.mean(t["budget"] for t in x),
            "mean_selected_distinct_titles":statistics.mean(t["selected_source_titles"] for t in x),
            "intermediate_node_examples":sum(t["intermediate_entity_n"]>0 for t in x),
            "all_intermediate_nodes_mentioned_rate":statistics.mean(
                t["all_intermediate_entities_mentioned"] for t in x
                if t["all_intermediate_entities_mentioned"] is not None)
                if any(t["intermediate_entity_n"] for t in x) else None,
            "mean_triple_endpoints_mentioned":statistics.mean(
                t["triple_endpoint_mention_fraction"] for t in x
                if t["triple_endpoint_mention_fraction"] is not None),
            "token_cap_violations":sum(t["real_cl100k_tokens"]>t["budget"] for t in x)}
    pairs=list(grouped.values())
    deltas=[x["hop-planner"]["supporting_sentences_all"]-
            x["crossdoc-bridge"]["supporting_sentences_all"] for x in pairs]
    rec=[x["hop-planner"]["supporting_sentence_recall"]-
         x["crossdoc-bridge"]["supporting_sentence_recall"] for x in pairs]
    wins=sum(v==1 for v in deltas)
    losses=sum(v==-1 for v in deltas)
    comparison={"all_supporting_sentences":{
        "difference":statistics.mean(deltas),
        "paired_question_bootstrap_CI95":paired_ci(deltas),
        "hop_only_all":wins,"bridge_only_all":losses,
        "exact_mcnemar_p":binomtest(min(wins,losses),wins+losses,.5).pvalue if wins+losses else 1.0},
        "supporting_sentence_recall":{"difference":statistics.mean(rec),
                                      "paired_question_bootstrap_CI95":paired_ci(rec)}}
    bytype={}
    for typ in sorted({x["type"] for x in obs}):
        xs=[p for p in pairs if p[METHODS[0]]["type"]==typ]
        bytype[typ]={"n":len(xs),
            "bridge_all_support":sum(x[METHODS[0]]["supporting_sentences_all"] for x in xs),
            "hop_all_support":sum(x[METHODS[1]]["supporting_sentences_all"] for x in xs),
            "mean_bridge_recall":statistics.mean(x[METHODS[0]]["supporting_sentence_recall"] for x in xs),
            "mean_hop_recall":statistics.mean(x[METHODS[1]]["supporting_sentence_recall"] for x in xs)}
    readercases={x["id"]:x for x in reader["per_case"]}
    outcomes=[]
    for id,x in grouped.items():
        if id not in readercases:continue
        r=readercases[id]
        diff=r["hop-planner"]["EM"]-r["crossdoc-bridge"]["EM"]
        outcomes.append({"id":id,"type":x[METHODS[0]]["type"],"EM_difference":diff,
            "support_sentence_all_delta":x[METHODS[1]]["supporting_sentences_all"]-x[METHODS[0]]["supporting_sentences_all"],
            "support_sentence_recall_delta":x[METHODS[1]]["supporting_sentence_recall"]-x[METHODS[0]]["supporting_sentence_recall"]})
    result={"study":"Faz12 2WikiMultiHopQA new benchmark evidence-relation diagnostic",
        "n_retrieval":64,"n_reader":32,
        "methods":METHODS,"metric_caveat":"Supporting sentence index is exact; intermediate/triple endpoint mentions are lexical proxies, NOT verified reasoning edges.",
        "gold_source_not_used_in_inference":True,
        "gold_first_opened_after_context_and_reader_freeze":True,
        "support_annotation_not_found_in_sources_cases":missing_examples,
        "arms":averages,"paired":comparison,"by_type_descriptive":bytype,
        "reader":{k:v for k,v in reader.items() if k!="per_case"},
        "reader_vs_support_paired_cases":outcomes,
        "per_case_evidence_metrics":obs,
        "limitations":["Source only ten supplied Wikipedia contexts, not open-Wikipedia retrieval",
                       "Public validation split is not a hidden official test",
                       "Exact support sentence coverage does not guarantee answer sufficiency",
                       "Lexical intermediate entity and triple endpoint mentions are only weak proxies",
                       "Small stratification groups and posthoc relation analysis exploratory",
                       "Same maximum cap but realized tokens may vary"],
        "reserved_unused":128,
        "QASPER_frozen_test_unchanged":True,"production_changed":False}
    f=OUT/"RELATION_SUPPORT_POST_GOLD_RESULTS.json"
    if f.exists():raise FileExistsError(f)
    new_json(f,result)
    print("PHASE12_FINAL_DIAGNOSTIC",json.dumps({
        "n":64,"availability_missing_examples":missing_examples,
        "arms":averages,"paired":comparison,"by_type":bytype,
        "reader":result["reader"]["arms"]},ensure_ascii=False),flush=True)

if __name__=="__main__":run()
