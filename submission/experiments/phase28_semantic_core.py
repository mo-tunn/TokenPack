"""Phase28 gold-blind source-grounded semantic chain citation selector.

Frozen per-question candidate pool = union of top24 MiniLM sentences and
top1 per original source article (deduped by original position).
Qwen chooses 2-3 original source sentence IDs that jointly support the
question, citing a short intermediate relation. ALL citations must belong
to different articles at least twice and fit exact 25% render cap.
Gold/supporting facts NEVER used; no semantic-accuracy guarantee.
"""
from __future__ import annotations
import json,re
from submission.experiments.phase27_joint_chain_core import baseline_indices
from submission.experiments.phase8_atomic_contexts import render
MAX_TOP=24
MAX_CANDIDATES=36
SYSTEM=(
 "You are an evidence-only multi-hop Wikipedia citation verifier. Given a "
 "question and numbered verbatim SOURCE sentences from source articles, "
 "identify TWO or THREE distinct sentence IDs that together justify the "
 "intermediate entity and the TARGET answer relation in the question. "
 "At least TWO DIFFERENT Wikipedia source articles MUST contribute sentences. "
 "Do NOT rely on world knowledge, inferred titles alone or unquoted facts. "
 "If the listed sources do not jointly justify both hops, ABSTAIN. "
 "Return one valid JSON object ONLY with keys "
 "'citation_ids' (array of 2 or 3 sentence IDs, or [] on abstention) and "
 "'intermediate_relation' (at most 20 words summarizing the grounded cross-source hop). "
 "DO NOT output the target answer as an extra unsupported claim; use exactly "
 "the provided candidate sentence IDs, no hallucinated identifiers."
)
USER="Multi-hop question: {question}\n\nOriginal source sentences:\n{sentences}\n\nOutput grounded citation_ids JSON, [] if two-hop relation cannot be verified."
SEED=20261013
MAX_OUTPUT_TOKENS=190
def candidate_ids(atoms,scores):
    ranked=sorted(range(len(atoms)),key=lambda i:(
       -scores[i],atoms[i]["article"],atoms[i]["sent_id"]))
    chosen=ranked[:MAX_TOP]
    present=set(chosen)
    seen_articles=set()
    for i in ranked:
        article=atoms[i]["article"]
        if article not in seen_articles:
            seen_articles.add(article)
            if i not in present and len(chosen)<MAX_CANDIDATES:
                chosen.append(i);present.add(i)
    return chosen
def make_task(id,question,atoms,scores,cap,source_total):
    eligible=candidate_ids(atoms,scores)
    choices=[]
    for ordinal,i in enumerate(eligible):
        a=atoms[i]
        choices.append({"cid":f"S{ordinal:02d}","atom_index":i,
           "article":a["article"],"title":a["title"],
           "sent_id":a["sent_id"],"sentence":a["sentence"]})
    return {"id":id,"question":question,"cap":cap,"source_total":source_total,
       "candidates":choices}
def user_message(task):
    lines=[f'{x["cid"]} | {x["title"]} | {x["sentence"]}' for x in task["candidates"]]
    return USER.format(question=task["question"],sentences="\n".join(lines))
def payload(task,model):
    return {"model":model,"messages":[
         {"role":"system","content":SYSTEM},
         {"role":"user","content":user_message(task)}],
        "temperature":0,"top_k":1,"top_p":1,"min_p":0,
        "seed":SEED,"max_tokens":MAX_OUTPUT_TOKENS,
        "reasoning_effort":"none","stream":False}
def parse_ids(raw,task):
    text=str(raw or "").strip()
    if text.startswith("```"):
        text="\n".join(text.splitlines()[1:]).split("```")[0].strip()
    try:
        if "{" in text and "}" in text:
            text=text[text.find("{"):text.rfind("}")+1]
        d=json.loads(text)
        ids=d["citation_ids"]
        relation=d["intermediate_relation"]
        if not isinstance(ids,list) or not isinstance(relation,str) or len(relation)>220:
            return [],"invalid_schema"
        if not ids:return [],"abstained"
        if len(ids) not in (2,3) or not all(isinstance(x,str) for x in ids) or len(set(ids))!=len(ids):
            return [],"invalid_citation_list"
        known={x["cid"]:x for x in task["candidates"]}
        if not set(ids)<=set(known):return [],"unlisted_citations"
        if len({known[x]["article"] for x in ids})<2:
            return [],"single_article_not_chain"
        return ids,"valid_cross_article_citations"
    except (KeyError,ValueError,TypeError,json.JSONDecodeError):
        return [],"invalid_json"
def select(task,atoms,scores,raw,counter):
    baseline,indices=baseline_indices(atoms,scores,task["cap"],counter)
    ids,status=parse_ids(raw,task)
    if not ids:
        return baseline,indices,"fallback_"+status
    known={x["cid"]:x for x in task["candidates"]}
    seeds=[known[x]["atom_index"] for x in ids]
    if counter.count(render(seeds,atoms))>task["cap"]:
        return baseline,indices,"fallback_citations_exceed_cap"
    ranked=sorted(range(len(atoms)),key=lambda i:(
       -scores[i],atoms[i]["article"],atoms[i]["sent_id"]))
    selected=list(seeds);used=set(selected)
    for i in ranked:
        if i in used:continue
        if counter.count(render(selected+[i],atoms))<=task["cap"]:
            selected.append(i);used.add(i)
    selected_text=render(selected,atoms)
    actual=counter.count(selected_text)
    assert actual<=task["cap"]
    return {"text":selected_text,"tokens":actual,"budget":task["cap"],
       "source_articles":len({atoms[i]["article"] for i in selected}),
       "sentences":len(selected),"candidate_count":len(atoms)},selected,"verified_citation_anchors_selected"
