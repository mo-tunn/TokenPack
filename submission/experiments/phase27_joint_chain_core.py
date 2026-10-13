"""Phase27 gold-BLIND two-article citation-link chain seed selector.

Experimental change relative to untouched Faz8 greedy sentence rank:
find one actual cross-article Wikipedia title mention in original sentence A
(pointing at article B), then force source sentence A and top-scoring
sentence from B into the %25 token budget, fill with SAME greedy ranking.
If no fitting link, EXACT Faz8 greedy baseline. No Qwen or gold.
"""
from __future__ import annotations
import re
from submission.experiments.phase8_atomic_contexts import render,choose_atoms

def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+",str(text).casefold()))

def contains_title(sentence,title):
    s=normalize(sentence);t=normalize(title)
    if len(t)<4:return False
    return bool(re.search(r"(?<!\w)"+re.escape(t)+r"(?!\w)",s))

def seed_link_pair(atoms,scores,cap,counter):
    """Return fixed best scoring link if both originals fit, else None.

    Candidate A: top-24 relevance sentences that explicitly MENTION the
    title of a different input document B. B: highest scoring sentence
    belonging to that document. Link must be literal, case-folded,
    punctuation-normalized, whole phrase. Pair maximizes sum of relevance
    ranks (not gold); ties broken by input/source order.
    """
    assert len(atoms)==len(scores)
    ranked=sorted(range(len(atoms)),key=lambda i:(
       -scores[i],atoms[i]["article"],atoms[i]["sent_id"]))
    best_by_article={}
    titles={}
    for i in ranked:
        article=atoms[i]["article"]
        if article not in best_by_article:best_by_article[article]=i
        if article not in titles:titles[article]=atoms[i]["title"]
    pairs=[]
    for rank_a,i in enumerate(ranked[:24]):
        a=atoms[i]["article"]
        for b,title in titles.items():
            if b==a or not contains_title(atoms[i]["sentence"],title):continue
            j=best_by_article[b]
            test=render([i,j],atoms)
            if counter.count(test)>cap:continue
            # Prefer linked pairs with high cross-sentence relevance; all
            # components strictly gold-blind. Max score vs baseline requires
            # two original sentences, not extrapolated graph facts.
            pair_score=float(scores[i]+scores[j])
            pairs.append((-pair_score,rank_a,b,atoms[j]["sent_id"],i,j))
    if not pairs:return None
    pairs.sort()
    x=pairs[0]
    return (x[-2],x[-1])

def joint_chain_select(atoms,scores,cap,counter):
    baseline=choose_atoms(atoms,scores,{},cap,counter,bridge=False)
    seed=seed_link_pair(atoms,scores,cap,counter)
    if seed is None:
        return dict(baseline),[], "no_fitting_original_title_link"
    rank=sorted(range(len(atoms)),key=lambda i:(
         -scores[i],atoms[i]["article"],atoms[i]["sent_id"]))
    selected=list(seed)
    used=set(selected)
    for i in rank:
        if i in used:continue
        if counter.count(render([*selected,i],atoms))<=cap:
            selected.append(i);used.add(i)
    text=render(selected,atoms)
    assert counter.count(text)<=cap
    # Provenance and source fact indices are gold-blind. No phrase rewriting.
    result={"text":text,"tokens":counter.count(text),"budget":cap,
      "source_articles":len({atoms[i]["article"] for i in selected}),
      "sentences":len(selected),"candidate_count":len(atoms)}
    return result,selected,"fitting_cross_article_title_link"

def baseline_indices(atoms,scores,cap,counter):
    rank=sorted(range(len(atoms)),key=lambda i:(
       -scores[i],atoms[i]["article"],atoms[i]["sent_id"]))
    selected=[]
    for i in rank:
        if counter.count(render([*selected,i],atoms))<=cap:
            selected.append(i)
    baseline=choose_atoms(atoms,scores,{},cap,counter,bridge=False)
    assert render(selected,atoms)==baseline["text"]
    return baseline,selected
