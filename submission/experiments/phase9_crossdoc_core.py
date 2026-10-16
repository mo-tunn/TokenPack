"""Deterministic label-free cross-document sentence graph selection.

No embedding or annotation dependencies. The linked pair is motivated
by explicit entity mentions between two DIFFERENT source paragraphs;
subsequent greed uses pretrained sentence relevance and diminishing
returns to already selected paragraphs.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from submission.experiments.phase9_musique_protocol import (
    PAIR_CANDIDATE_POOL,PAIR_LINK_BONUS,DOC_DIVERSITY_DISCOUNT,
    LINKED_DOC_BONUS,MIN_SEED_RELEVANCE,MAX_PAIR_VS_UNLINKED_LOSS,
)

STOP={
    "the","a","an","in","and","of","or","to","for","by","on","from",
    "is","are","at","as","what","who","where","was","were","with",
    "which","when","how","this","that","their","its","after","before",
    "american","british","film","album","series","school","country",
}


def words(s):
    return re.findall(r"[a-z0-9]+",str(s).casefold())


def content_words(s):
    return {w for w in words(s) if len(w)>=4 and w not in STOP}


def title_key(title):
    """Avoid Wikipedia disambiguation brackets in lexical bridge matching."""
    base=str(title).split(" (",1)[0].strip()
    return " ".join(words(base))


def title_in_sentence(title,sentence):
    key=title_key(title)
    if len(key)<5:return False
    sentence_text=" ".join(words(sentence))
    return bool(re.search(r"(?<![a-z0-9])"+re.escape(key)+r"(?![a-z0-9])",
                          sentence_text))


def bridge_link(first,second):
    """Edge only between *different source paragraphs*, no gold."""
    if first["article"]==second["article"]:return 0.
    if (title_in_sentence(second["title"],first["sentence"])
        or title_in_sentence(first["title"],second["sentence"])):
        return 1.
    # Soft cross-sentence entity bridge: two uncommon words repeated.
    wa=content_words(first["sentence"])-content_words(first["title"])
    wb=content_words(second["sentence"])-content_words(second["title"])
    shared=wa&wb
    return .65 if len(shared)>=2 else 0.


def rank_rescale(scores):
    if not scores:return []
    vals=[float(v) for v in scores]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("nonfinite reranker output")
    lo,hi=min(vals),max(vals)
    if hi==lo:return [.5]*len(vals)
    return [(v-lo)/(hi-lo) for v in vals]


def render_atomic(selected,atoms):
    """Whole unmodified original sentences, grouped by provenance.

    Same renderer for both control and new policy. Output can be
    checked against actual original source spans.
    """
    ordered=sorted(selected,key=lambda i:(atoms[i]["article"],
                                            atoms[i]["sent_id"],i))
    articles={}
    for i in ordered:
        a=atoms[i]
        articles.setdefault(a["article"],[]).append(a)
    texts=[]
    for entries in articles.values():
        texts.append("[Source: "+entries[0]["title"]+"]\n"+
                     " ".join(x["sentence"] for x in entries))
    return "\n\n".join(texts).strip()+("\n" if texts else "")


def find_linked_seed(atoms,scaled,cost_budget,counter,renderer=render_atomic):
    """Best affordable strongly linked pair among top 48.

    An explicit linked pair must have enough relevance to not displace
    the relevance-optimal unlinked pair by an arbitrary large margin.
    """
    order=sorted(range(len(atoms)),
                 key=lambda i:(-scaled[i],atoms[i]["article"],
                               atoms[i]["sent_id"],i))
    top=order[:PAIR_CANDIDATE_POOL]
    # Cost cache per pair; real final-render accounting is enforced.
    baseline=sum(scaled[i] for i in top[:2])
    best=None
    for pos,i in enumerate(top):
        if scaled[i]<MIN_SEED_RELEVANCE:continue
        for j in top[pos+1:]:
            if scaled[j]<MIN_SEED_RELEVANCE:continue
            if atoms[i]["article"]==atoms[j]["article"]:continue
            link=bridge_link(atoms[i],atoms[j])
            if link<.60:continue
            relevance=scaled[i]+scaled[j]
            if relevance<baseline-MAX_PAIR_VS_UNLINKED_LOSS:continue
            if counter.count(renderer([i,j],atoms))>cost_budget:continue
            score=relevance+PAIR_LINK_BONUS*link
            row=(score,relevance,-atoms[i]["article"],-atoms[j]["article"],i,j)
            if best is None or row[:4]>best[:4]:
                best=row
    return (best[-2],best[-1]) if best else ()


def choose(atoms,raw_scores,budget,counter,*,graph):
    """Base ranking or graph-aware ranking, same compact renderer and cap."""
    if budget<=0 or len(atoms)!=len(raw_scores) or not atoms:
        raise ValueError("Invalid sentence candidate list or budget")
    norm=rank_rescale(raw_scores)
    @lru_cache(maxsize=None)
    def link(i,j):
        return bridge_link(atoms[min(i,j)],atoms[max(i,j)])
    sort_order=sorted(range(len(atoms)),
                      key=lambda i:(-norm[i],atoms[i]["article"],
                                    atoms[i]["sent_id"],i))
    chosen=[]
    seen=set()
    by_doc={}
    links=0
    max_checks=0
    def add(i):
        nonlocal max_checks
        if i in seen:return False
        max_checks+=1
        if counter.count(render_atomic([*chosen,i],atoms))>budget:
            return False
        chosen.append(i)
        seen.add(i)
        by_doc[atoms[i]["article"]]=by_doc.get(atoms[i]["article"],0)+1
        return True

    if graph:
        seed=find_linked_seed(atoms,norm,budget,counter)
        for i in seed:
            if not add(i):raise AssertionError("seed no longer fits")
        links=int(bool(seed))

    remaining=set(range(len(atoms)))-seen
    while remaining:
        if not graph:
            queue=[i for i in sort_order if i in remaining]
        else:
            def utility(i):
                num_already=by_doc.get(atoms[i]["article"],0)
                # Bridge bonus applies ONLY across distinct articles.
                edge=max((link(i,j) for j in chosen
                          if atoms[i]["article"]!=atoms[j]["article"]),
                         default=0.)
                return norm[i]+LINKED_DOC_BONUS*edge-(
                    DOC_DIVERSITY_DISCOUNT*min(3,num_already))
            queue=sorted(remaining,key=lambda i:(-utility(i),
                                                  atoms[i]["article"],
                                                  atoms[i]["sent_id"],i))
        added=None
        for i in queue:
            if add(i):
                added=i
                break
        if added is None:break
        remaining.remove(added)
    text=render_atomic(chosen,atoms)
    used=counter.count(text)
    if used>budget:raise AssertionError("actual rendered cap exceeded")
    return {
        "text":text,"tokens":used,"budget":budget,
        "selected_sentence_indexes":sorted(chosen),
        "selected_source_idxs":sorted({
            atoms[i]["original_paragraph_idx"] for i in chosen}),
        "selected_paragraph_count":len(by_doc),
        "selected_sentence_count":len(chosen),
        "seed_graph_activated":bool(links),
        "render_checks":max_checks,
    }
