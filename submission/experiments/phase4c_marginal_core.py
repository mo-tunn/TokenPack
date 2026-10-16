"""Gold-supervised marginal gain features for TRAIN, gold-blind for inference.

Targets: incremental 5gram evidence coverage of a candidate given currently
selected chunks. Separate annotator groups are alternatives: maximum marginal
recall improvement across groups, NOT merged evidence texts.
"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np

from submission.experiments.phase3a_oracle_core import (
    gram_counts,gold_counts,matched_count,
)
from submission.experiments.phase3b_learned_scoring import FEATURES
from submission.experiments.phase3c_set_aware import terms
from tokenpack.embeddings import cosine

ADDITIONAL=(
    "log_candidate_source_tokens",
    "budget_ratio",
    "cost_fraction_budget",
    "selected_count_fraction",
    "selected_budget_fraction",
    "mean_selected_hybrid_score",
    "new_query_terms_fraction",
    "candidate_new_5grams_fraction",
    "max_text_jaccard_to_selected",
    "max_embedding_cosine_to_selected",
    "max_source_block_containment",
)
NAMES=tuple(FEATURES)+ADDITIONAL
RATIOS=(.25,.5)


def feature_rows(q,indices,selected,budget,ratio,precomputed=None):
    """Compute candidate+state features without gold or answer annotations."""
    if not indices:return np.empty((0,len(NAMES)),dtype=np.float32)
    items=q.scored
    n=len(items)
    selected=set(selected)
    if any(i<0 or i>=n for i in selected):raise ValueError("Bad selected index")
    tok=precomputed if precomputed is not None else [gram_counts(x.chunk.text) for x in items]
    selected_grams=Counter()
    selected_terms=frozenset()
    query_terms=terms(q.text)
    selected_weight=0
    for i in selected:
        selected_grams.update(tok[i])
        selected_terms|=terms(items[i].chunk.text)
        selected_weight+=items[i].weight
    rows=[]
    for i in indices:
        candidate=items[i]
        cterms=terms(candidate.chunk.text)
        new_query=len((cterms&query_terms)-selected_terms)/max(1,len(query_terms))
        total=sum(tok[i].values())
        new_5grams=sum(max(0,val-selected_grams.get(k,0)) for k,val in tok[i].items())/max(1,total)
        max_lex=0.
        max_sem=0.
        max_blocks=0.
        for j in selected:
            other=items[j]
            ot=terms(other.chunk.text)
            max_lex=max(max_lex,len(cterms&ot)/max(1,len(cterms|ot)))
            max_sem=max(max_sem,max(0.,cosine(
                candidate.embedding or [],other.embedding or [])))
            blocks_a=set(candidate.chunk.block_ids)
            blocks_b=set(other.chunk.block_ids)
            if blocks_a and blocks_b:
                max_blocks=max(max_blocks,len(blocks_a&blocks_b)/
                               max(1,min(len(blocks_a),len(blocks_b))))
        contextual=[
            math.log1p(candidate.weight),
            ratio,
            candidate.weight/max(1,budget),
            len(selected)/max(1,n),
            selected_weight/max(1,budget),
            (sum(items[j].value for j in selected)/len(selected)) if selected else 0.,
            new_query,
            new_5grams,
            max_lex,
            max_sem,
            max_blocks,
        ]
        rows.append([*q.feature_matrix[i].tolist(),*contextual])
    return np.asarray(rows,dtype=np.float32)


def marginal_gain_targets(q,indices,selected,precomputed=None):
    """TRAIN only: max increase in one valid alternative gold group recall."""
    groups=[gold_counts(g)[0] for g in q.evidence_groups]
    groups=[g for g in groups if g]
    if not groups:raise ValueError("Missing gold 5grams for train")
    grams=precomputed if precomputed is not None else [
        gram_counts(x.chunk.text) for x in q.scored]
    current=Counter()
    for j in selected:current.update(grams[j])
    y=[]
    for i in indices:
        candidate=current+grams[i]
        gains=[
            max(0,matched_count(candidate,gold)-matched_count(current,gold))/
            sum(gold.values())
            for gold in groups
        ]
        y.append(max(gains))
    return np.asarray(y,dtype=np.float32)


def training_states(q):
    """All chosen prefixes are constructed from gold-blind manual scores.

    The response label may use gold, but state feature construction does not.
    """
    indices=sorted(range(len(q.scored)),
                   key=lambda i:(-q.scored[i].value,
                                 q.scored[i].chunk.order_key,i))
    return ((),tuple(indices[:1]),tuple(indices[:3]))
