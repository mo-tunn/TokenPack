"""Train-only, alternative-group-aligned marginal coverage targets.

The earlier Phase4C label was max_g [coverage_g(S+i) - coverage_g(S)].
But final QASPER evaluation is max_g coverage_g(S): distinct annotation
groups are ALTERNATIVES. The aligned target is therefore
    max_g coverage_g(S+i) - max_g coverage_g(S).
These objectives differ when the currently best annotation changes.

This file is opt-in research code. No gold information enters inference.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from submission.experiments.phase3a_oracle_core import (
    gold_counts, gram_counts, matched_count,
)


def aligned_marginal_gain_targets(q,indices,selected,precomputed=None):
    """Train label: increment to best *single* annotation group's 5gram recall."""
    gold_options=[gold_counts(group)[0] for group in q.evidence_groups]
    gold_options=[g for g in gold_options if g]
    if not gold_options:
        raise ValueError("No gold evidence 5grams")
    grams=precomputed if precomputed is not None else [
        gram_counts(x.chunk.text) for x in q.scored]
    existing=Counter()
    for j in selected:
        existing.update(grams[j])
    baseline=max(
        matched_count(existing,g)/sum(g.values()) for g in gold_options
    )
    labels=[]
    for i in indices:
        proposed=existing+grams[i]
        next_best=max(
            matched_count(proposed,g)/sum(g.values()) for g in gold_options
        )
        labels.append(max(0.,next_best-baseline))
    return np.asarray(labels,dtype=np.float32)
