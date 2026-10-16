"""Frozen research protocol for out-of-domain, no-training HotpotQA evaluation.

This protocol is meant to be reviewed before any HotpotQA answers/gold facts
are inspected. It is not a training or hyperparameter-tuning script.
"""
from __future__ import annotations

BENCHMARK="hotpotqa/hotpot_qa"
CONFIG="distractor"
SPLIT="validation"
N_CASES=512
SAMPLE_SEED="tokenpack-phase6-first-unseen-20261011"
BUDGET_RATIOS=(.25,.5)
METHODS=(
    "manual-650",
    "lambdamart-650",
    "manual-300",
    "msmarco-zero-shot-300",
)
CHUNK_CONFIG={
    650:{"target_tokens":650,"min_tokens":120,"max_tokens":900},
    300:{"target_tokens":300,"min_tokens":70,"max_tokens":420},
}
SEMANTIC_THRESHOLD=.35
SELECTOR="render-value"
BOOTSTRAP_REPLICATES=4000

# Gold-free sample selection: lowest hash values over question IDs.
# Never select or exclude based on answer, supporting fact, or retrieval result.
#
# More stringent primary metric (multi-hop ALL support):
# 1 only if every annotated supporting sentence is present contiguously
# in the FINAL RENDERED context (whitespace-normalized).
#
# Secondary metrics:
# * supporting fact sentence recall = fraction of all annotated supporting
#   sentences present in the final context;
# * BOTH support articles retained (title-matched sentences evidence).
# * "answer string present": factoid answers ONLY (exclude yes/no),
#   a weak extractive proxy, NOT answer correctness/sufficiency.
#
# Gold labels and answer strings are accessed ONLY after every arm's
# contexts have been selected for a question. They never enter features,
# chunking or selection.
#
# The benchmark uses PROVIDED 10 distractor contexts per HotpotQA question,
# not full-Wikipedia retriever search. QASPER and HotpotQA have different
# context distributions; compare WITHIN HotpotQA as an out-of-domain check.
#
# Important: This is the HotpotQA publicly labeled validation split, not
# hidden test leaderboard. It has never been used for model tuning in this
# research; methods are fixed before viewing any labels.
