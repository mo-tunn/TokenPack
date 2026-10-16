"""Pre-registered Faz19: soft cross-document bridge exception, token matched.

All decisions made on new, gold-unseen 2Wiki public validation IDs.
This pilot separates the obvious token-length confound of Phase18.
No theorem that a Qwen-positive citation is semantically entailed.
"""
SEED="TokenPack-phase19-soft-bridge-matched-20261012-v1"
PILOT_N=32
HOLDOUT_N=96
ARMS=("branch-lock","soft-bridge-matched")
GATE_NET_EM=3
STRUCTURAL_CHANGED_MIN=8
TOKEN_RATIO_TOLERANCE=.05
TOKEN_ABSOLUTE_TOLERANCE=8
# For *every question*, abs(new cl100k tokens - control actual tokens)
# <= max(8,ceil(0.05*control actual tokens)), else return the exact baseline
# context unchanged. Means a matched cap does not imply sufficient evidence.
BUDGET_RATIO=.25
MAX_BRIDGE_DOCS=2
# For each question, (A) anchor a maximal feasible set of positive
# Qwen cross-edge citations; (B) add up to TWO noncited original Wikipedia
# sources that have high graph edge relevance and title/mention bridge to
# a Qwen-positive citation source; preference first for candidates absent
# from original baseline, then present; (C) fill with high-relevance baseline
# sentences and cross-doc candidates under exact token parity & final cap.
# Qwen-rejected pool items penalized but NOT prohibited.
# This ranking and optional baseline fallback are frozen BEFORE inference.
MAX_PLANNER_CALLS=32
MAX_INITIAL_VERIFY_CALLS=32
MAX_RESCUE_CALLS=32
MAX_COHERENCE_CALLS=32
MAX_READER_CALLS=64
ESTIMATED_HOURLY_USD=2.60
PER_SESSION_EST_USD_GUARD=1.35
