"""Faz18 preregistration: citation-closed evidence selection (no gold).
Primary scientific outcome: does directly acting on the cross-edge verdict
change the actual rendered evidence context and downstream reader EM?
"""
SEED="TokenPack-phase18-2Wiki-citation-closed-20261012-v1"
PILOT_N=32
VALIDATION_N=96
ARMS=("branch-lock","citation-closed")
CAP_RATIO=.25
STRUCTURAL_DIFFERENT_MIN=8
STRUCTURAL_DIFFERENT_PCT=STRUCTURAL_DIFFERENT_MIN/PILOT_N
GATE_NET_EM=3
# Selection: anchor every question-level Qwen-positive edge original sentence
# subject to exact cap; prohibit *all* candidate source sentences for edges
# that Qwen judged unsupported unless they ALSO represent a Qwen-approved edge.
# For filling, use only positive-anchor Wikipedia article IDs.
# Missing source article IDs never imply entailment. If no positive evidence
# on ANY edge, preserve branch-lock exactly and tag the fallback.
# Candidate list is unchanged and no new evidence comes from cloud.
MAX_PLANNER_CALLS=32
MAX_FIRST_VERIFIER_CALLS=32
MAX_RESCUE_CALLS=32
MAX_COHERENCE_CALLS=32
MAX_READER_CALLS=64
MAX_SESSION_SECONDS=1800
ESTIMATED_USD_PER_HOUR=2.60
SESSION_ESTIMATED_USD_GUARD=1.35
