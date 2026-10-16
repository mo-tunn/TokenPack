"""Faz16 preregistered edge-complete hard evidence packing pilot."""
SEED="TokenPack-phase16-2Wiki-branch-lock-20261012-v1"
PILOT_N=32
VALIDATION_N=96
ARMS=("conditional-rescue","branch-lock")
PILOT_GATE_EM_NET=3
TOKEN_CAP_RATIO=.25
EDGE_ALTERNATIVES=3
# Strict priority under budget: (1) number of fully represented branches,
# (2) graph edges represented, (3) verified positive edges kept,
# (4) MS MARCO rank, (5) smaller render cost.
# For unverified edges, highest scored original/expanded candidate can be used
# but MUST be explicitly labelled provisional (NOT verified evidence).
MAX_GRAPH_EDGES=4
MAX_PLANNER_CALLS=32
MAX_INITIAL_VERIFIER_CALLS=32
MAX_RESCUE_VERIFIER_CALLS=32
MAX_READER_CALLS=64
MAX_ESTIMATED_SESSION_USD=1.35
ESTIMATED_HOURLY_USD=2.60
