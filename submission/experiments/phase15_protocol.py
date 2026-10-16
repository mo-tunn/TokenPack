"""Frozen Faz15 design. Source-only targeted retrieval after verifier abstention."""
SALT="tokenpack-phase15-2wiki-conditional-edge-rescue-20261012-v1"
PILOT_N=32
HOLDOUT_N=96
METHODS=("top4-verified","conditional-rescue")
EDGE_TOP_K=4
EXPANSION_TOP_K=4
MAX_EDGE_COUNT=4
ORIGINAL_WEIGHT=.15
EDGE_WEIGHT=.60
REFORM_WEIGHT=.25
EXPANSION_UNIQUE_DOC_PRIORITY=True
TOKEN_CAP_RATIO=.25
ANCHOR_SHARE=.78
GATE_NET_EM=3
MAX_PLANNER=32
MAX_INITIAL_VERIFY=32
MAX_RESCUE_VERIFY=32
MAX_READER=64
ESTIMATED_HOUR_USD=2.6
MAX_EST_SESSION_USD=1.35
MAX_EST_SESSION_SECONDS=1800
# Existing Phase13 question-to-graph planner and Phase14 source-only
# verifier reused byte-for-byte in the calls.
