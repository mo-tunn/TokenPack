"""Faz14 immutable experiment constants: gold-blind model-judged relation evidence.
Warning: Qwen verification is a model judgement, NOT formal NLI proof.
"""
PHASE="phase14"
SALT="TokenPack-phase14-2wiki-20261012-edge-citation-verification-v1"
PILOT_N=32
VALIDATION_N=96
EDGE_TOP_K=4
EDGE_MAX_COUNT=4
BUDGET_RATIO=.25
EDGE_ANCHOR_SHARE=.78
MAX_VERIFIER_OUTPUT_TOKENS=330
MAX_PLAN_OUTPUT_TOKENS=520
VERIFIER_SYSTEM=(
    "You are a STRICT evidence verifier, NOT a question answering assistant. "
    "You will receive a question, hypothesized relation edges, and numbered "
    "ORIGINAL source sentences for each edge. Determine which candidate sentence "
    "DIRECTLY supports the relation described by the edge, based on its text. "
    "Do not infer facts from unrelated world knowledge. When the object is unknown, "
    "the candidate must explicitly identify or link the requested subject and relation. "
    "A name appearing in the text is not enough. "
    "Return ONLY JSON object with key 'decisions' containing exactly one object per "
    "edge with keys 'edge_id' (string) and 'candidate_id' (string or null). "
    "For each edge, select at most one provided candidate ID, or null if none "
    "directly supports the edge; do NOT invent IDs or explanations."
)
ARMS=("flat-edge","graph-path","edge-verified")
GATE_NET_EM=3
MAX_PLANNER_CALLS=32
MAX_VERIFIER_CALLS=32
MAX_READER_CALLS=96
ESTIMATED_HOUR_USD=2.60
MAX_SESSION_ESTIMATED_USD=1.35
MAX_SESSION_SECONDS=1800
