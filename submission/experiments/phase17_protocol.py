"""Faz17 fixed question-to-source cross-edge consistency protocol.
The LLM judgment is NOT a verified entailment relation or gold ground truth.
"""
SEED="TokenPack-phase17-2Wiki-cross-edge-coherence-20261012-v1"
PILOT_N=32
HOLDOUT_N=96
ARMS=("branch-lock","coherence-gated")
GATE_NET_EM=3
CAP_RATIO=.25
MAX_GRAPH_EDGES=4
TOP_CANDIDATES_PER_EDGE=3
MAX_PLANNER_CALLS=32
MAX_INITIAL_VERIFY_CALLS=32
MAX_RESCUE_VERIFY_CALLS=32
MAX_CONSISTENCY_CALLS=32
MAX_READER_CALLS=64
MAX_SECONDS=1800
ESTIMATED_HOURLY_USD=2.60
MAX_ESTIMATED_USD=1.35
COHERENCE_SYSTEM=(
    "You audit SOURCE-GROUNDED evidence links, not answer the question. "
    "Given a question, hypothesized relation graph, and numbered ORIGINAL "
    "Wikipedia source sentences for each edge, select at most one supplied "
    "candidate sentence per relation edge, ensuring that adjacent edges in "
    "each branch refer to the SAME intermediate entity and do not contradict "
    "one another. Treat a mere name mention as INSUFFICIENT; direct relation "
    "support is required. Do not fill missing evidence from prior knowledge. "
    "If a relation lacks direct evidence or the proposed graph edges contradict, "
    "return null for unsupported edges. Never guess the final answer. "
    "Output ONLY compact JSON with property 'decisions', array with exactly "
    "one entry per input edge: {'edge_id':'e0','candidate_id':'e0_c0' or null}. "
    "Use only provided candidate_id values; no explanations or extra fields."
)
