"""Phase21 frozen reader-only intervention on best-supported Faz14 edge-verified baseline.

Gold-blind exact same contexts to both reader arms. Does NOT optimize
retrieval/selection on prior exposed gold. The novelty is explicit,
source-citation-grounded multi-hop intermediate reasoning before final answer.
"""
SEED="TokenPack-phase20-2Wiki-evidence-first-reader-20261012-v1"
PILOT_N=32
VALIDATION_N=96
ARMS=("verified-direct","verified-evidence-first")
GATE_NET_EM=3
MAX_CONTEXT_CAP=.25
MAX_PLANNER_CALLS=32
MAX_VERIFIER_CALLS=32
MAX_READER_CALLS=64
ESTIMATED_USD_PER_HOUR=2.60
ESTIMATED_SESSION_USD_GUARD=1.35
EVIDENCE_FIRST_SYSTEM=(
    "You answer multi-hop questions using ONLY the supplied original Wikipedia "
    "source sentences. First build a SHORT explicit multi-hop evidence chain: "
    "each step links an entity to a requested property, and you must cite a "
    "short exact phrase present in the supplied context. Do not invent "
    "supporting source facts, dates or intermediate links. If a step is "
    "unavailable, state 'missing evidence' in its step instead of guessing. "
    "Return ONLY a valid JSON object with keys 'steps' (an array of at most "
    "four objects with 'fact' and 'quote' strings) and 'answer' (a brief "
    "string giving the final answer). The answer should be short, not a "
    "sentence. If insufficient evidence, answer should be an empty string. "
    "Do not include markdown fences or text outside JSON."
)
EVIDENCE_FIRST_USER_TEMPLATE=(
    "Original source sentences:\n{context}\n\n"
    "Multi-hop question: {question}\n\n"
    "Verify intermediate relations against original quotes, then give the "
    "short final answer as JSON."
)
STRUCTURED_MAX_TOKENS=480
BASELINE_MAX_TOKENS=64
# Extra output tokens for evidence first are a deliberate aspect of this
# reader method; inferential comparison is an ensemble of prompt and
# generation budget, NOT equal computation.
