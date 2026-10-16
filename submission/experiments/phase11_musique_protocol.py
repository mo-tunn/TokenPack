"""Preregistered Phase11 protocol: question-only Qwen planner -> multi-query rerank.
No gold in planning, ranking, rendering, or remote prompts.
Prior Phase9 256 and Phase10 128 IDs are excluded before hash sampling.
"""
DATA_SEED="TokenPack_phase11_musique_question_planning_v1_20261012"
PILOT_N=32
VALIDATION_N=96
BUDGET_RATIO=.25
METHODS=("crossdoc-bridge","hop-planner")
# Algorithm parameters frozen ahead of model calls/evaluation:
MAX_PLANNED_QUERIES=3
ANCHOR_BUDGET_SHARE=.40
QUERY_WEIGHT=.52
ORIGINAL_WEIGHT=.48
DOC_REPEAT_PENALTY=.10
CROSSDOC_LINK_BONUS=.10
MAX_CANDIDATE_SCORE_INPUT=256
RANK_BATCH_SIZE=24
PILOT_GO_NET_EM=3
PLANNER_MAX_REMOTE_CALLS=32
READER_PILOT_MAX_REMOTE_CALLS=64
MAX_SESSION_SECONDS=1800
MAX_ESTIMATED_SESSION_USD=1.35
ESTIMATED_USD_HOUR=2.60
PLANNER_SYSTEM=(
    "You plan evidence retrieval for multi-hop questions. Do not answer the question. "
    "Return exactly a JSON array of three short search queries, each no more than 18 words. "
    "Queries should target different possible pieces of evidence needed to answer, "
    "not three paraphrases of the same question. No explanations, no markdown, "
    "no answer statements, no gold labels."
)
