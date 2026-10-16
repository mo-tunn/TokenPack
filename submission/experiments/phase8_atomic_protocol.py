"""Predefined research-only answer-readiness experiment (no label access).

Phase7 first 8 Qwen pilot IDs: indices 0..7.
Phase7 next 96 Qwen experiment IDs: indices 8..103.
Phase8 development *follow-up* IDs: indices 104..255 (152 questions).
All IDs were already available in frozen gold-blind Phase7 contexts.

Training: NONE. Model: pinned MS MARCO MiniLM-L6-v2, no adaptation.
Source: HotpotQA supplied 10 distractor Wikipedia passages per question.
Actual output cap: existing Phase7 25% cl100k_base per source.
Methods:
  1. original ms-marco-25 Phase7 output (frozen)
  2. atomic-sentence-rank (individual full source sentences + article title)
  3. atomic-bridge-first (force relevance-selected second distinct article
     when both first sentences fit; then continue sentence-level rank)

Both atomic methods output grouped source headings, NOT previous verbose
technical chunk headers. Consequently gains cannot be attributed to
candidate selection alone: formatting is part of the new pipeline.

Only AFTER all 152 contexts are written / SHA256 frozen, compare literal
full annotated supporting-sentence recall to saved Qwen experiment baseline.
No tuning or candidate selection based on gold.

Goal to justify extra paid Qwen experiment: prospective strict all-fact
recall +8.0 percentage points or more on 152 development cases, no
budget violations and no missing/synthetic source content.
This threshold is NOT a statistical-significance claim.
"""
EVAL_START=104
EVAL_STOP=256
EVAL_COUNT=152
BUDGET_RATIO=.25
ARMS=("msmarco-25","atomic-sentence-rank","atomic-bridge-first")
MSMARCO_MODEL="cross-encoder/ms-marco-MiniLM-L6-v2"
SCORER_MAX_INPUT_TOKENS=256
DOCUMENT_MAX_INPUT_TOKENS=512
BATCH_SIZE=16
READER_ADVANCE_THRESHOLD_PP=8.0
