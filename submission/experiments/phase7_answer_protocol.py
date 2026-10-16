"""Preregistered Phase7 reader QA evaluation, frozen before gold inspection.

Pre-existing HotpotQA 512-ID sample; new deterministic ID-only subsample.
Single reader Qwen2.5-0.5B-Instruct frozen across 5 conditions:
full supplied distractor source, manual TokenPack at .25/.50, MS MARCO
zero-shot 300-token selector at .25/.50.

Primary: official HotpotQA answer normalized exact match (EM).
Secondary: official HotpotQA answer token F1.  Paired question bootstrap
5000 replicates; no prompt selection, training or decoding tuning on gold.
No QASPER test access, no paid API, no production edits.
"""

SOURCE_QUESTION_IDS="submission/results/phase6_hotpotqa_preregistered512_20261011/question_ids.txt"
SAMPLE_COUNT=256
SAMPLE_SALT="TokenPack_phase7_reader_id_only_20261012_v1"
BUDGET_RATIOS=(.25,.50)
ARMS=("full-source","manual-25","msmarco-25","manual-50","msmarco-50")
READER_ID="Qwen/Qwen2.5-0.5B-Instruct"
READER_REVISION="7ae557604adf67be50417f59c2c2f167def9a775"
READER_MAX_NEW_TOKENS=32
READER_DO_SAMPLE=False
READER_MAX_INPUT_TOKENS=4096
READER_PROMPT=(
    "Answer the question using only the supplied context. "
    "Reply with only the short answer (a name, phrase, number, yes or no). "
    "If the context does not contain enough information, reply unknown."
)
BOOTSTRAP_REPLICATES=5000
