"""Pre-registered (before MuSiQue gold inspection) external benchmark protocol.

Data: awinml/musique validation ONLY, pinned revision and parquet SHA256.
No training, no test split, no production updates.
Sample 256 distinct cases by smallest SHA256 of (seed + question ID).
Include all 2/3/4-hop categories without filtering by answerability or gold.
Inference accesses id, question, paragraph idx/title/paragraph_text ONLY:
ParquetFile columnar leaf-level projection strictly excludes is_supporting,
question_decomposition, answer and answerable.

Methods: original Phase8 unadapted MS MARCO atomic sentence ranking, and
new gold-blind cross-document lexical bridge pair + diminishing-doc
returns graph selector (different score policy, SAME source rendering).
The previous 300-token chunk method is not necessary for isolating the
bridge's incremental value; we report only comparable two atomic arms.
Hard budget: floor(0.25*sum cl100k source paragraph tokens), exact final
rendered count <= cap for every case and method.

Primary gold-local retrieval: all supporting PARAGRAPH indices have >=1
selected sentence (not full paragraph evidence!).
Secondary: per-support-paragraph hit fraction, answer span presence.
Gold read ONLY when selected contexts and SHA256 are frozen.
Unseen dataset relative to TokenPack's development, but publicly
labeled validation—not a hidden test. No tuning based on this dataset.

Reader: only if downstream is justified; run Qwen27B on 64 ID-only selected
questions, 3 arms (full, old atomic, bridge) with existing fixed prompt.
Reader EM primary; no external invoice hard cap available.
"""

DATA_REPO="awinml/musique"
DATA_REV="3ac762478df3609852f18dd33652a16820660a5e"
DATA_SHA256="f3b7c998962b0282479ddf58d2ea168f692adac391b88c03c87f66df3dcdd94a"
DATA_SPLIT="validation"
EXPECTED_DATASET_ROWS=2417
SAMPLE_SEED="TokenPack_phase9_MuSiQue_independent_validation_ids_v1_20261012"
N_QUESTIONS=256
BUDGET_RATIO=0.25
METHODS=("atomic-rank","crossdoc-bridge")
READER_SAMPLES=64
MAX_SCORE_INPUT_TOKENS=256
RANK_BATCH_SIZE=24
PAIR_CANDIDATE_POOL=48
PAIR_LINK_BONUS=.32
DOC_DIVERSITY_DISCOUNT=.12
LINKED_DOC_BONUS=.22
MIN_SEED_RELEVANCE=.20
MAX_PAIR_VS_UNLINKED_LOSS=.10
