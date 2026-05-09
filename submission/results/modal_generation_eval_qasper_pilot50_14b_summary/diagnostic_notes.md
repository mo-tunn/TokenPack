# QASPER 50-Question Generation Pilot Diagnostics

Run: `submission/results/modal_generation_eval_qasper_pilot50/qasper_generation_judged_14b.jsonl`

Model/judge: `Qwen/Qwen2.5-14B-Instruct`

Rows: 200 answers, 50 questions, 4 methods.

Judge parse failures: 0.

## Main Result

The 50-question pilot does not support scaling directly to the 200-question generation-quality run as a main paper result. TokenPack is close to `budget-top-k-50`, but it is not ahead in this pilot.

| Method | Avg judge | Correct | Grounded | Hallucination | Insufficient |
|---|---:|---:|---:|---:|---:|
| Full document | 1.45 | 0.56 | 0.46 | 0.14 | 0.02 |
| Budget-top-k 50% | 1.18 | 0.52 | 0.42 | 0.18 | 0.18 |
| TokenPack 50% | 1.14 | 0.46 | 0.40 | 0.20 | 0.16 |
| Only LLMLingua-2 50% | 1.27 | 0.50 | 0.42 | 0.14 | 0.10 |

## Pairwise TokenPack Result

| Baseline | Wins | Ties | Losses | Mean score delta |
|---|---:|---:|---:|---:|
| Full document | 6 | 26 | 18 | -0.313 |
| Budget-top-k 50% | 6 | 36 | 8 | -0.040 |
| Only LLMLingua-2 50% | 11 | 23 | 16 | -0.133 |

Interpretation: TokenPack is nearly tied with `budget-top-k-50`, but the result is not positive enough to justify a full 200-question claim. Against LLMLingua-2, the pilot leans negative.

## Evidence-Retention Check

Approximate gold-evidence token recall in the supplied context:

| Method | Avg evidence recall | Cases >= 0.8 | Cases < 0.5 |
|---|---:|---:|---:|
| Full document | 1.000 | 50 | 0 |
| Budget-top-k 50% | 0.881 | 37 | 3 |
| TokenPack 50% | 0.879 | 37 | 3 |
| Only LLMLingua-2 50% | 0.754 | 9 | 0 |

Interpretation: TokenPack is not mainly failing because it drops gold evidence relative to `budget-top-k-50`. The recall is almost identical. Losses are more likely caused by context organization, distractor chunks, answer prompt sensitivity, or judge sensitivity.

## Loss Pattern

TokenPack losses vs `budget-top-k-50`: 8 cases.

Breakdown by evidence recall difference:

- Similar evidence recall: 6
- TokenPack less evidence: 2

TokenPack losses vs `Only LLMLingua-2`: 16 cases.

Breakdown by evidence recall difference:

- TokenPack more evidence but still lost: 6
- Similar evidence recall: 5
- TokenPack less evidence: 5

This points away from a simple evidence-retention failure. The stronger hypothesis is that TokenPack often retains enough relevant evidence, but generation is less robust when context order or neighboring chunks differ.

## Representative Failure

Case: `1910.10781::0810b43404686ddfe4ca84783477ae300fdd2ea4`

Question: "On top of BERT does the RNN layer work better or the transformer layer?"

Gold answer: "Transformer over BERT (ToBERT)"

TokenPack evidence recall: 0.83.

Budget-top-k evidence recall: 1.00.

TokenPack answer incorrectly says the context is insufficient, despite the context containing RoBERT/ToBERT evidence. This suggests the issue may be answer extraction from the packed context, not only retrieval.

## Recommended Next Step

Do not run the 200-question generation-quality eval yet.

Instead, run small targeted ablations on the same 50-question pilot:

1. TokenPack with evidence-/score-sorted output order instead of original/packed order.
2. TokenPack with a stricter redundancy penalty or smaller candidate pool to reduce distractors.
3. Answer prompt variant that says: "If the answer is present anywhere in the context, answer directly; do not say insufficient unless no relevant evidence appears."
4. Optional: judge with a different open model or a second shuffled judge pass before making any paper claim.

Acceptance criterion for continuing to 200 questions: TokenPack should at least match `budget-top-k-50` on average judge score and reduce losses against LLMLingua-2 on the 50-question pilot.
