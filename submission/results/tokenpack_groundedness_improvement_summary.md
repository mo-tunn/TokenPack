# TokenPack Groundedness Improvement Summary

## What changed

- The LongBench groundedness evaluator now separates narrow hallucination (`unsupported_claims`) from strict grounding failures such as missing quotes, unsupported rationales, or quotes that do not directly support the selected answer.
- The grounded answer prompt is stricter: it asks for a directly supported option, a short rationale, and one exact contiguous quote from the context.
- A new experimental selector, `knapsack-coverage`, was added. It keeps the redundancy-aware TokenPack idea but greedily favors complementary query-term coverage under the same token budget.

## Why this matters

The earlier single hallucination rate was too coarse. It counted several different failures together, which made full-context prompting look extremely hallucination-heavy and made it hard to tell whether TokenPack was failing because it selected bad context or because the generated answer/quote format was not cleanly grounded.

The new diagnostic split lets us read future runs as:

- `unsupported_claim_rate`: closest to true hallucination.
- `strict_grounding_failure_rate`: broader evidence/quote/rationale failure.
- `answer_supported_rate`: whether the selected answer is supported by context.
- `quote_supports_answer_rate`: whether the exact quote actually proves the answer.
- `quote_missing_rate`: whether the model failed to copy a usable quote.

## Current interpretation

TokenPack's strongest current claim remains accuracy retention under large token savings, not hallucination superiority over LongLLMLingua. Existing LongBench results show TokenPack-50 nearly preserves full-context accuracy at about 50% token saving, while the TokenPack+LongLLMLingua cascade reaches about 75% token saving with moderate accuracy loss.

The groundedness result should still be treated as preliminary, but reparsing the existing 20-case run with split metrics gives a clearer reading:

| Method | Accuracy | Grounded acc. | Unsupported claims | Strict grounding failure | Quote found |
|---|---:|---:|---:|---:|---:|
| full-context | 0.450 | 0.050 | 0.400 | 0.900 | 0.350 |
| tokenpack-50 | 0.350 | 0.050 | 0.300 | 0.750 | 0.450 |
| longllmlingua-50 | 0.350 | 0.150 | 0.150 | 0.500 | 0.600 |
| tokenpack-50 + longllmlingua-50 | 0.350 | 0.100 | 0.300 | 0.650 | 0.450 |

This suggests TokenPack improves over full-context on unsupported claims and quote grounding, but LongLLMLingua remains stronger on groundedness. The paper claim should therefore stay focused on token-budget allocation and accuracy retention, with groundedness framed as an active ablation axis.

## Next ablation to run

Use the same 30-case or 100-case LongBench subset and compare:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m modal run submission/longbench_eval/app.py::build_and_run --output-dir submission/results/longbench_v2_modal_pilot30_coverage --limit 30 --source-min-tokens 8000 --source-max-tokens 24000 --max-scanned 503 --batch-size 2 --selection-strategy knapsack-coverage
```

Then run groundedness on that task file and compare the split metrics against the existing `knapsack-redundancy` run.

## Coverage ablation result

The 30-case coverage ablation was run on Modal. It did not improve the main generation result:

| Setting | TokenPack-50 acc. | Saving | Cascade acc. | Cascade saving |
|---|---:|---:|---:|---:|
| evidence-hybrid + knapsack-redundancy | 0.400 | 0.504 | 0.300 | 0.746 |
| evidence-hybrid + knapsack-coverage | 0.367 | 0.512 | 0.267 | 0.749 |

The 20-case grounded probe with the stricter prompt looked better for unsupported claims, but not enough to make coverage the default:

| Method | Acc. | Grounded acc. | Unsupported claims | Strict grounding failure | Saving |
|---|---:|---:|---:|---:|---:|
| tokenpack-50 | 0.350 | 0.150 | 0.000 | 0.500 | 0.510 |
| longllmlingua-50 | 0.350 | 0.250 | 0.100 | 0.500 | 0.515 |
| tokenpack-50 + longllmlingua-50 | 0.350 | 0.250 | 0.100 | 0.400 | 0.749 |

Decision: keep `evidence-hybrid + knapsack-redundancy` as the primary setting. Keep `knapsack-coverage` as an experimental groundedness ablation.

## Decision-aware ablation result

The `decision-aware` scoring profile was also tested on the same 30-case LongBench v2 window. It parses explicit candidate answers and adds candidate-support/contrast signals to the evidence-hybrid base.

| Setting | TokenPack-50 acc. | Saving | Cascade acc. | Cascade saving |
|---|---:|---:|---:|---:|
| evidence-hybrid + knapsack-redundancy | 0.400 | 0.504 | 0.300 | 0.746 |
| query-support + knapsack-redundancy | 0.367 | 0.502 | 0.333 | 0.745 |
| decision-aware + knapsack-redundancy | 0.367 | 0.502 | 0.333 | 0.745 |

Decision: `decision-aware` is useful as an ablation, but it should not replace the main `evidence-hybrid` profile. The simple lexical candidate-contrast signal did not recover the lost TokenPack-50 accuracy.
