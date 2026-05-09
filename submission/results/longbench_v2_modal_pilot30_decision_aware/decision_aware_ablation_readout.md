# LongBench v2 Decision-Aware Scoring Ablation

## Modal Run

- Generation: https://modal.com/apps/metehankizilcik09/main/ap-j0WbIfadeiuY9IaxmGThrs

## Result

The `decision-aware` scoring profile was tested on the same 30-case LongBench v2 window as the earlier pilots. It parses explicit candidate answers from the selection query and adds candidate-support and candidate-contrast signals to the evidence-hybrid base.

| Setting | TokenPack-50 acc. | Avg context toks | Saving | Cascade acc. | Cascade saving |
|---|---:|---:|---:|---:|---:|
| evidence-hybrid + knapsack-redundancy | 0.400 | 8750 | 0.504 | 0.300 | 0.746 |
| query-support + knapsack-redundancy | 0.367 | 8778 | 0.502 | 0.333 | 0.745 |
| decision-aware + knapsack-redundancy | 0.367 | 8774 | 0.502 | 0.333 | 0.745 |
| evidence-hybrid + knapsack-coverage | 0.367 | 8631 | 0.512 | 0.267 | 0.749 |

## Interpretation

The candidate-answer signals did not recover the lost accuracy. `decision-aware` matches the earlier `query-support` result but remains below the current `evidence-hybrid + knapsack-redundancy` default for TokenPack-50.

This suggests the current LongBench bottleneck is not solved by simple lexical candidate discrimination. The safer paper story remains:

- keep `evidence-hybrid + knapsack-redundancy` as the main TokenPack setting;
- keep `query-support`, `decision-aware`, and `knapsack-coverage` as experimental ablations;
- emphasize TokenPack's accuracy retention at about 50% token saving and the TokenPack+LongLLMLingua cascade for aggressive compression.
