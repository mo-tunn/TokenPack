# LongBench v2 Coverage Selector Ablation

## Modal Runs

- Generation: https://modal.com/apps/metehankizilcik09/main/ap-YrR9VzP39eFIG6X1OAAqwn
- Grounded probe: https://modal.com/apps/metehankizilcik09/main/ap-NrxvNBXvu7cNYe1ft8tte8

## Generation Accuracy

The `knapsack-coverage` selector was tested on the same 30-case LongBench v2 window used by the earlier pilot.

| Setting | TokenPack-50 acc. | Avg context toks | Saving | Cascade acc. | Cascade saving |
|---|---:|---:|---:|---:|---:|
| evidence-hybrid + knapsack-redundancy | 0.400 | 8750 | 0.504 | 0.300 | 0.746 |
| query-support + knapsack-redundancy | 0.367 | 8778 | 0.502 | 0.333 | 0.745 |
| evidence-hybrid + knapsack-coverage | 0.367 | 8631 | 0.512 | 0.267 | 0.749 |

Interpretation: `knapsack-coverage` does not improve answer accuracy. It saves slightly more tokens, but TokenPack-50 drops one correct answer relative to the current `knapsack-redundancy` default, and the TokenPack+LongLLMLingua cascade drops further.

## Grounded Probe

The grounded probe used the stricter grounded prompt and split groundedness metrics, so it should be compared within this run rather than directly against older grounded runs.

| Method | Acc. | Grounded acc. | Unsupported claims | Strict grounding failure | Quote found | Saving |
|---|---:|---:|---:|---:|---:|---:|
| full-context | 0.450 | 0.300 | 0.050 | 0.450 | 0.550 | 0.000 |
| tokenpack-50 | 0.350 | 0.150 | 0.000 | 0.500 | 0.500 | 0.510 |
| longllmlingua-50 | 0.350 | 0.250 | 0.100 | 0.500 | 0.500 | 0.515 |
| tokenpack-50 + longllmlingua-50 | 0.350 | 0.250 | 0.100 | 0.400 | 0.650 | 0.749 |

Interpretation: TokenPack-50 has the lowest unsupported-claim rate in this stricter run, but it still trails LongLLMLingua and the cascade on grounded accuracy. The cascade gives the best grounded/cost tradeoff in this 20-case probe.

## Decision

Do not promote `knapsack-coverage` to the main TokenPack mode. Keep `evidence-hybrid + knapsack-redundancy` as the primary setting for accuracy retention, and keep `knapsack-coverage` as an experimental groundedness ablation.
