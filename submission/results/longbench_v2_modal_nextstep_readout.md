# LongBench v2 Next-Step Ablation Readout

30-case pilot, Qwen/Qwen2.5-14B-Instruct, LongBench v2 8k-24k source-token window.

## Main Finding

The strongest new signal is context ordering, not learned reranking.

`score-then-source` improves TokenPack-50 from the previous `0.400` accuracy to `0.433` while preserving the same average token saving (`0.504`). It also beats LongLLMLingua more often pairwise: `5 win / 23 tie / 2 loss` in the dedicated ordering run, and `6 win / 21 tie / 3 loss` in the cascade-frontier run.

## Ordering Ablation

| Variant | TokenPack-50 acc. | Saving | Pairwise vs LLL |
|---|---:|---:|---:|
| score baseline | 0.400 | 0.504 | 4 win / 24 tie / 2 loss |
| source | 0.367 | 0.504 | 3 win / 25 tie / 2 loss |
| score-then-source | 0.433 | 0.504 | 5 win / 23 tie / 2 loss |

Interpretation: preserving pure document order hurts. Keeping a small priority evidence block first, then rendering the remainder in source order is the best candidate for the next 100-case run.

## Learned Reranker Ablation

| Variant | TokenPack-50 acc. | Saving | Pairwise vs LLL |
|---|---:|---:|---:|
| BGE cross-encoder | 0.400 | 0.503 | 4 win / 24 tie / 2 loss |

Interpretation: the BGE reranker did not improve the 30-case result over baseline and underperformed `score-then-source`. Keep it experimental unless error analysis shows it fixes specific LongLLMLingua-loss cases.

## Cascade Frontier

| Variant | Accuracy | Avg context tokens | Saving | Pairwise vs LLL |
|---|---:|---:|---:|---:|
| TP50 + LLL50 | 0.300 | 4454 | 0.746 | 1 win / 27 tie / 2 loss |
| TP60 + LLL50 | 0.267 | 5284 | 0.699 | 0 win / 28 tie / 2 loss |
| TP50 + LLL65 | 0.333 | 5716 | 0.675 | 2 win / 26 tie / 2 loss |

Interpretation: cascade is useful as an aggressive compression frontier, not as the main quality claim. `TP50 + LLL65` is the least damaging cascade point in this pilot, but it still trails TokenPack-only.

## Recommendation

Run a 100-case pilot for:

```text
evidence-hybrid + knapsack-redundancy + context-order score-then-source
```

Do not promote BGE reranking or cascade as main paper claims yet. Mention cascade only as a token-saving frontier if the paper needs an aggressive-compression point.

## 83-Case Follow-Up

The planned 100-case `score-then-source` run yielded 83 eligible LongBench v2 cases after the 8k-24k token filter.

| Method | Runs | Accuracy | Avg context tokens | Saving | Parse fail |
|---|---:|---:|---:|---:|---:|
| full-context | 83 | 0.398 | 17646 | 0.000 | 0.000 |
| TokenPack-50, score-then-source | 83 | 0.386 | 8758 | 0.504 | 0.000 |
| LongLLMLingua-50 | 83 | 0.402 | 8560 | 0.512 | 0.012 |
| TokenPack-50 + LongLLMLingua-50 | 83 | 0.410 | 4486 | 0.745 | 0.000 |

Pairwise vs LongLLMLingua:

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 82 | 8 | 66 | 8 |
| TokenPack-50, score-then-source | 82 | 10 | 61 | 11 |
| TokenPack-50 + LongLLMLingua-50 | 82 | 5 | 73 | 4 |

Interpretation: the 30-case ordering gain did not fully hold on the larger eligible set. TokenPack-only remains near full-context accuracy with ~50% saving, but it does not beat LongLLMLingua on aggregate accuracy here. The strongest 83-case result is the cascade point: it slightly beats full-context and LongLLMLingua while saving ~74.5% of context tokens. This should be framed carefully as an aggressive compression result, not as proof that TokenPack-only dominates LongLLMLingua.
