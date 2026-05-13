# LongBench v2 Modal Production-RAG Baseline Summary

Date: 2026-05-12

Model: Qwen/Qwen2.5-14B-Instruct via vLLM on Modal.

Embedding: sentence-transformers, downloaded on Modal cache when needed.

Methods:

- `full-context`: full source context inside the selected token window.
- `production-rag-50`: raw embedding similarity top chunks, greedy-packed to 50% of source tokens.
- `tokenpack-50`: historical rows before the hybrid-greedy pivot used evidence-hybrid scoring plus knapsack-redundancy selection; the 83-case rerun below uses `budget-top-k` as TokenPack hybrid-greedy.

## 64K Pilot

Run URL: https://modal.com/apps/metehankizilcik09/main/ap-wXT8Jsjah2ZLfjq6NjKO93

Output directory: `submission/results/longbench_v2_modal_64k_pilot_32k58k_production_rag`

Window: 32K-58K source tokens, 30 cases, 90 generations.

| Method | Accuracy | Avg Context Tokens | Saving | Avg Total Latency | Speedup |
|---|---:|---:|---:|---:|---:|
| full-context | 0.400 | 43,704 | 0.000 | 13.279s | 1.00x |
| production-rag-50 | 0.433 | 21,770 | 0.502 | 6.314s | 2.10x |
| tokenpack-50 | 0.400 | 21,803 | 0.501 | 6.691s | 1.98x |

Pairwise vs `production-rag-50`:

| Method | Compared | Wins | Ties | Losses |
|---|---:|---:|---:|---:|
| full-context | 30 | 4 | 21 | 5 |
| tokenpack-50 | 30 | 4 | 21 | 5 |

Interpretation: TokenPack preserved full-context accuracy while using about half the context and nearly halving latency. In this pilot it did not beat the strong production-RAG baseline; production-RAG was slightly ahead.

## 128K Smoke

Run URL: https://modal.com/apps/metehankizilcik09/main/ap-xll1mDwFbXwhvEFRUvO1KQ

Output directory: `submission/results/longbench_v2_modal_128k_smoke_production_rag`

Window: 64K-112K source tokens, 2 cases, 6 generations.

| Method | Accuracy | Avg Context Tokens | Saving | Avg Total Latency | Speedup |
|---|---:|---:|---:|---:|---:|
| full-context | 0.000 | 74,938 | 0.000 | 8.677s | 1.00x |
| production-rag-50 | 0.500 | 37,402 | 0.501 | 3.549s | 2.45x |
| tokenpack-50 | 0.000 | 37,460 | 0.500 | 3.884s | 2.23x |

Interpretation: This is only a feasibility smoke test. It confirms Modal H100 + 131K max context + YaRN configuration runs successfully.

## 128K Micro

Run URL: https://modal.com/apps/metehankizilcik09/main/ap-uWIGCHZ2tW5UA1MKSos16U

Output directory: `submission/results/longbench_v2_modal_128k_micro_production_rag`

Window: 64K-112K source tokens, 8 cases, 24 generations.

| Method | Accuracy | Avg Context Tokens | Saving | Avg Total Latency | Speedup |
|---|---:|---:|---:|---:|---:|
| full-context | 0.375 | 78,844 | 0.000 | 9.618s | 1.00x |
| production-rag-50 | 0.500 | 39,300 | 0.501 | 4.144s | 2.32x |
| tokenpack-50 | 0.500 | 39,340 | 0.501 | 5.266s | 1.83x |

Pairwise vs `production-rag-50`:

| Method | Compared | Wins | Ties | Losses |
|---|---:|---:|---:|---:|
| full-context | 8 | 0 | 7 | 1 |
| tokenpack-50 | 8 | 1 | 6 | 1 |

Interpretation: At 128K-scale contexts, both compressed methods beat full-context accuracy in this small sample while using about half the tokens. TokenPack tied production-RAG in aggregate accuracy, but production-RAG was faster because it avoids TokenPack preprocessing.

## Current Takeaway

The production-RAG baseline is strong, but the follow-up diagnostic changed the main pipeline decision: evidence-hybrid scoring works best here when paired with greedy budget filling rather than the older knapsack-redundancy selector. The current paper therefore uses `budget-top-k` as TokenPack hybrid-greedy.

## Selector Diagnostic

Date: 2026-05-13

Purpose: isolate whether the loss against `production-rag-50` comes from scoring or from the selector.

Diagnostic methods:

- `production-rag-50`: raw similarity + greedy budget fill.
- `similarity-knapsack-50`: raw similarity as value + knapsack.
- `hybrid-greedy-50`: evidence-hybrid value + greedy budget fill.
- `hybrid-knapsack-50`: evidence-hybrid value + knapsack-redundancy.

### 64K Diagnostic

Run URL: https://modal.com/apps/metehankizilcik09/main/ap-dQaxFYRnnDu7WfGWtkTSa0

Output directory: `submission/results/longbench_v2_modal_64k_diagnostic_selectors`

Window: 32K-58K source tokens, 30 cases, 150 generations.

| Method | Accuracy | Correct | Avg Context Tokens | Saving | Avg Total Latency | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| full-context | 0.400 | 12/30 | 43,704 | 0.000 | 13.580s | 1.00x |
| production-rag-50 | 0.433 | 13/30 | 21,770 | 0.502 | 6.440s | 2.11x |
| similarity-knapsack-50 | 0.300 | 9/30 | 21,813 | 0.501 | 6.883s | 1.97x |
| hybrid-greedy-50 | 0.500 | 15/30 | 21,767 | 0.502 | 6.313s | 2.15x |
| hybrid-knapsack-50 | 0.367 | 11/30 | 21,803 | 0.501 | 6.718s | 2.02x |

Pairwise vs `production-rag-50`:

| Method | Wins | Ties | Losses |
|---|---:|---:|---:|
| similarity-knapsack-50 | 0 | 26 | 4 |
| hybrid-greedy-50 | 4 | 24 | 2 |
| hybrid-knapsack-50 | 3 | 22 | 5 |

Interpretation: This points strongly at the selector layer. Evidence-hybrid scoring is useful when used greedily, but knapsack/redundancy hurts on this 64K pilot. Raw-similarity knapsack is especially weak, so the problem is not simply that raw similarity is better.

### 128K Diagnostic

Run URL: https://modal.com/apps/metehankizilcik09/main/ap-ny9mla49DrOz1mZmheO5K5

Output directory: `submission/results/longbench_v2_modal_128k_diagnostic_selectors`

Window: 64K-112K source tokens, 8 cases, 40 generations.

| Method | Accuracy | Correct | Avg Context Tokens | Saving | Avg Total Latency | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| full-context | 0.375 | 3/8 | 78,844 | 0.000 | 9.317s | 1.00x |
| production-rag-50 | 0.500 | 4/8 | 39,300 | 0.501 | 4.049s | 2.30x |
| similarity-knapsack-50 | 0.375 | 3/8 | 39,401 | 0.500 | 4.530s | 2.06x |
| hybrid-greedy-50 | 0.375 | 3/8 | 39,290 | 0.501 | 3.991s | 2.33x |
| hybrid-knapsack-50 | 0.500 | 4/8 | 39,340 | 0.501 | 4.666s | 2.00x |

Pairwise vs `production-rag-50`:

| Method | Wins | Ties | Losses |
|---|---:|---:|---:|
| similarity-knapsack-50 | 0 | 7 | 1 |
| hybrid-greedy-50 | 0 | 7 | 1 |
| hybrid-knapsack-50 | 1 | 6 | 1 |

Interpretation: The 128K sample is too small for a stable ranking, but it does not rescue raw-similarity knapsack. The most actionable result remains the 64K diagnostic: keep evidence-hybrid, reconsider knapsack-redundancy as the default selector for LongBench-style generation.

## 83-Case Hybrid-Greedy Rerun

Date: 2026-05-13

Run URL: https://modal.com/apps/metehankizilcik09/main/ap-OmDEeJ73KkJsFOr4mblKRl

Output directory: `submission/results/longbench_v2_modal_hybrid_greedy_83_latency`

Window: 8K-24K source tokens, 83 eligible cases, 415 generations. This run uses `budget-top-k` as TokenPack hybrid-greedy and records batch-size-one hot-model latency.

| Method | Accuracy | Avg Context Tokens | Saving | Avg Total Latency | Speedup |
|---|---:|---:|---:|---:|---:|
| full-context | 0.386 | 17,646 | 0.000 | 4.140s | 1.00x |
| production-rag-50 | 0.410 | 8,720 | 0.506 | 2.223s | 1.86x |
| tokenpack-50 | 0.446 | 8,731 | 0.506 | 2.196s | 1.89x |
| only-longllmlingua-rate050 | 0.366 | 8,570 | 0.511 | 2.437s | 1.70x |
| tokenpack-50+longllmlingua-rate050 | 0.446 | 4,468 | 0.746 | 1.060s | 3.90x |

Pairwise vs `production-rag-50`:

| Method | Compared | Wins | Ties | Losses |
|---|---:|---:|---:|---:|
| full-context | 83 | 10 | 61 | 12 |
| tokenpack-50 | 83 | 8 | 70 | 5 |
| only-longllmlingua-rate050 | 82 | 9 | 60 | 13 |
| tokenpack-50+longllmlingua-rate050 | 83 | 10 | 66 | 7 |

Interpretation: this rerun supports the paper pivot. Hybrid-greedy is now the main TokenPack operating point: it beats production-RAG in aggregate accuracy on this 83-case pilot while using the same budget scale, and the selection-then-compression cascade keeps that accuracy while cutting average context to 4,468 tokens.
