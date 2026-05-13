# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 8 | 0.375 | 78844 | 0.000 | 0.000 | 9.317 | 9.317 | 11.565 | 1.00x | 0.000 |
| production-rag-50 | 8 | 0.500 | 39300 | 0.501 | 0.000 | 4.048 | 4.049 | 4.878 | 2.30x | 0.000 |
| similarity-knapsack-50 | 8 | 0.375 | 39401 | 0.500 | 0.313 | 4.217 | 4.530 | 5.590 | 2.06x | 0.000 |
| hybrid-greedy-50 | 8 | 0.375 | 39290 | 0.501 | 0.000 | 3.991 | 3.991 | 4.826 | 2.33x | 0.000 |
| hybrid-knapsack-50 | 8 | 0.500 | 39340 | 0.501 | 0.553 | 4.113 | 4.666 | 5.941 | 2.00x | 0.000 |
| tokenpack-50 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| only-longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 0 | 0 | 0 | 0 |
| production-rag-50 | 0 | 0 | 0 | 0 |
| similarity-knapsack-50 | 0 | 0 | 0 | 0 |
| hybrid-greedy-50 | 0 | 0 | 0 | 0 |
| hybrid-knapsack-50 | 0 | 0 | 0 | 0 |
| tokenpack-50 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0 | 0 | 0 |
