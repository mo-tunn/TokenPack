# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 30 | 0.400 | 43704 | 0.000 | 0.000 | 13.580 | 13.580 | 17.257 | 1.00x | 0.000 |
| production-rag-50 | 30 | 0.433 | 21770 | 0.502 | 0.000 | 6.440 | 6.440 | 8.161 | 2.11x | 0.000 |
| similarity-knapsack-50 | 30 | 0.300 | 21813 | 0.501 | 0.111 | 6.773 | 6.883 | 8.703 | 1.97x | 0.000 |
| hybrid-greedy-50 | 30 | 0.500 | 21767 | 0.502 | 0.000 | 6.313 | 6.313 | 8.028 | 2.15x | 0.000 |
| hybrid-knapsack-50 | 30 | 0.367 | 21803 | 0.501 | 0.180 | 6.538 | 6.718 | 8.458 | 2.02x | 0.000 |
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
