# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 83 | 0.386 | 17646 | 0.000 | 0.000 | 4.140 | 4.140 | 5.715 | 1.00x | 0.000 |
| production-rag-50 | 83 | 0.410 | 8720 | 0.506 | 0.000 | 2.223 | 2.223 | 2.959 | 1.86x | 0.000 |
| similarity-knapsack-50 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| hybrid-greedy-50 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| hybrid-knapsack-50 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50 | 83 | 0.446 | 8731 | 0.506 | 0.000 | 2.196 | 2.196 | 2.959 | 1.89x | 0.000 |
| only-longllmlingua-rate050 | 83 | 0.366 | 8570 | 0.511 | 0.523 | 1.914 | 2.437 | 3.133 | 1.70x | 0.012 |
| tokenpack-50+longllmlingua-rate050 | 83 | 0.446 | 4468 | 0.746 | 0.261 | 0.800 | 1.060 | 1.394 | 3.90x | 0.000 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 82 | 9 | 66 | 7 |
| production-rag-50 | 82 | 13 | 60 | 9 |
| similarity-knapsack-50 | 0 | 0 | 0 | 0 |
| hybrid-greedy-50 | 0 | 0 | 0 | 0 |
| hybrid-knapsack-50 | 0 | 0 | 0 | 0 |
| tokenpack-50 | 82 | 12 | 65 | 5 |
| tokenpack-50+longllmlingua-rate050 | 82 | 9 | 71 | 2 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0 | 0 | 0 |
