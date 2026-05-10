# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 83 | 0.386 | 17646 | 0.000 | 0.000 | 4.106 | 4.106 | 5.733 | 1.00x | 0.000 |
| tokenpack-50 | 83 | 0.386 | 8758 | 0.504 | 0.019 | 2.166 | 2.184 | 2.952 | 1.88x | 0.000 |
| only-longllmlingua-rate050 | 83 | 0.402 | 8560 | 0.512 | 0.537 | 1.895 | 2.432 | 3.172 | 1.69x | 0.012 |
| tokenpack-50+longllmlingua-rate050 | 83 | 0.410 | 4486 | 0.745 | 0.289 | 0.835 | 1.124 | 1.441 | 3.65x | 0.000 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 82 | 8 | 65 | 9 |
| tokenpack-50 | 82 | 10 | 61 | 11 |
| tokenpack-50+longllmlingua-rate050 | 82 | 5 | 73 | 4 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0 | 0 | 0 |
