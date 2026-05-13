# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 3 | 0.667 | 46853 | 0.000 | 0.000 | 9.105 | 9.105 | 9.105 | 1.00x | 0.000 |
| production-rag-50 | 3 | 0.333 | 23246 | 0.504 | 0.000 | 9.105 | 9.105 | 9.105 | 1.00x | 0.000 |
| tokenpack-50 | 3 | 0.333 | 23237 | 0.505 | 0.120 | 9.105 | 9.225 | 9.301 | 0.99x | 0.000 |
| only-longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00x | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 0 | 0 | 0 | 0 |
| production-rag-50 | 0 | 0 | 0 | 0 |
| tokenpack-50 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0 | 0 | 0 |
