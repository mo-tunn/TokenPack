# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 2 | 0.000 | 74938 | 0.000 | 0.000 | 8.677 | 8.677 | 9.631 | 1.00x | 0.000 |
| production-rag-50 | 2 | 0.500 | 37402 | 0.501 | 0.000 | 3.549 | 3.549 | 3.951 | 2.45x | 0.000 |
| tokenpack-50 | 2 | 0.000 | 37460 | 0.500 | 0.346 | 3.539 | 3.884 | 4.351 | 2.23x | 0.000 |
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
