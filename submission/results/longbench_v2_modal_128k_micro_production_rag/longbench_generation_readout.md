# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Prep s | LLM s | Total s | P90 s | Speedup | Parse fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-context | 8 | 0.375 | 78844 | 0.000 | 0.000 | 9.618 | 9.618 | 11.813 | 1.00x | 0.000 |
| production-rag-50 | 8 | 0.500 | 39300 | 0.501 | 0.000 | 4.144 | 4.144 | 5.004 | 2.32x | 0.000 |
| tokenpack-50 | 8 | 0.500 | 39340 | 0.501 | 1.030 | 4.236 | 5.266 | 6.905 | 1.83x | 0.000 |
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
