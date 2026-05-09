# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Parse fail |
|---|---:|---:|---:|---:|---:|
| full-context | 30 | 0.400 | 17623 | 0.000 | 0.000 |
| tokenpack-50 | 30 | 0.367 | 8778 | 0.502 | 0.000 |
| only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 | 0.000 |
| tokenpack-50+longllmlingua-rate050 | 30 | 0.333 | 4483 | 0.745 | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 30 | 3 | 26 | 1 |
| tokenpack-50 | 30 | 3 | 25 | 2 |
| tokenpack-50+longllmlingua-rate050 | 30 | 2 | 26 | 2 |
