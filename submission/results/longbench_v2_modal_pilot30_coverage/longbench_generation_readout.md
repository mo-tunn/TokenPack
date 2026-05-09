# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Parse fail |
|---|---:|---:|---:|---:|---:|
| full-context | 30 | 0.400 | 17623 | 0.000 | 0.000 |
| tokenpack-50 | 30 | 0.367 | 8631 | 0.512 | 0.000 |
| only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 | 0.000 |
| tokenpack-50+longllmlingua-rate050 | 30 | 0.267 | 4414 | 0.749 | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 30 | 3 | 26 | 1 |
| tokenpack-50 | 30 | 4 | 23 | 3 |
| tokenpack-50+longllmlingua-rate050 | 30 | 1 | 26 | 3 |
