# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Parse fail |
|---|---:|---:|---:|---:|---:|
| full-context | 30 | 0.400 | 17623 | 0.000 | 0.000 |
| tokenpack-50 | 30 | 0.433 | 8750 | 0.504 | 0.000 |
| only-longllmlingua-rate050 | 30 | 0.333 | 8564 | 0.512 | 0.000 |
| tokenpack-50+longllmlingua-rate050 | 30 | 0.300 | 4454 | 0.746 | 0.000 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0.000 | 0 | 0.000 | 0.000 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0.000 | 0 | 0.000 | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 30 | 3 | 26 | 1 |
| tokenpack-50 | 30 | 5 | 23 | 2 |
| tokenpack-50+longllmlingua-rate050 | 30 | 1 | 27 | 2 |
| tokenpack-60+longllmlingua-rate050 | 0 | 0 | 0 | 0 |
| tokenpack-50+longllmlingua-rate065 | 0 | 0 | 0 | 0 |
