# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Parse fail |
|---|---:|---:|---:|---:|---:|
| full-context | 100 | 0.420 | 19175 | 0.000 | 0.000 |
| tokenpack-50 | 100 | 0.410 | 9513 | 0.504 | 0.000 |
| only-longllmlingua-rate050 | 100 | 0.434 | 9239 | 0.515 | 0.010 |
| tokenpack-50+longllmlingua-rate050 | 100 | 0.414 | 4848 | 0.746 | 0.010 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 99 | 10 | 78 | 11 |
| tokenpack-50 | 99 | 11 | 75 | 13 |
| tokenpack-50+longllmlingua-rate050 | 98 | 6 | 85 | 7 |
