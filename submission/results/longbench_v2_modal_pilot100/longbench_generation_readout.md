# LongBench v2 Modal Pilot Readout

| Method | Runs | Accuracy | Avg context toks | Saving | Parse fail |
|---|---:|---:|---:|---:|---:|
| full-context | 83 | 0.398 | 17646 | 0.000 | 0.000 |
| tokenpack-50 | 83 | 0.398 | 8758 | 0.504 | 0.000 |
| only-longllmlingua-rate050 | 83 | 0.402 | 8560 | 0.512 | 0.012 |
| tokenpack-50+longllmlingua-rate050 | 83 | 0.410 | 4486 | 0.745 | 0.000 |

## Pairwise vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 82 | 8 | 66 | 8 |
| tokenpack-50 | 82 | 10 | 62 | 10 |
| tokenpack-50+longllmlingua-rate050 | 82 | 5 | 73 | 4 |
