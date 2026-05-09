# LongBench v2 Grounded Readout

| Method | Runs | Acc. | Grounded acc. | Halluc. | Quote found | Correct unsupported | Saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-context | 20 | 0.450 | 0.050 | 0.900 | 0.350 | 0.889 | 0.000 |
| tokenpack-50 | 20 | 0.350 | 0.050 | 0.750 | 0.450 | 0.857 | 0.504 |
| only-longllmlingua-rate050 | 20 | 0.350 | 0.150 | 0.450 | 0.600 | 0.571 | 0.518 |
| tokenpack-50+longllmlingua-rate050 | 20 | 0.350 | 0.100 | 0.650 | 0.450 | 0.714 | 0.747 |

## Pairwise Grounded Accuracy vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 20 | 0 | 18 | 2 |
| tokenpack-50 | 20 | 1 | 16 | 3 |
| tokenpack-50+longllmlingua-rate050 | 20 | 1 | 17 | 2 |
