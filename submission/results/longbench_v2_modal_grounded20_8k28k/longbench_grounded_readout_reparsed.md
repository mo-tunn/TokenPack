# LongBench v2 Grounded Readout (Reparsed)

| Method | Runs | Acc. | Grounded acc. | Halluc. | Quote found | Correct unsupported | Saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-context | 20 | 0.450 | 0.150 | 0.650 | 0.550 | 0.667 | 0.000 |
| tokenpack-50 | 20 | 0.350 | 0.150 | 0.500 | 0.600 | 0.571 | 0.504 |
| only-longllmlingua-rate050 | 20 | 0.350 | 0.200 | 0.250 | 0.750 | 0.429 | 0.518 |
| tokenpack-50+longllmlingua-rate050 | 20 | 0.350 | 0.150 | 0.400 | 0.650 | 0.571 | 0.747 |

## Pairwise Grounded Accuracy vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 20 | 0 | 19 | 1 |
| tokenpack-50 | 20 | 1 | 17 | 2 |
| tokenpack-50+longllmlingua-rate050 | 20 | 1 | 17 | 2 |
