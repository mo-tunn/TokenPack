# LongBench v2 Grounded Readout

| Method | Runs | Acc. | Grounded acc. | Halluc. claims | Quote found | Correct unsupported | Saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-context | 20 | 0.450 | 0.300 | 0.050 | 0.550 | 0.333 | 0.000 |
| tokenpack-50 | 20 | 0.350 | 0.150 | 0.000 | 0.500 | 0.571 | 0.510 |
| only-longllmlingua-rate050 | 20 | 0.350 | 0.250 | 0.100 | 0.500 | 0.286 | 0.515 |
| tokenpack-50+longllmlingua-rate050 | 20 | 0.350 | 0.250 | 0.100 | 0.650 | 0.286 | 0.749 |

## Grounding Diagnostic Split

| Method | Unsupported claims | Strict grounding failure | Answer supported | Rationale supported | Quote supports answer | Quote missing |
|---|---:|---:|---:|---:|---:|---:|
| full-context | 0.050 | 0.450 | 0.950 | 0.950 | 0.900 | 0.450 |
| tokenpack-50 | 0.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 |
| only-longllmlingua-rate050 | 0.100 | 0.500 | 0.900 | 0.950 | 0.900 | 0.500 |
| tokenpack-50+longllmlingua-rate050 | 0.100 | 0.400 | 0.900 | 0.950 | 0.900 | 0.350 |

## Pairwise Grounded Accuracy vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| full-context | 20 | 3 | 15 | 2 |
| tokenpack-50 | 20 | 1 | 16 | 3 |
| tokenpack-50+longllmlingua-rate050 | 20 | 1 | 18 | 1 |
