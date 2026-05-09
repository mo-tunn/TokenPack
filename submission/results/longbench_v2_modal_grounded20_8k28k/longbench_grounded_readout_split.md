# LongBench v2 Grounded Readout

| Method | Runs | Acc. | Grounded acc. | Halluc. claims | Quote found | Correct unsupported | Saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-context | 20 | 0.450 | 0.050 | 0.400 | 0.350 | 0.889 | 0.000 |
| tokenpack-50 | 20 | 0.350 | 0.050 | 0.300 | 0.450 | 0.857 | 0.504 |
| only-longllmlingua-rate050 | 20 | 0.350 | 0.150 | 0.150 | 0.600 | 0.571 | 0.518 |
| tokenpack-50+longllmlingua-rate050 | 20 | 0.350 | 0.100 | 0.300 | 0.450 | 0.714 | 0.747 |

## Grounding Diagnostic Split

| Method | Unsupported claims | Strict grounding failure | Answer supported | Rationale supported | Quote supports answer | Quote missing |
|---|---:|---:|---:|---:|---:|---:|
| full-context | 0.400 | 0.900 | 0.600 | 0.600 | 0.000 | 0.650 |
| tokenpack-50 | 0.300 | 0.750 | 0.700 | 0.700 | 0.000 | 0.550 |
| only-longllmlingua-rate050 | 0.150 | 0.500 | 0.850 | 0.850 | 0.000 | 0.400 |
| tokenpack-50+longllmlingua-rate050 | 0.300 | 0.650 | 0.700 | 0.700 | 0.000 | 0.550 |

## Pairwise Grounded Accuracy vs LongLLMLingua

| Method | Compared | Win | Tie | Loss |
|---|---:|---:|---:|---:|
