# TokenPack Generation Ablation Readout

Run: `submission/results/modal_generation_eval_qasper_ablation50/qasper_generation_judged_14b.jsonl`

Model/judge: `Qwen/Qwen2.5-14B-Instruct`

Scope: 50 QASPER questions, 350 judged answers, 7 context/prompt methods.

Judge parse failures: 0.

## Summary

The ablation partially recovers the generation-quality claim. The strongest TokenPack variant, `tokenpack-score-sorted-strong-50`, matches LLMLingua-2 on average judge score and is slightly ahead in paired win/loss, while preserving the same TokenPack evidence-retention advantage.

| Method | Judge | Correct | Grounded | Halluc. | Insuff. |
|---|---:|---:|---:|---:|---:|
| Full document | 1.493 | 0.620 | 0.520 | 0.160 | 0.000 |
| Budget-top-k 50% | 1.127 | 0.500 | 0.380 | 0.160 | 0.280 |
| TokenPack original | 1.180 | 0.440 | 0.380 | 0.160 | 0.160 |
| TokenPack score-sorted | 1.200 | 0.520 | 0.400 | 0.180 | 0.160 |
| TokenPack density-sorted | 1.153 | 0.540 | 0.360 | 0.140 | 0.220 |
| TokenPack score-sorted + strict prompt | 1.273 | 0.580 | 0.460 | 0.160 | 0.140 |
| Only LLMLingua-2 50% | 1.253 | 0.500 | 0.420 | 0.080 | 0.120 |

## Paired Comparison

`tokenpack-score-sorted-strong-50` vs `only-llmlingua2-rate050`:

- Mean paired judge-score delta: `+0.020`
- Wins/ties/losses: `14 / 25 / 11`
- Correct rate: `0.580` vs `0.500`
- Grounded rate: `0.460` vs `0.420`
- Hallucination rate: `0.160` vs `0.080`
- Insufficient rate: `0.140` vs `0.120`

Interpretation: score-sorted presentation plus a stricter answer prompt converts TokenPack's evidence-retention advantage into roughly tied or slightly better judged correctness, but the hallucination rate is still worse than LLMLingua-2.

## Recommended Next Step

Do not run the 200-question full generation eval yet.

Run one more 50-question targeted ablation focused on reducing hallucinations while keeping the `score-sorted` presentation:

1. `tokenpack-score-sorted-grounded-50`: stricter prompt against unsupported details.
2. `tokenpack-score-sorted-extractive-50`: answer with the shortest directly supported phrase when possible.
3. Optional selector tweak: lower the candidate pool or increase redundancy penalty to reduce distractor chunks.

Acceptance criterion: beat LLMLingua-2 on average judge score and paired win/loss while keeping hallucination rate no worse than LLMLingua-2 by more than a small margin.
