# TokenPack Grounded Prompt Ablation Readout

Run: `submission/results/modal_generation_eval_qasper_grounded_ablation50/qasper_generation_judged.jsonl`

Reference run: `submission/results/modal_generation_eval_qasper_ablation50/qasper_generation_judged_14b.jsonl`

Model/judge: `Qwen/Qwen2.5-14B-Instruct`

Scope: same 50 QASPER questions, 100 newly judged answers, 2 score-sorted TokenPack prompt methods.

Judge parse failures: 0.

## Summary

The targeted prompt-only ablation did not improve the generation-quality tradeoff. Both new prompts reduced average judge score relative to `tokenpack-score-sorted-strong-50`, and hallucination did not move toward the LLMLingua-2 rate.

| Method | Judge | Correct | Grounded | Halluc. | Insuff. | Answer toks |
|---|---:|---:|---:|---:|---:|---:|
| LLMLingua-2 50% | 1.253 | 0.50 | 0.42 | 0.08 | 0.12 | 137.5 |
| TokenPack score-sorted + strict | 1.273 | 0.58 | 0.46 | 0.16 | 0.14 | 138.6 |
| TokenPack score-sorted + grounded | 1.133 | 0.52 | 0.44 | 0.18 | 0.20 | 173.3 |
| TokenPack score-sorted + extractive | 0.900 | 0.38 | 0.30 | 0.20 | 0.28 | 169.0 |

## Paired Comparison vs LLMLingua-2

| Method | Mean delta | Win | Tie | Loss |
|---|---:|---:|---:|---:|
| TokenPack score-sorted + strict | +0.020 | 14 | 25 | 11 |
| TokenPack score-sorted + grounded | -0.120 | 11 | 23 | 16 |
| TokenPack score-sorted + extractive | -0.353 | 7 | 21 | 22 |

## Interpretation

The new prompts over-triggered insufficiency/refusal language and often produced long repetitive answers despite asking for concise output. This hurt correctness and did not reduce hallucination. The prior `strict` prompt remains the best 50-question TokenPack generation result.

Recommended next step: if one more targeted run is worth the Modal cost, keep the `strict` prompt wording but lower answer generation length, e.g. `--max-answer-tokens 80`, instead of adding stronger insufficiency/extractive language.
