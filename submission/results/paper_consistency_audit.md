# Paper Consistency Audit

Date: 2026-05-10

Scope: `submission/paper/main.tex`, all `submission/paper/tables/*.tex` files that are actually included by the paper, and the visible paper figures.

## Summary

The main stale-data issue found during this audit was the LongBench/cost line:

- The paper was still using the earlier `longbench_v2_modal_pilot100_8k28k` result.
- The newer run is `submission/results/longbench_v2_modal_pilot100_score_then_source`, with `context_order=score-then-source`.
- That newer run scanned 503 LongBench v2 examples and produced 83 eligible cases under the 8k--24k token window.
- The paper table, setup text, abstract/discussion/conclusion savings, cost table, and cost figure were updated to this newer 83-case run.

## Table Audit

| Paper Asset | Source Checked | Status |
|---|---|---|
| `algorithm_comparison_table.tex` | `submission/results/knapsack_summary.csv`, N=1000 rows | OK |
| `knapsack_summary_table.tex` | `submission/results/knapsack_summary.csv` | OK |
| `knapsack_timeout_table.tex` | `submission/results/knapsack_summary.csv`, timeout/state rows | OK |
| inline retrieval feasibility table | old local LLM/retrieval feasibility artifact; still framed as small engineering sanity check | OK, but not a main claim |
| `chunking_ablation_table.tex` | `submission/results/chunking_ablation_strong_rerun/chunking_ablation.csv` | OK |
| `value_ablation_table.tex` | `submission/results/value_ablation_strong_rerun/value_ablation.csv` | OK |
| `qasper_selector_eval_table.tex` | `submission/results/qasper_selector_eval_strong_rerun/qasper_selector_eval.csv` | OK |
| `qasper_cost_quality_table.tex` | `submission/results/qasper_cost_quality_frontier_evidence/qasper_cost_quality_summary.csv` | OK |
| `qasper_compression_comparison_table.tex` | `submission/results/qasper_compression_report/qasper_compression_methods.csv` | OK |
| `qasper_compression_frontier_table.tex` | `submission/results/qasper_compression_report/qasper_compression_methods.csv` | OK |
| `longbench_generation_table.tex` | updated to `submission/results/longbench_v2_modal_pilot100_score_then_source/longbench_generation_summary.csv` | Fixed |
| `cost_savings_table.tex` | updated from the same LongBench score-then-source saving ratios | Fixed |

## LongBench Values Now Used In Paper

Source: `submission/results/longbench_v2_modal_pilot100_score_then_source/longbench_generation_summary.csv`

| Method | Runs | Accuracy | Avg context tokens | Saving |
|---|---:|---:|---:|---:|
| full-context | 83 | 0.398 | 17646 | 0.000 |
| tokenpack-50 | 83 | 0.386 | 8758 | 0.504 |
| only-longllmlingua-rate050 | 83 | 0.402 | 8560 | 0.512 |
| tokenpack-50+longllmlingua-rate050 | 83 | 0.410 | 4486 | 0.745 |

Task report:

- scanned examples: 503
- eligible cases: 83
- skipped too short: 0
- skipped too long: 420
- source-token window: 8k--24k
- scoring: `evidence-hybrid`
- selector: `knapsack-redundancy`
- context order: `score-then-source`

## Removed/Updated Stale Claims

- Replaced the earlier 100-case 8k--28k LongBench description with the newer 83-case 8k--24k score-then-source setup.
- Replaced LongBench values `0.420 / 0.410 / 0.434 / 0.414` with `0.398 / 0.386 / 0.402 / 0.410`.
- Replaced context-token values `19175 / 9513 / 9239 / 4848` with `17646 / 8758 / 8560 / 4486`.
- Replaced cascade saving `74.6%` with `74.5%`.
- Replaced the cost table/figure derived from the older small `llm_quality` scenario with LongBench-derived scaled costs.
- Updated `submission/source_code_manifest.md` so the paper reproduction command points to `longbench_v2_modal_pilot100_score_then_source` with `--source-max-tokens 24000 --context-order score-then-source`.

## Remaining Caveats

- The inline retrieval feasibility table is still a small engineering sanity check, not a large benchmark. The paper text frames it as such.
- QASPER answer overlap F1 remains a retention proxy, not generated-answer correctness.
- The BudgetMem-style baseline is a proxy only; the paper does not claim direct BudgetMem reproduction.
