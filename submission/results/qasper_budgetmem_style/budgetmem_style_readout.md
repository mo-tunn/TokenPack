# BudgetMem-Style QASPER Proxy Readout

This is not a direct BudgetMem reproduction. No official BudgetMem code or exact reproduction artifact was available in this workspace, so this run uses a hand-designed feature-family proxy:

- BM25 lexical relevance
- query coverage
- source-position prior
- term specificity
- entity density
- numerical density
- discourse markers
- length utility

Setup:

- Dataset: QASPER validation
- Scope: 861 questions from 276 papers
- Embeddings: deprecated historical hash run; regenerate with `sentence-transformers/all-MiniLM-L6-v2` before using this result in paper claims.
- Chunker: `structure-aware`
- Chunk preset: `low-budget`
- Candidate pool: 300
- Selectors: `budget-top-k`, `knapsack-redundancy`
- Budgets: 20%, 30%, 40%, 50%

## Result

| Selector | Budget | Evidence recall | Complete evidence | Answer overlap F1 |
|---|---:|---:|---:|---:|
| budget-top-k | 20% | 0.692 | 0.386 | 0.027 |
| knapsack-redundancy | 20% | 0.666 | 0.324 | 0.026 |
| budget-top-k | 30% | 0.795 | 0.548 | 0.020 |
| knapsack-redundancy | 30% | 0.781 | 0.511 | 0.020 |
| budget-top-k | 40% | 0.848 | 0.662 | 0.016 |
| knapsack-redundancy | 40% | 0.846 | 0.642 | 0.016 |
| budget-top-k | 50% | 0.895 | 0.775 | 0.014 |
| knapsack-redundancy | 50% | 0.889 | 0.760 | 0.013 |

## Comparison To Paper Default

The paper's default `evidence-hybrid` profile remains stronger on the same 861-question QASPER selector setup:

| Selector | Budget | BudgetMem-style recall | Evidence-hybrid recall | Delta |
|---|---:|---:|---:|---:|
| budget-top-k | 20% | 0.692 | 0.713 | -0.021 |
| budget-top-k | 30% | 0.795 | 0.815 | -0.020 |
| budget-top-k | 40% | 0.848 | 0.872 | -0.024 |
| budget-top-k | 50% | 0.895 | 0.911 | -0.016 |
| knapsack-redundancy | 20% | 0.666 | 0.707 | -0.041 |
| knapsack-redundancy | 30% | 0.781 | 0.807 | -0.026 |
| knapsack-redundancy | 40% | 0.846 | 0.871 | -0.025 |
| knapsack-redundancy | 50% | 0.889 | 0.906 | -0.017 |

Interpretation: the proxy is useful as a sanity check, but it does not replace a direct BudgetMem baseline and does not beat TokenPack's current evidence-hybrid scoring. It supports the paper's conservative claim that utility estimation is the real bottleneck.
