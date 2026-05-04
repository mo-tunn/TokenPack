# Without TokenPack vs TokenPack

This report uses the simple reviewed PDF dataset:

- Source PDF: `submission/gold/simple_corpus/tokenpack_demo_context.pdf`
- Gold records: `submission/gold/gold.jsonl`
- Index: `submission/gold/simple_corpus/simple-index.json`

## What "without TokenPack" means here

- `document-prefix`: Put chunks into the prompt from the beginning of the PDF until the budget is full. This is budget-safe, but it does not use query relevance.
- `full-document`: Put the whole document into the prompt. This has high recall on a tiny document, but it is invalid when the document exceeds the context budget.
- `top-k`: Retrieve the highest-similarity chunks without enforcing the final token budget.

## Key Result

At an effective 250-token budget:

| Method | Evidence Recall | Avg. Tokens | Over Budget Rate | Interpretation |
|---|---:|---:|---:|---|
| document-prefix | 0.25 | 180 | 0.00 | Valid but misses most evidence. |
| full-document | 1.00 | 1539 | 1.00 | Finds evidence but violates budget. |
| top-k | 0.94 | 809 | 1.00 | Relevant but not context-safe. |
| knapsack | 0.75 | 213 | 0.00 | Good recall while respecting budget. |

At an effective 700-token budget:

| Method | Evidence Recall | Avg. Tokens | Over Budget Rate | Interpretation |
|---|---:|---:|---:|---|
| document-prefix | 0.69 | 698 | 0.00 | Uses budget but still misses evidence. |
| full-document | 1.00 | 1539 | 1.00 | Still invalid under the budget. |
| top-k | 0.94 | 809 | 0.88 | Usually exceeds the budget. |
| knapsack | 0.94 | 657 | 0.00 | Matches top-k recall while staying valid. |

## Conclusion

TokenPack improves the process because it optimizes semantic value and token cost together. The baseline methods demonstrate the trade-off: prefix selection is safe but weak, full-document selection is complete but invalid, and naive top-k is relevant but often over budget. Knapsack gives a budget-valid selection with high evidence recall.
