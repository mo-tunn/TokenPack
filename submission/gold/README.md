# TokenPack Gold Evidence Review

This folder now contains two gold-evidence paths:

- `simple_gold.jsonl`: Recommended small review dataset based on a controlled 3-page demo PDF.
- `candidate_gold.jsonl`: Same simple dataset, kept as the default candidate file for the review script.
- `gold.jsonl`: Human-reviewed simple dataset used as the small reviewed benchmark baseline.
- `gold_reviewed.jsonl`: Interactive review output; currently synchronized with `gold.jsonl`.
- `academic_candidate_gold.jsonl`: Older auto-proposed dataset from academic research PDFs. Keep it only as an optional advanced dataset.

For the project submission, use the simple dataset first. It is intentionally easy to inspect because every answer is supported by the local PDF at `simple_corpus/tokenpack_demo_context.pdf`.

## Simple Dataset Files

- `simple_corpus/tokenpack_demo_context.pdf`: Small PDF used as the source document.
- `simple_corpus/simple-index.json`: TokenPack index built from the small PDF.
- `simple_gold.jsonl`: Original proposed simple query, answer, and evidence chunk records.
- `simple_review_packet.md`: Human-readable review packet showing each query, answer, source chunk id, page range, and evidence text.
- `simple_benchmark_report.md`: Benchmark comparison for top-k, budget-top-k, MMR, knapsack, and knapsack with redundancy.
- `simple_benchmark_report.csv`: CSV version of the benchmark report.
- `simple_benchmark.json`: Raw benchmark output.
- `review_packet.md`: Same content as `simple_review_packet.md`, kept as the default review packet name.
- `without_tokenpack_comparison.md`: Short before/after explanation comparing naive baselines with TokenPack selection.

## Fast Review

Open:

```powershell
submission\gold\simple_review_packet.md
```

For each record, check only three things:

- The question is meaningful.
- The answer is directly supported by the evidence text shown below it.
- The evidence chunk id belongs to `simple_corpus/tokenpack_demo_context.pdf`.

If all three are true, the record can be accepted as gold evidence for a small feasibility benchmark.

## Re-run Commands

Validate the simple gold set:

```powershell
$env:PYTHONPATH='src'
python -m tokenpack.cli --backend hash dataset validate --index submission\gold\simple_corpus\simple-index.json --gold submission\gold\simple_gold.jsonl
```

Recreate the review packet:

```powershell
python submission\gold\review_gold.py --index submission\gold\simple_corpus\simple-index.json --input submission\gold\simple_gold.jsonl --export-markdown submission\gold\simple_review_packet.md --max-chars 1400
```

Re-run the benchmark:

```powershell
$env:PYTHONPATH='src'
python -m tokenpack.cli --backend hash benchmark --index submission\gold\simple_corpus\simple-index.json --gold submission\gold\simple_gold.jsonl --budgets 350,500,800 --reserve-output 100 --markdown-output submission\gold\simple_benchmark_report.md --csv-output submission\gold\simple_benchmark_report.csv --output submission\gold\simple_benchmark.json
```

The old academic candidate file is backed up as `academic_candidate_gold.jsonl`. It is harder to review because it is generated from technical research papers. Do not use it as the first manual review target.
