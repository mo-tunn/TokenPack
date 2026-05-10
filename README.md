# TokenPack

**TokenPack packs the most useful evidence chunks into a limited LLM context window.**

It turns long-context selection into a 0/1 knapsack problem: chunks are items, token counts are weights, and query-conditioned evidence scores are values. The default mode is the strongest current research setting:

```text
structure-aware chunks + evidence-hybrid scoring + redundancy-aware knapsack
```

This is useful when a RAG or long-document QA system has too much context, uneven chunk lengths, and a hard input-token budget.

## What You Get

- A reusable Python package and `tokenpack` CLI.
- Budget-valid context selection for long documents, code, PDFs, or mixed folders.
- Exact/heuristic selector baselines: top-k, budget-top-k, MMR, greedy, knapsack, redundancy-aware knapsack.
- Evidence-oriented scoring profiles: cosine, hybrid, evidence-hybrid, knapsack-aware, query-support, decision-aware, BudgetMem-style proxy.
- Optional second-stage prompt compression with LLMLingua / LongLLMLingua.
- Reproducible paper artifacts under [`submission/`](submission).

## Headline Results

These are the cleanest results from the current paper artifacts. The paper is intentionally conservative: TokenPack does **not** claim universal knapsack dominance, but it does show that selection-first context packing is a strong budget-control layer.

| Setting | Main Result |
|---|---|
| **QASPER, matched ~50% saving** | Only TokenPack preserves **0.900 evidence recall** vs **0.714** for Only LLMLingua-2. |
| **QASPER complete evidence** | Only TokenPack preserves complete evidence on **0.785** of questions vs **0.110** for Only LLMLingua-2. |
| **QASPER cascade frontier** | TokenPack + LLMLingua-2 at rate 0.85 reaches **58.3% saving** with **0.823 evidence recall**. |
| **LongBench v2 generation pilot** | TP-50 stays near full context: **0.386 acc.** vs **0.398 full context**, with **50.4% saving**. |
| **LongBench aggressive cascade** | TP-50 + LongLLMLingua-50 reaches **74.5% context saving** with **0.410 acc.** on the 83-case eligible pilot. |

The strongest claim is not “knapsack always beats every retriever.” The stronger, cleaner claim is:

> Select evidence first, then optionally compress it. Retrieval-time budget selection and prompt compression are not interchangeable.

## Install

From PyPI, once published:

```bash
pip install tokenpack
```

From GitHub today:

```bash
pip install "git+https://github.com/mo-tunn/TokenPack.git"
```

For PDF parsing, neural embeddings, compression, and development tools:

```bash
pip install "tokenpack[embeddings,pdf,tokens,compression,dev] @ git+https://github.com/mo-tunn/TokenPack.git"
```

For local editable development:

```bash
git clone https://github.com/mo-tunn/TokenPack.git
cd TokenPack
pip install -e ".[embeddings,pdf,tokens,compression,dev]"
```

The core package has no mandatory heavy dependencies. If optional models are unavailable, TokenPack can still run offline with deterministic hash embeddings.

## Quick Start

Index a document or folder:

```bash
tokenpack --backend hash ingest README.md --index .tokenpack/readme-index.json
```

Select evidence under a token budget:

```bash
tokenpack --backend hash select \
  --index .tokenpack/readme-index.json \
  --query "How does TokenPack reduce LLM context cost?" \
  --budget 3000 \
  --reserve-output 500 \
  --output .tokenpack/selection.json
```

Export the selected context:

```bash
tokenpack export-context \
  --selection .tokenpack/selection.json \
  --output .tokenpack/context.txt
```

By default, these commands use:

```text
chunker: structure-aware
chunk-size-preset: low-budget
scoring: evidence-hybrid
selector: knapsack-redundancy
```

Use `--scoring cosine`, `--strategy budget-top-k`, or `--chunker paragraph` when you want simpler baselines or ablations.

## Compression Cascade

TokenPack can select evidence first and then pass only that selected evidence to LLMLingua / LongLLMLingua:

```bash
tokenpack export-context \
  --selection .tokenpack/selection.json \
  --output .tokenpack/compressed-context.txt \
  --compressor llmlingua \
  --llmlingua2 \
  --compression-rate 0.85 \
  --compression-question "How does TokenPack reduce LLM context cost?"
```

This mirrors the paper’s selection-first framing: TokenPack controls which evidence enters the prompt, while compression optionally reduces the selected text further.

## Python API

```python
from tokenpack.embeddings import HashingEmbedder
from tokenpack.pipeline import ingest_path
from tokenpack.scoring import score_chunks
from tokenpack.selectors import select_chunks

embedder = HashingEmbedder(dimensions=384)
index = ingest_path(
    "README.md",
    ".tokenpack/readme-index.json",
    embedder=embedder,
    chunker_name="structure-aware",
    target_tokens=250,
    min_tokens=40,
    max_tokens=320,
)

query = "How does TokenPack reduce LLM context cost?"
query_embedding = embedder.embed([query])[0]

scored = score_chunks(
    query_embedding,
    index.chunks,
    index.embeddings,
    scoring="evidence-hybrid",
    query_text=query,
    redundancy_penalty=0.35,
)

result = select_chunks(
    scored,
    strategy="knapsack-redundancy",
    budget=3000,
    candidate_pool=250,
)

print(result.used_tokens, [item.chunk.id for item in result.selected])
```

## Reproduce Paper Runs

Fast local tests:

```bash
python -m pytest -q
```

QASPER selector baseline:

```bash
python submission/experiments/qasper_selector_eval.py \
  --data-file .tokenpack/data/qasper-validation.parquet \
  --backend hash \
  --chunker structure-aware \
  --scoring evidence-hybrid \
  --strategies budget-top-k,greedy-density,knapsack,knapsack-redundancy \
  --budget-ratios 0.20,0.30,0.40,0.50 \
  --max-papers 500 \
  --max-questions 861 \
  --candidate-pool 300 \
  --chunk-size-preset low-budget \
  --output-dir submission/results/qasper_selector_eval_strong_rerun
```

LongBench v2 Modal pilot used in the current paper:

```bash
python -m modal run submission/longbench_eval/app.py::build_and_run \
  --output-dir submission/results/longbench_v2_modal_pilot100_score_then_source \
  --limit 100 \
  --source-min-tokens 8000 \
  --source-max-tokens 24000 \
  --max-scanned 503 \
  --batch-size 1 \
  --context-order score-then-source
```

See [`submission/source_code_manifest.md`](submission/source_code_manifest.md) for the full artifact map and [`submission/results/paper_consistency_audit.md`](submission/results/paper_consistency_audit.md) for the latest paper-data consistency audit.

## Repository Layout

```text
src/tokenpack/                 Python package and CLI implementation
tests/                         Unit and smoke tests
submission/paper/              LaTeX paper source, tables, figures
submission/experiments/        QASPER, LongBench, compression, and ablation scripts
submission/results/            Paper result artifacts and readouts
submission/longbench_eval/     Modal LongBench v2 generation harness
submission/modal_generation_eval/  Modal QASPER generation/judge harness
```

## Notes

- QASPER metrics are evidence-retention and answer-token-retention proxies, not human-judged generated-answer quality.
- LongBench v2 accuracy numbers are pilot-scale and should be read descriptively, not as statistically significant wins.
- BudgetMem is discussed as related work; this repo includes a `budgetmem-style` feature proxy, not a direct BudgetMem reproduction.
- The default CLI mode is optimized for the current paper setting. Use explicit flags for ablations.

## License

TokenPack is licensed under the Business Source License 1.1. See [`LICENSE`](LICENSE).

## Citation

If you use TokenPack in research, cite the paper PDF in [`submission/TokenPack-paper.pdf`](submission/TokenPack-paper.pdf). A BibTeX entry will be added when the public preprint is available.
