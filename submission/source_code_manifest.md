# Source Code Manifest

The main project source code is located in the repository root under `src/tokenpack`.

## Important Modules

- `src/tokenpack/selectors.py`: top-k, budget-top-k, MMR, 0/1 knapsack, redundancy-aware knapsack, and experimental query-coverage selection.
- `src/tokenpack/scoring.py`: baseline (`cosine`, `hybrid`), default (`evidence-hybrid`), budget-aware (`knapsack-aware`), related-work proxy (`budgetmem-style`), and experimental (`query-support`, `decision-aware`, `instruction-ami`) value scoring.
- `src/tokenpack/embeddings.py`: deterministic `hashing-384` embeddings and optional `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- `src/tokenpack/ami.py`: optional time-budgeted AMI reranking with local Hugging Face causal language models.
- `src/tokenpack/chunk_profiles.py`: named chunk-size presets, including the low-budget evidence preset used for aggressive compression experiments.
- `src/tokenpack/chunking.py`: paragraph-group, semantic-threshold, and structure-aware chunking.
- `src/tokenpack/benchmark.py`: gold/smoke benchmark metrics.
- `src/tokenpack/generation.py`: local Ollama and OpenAI generation adapters.
- `src/tokenpack/cli.py`: `tokenpack` command-line interface.
- `tests/test_core.py`: unit and integration smoke tests.
- `submission/experiments/knapsack_performance.py`: repeated algorithm-analysis experiment comparing exact DP, value-density greedy, simulated annealing, value greedy, lightest-first greedy, and random feasible selection over 100 runs per problem size.
- `submission/modal_generation_eval/`: Modal QASPER generation and judge harness used for strict/grounded prompt ablations.
- `submission/longbench_eval/`: Modal LongBench v2 generation and groundedness harness comparing full context, TokenPack, LongLLMLingua context-level filtering, and the TokenPack+LongLLMLingua cascade. The groundedness report now separates unsupported claims from strict grounding failures such as quote mismatch or unsupported rationale.

## Installation

```powershell
pip install -e ".[embeddings,pdf,tokens,dev]"
```

Fast offline test without external model downloads:

```powershell
$env:PYTHONPATH="src"
python -m tokenpack.cli --backend hash ingest README.md --index .tokenpack/demo-index.json
python -m tokenpack.cli --backend hash select --index .tokenpack/demo-index.json --query "knapsack retrieval context budget" --budget 300 --reserve-output 50
python -m pytest -p no:cacheprovider
```

## Reproducibility Defaults

- Paper QASPER and LongBench runs use `--backend hash`, implemented as deterministic `hashing-384` embeddings.
- If `--backend sentence-transformers` is selected, the default neural embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- QASPER 200-question compression runs stream the validation split in dataset order and stop after the first 200 questions from papers with parseable text blocks and questions; they are not random subsamples.
- Synthetic simulated annealing starts from density-greedy selection, uses temperature `max(1, n/2)`, iterations `min(25000, max(2000, 12*n))`, cooling `0.9995`, and deterministic reheating by `1.02` every 250 steps.
- `budgetmem-style` is an artifact-local proxy baseline using BM25, query coverage, position, term specificity, entity density, numerical density, discourse markers, and length utility. It is not a reproduction of BudgetMem's learned policy.

BudgetMem-style QASPER proxy run:

```powershell
python submission\experiments\qasper_selector_eval.py --data-file .tokenpack\data\qasper-validation.parquet --backend hash --chunker structure-aware --scoring budgetmem-style --strategies budget-top-k,knapsack-redundancy --budget-ratios 0.20,0.30,0.40,0.50 --max-papers 500 --max-questions 861 --candidate-pool 300 --chunk-size-preset low-budget --output-dir submission\results\qasper_budgetmem_style --no-paper-table
```

Local LLM test with Ollama:

```powershell
ollama list
python -m tokenpack.cli answer --query "What does TokenPack do?" --selection .tokenpack/selection.json --provider ollama --model qwen3:0.6b
```

Repeated knapsack experiment used by the paper:

```powershell
python submission\experiments\knapsack_performance.py --output-dir submission\results --repetitions 100
```

LongBench v2 Modal pilot used by the paper:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m modal run submission/longbench_eval/app.py::build_and_run --output-dir submission/results/longbench_v2_modal_pilot100_score_then_source --limit 100 --source-min-tokens 8000 --source-max-tokens 24000 --max-scanned 503 --batch-size 1 --context-order score-then-source
```

Coverage-selector LongBench ablation:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m modal run submission/longbench_eval/app.py::build_and_run --output-dir submission/results/longbench_v2_modal_pilot30_coverage --limit 30 --source-min-tokens 8000 --source-max-tokens 24000 --max-scanned 503 --batch-size 2 --selection-strategy knapsack-coverage
```

## Submission Note

For ZIP or GitHub submission, include the repository root with `src`, `tests`, `pyproject.toml`, `README.md`, and the `submission` folder.
