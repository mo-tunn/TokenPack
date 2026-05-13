# Source Code Manifest

The main project source code is located in the repository root under `src/tokenpack`.

## Important Modules

- `src/tokenpack/selectors.py`: production-style dense RAG, top-k, budget-top-k, MMR, 0/1 knapsack, redundancy-aware knapsack, and experimental query-coverage selection.
- `src/tokenpack/scoring.py`: production `evidence-hybrid` value scoring only.
- `src/tokenpack/scoring_experimental.py`: historical and ablation-only scoring profiles (`cosine`, `hybrid`, `knapsack-aware`, `budgetmem-style`, `query-support`, `decision-aware`).
- `src/tokenpack/embeddings.py`: required `sentence-transformers/all-MiniLM-L6-v2` embeddings by default.
- `src/tokenpack/chunk_profiles.py`: named chunk-size presets, including the low-budget evidence preset used for aggressive compression experiments.
- `src/tokenpack/chunking.py`: standalone semantic-threshold ablation plus the default structure-aware chunker with semantic drift boundaries.
- `src/tokenpack/benchmark.py`: lightweight local benchmark helpers kept for smoke tests, not used for final paper claims.
- `src/tokenpack/generation.py`: local Ollama plus optional non-paper cloud generation adapters.
- `src/tokenpack/cli.py`: `tokenpack` command-line interface.
- `tests/test_core.py`: unit and integration smoke tests.
- `submission/experiments/knapsack_performance.py`: repeated algorithm-analysis experiment comparing exact DP, value-density greedy, simulated annealing, value greedy, lightest-first greedy, and random feasible selection over 100 runs per problem size.
- `submission/modal_generation_eval/`: Modal QASPER generation and judge harness used for strict/grounded prompt ablations.
- `submission/longbench_eval/`: Modal LongBench v2 generation and groundedness harness comparing full context, TokenPack, LongLLMLingua context-level filtering, and the TokenPack+LongLLMLingua cascade. The groundedness report now separates unsupported claims from strict grounding failures such as quote mismatch or unsupported rationale.

## Installation

```powershell
pip install -e ".[pdf,tokens,dev]"
```

Fast local smoke test. Use `--offline-models` only if the sentence-transformers model is already cached locally:

```powershell
$env:PYTHONPATH="src"
python -m tokenpack.cli --offline-models ingest README.md --index .tokenpack/demo-index.json
python -m tokenpack.cli --offline-models select --index .tokenpack/demo-index.json --query "knapsack retrieval context budget" --budget 300 --reserve-output 50
python -m pytest -p no:cacheprovider
```

## Reproducibility Defaults

- New QASPER and LongBench runs use `sentence-transformers/all-MiniLM-L6-v2`; older deterministic lexical-embedding artifacts are deprecated and should not be used for final paper claims.
- LongBench v2 generation and latency tables use `Qwen/Qwen2.5-14B-Instruct` via vLLM on Modal.
- The main chunking path is `structure-aware`: it preserves metadata for code/PDF sources and now also uses adjacent block embedding similarity to split semantic topic shifts inside compatible document sections.
- QASPER 200-question compression runs stream the validation split in dataset order and stop after the first 200 questions from papers with parseable text blocks and questions; they are not random subsamples.
- Synthetic simulated annealing starts from density-greedy selection, uses temperature `max(1, n/2)`, iterations `min(25000, max(2000, 12*n))`, cooling `0.9995`, and deterministic reheating by `1.02` every 250 steps.
- `budgetmem-style` is retained only as an artifact-local proxy in `scoring_experimental.py`. It is not a reproduction of BudgetMem's learned policy and is no longer selectable from production experiment CLIs.

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
python -m modal run submission/longbench_eval/app.py::build_and_run --output-dir submission/results/longbench_v2_modal_hybrid_greedy_83_latency --limit 83 --source-min-tokens 8000 --source-max-tokens 24000 --max-scanned 503 --model-id Qwen/Qwen2.5-14B-Instruct --batch-size 1 --context-order score-then-source --latency-mode
```

Coverage-selector LongBench ablation:

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m modal run submission/longbench_eval/app.py::build_and_run --output-dir submission/results/longbench_v2_modal_pilot30_coverage --limit 30 --source-min-tokens 8000 --source-max-tokens 24000 --max-scanned 503 --batch-size 2 --selection-strategy knapsack-coverage
```

## Submission Note

For ZIP or GitHub submission, include the repository root with `src`, `tests`, `pyproject.toml`, `README.md`, and the `submission` folder.
