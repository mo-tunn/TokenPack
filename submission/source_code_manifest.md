# Source Code Manifest

The main project source code is located in the repository root under `src/tokenpack`.

## Important Modules

- `src/tokenpack/selectors.py`: top-k, budget-top-k, MMR, and 0/1 knapsack selection algorithms.
- `src/tokenpack/scoring.py`: cosine similarity, value normalization, and redundancy penalty.
- `src/tokenpack/chunking.py`: paragraph-group and semantic-threshold chunking.
- `src/tokenpack/benchmark.py`: gold/smoke benchmark metrics.
- `src/tokenpack/generation.py`: local Ollama and OpenAI generation adapters.
- `src/tokenpack/cli.py`: `tokenpack` command-line interface.
- `tests/test_core.py`: unit and integration smoke tests.
- `submission/experiments/knapsack_performance.py`: repeated algorithm-analysis experiment comparing exact DP, value-density greedy, simulated annealing, value greedy, lightest-first greedy, and random feasible selection over 100 runs per problem size.

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

Local LLM test with Ollama:

```powershell
ollama list
python -m tokenpack.cli answer --query "What does TokenPack do?" --selection .tokenpack/selection.json --provider ollama --model qwen3:0.6b
```

Repeated knapsack experiment used by the paper:

```powershell
python submission\experiments\knapsack_performance.py --output-dir submission\results --repetitions 100
```

## Submission Note

For ZIP or GitHub submission, include the repository root with `src`, `tests`, `pyproject.toml`, `README.md`, and the `submission` folder.
