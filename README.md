# TokenPack

TokenPack is a Python package and CLI for selecting the most valuable semantic chunks from long documents under a fixed LLM context-window budget.

The first version focuses on benchmarkable retrieval rather than a web app:

```powershell
tokenpack ingest resources --index .tokenpack/index.json
tokenpack select --query "How can knapsack optimize LLM context windows?" --budget 50000
tokenpack benchmark --budget 50000
tokenpack export-context --selection .tokenpack/selection.json --output .tokenpack/context.txt
```

Gold evidence evaluation workflow:

```powershell
tokenpack --backend hash ingest README.md --index .tokenpack/readme-index.json
tokenpack dataset propose --index .tokenpack/readme-index.json --output .tokenpack/gold-proposed.jsonl
tokenpack dataset validate --index .tokenpack/readme-index.json --gold .tokenpack/gold-proposed.jsonl
tokenpack --backend hash benchmark --index .tokenpack/readme-index.json --gold .tokenpack/gold-proposed.jsonl --budgets 300,500 --markdown-output .tokenpack/report.md --csv-output .tokenpack/report.csv
```

Semantic threshold chunking can be enabled during ingest:

```powershell
tokenpack --backend hash ingest README.md --chunker semantic-threshold --semantic-threshold 0.35
```

Optional answer generation is intentionally separate from retrieval benchmarking:

```powershell
tokenpack answer --query "..." --selection .tokenpack/selection.json --provider none
```

Local Ollama answer generation:

```powershell
tokenpack doctor
tokenpack answer --query "What does TokenPack do?" --selection .tokenpack/selection.json --provider ollama --model qwen3:0.6b
```

For cached sentence-transformers usage without Hugging Face network checks:

```powershell
tokenpack --offline-models --backend sentence-transformers benchmark --index .tokenpack/index.json --gold .tokenpack/gold-proposed.jsonl
```

Optional dependencies improve quality:

```powershell
pip install -e ".[embeddings,pdf,tokens,dev]"
```

If optional libraries are unavailable, the package falls back to deterministic token counting, text/PDF extraction fallbacks, and hash embeddings so the core algorithms remain testable offline.

