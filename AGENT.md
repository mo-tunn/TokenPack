# TokenPack-RAG Agent Guide

This file is for coding agents and local assistant tools that need to understand how to use this repository without rereading the whole paper.

## What TokenPack-RAG Does

TokenPack-RAG turns a long file or folder into a smaller Markdown context file for an LLM. It does not call an LLM during the normal `pack` workflow. It performs local embedding, query-aware scoring, and token-budget-aware selection, then writes a `-tp.md` packed context artifact.

Default production pipeline:

```text
structure-aware semantic chunks + evidence-hybrid scoring + hybrid-greedy packing
```

Plain meaning:

- Split sources into chunks that respect structure such as headings, paragraphs, code blocks, and semantic shifts.
- Score chunks by estimated evidence value for the user query.
- Fill the token budget with the highest-value chunks instead of blindly using full context or naive top-k retrieval.

## Default Agent Behavior

When a user asks to compress, pack, summarize, prepare context, or make a large local source easier for an LLM to read, prefer:

```bash
tokenpack-rag pack SOURCE --query "USER QUESTION OR TASK"
```

The default output path is inferred:

- `paper.pdf` -> `paper-tp.md`
- `notes.txt` -> `notes-tp.md`
- `docs/` -> `docs-tp.md`

Existing outputs are protected. Add `--overwrite` only when the user clearly wants replacement.

## Best Combination

For maximum context saving, use TokenPack selection followed by LongLLMLingua compression:

```bash
tokenpack-rag pack SOURCE \
  --query "USER QUESTION OR TASK" \
  --compress llmlingua \
  --longllmlingua \
  --compression-rate 0.50
```

Use this when the user asks for strongest compression, best current setup, paper-style setting, or the TokenPack + LongLLMLingua cascade. It requires the `compression` extra and a locally cached compression model unless `--allow-download` is intentionally provided.

## Installation Profiles

Basic:

```bash
pip install tokenpack-rag
```

Documents and token counting:

```bash
pip install "tokenpack-rag[pdf,office,tokens]"
```

Agents/MCP:

```bash
pip install "tokenpack-rag[mcp,pdf,office,tokens]"
```

Best-combination compression:

```bash
pip install "tokenpack-rag[pdf,office,tokens,compression]"
```

## MCP Usage

TokenPack-RAG can run as a local stdio MCP server:

```bash
tokenpack-rag-mcp --workspace /path/to/project
```

MCP config shape:

```json
{
  "mcpServers": {
    "tokenpack-rag": {
      "command": "tokenpack-rag-mcp",
      "args": ["--workspace", "/path/to/project"]
    }
  }
}
```

Important MCP behavior:

- The server only reads and writes inside `--workspace` by default.
- Use `--allow-any-path` only in trusted local setups.
- Main tool: `pack_context`.
- Helper tool: `read_packed_context` for large output files.

## Supported Inputs

TokenPack-RAG accepts a file or folder. Folders are scanned recursively and unsupported binary/media files are skipped.

Common supported types:

- Text/docs: `.txt`, `.md`, `.rst`, `.adoc`, `.tex`, `.log`
- PDF: `.pdf` with the `pdf` extra
- Web/data: `.html`, `.json`, `.jsonl`, `.csv`, `.tsv`, `.yaml`, `.toml`
- Office: `.docx`, `.pptx`, `.xlsx` with the `office` extra
- Code: Python, JavaScript/TypeScript, Java, Go, Rust, C/C++, C#, PHP, Ruby, Swift, Kotlin, Scala, shell, SQL, CSS/XML, and related variants

## Budget Behavior

If the user does not provide `--budget`, TokenPack-RAG estimates one:

```text
raw_budget = ceil(source_tokens * 0.50)
budget = clamp(raw_budget, min_budget=1200, max_budget=64000)
reserve_output = min(4000, max(512, int(budget * 0.10)))
selection_budget = budget - reserve_output
```

Override controls:

```bash
--budget 32000
--budget-ratio 0.35
--max-budget 128000
--reserve-output 2000
```

## Output Artifacts

Main output:

- Packed Markdown context file, usually `<source>-tp.md`.

Optional explicit artifacts:

```bash
--index-out .tokenpack/source.index.json
--selection-out source-tp.selection.json
```

If explicit artifact paths are not provided, internal run files go under `.tokenpack/runs/<timestamp>/`.

## Important Limitations

- Normal packing does not generate an answer; it prepares context for another LLM or agent.
- QASPER paper metrics are evidence-retention proxies, not human-judged answer quality.
- LongBench v2 answer results are pilot-scale and descriptive.
- LLM answer-quality experiments were not fully human-reviewed.
- Evidence-hybrid scoring weights are engineering defaults and are future calibration work.
- TokenPack cannot recover information missing from the source or lost during extraction.

## Repository Map

- `src/tokenpack/packing.py`: shared one-command packing service used by CLI and MCP.
- `src/tokenpack/cli.py`: `tokenpack-rag` command-line interface.
- `src/tokenpack/mcp_server.py`: local stdio MCP server.
- `src/tokenpack/chunking.py`: structure-aware semantic chunking.
- `src/tokenpack/scoring.py`: production evidence-hybrid scoring.
- `src/tokenpack/scoring_experimental.py`: historical/ablation scoring profiles.
- `src/tokenpack/selectors.py`: budget-aware selectors, including hybrid-greedy as `budget-top-k`.
- `src/tokenpack/loaders.py`: file and folder loading for documents, code, data, PDF, and Office formats.
- `submission/`: paper source, experiments, and result artifacts.

## Agent Rules of Thumb

- Prefer `pack` over lower-level `ingest` + `select` + `export-context` unless the user is doing experiments.
- Prefer the default selection-only command for quick local work.
- Use the LongLLMLingua cascade only when the user wants aggressive compression or the documented best-combination setup.
- Do not claim statistically definitive benchmark wins from the pilot results.
- Do not upload user documents unless the user explicitly asks for a remote workflow.
- Do not overwrite `-tp.md` outputs unless the user asks or passes `--overwrite`.
