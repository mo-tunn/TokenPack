# One-Command Demo

Run this from the repository root:

```bash
tokenpack-rag pack examples/sample_docs/mini_context.txt --query "How does TokenPack-RAG choose a context budget?"
```

Expected output:

```text
examples/sample_docs/mini_context-tp.md
```

The generated Markdown file includes a metadata header, auto-budget details, and
the selected context chunks.
