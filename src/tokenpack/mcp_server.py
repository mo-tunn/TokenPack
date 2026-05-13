from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenpack.embeddings import DEFAULT_EMBEDDING_MODEL, make_embedder
from tokenpack.packing import (
    AUTO_BUDGET_RATIO,
    AUTO_MAX_BUDGET,
    AUTO_MIN_BUDGET,
    _infer_pack_output_path,
    pack_source,
)

INLINE_MARKDOWN_LIMIT = 200_000


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    workspace: Path
    allow_any_path: bool = False
    model_name: str = DEFAULT_EMBEDDING_MODEL
    offline_models: bool = False


def pack_context_tool(
    *,
    source: str,
    query: str,
    config: McpServerConfig,
    out: str | None = None,
    overwrite: bool = False,
    budget: int | None = None,
    budget_ratio: float = AUTO_BUDGET_RATIO,
    max_budget: int = AUTO_MAX_BUDGET,
    reserve_output: int | None = None,
    compress: str = "none",
    compression_rate: float = 0.85,
) -> dict[str, Any]:
    source_path = _resolve_workspace_path(source, config)
    output_path = _resolve_workspace_path(out, config) if out else None
    inferred_output = output_path or _infer_pack_output_path(source_path)
    if inferred_output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {inferred_output}\nUse overwrite=True or choose out.")
    embedder = _make_mcp_embedder(config)
    result = pack_source(
        source=source_path,
        query=query,
        embedder=embedder,
        out=output_path,
        overwrite=overwrite,
        budget=budget,
        budget_ratio=budget_ratio,
        min_budget=AUTO_MIN_BUDGET,
        max_budget=max_budget,
        reserve_output=reserve_output,
        compress=compress,
        compression_rate=compression_rate,
        run_root=_run_root(config),
    )
    return _pack_result_payload(result)


def read_packed_context_tool(
    *,
    path: str,
    config: McpServerConfig,
    offset: int = 0,
    max_chars: int = INLINE_MARKDOWN_LIMIT,
) -> dict[str, Any]:
    if offset < 0:
        raise ValueError("offset must be greater than or equal to 0.")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0.")
    target = _resolve_workspace_path(path, config)
    if not target.exists():
        raise ValueError(f"Packed context does not exist: {target}")
    text = target.read_text(encoding="utf-8")
    end = min(len(text), offset + max_chars)
    return {
        "path": str(target),
        "offset": offset,
        "max_chars": max_chars,
        "text": text[offset:end],
        "total_chars": len(text),
        "next_offset": end if end < len(text) else None,
        "truncated": end < len(text),
    }


def build_server(config: McpServerConfig):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP support is not installed. Install it with `pip install tokenpack-rag[mcp]`."
        ) from exc

    mcp = FastMCP("TokenPack-RAG")

    @mcp.tool()
    def pack_context(
        source: str,
        query: str,
        out: str | None = None,
        overwrite: bool = False,
        budget: int | None = None,
        budget_ratio: float = AUTO_BUDGET_RATIO,
        max_budget: int = AUTO_MAX_BUDGET,
        reserve_output: int | None = None,
        compress: str = "none",
        compression_rate: float = 0.85,
    ) -> dict[str, Any]:
        """Pack a local file or folder into an LLM-ready TokenPack Markdown context."""

        return pack_context_tool(
            source=source,
            query=query,
            out=out,
            overwrite=overwrite,
            budget=budget,
            budget_ratio=budget_ratio,
            max_budget=max_budget,
            reserve_output=reserve_output,
            compress=compress,
            compression_rate=compression_rate,
            config=config,
        )

    @mcp.tool()
    def read_packed_context(path: str, offset: int = 0, max_chars: int = INLINE_MARKDOWN_LIMIT) -> dict[str, Any]:
        """Read a packed context artifact, optionally in chunks for very large files."""

        return read_packed_context_tool(path=path, offset=offset, max_chars=max_chars, config=config)

    return mcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenpack-rag-mcp",
        description="Local stdio MCP server for TokenPack-RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", default=".", help="Root folder the MCP server is allowed to read and write.")
    parser.add_argument("--allow-any-path", action="store_true", help="Allow paths outside --workspace.")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence-transformers embedding model.")
    parser.add_argument("--offline-models", action="store_true", help="Use only locally cached embedding model files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = McpServerConfig(
        workspace=Path(args.workspace).resolve(),
        allow_any_path=args.allow_any_path,
        model_name=args.model,
        offline_models=args.offline_models,
    )
    try:
        server = build_server(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    server.run(transport="stdio")
    return 0


def _pack_result_payload(result) -> dict[str, Any]:
    metadata = result.to_metadata()
    markdown = result.markdown
    truncated = len(markdown) > INLINE_MARKDOWN_LIMIT
    if truncated:
        markdown = markdown[:INLINE_MARKDOWN_LIMIT]
    metadata.update(
        {
            "markdown": markdown,
            "markdown_chars": len(result.markdown),
            "markdown_truncated": truncated,
            "next_offset": INLINE_MARKDOWN_LIMIT if truncated else None,
        }
    )
    return metadata


def _resolve_workspace_path(path: str | Path, config: McpServerConfig) -> Path:
    raw = Path(path)
    resolved = (raw if raw.is_absolute() else config.workspace / raw).resolve()
    if config.allow_any_path:
        return resolved
    workspace = config.workspace.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Path is outside the MCP workspace: {resolved}") from exc
    return resolved


def _run_root(config: McpServerConfig) -> Path:
    if config.allow_any_path:
        return Path(".tokenpack/runs")
    return config.workspace.resolve() / ".tokenpack" / "runs"


def _make_mcp_embedder(config: McpServerConfig):
    return make_embedder(
        model_name=config.model_name,
        local_files_only=True if config.offline_models else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
