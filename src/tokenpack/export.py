from __future__ import annotations

import json
from pathlib import Path

from tokenpack.compression import CompressionConfig, CompressionResult, compress_chunks
from tokenpack.models import Chunk


def ordered_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=lambda chunk: chunk.order_key)


def render_context(
    chunks: list[Chunk],
    include_headers: bool = True,
    header_style: str = "technical",
) -> str:
    parts: list[str] = []
    for number, chunk in enumerate(ordered_chunks(chunks), start=1):
        if include_headers:
            page_info = f", pages {chunk.start_page}-{chunk.end_page}" if chunk.start_page is not None else ""
            if header_style == "technical":
                parts.append(
                    f"[Chunk {number}: id={chunk.id}, source={chunk.source_path}{page_info}, tokens={chunk.token_count}]"
                )
            elif header_style == "source":
                parts.append(f"[Source: {chunk.source_path}{page_info}]")
            elif header_style != "none":
                raise ValueError(f"Unknown context header style: {header_style}")
        parts.append(chunk.text)
    return "\n\n".join(parts).strip() + "\n"


def render_compressed_context(
    chunks: list[Chunk],
    config: CompressionConfig,
    include_headers: bool = True,
) -> tuple[str, CompressionResult]:
    ordered = ordered_chunks(chunks)
    result = compress_chunks(ordered, config)
    parts: list[str] = []
    if include_headers:
        parts.append(
            "[Compressed context: "
            f"compressor={result.metadata.get('compressor')}, "
            f"origin_tokens={result.origin_tokens}, "
            f"compressed_tokens={result.compressed_tokens}, "
            f"saving_rate={result.saving_rate:.1%}]"
        )
    parts.append(result.compressed_prompt)
    return "\n\n".join(part for part in parts if part).strip() + "\n", result


def export_selection(
    selection_path: str | Path,
    output_path: str | Path,
    include_headers: bool = True,
    compression_config: CompressionConfig | None = None,
) -> CompressionResult | None:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(item["chunk"]) for item in payload.get("selected", [])]
    compression_result = None
    if compression_config and compression_config.compressor != "none":
        rendered, compression_result = render_compressed_context(
            chunks,
            compression_config,
            include_headers=include_headers,
        )
    else:
        rendered = render_context(chunks, include_headers=include_headers)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(rendered, encoding="utf-8")
    return compression_result

