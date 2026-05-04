from __future__ import annotations

import json
from pathlib import Path

from tokenpack.models import Chunk


def ordered_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=lambda chunk: chunk.order_key)


def render_context(chunks: list[Chunk], include_headers: bool = True) -> str:
    parts: list[str] = []
    for number, chunk in enumerate(ordered_chunks(chunks), start=1):
        if include_headers:
            page_info = f", pages {chunk.start_page}-{chunk.end_page}" if chunk.start_page is not None else ""
            parts.append(
                f"[Chunk {number}: id={chunk.id}, source={chunk.source_path}{page_info}, tokens={chunk.token_count}]"
            )
        parts.append(chunk.text)
    return "\n\n".join(parts).strip() + "\n"


def export_selection(selection_path: str | Path, output_path: str | Path, include_headers: bool = True) -> None:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(item["chunk"]) for item in payload.get("selected", [])]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(render_context(chunks, include_headers=include_headers), encoding="utf-8")

