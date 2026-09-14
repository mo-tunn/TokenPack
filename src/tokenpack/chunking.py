from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from tokenpack.embeddings import cosine
from tokenpack.models import Chunk, TextBlock
from tokenpack.tokenization import TokenCounter

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class _ChunkGroupBase:
    """Shared token-bounded grouping helpers for structure-aware chunkers."""

    def __init__(
        self,
        target_tokens: int = 650,
        min_tokens: int = 120,
        max_tokens: int = 900,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if min_tokens > target_tokens or target_tokens > max_tokens:
            raise ValueError("Expected min_tokens <= target_tokens <= max_tokens.")
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()

    def _split_large_block(self, block_id: int, block: TextBlock) -> list[Chunk]:
        units = self._split_block_units(block)
        chunks: list[Chunk] = []
        current_units: list[str] = []
        current_tokens = 0
        split_offset = 0
        source_cursor = 0

        def flush() -> None:
            nonlocal current_units, current_tokens, split_offset, source_cursor
            if not current_units:
                return
            separator = "\n" if block.metadata.get("content_type") == "code" else " "
            text = separator.join(current_units).strip()
            if text:
                span_start, span_end = self._locate_source_span(block.text, current_units, source_cursor)
                source_cursor = span_end
                chunks.append(
                    self._make_chunk(
                        [(block_id, block, self.token_counter.count(text))],
                        text_override=text,
                        suffix=f"split-{split_offset}",
                        char_start=block.char_start + span_start,
                        char_end=block.char_start + span_end,
                        metadata_overrides=self._split_metadata(block, span_start, span_end),
                    )
                )
            split_offset += 1
            current_units = []
            current_tokens = 0

        for unit in units:
            unit_tokens = max(1, self.token_counter.count(unit))
            if unit_tokens > self.max_tokens:
                flush()
                for piece in self._split_oversized_unit(unit):
                    span_start, span_end = self._locate_source_span(block.text, [piece], source_cursor)
                    source_cursor = span_end
                    chunks.append(
                        self._make_chunk(
                            [(block_id, block, self.token_counter.count(piece))],
                            text_override=piece,
                            suffix=f"split-{split_offset}",
                            char_start=block.char_start + span_start,
                            char_end=block.char_start + span_end,
                            metadata_overrides=self._split_metadata(block, span_start, span_end),
                        )
                    )
                    split_offset += 1
                continue
            if current_units and current_tokens + unit_tokens > self.max_tokens:
                flush()
            current_units.append(unit)
            current_tokens += unit_tokens
            if current_tokens >= self.target_tokens:
                flush()
        flush()
        return chunks

    def _split_block_units(self, block: TextBlock) -> list[str]:
        text = block.text.strip()
        if not text:
            return []
        if block.metadata.get("content_type") == "code":
            units = [line.rstrip() for line in text.splitlines() if line.strip()]
            return units or [text]
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
        units: list[str] = []
        for paragraph in paragraphs or [text]:
            sentences = [sentence.strip() for sentence in SENTENCE_RE.split(paragraph) if sentence.strip()]
            units.extend(sentences or [paragraph])
        return units

    def _split_oversized_unit(self, unit: str) -> list[str]:
        words = list(re.finditer(r"\S+", unit))
        if not words:
            return [unit]
        pieces: list[str] = []
        piece_start = words[0].start()
        for word in words:
            candidate = unit[piece_start : word.end()]
            if self.token_counter.count(candidate) >= self.target_tokens:
                pieces.append(candidate.strip())
                piece_start = word.end()
                while piece_start < len(unit) and unit[piece_start].isspace():
                    piece_start += 1
        if piece_start < len(unit):
            pieces.append(unit[piece_start:].strip())
        return [piece for piece in pieces if piece] or [unit]

    def _locate_source_span(self, source: str, parts: list[str], cursor: int) -> tuple[int, int]:
        span_start: int | None = None
        span_end = cursor
        search_from = cursor
        for part in parts:
            index = source.find(part, search_from)
            if index < 0:
                fallback_start = cursor if span_start is None else span_start
                fallback_end = min(len(source), max(search_from, fallback_start) + len(part))
                return fallback_start, fallback_end
            if span_start is None:
                span_start = index
            span_end = index + len(part)
            search_from = span_end
        return (span_start if span_start is not None else cursor), span_end

    def _split_metadata(self, block: TextBlock, span_start: int, span_end: int) -> dict[str, Any]:
        start_line = block.metadata.get("start_line")
        if not isinstance(start_line, int):
            return {}
        return {
            "start_line": start_line + block.text[:span_start].count("\n"),
            "end_line": start_line + block.text[:span_end].count("\n"),
        }

    def _flush(self, items: list[tuple[int, TextBlock, int]]) -> list[Chunk]:
        if not items:
            return []
        return [self._make_chunk(items)]

    def _make_chunk(
        self,
        items: list[tuple[int, TextBlock, int]],
        text_override: str | None = None,
        suffix: str = "",
        char_start: int | None = None,
        char_end: int | None = None,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> Chunk:
        blocks = [item[1] for item in items]
        text = text_override if text_override is not None else "\n\n".join(block.text for block in blocks)
        token_count = max(1, self.token_counter.count(text))
        start_page = next((block.page for block in blocks if block.page is not None), None)
        end_page = next((block.page for block in reversed(blocks) if block.page is not None), start_page)
        digest_input = (
            f"{blocks[0].source_path}:{blocks[0].paragraph_index}:"
            f"{blocks[-1].paragraph_index}:{suffix}:{text[:80]}"
        )
        digest = hashlib.sha1(digest_input.encode("utf-8", errors="replace")).hexdigest()[:12]
        metadata = self._chunk_metadata(blocks)
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return Chunk(
            id=f"chunk-{digest}",
            text=text,
            source_path=blocks[0].source_path,
            document_index=blocks[0].document_index,
            start_page=start_page,
            end_page=end_page,
            start_paragraph=blocks[0].paragraph_index,
            end_paragraph=blocks[-1].paragraph_index,
            char_start=blocks[0].char_start if char_start is None else char_start,
            char_end=blocks[-1].char_end if char_end is None else char_end,
            token_count=token_count,
            block_ids=[item[0] for item in items],
            metadata=metadata,
        )

    def _chunk_metadata(self, blocks: list[TextBlock]) -> dict[str, Any]:
        metadata: dict[str, Any] = {"bbox": [block.bbox for block in blocks if block.bbox is not None]}
        all_keys = {key for block in blocks for key in block.metadata}
        for key in all_keys:
            values = [block.metadata.get(key) for block in blocks if key in block.metadata]
            unique_values = {repr(value): value for value in values}
            if len(unique_values) == 1:
                metadata[key] = values[0]
            elif key == "content_type":
                metadata[key] = "mixed"
        start_lines = [block.metadata.get("start_line") for block in blocks if isinstance(block.metadata.get("start_line"), int)]
        end_lines = [block.metadata.get("end_line") for block in blocks if isinstance(block.metadata.get("end_line"), int)]
        if start_lines:
            metadata["start_line"] = min(start_lines)
        if end_lines:
            metadata["end_line"] = max(end_lines)
        return metadata


class StructureAwareChunker(_ChunkGroupBase):
    """Chunk documents and code using structural metadata plus semantic drift."""

    def __init__(
        self,
        target_tokens: int = 650,
        min_tokens: int = 120,
        max_tokens: int = 900,
        token_counter: TokenCounter | None = None,
        block_embeddings: Sequence[list[float]] | None = None,
        semantic_threshold: float = 0.35,
    ) -> None:
        super().__init__(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        self.block_embeddings = block_embeddings
        self.semantic_threshold = semantic_threshold

    def chunk(self, blocks: Sequence[TextBlock]) -> list[Chunk]:
        if self.block_embeddings is not None and len(self.block_embeddings) != len(blocks):
            raise ValueError("StructureAwareChunker requires one embedding per text block when semantic boundaries are enabled.")

        chunks: list[Chunk] = []
        current: list[tuple[int, TextBlock, int]] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            chunks.extend(self._flush(current))
            current = []
            current_tokens = 0

        for block_id, block in enumerate(blocks):
            block_tokens = max(1, self.token_counter.count(block.text))
            content_type = block.metadata.get("content_type", "document")
            is_symbol = bool(block.metadata.get("symbol_name") or block.metadata.get("symbol_kind"))

            if content_type == "code" and is_symbol:
                flush()
                if block_tokens > self.max_tokens:
                    chunks.extend(self._split_large_block(block_id, block))
                else:
                    chunks.extend(self._flush([(block_id, block, block_tokens)]))
                continue

            if block_tokens > self.max_tokens:
                flush()
                chunks.extend(self._split_large_block(block_id, block))
                continue

            if current and self._structural_boundary(current[-1][1], block):
                flush()

            would_exceed = current_tokens + block_tokens > self.max_tokens
            topic_shift = self._semantic_boundary(block_id, block, current)
            good_enough = current_tokens >= self.min_tokens
            if current and good_enough and (would_exceed or topic_shift):
                flush()

            current.append((block_id, block, block_tokens))
            current_tokens += block_tokens

            if current_tokens >= self.target_tokens:
                flush()

        flush()
        for chunk in chunks:
            chunk.metadata["chunker"] = "structure-aware"
            if self.block_embeddings is not None:
                chunk.metadata["semantic_threshold"] = self.semantic_threshold
        return chunks

    def _structural_boundary(self, previous: TextBlock, current: TextBlock) -> bool:
        if previous.source_path != current.source_path:
            return True
        previous_type = previous.metadata.get("content_type", "document")
        current_type = current.metadata.get("content_type", "document")
        if previous_type != current_type:
            return True
        if previous.metadata.get("section_hint") != current.metadata.get("section_hint"):
            return bool(previous.metadata.get("section_hint") or current.metadata.get("section_hint"))
        return False

    def _semantic_boundary(self, block_id: int, block: TextBlock, current: list[tuple[int, TextBlock, int]]) -> bool:
        if self.block_embeddings is None or not current:
            return False
        previous_id, previous, _ = current[-1]
        if previous.source_path != block.source_path:
            return False
        if previous.metadata.get("content_type", "document") != "document":
            return False
        if block.metadata.get("content_type", "document") != "document":
            return False
        similarity = cosine(self.block_embeddings[previous_id], self.block_embeddings[block_id])
        return similarity < self.semantic_threshold


class SemanticThresholdChunker(_ChunkGroupBase):
    """Start a new chunk when adjacent block embeddings indicate topic drift."""

    def __init__(
        self,
        block_embeddings: Sequence[list[float]],
        similarity_threshold: float = 0.35,
        target_tokens: int = 650,
        min_tokens: int = 120,
        max_tokens: int = 900,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        self.block_embeddings = block_embeddings
        self.similarity_threshold = similarity_threshold

    def chunk(self, blocks: Sequence[TextBlock]) -> list[Chunk]:
        if len(self.block_embeddings) != len(blocks):
            raise ValueError("SemanticThresholdChunker requires one embedding per text block.")

        chunks: list[Chunk] = []
        current: list[tuple[int, TextBlock, int]] = []
        current_tokens = 0

        for block_id, block in enumerate(blocks):
            block_tokens = max(1, self.token_counter.count(block.text))
            if block_tokens > self.max_tokens:
                chunks.extend(self._flush(current))
                current = []
                current_tokens = 0
                chunks.extend(self._split_large_block(block_id, block))
                continue

            starts_new_doc = current and block.source_path != current[-1][1].source_path
            would_exceed = current_tokens + block_tokens > self.max_tokens
            topic_shift = self._topic_shift(block_id, current)
            good_enough = current_tokens >= self.min_tokens

            if current and (starts_new_doc or (would_exceed and good_enough) or (topic_shift and good_enough)):
                chunks.extend(self._flush(current))
                current = []
                current_tokens = 0

            current.append((block_id, block, block_tokens))
            current_tokens += block_tokens

            if current_tokens >= self.target_tokens:
                chunks.extend(self._flush(current))
                current = []
                current_tokens = 0

        chunks.extend(self._flush(current))
        for chunk in chunks:
            chunk.metadata["chunker"] = "semantic-threshold"
            chunk.metadata["similarity_threshold"] = self.similarity_threshold
        return chunks

    def _topic_shift(self, block_id: int, current: list[tuple[int, TextBlock, int]]) -> bool:
        if not current:
            return False
        previous_id = current[-1][0]
        if current[-1][1].source_path != current[0][1].source_path:
            return False
        similarity = cosine(self.block_embeddings[previous_id], self.block_embeddings[block_id])
        return similarity < self.similarity_threshold

