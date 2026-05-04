from __future__ import annotations

import hashlib
from collections.abc import Sequence

from tokenpack.embeddings import cosine
from tokenpack.models import Chunk, TextBlock
from tokenpack.tokenization import TokenCounter


class ParagraphGroupChunker:
    """Group neighboring paragraphs while preserving document order and metadata."""

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

    def chunk(self, blocks: Sequence[TextBlock]) -> list[Chunk]:
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
            good_enough = current_tokens >= self.min_tokens
            if current and (starts_new_doc or (would_exceed and good_enough)):
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
        return chunks

    def _split_large_block(self, block_id: int, block: TextBlock) -> list[Chunk]:
        words = block.text.split()
        chunks: list[Chunk] = []
        current_words: list[str] = []
        token_start = 0
        paragraph_offset = 0
        for word in words:
            current_words.append(word)
            if self.token_counter.count(" ".join(current_words)) >= self.target_tokens:
                text = " ".join(current_words)
                chunks.append(
                    self._make_chunk(
                        [(block_id, block, self.token_counter.count(text))],
                        text_override=text,
                        suffix=f"split-{paragraph_offset}",
                        char_start=token_start,
                        char_end=token_start + len(text),
                    )
                )
                token_start += len(text) + 1
                paragraph_offset += 1
                current_words = []
        if current_words:
            text = " ".join(current_words)
            chunks.append(
                self._make_chunk(
                    [(block_id, block, self.token_counter.count(text))],
                    text_override=text,
                    suffix=f"split-{paragraph_offset}",
                    char_start=token_start,
                    char_end=token_start + len(text),
                )
            )
        return chunks

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
    ) -> Chunk:
        blocks = [item[1] for item in items]
        text = text_override if text_override is not None else "\n\n".join(block.text for block in blocks)
        token_count = max(1, self.token_counter.count(text))
        start_page = next((block.page for block in blocks if block.page is not None), None)
        end_page = next((block.page for block in reversed(blocks) if block.page is not None), start_page)
        digest = hashlib.sha1(
            f"{blocks[0].source_path}:{blocks[0].paragraph_index}:{blocks[-1].paragraph_index}:{suffix}:{text[:80]}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
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
            metadata={"bbox": [block.bbox for block in blocks if block.bbox is not None]},
        )


class SemanticThresholdChunker(ParagraphGroupChunker):
    """Start a new paragraph group when adjacent block embeddings indicate topic drift."""

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

