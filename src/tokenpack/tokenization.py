from __future__ import annotations

import re


class TokenCounter:
    """Token counter backed by tiktoken when available, with an offline fallback."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding = None
        try:
            import tiktoken  # type: ignore

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        # A conservative-ish fallback: split words and punctuation separately.
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

