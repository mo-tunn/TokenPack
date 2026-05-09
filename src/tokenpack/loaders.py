from __future__ import annotations

import ast
import re
from pathlib import Path

from tokenpack.models import TextBlock

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
SUPPORTED_CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".h"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_CODE_EXTENSIONS | {".pdf"}
SOURCE_TYPES = {"auto", "document", "code", "mixed"}
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CODE_HINT_RE = re.compile(
    r"^\s*(def |class |function |const |let |var |import |from |public |private |protected |if |for |while |return|#include|package )"
)
SECTION_HINT_RE = re.compile(r"^\s*(?:[A-Z][A-Za-z0-9 ,:/-]{2,}|[0-9]+(?:\.[0-9]+)*\s+[A-Z][^\n]{2,})\s*$")


def iter_supported_files(path: str | Path, source_type: str = "auto") -> list[Path]:
    _validate_source_type(source_type)
    root = Path(path)
    if root.is_file():
        return [root] if _supports_file(root, source_type) else []
    return sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and _supports_file(candidate, source_type)
    )


def load_blocks(path: str | Path, source_type: str = "auto") -> list[TextBlock]:
    files = iter_supported_files(path, source_type=source_type)
    blocks: list[TextBlock] = []
    for document_index, file_path in enumerate(files):
        if file_path.suffix.lower() == ".pdf":
            blocks.extend(load_pdf_blocks(file_path, document_index))
        elif file_path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS:
            blocks.extend(load_code_blocks(file_path, document_index))
        else:
            blocks.extend(load_text_blocks(file_path, document_index))
    return blocks


def load_text_blocks(path: str | Path, document_index: int = 0) -> list[TextBlock]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    blocks: list[TextBlock] = []
    paragraph_index = 0
    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        paragraph = re.sub(r"[ \t]+", " ", match.group(0)).strip()
        if not paragraph:
            continue
        blocks.append(
            TextBlock(
                text=paragraph,
                source_path=str(file_path),
                document_index=document_index,
                page=None,
                paragraph_index=paragraph_index,
                char_start=match.start(),
                char_end=match.end(),
                metadata={"content_type": "document"},
            )
        )
        paragraph_index += 1
    return blocks


def load_code_blocks(path: str | Path, document_index: int = 0) -> list[TextBlock]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    language = _language_for_suffix(file_path.suffix)
    if file_path.suffix.lower() == ".py":
        blocks = _load_python_code_blocks(file_path, text, document_index, language)
    else:
        blocks = _load_regex_code_blocks(file_path, text, document_index, language)
    return blocks or [_make_code_block(file_path, document_index, 0, text, 1, _line_count(text), language)]


def load_pdf_blocks(path: str | Path, document_index: int = 0) -> list[TextBlock]:
    file_path = Path(path)
    try:
        return _load_pdf_blocks_pymupdf(file_path, document_index)
    except Exception:
        return _load_pdf_blocks_pypdf(file_path, document_index)


def _load_pdf_blocks_pymupdf(path: Path, document_index: int) -> list[TextBlock]:
    import fitz  # type: ignore

    blocks: list[TextBlock] = []
    paragraph_index = 0
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_blocks = page.get_text("blocks")
            for block in page_blocks:
                x0, y0, x1, y1, text, *_ = block
                for unit in _split_pdf_text(str(text)):
                    blocks.append(
                        TextBlock(
                            text=unit["text"],
                            source_path=str(path),
                            document_index=document_index,
                            page=page_index,
                            paragraph_index=paragraph_index,
                            char_start=0,
                            char_end=len(unit["text"]),
                            bbox=(float(x0), float(y0), float(x1), float(y1)),
                            metadata=unit["metadata"],
                        )
                    )
                    paragraph_index += 1
    return blocks


def _load_pdf_blocks_pypdf(path: Path, document_index: int) -> list[TextBlock]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    blocks: list[TextBlock] = []
    paragraph_index = 0
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for unit in _split_pdf_text(text):
            blocks.append(
                TextBlock(
                    text=unit["text"],
                    source_path=str(path),
                    document_index=document_index,
                    page=page_index,
                    paragraph_index=paragraph_index,
                    char_start=0,
                    char_end=len(unit["text"]),
                    metadata=unit["metadata"],
                )
            )
            paragraph_index += 1
    return blocks


def _validate_source_type(source_type: str) -> None:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unknown source_type: {source_type}")


def _supports_file(path: Path, source_type: str) -> bool:
    suffix = path.suffix.lower()
    if source_type == "document":
        return suffix in SUPPORTED_TEXT_EXTENSIONS or suffix == ".pdf"
    if source_type == "code":
        return suffix in SUPPORTED_CODE_EXTENSIONS
    return suffix in SUPPORTED_EXTENSIONS


def _split_pdf_text(text: str) -> list[dict[str, object]]:
    raw_lines = [line.rstrip() for line in text.replace("\x00", " ").splitlines()]
    units: list[dict[str, object]] = []
    current: list[str] = []
    current_is_code = False

    def flush() -> None:
        nonlocal current, current_is_code
        if not current:
            return
        raw = "\n".join(current).strip()
        current = []
        if not raw:
            return
        if current_is_code:
            units.append({"text": raw, "metadata": {"content_type": "code", "source_format": "pdf"}})
            return
        section_hint = _section_hint(raw)
        for sentence_group in _split_prose_unit(raw):
            metadata = {"content_type": "document", "source_format": "pdf"}
            if section_hint:
                metadata["section_hint"] = section_hint
            units.append({"text": sentence_group, "metadata": metadata})

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            flush()
            current_is_code = False
            continue
        is_code = _looks_like_code_line(stripped)
        if current and is_code != current_is_code:
            flush()
        current.append(stripped if is_code else stripped)
        current_is_code = is_code
    flush()

    if units:
        return units
    compact = re.sub(r"\s+", " ", text).strip()
    return [{"text": unit, "metadata": {"content_type": "document", "source_format": "pdf"}} for unit in _split_prose_unit(compact)]


def _split_prose_unit(text: str, max_sentences: int = 3) -> list[str]:
    clean = re.sub(r"[ \t]+", " ", text).strip()
    if not clean:
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(clean) if sentence.strip()]
    if len(sentences) <= max_sentences:
        return [clean]
    return [" ".join(sentences[index : index + max_sentences]) for index in range(0, len(sentences), max_sentences)]


def _looks_like_code_line(line: str) -> bool:
    if CODE_HINT_RE.search(line):
        return True
    symbols = sum(line.count(char) for char in "{}[]();=<>")
    return symbols >= 3 and len(line.split()) <= 18


def _section_hint(text: str) -> str | None:
    first_line = text.splitlines()[0].strip()
    if SECTION_HINT_RE.match(first_line):
        return first_line[:120]
    return None


def _language_for_suffix(suffix: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
    }.get(suffix.lower(), "text")


def _load_python_code_blocks(path: Path, text: str, document_index: int, language: str) -> list[TextBlock]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _load_regex_code_blocks(path, text, document_index, language)

    lines = text.splitlines()
    symbols = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "lineno")
    ]
    symbols.sort(key=lambda node: (node.lineno, getattr(node, "end_lineno", node.lineno)))

    blocks: list[TextBlock] = []
    cursor = 1
    paragraph_index = 0
    for node in symbols:
        start_line = max(1, node.lineno)
        end_line = max(start_line, getattr(node, "end_lineno", start_line))
        if cursor < start_line:
            prefix = "\n".join(lines[cursor - 1 : start_line - 1]).strip()
            if prefix:
                blocks.append(_make_code_block(path, document_index, paragraph_index, prefix, cursor, start_line - 1, language))
                paragraph_index += 1
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        symbol_text = "\n".join(lines[start_line - 1 : end_line]).strip()
        blocks.append(
            _make_code_block(
                path,
                document_index,
                paragraph_index,
                symbol_text,
                start_line,
                end_line,
                language,
                symbol_name=node.name,
                symbol_kind=kind,
            )
        )
        paragraph_index += 1
        cursor = max(cursor, end_line + 1)

    if cursor <= len(lines):
        suffix = "\n".join(lines[cursor - 1 :]).strip()
        if suffix:
            blocks.append(_make_code_block(path, document_index, paragraph_index, suffix, cursor, len(lines), language))
    return blocks


def _load_regex_code_blocks(path: Path, text: str, document_index: int, language: str) -> list[TextBlock]:
    lines = text.splitlines()
    starts = [index + 1 for index, line in enumerate(lines) if _looks_like_symbol_start(line, language)]
    if not starts:
        return [_make_code_block(path, document_index, 0, text.strip(), 1, _line_count(text), language)] if text.strip() else []

    blocks: list[TextBlock] = []
    paragraph_index = 0
    if starts[0] > 1:
        prefix = "\n".join(lines[: starts[0] - 1]).strip()
        if prefix:
            blocks.append(_make_code_block(path, document_index, paragraph_index, prefix, 1, starts[0] - 1, language))
            paragraph_index += 1
    for offset, start_line in enumerate(starts):
        end_line = starts[offset + 1] - 1 if offset + 1 < len(starts) else len(lines)
        symbol_text = "\n".join(lines[start_line - 1 : end_line]).strip()
        symbol_name, symbol_kind = _parse_symbol_header(lines[start_line - 1], language)
        blocks.append(
            _make_code_block(
                path,
                document_index,
                paragraph_index,
                symbol_text,
                start_line,
                end_line,
                language,
                symbol_name=symbol_name,
                symbol_kind=symbol_kind,
            )
        )
        paragraph_index += 1
    return blocks


def _looks_like_symbol_start(line: str, language: str) -> bool:
    stripped = line.strip()
    if language in {"javascript", "typescript"}:
        return bool(re.match(r"(export\s+)?(async\s+)?function\s+\w+|(?:export\s+)?class\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(", stripped))
    if language == "java":
        return bool(re.match(r"(public|private|protected|static|\s)+\s*(class|interface|enum|\w+[<>\w,\s]*\s+\w+\s*\()", stripped))
    if language == "go":
        return stripped.startswith("func ") or stripped.startswith("type ")
    if language == "rust":
        return bool(re.match(r"(pub\s+)?(fn|struct|enum|impl|trait)\s+", stripped))
    if language in {"c", "cpp"}:
        return bool(re.match(r"[\w:*&<>,\s]+\s+\w+\s*\([^;]*\)\s*\{?", stripped))
    return False


def _parse_symbol_header(line: str, language: str) -> tuple[str | None, str | None]:
    stripped = line.strip()
    if match := re.search(r"\b(class|interface|enum|struct|trait)\s+([A-Za-z_][\w]*)", stripped):
        return match.group(2), match.group(1)
    if match := re.search(r"\bfunction\s+([A-Za-z_][\w]*)", stripped):
        return match.group(1), "function"
    if match := re.search(r"\b(?:def|fn|func)\s+([A-Za-z_][\w]*)", stripped):
        return match.group(1), "function"
    if language in {"javascript", "typescript"} and (match := re.search(r"\b(?:const|let|var)\s+([A-Za-z_][\w]*)\s*=", stripped)):
        return match.group(1), "function"
    if match := re.search(r"\b([A-Za-z_][\w]*)\s*\(", stripped):
        return match.group(1), "function"
    return None, None


def _make_code_block(
    path: Path,
    document_index: int,
    paragraph_index: int,
    text: str,
    start_line: int,
    end_line: int,
    language: str,
    symbol_name: str | None = None,
    symbol_kind: str | None = None,
) -> TextBlock:
    metadata = {
        "content_type": "code",
        "language": language,
        "start_line": start_line,
        "end_line": end_line,
    }
    if symbol_name:
        metadata["symbol_name"] = symbol_name
    if symbol_kind:
        metadata["symbol_kind"] = symbol_kind
    return TextBlock(
        text=text,
        source_path=str(path),
        document_index=document_index,
        paragraph_index=paragraph_index,
        char_start=0,
        char_end=len(text),
        metadata=metadata,
    )


def _line_count(text: str) -> int:
    return max(1, len(text.splitlines()))

