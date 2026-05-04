from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from tokenpack.models import TextBlock

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | {".pdf"}


def iter_supported_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_blocks(path: str | Path) -> list[TextBlock]:
    files = iter_supported_files(path)
    blocks: list[TextBlock] = []
    for document_index, file_path in enumerate(files):
        if file_path.suffix.lower() == ".pdf":
            blocks.extend(load_pdf_blocks(file_path, document_index))
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
            )
        )
        paragraph_index += 1
    return blocks


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
                paragraph = re.sub(r"\s+", " ", str(text)).strip()
                if not paragraph:
                    continue
                blocks.append(
                    TextBlock(
                        text=paragraph,
                        source_path=str(path),
                        document_index=document_index,
                        page=page_index,
                        paragraph_index=paragraph_index,
                        char_start=0,
                        char_end=len(paragraph),
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
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
        for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
            paragraph = re.sub(r"\s+", " ", match.group(0)).strip()
            if not paragraph:
                continue
            blocks.append(
                TextBlock(
                    text=paragraph,
                    source_path=str(path),
                    document_index=document_index,
                    page=page_index,
                    paragraph_index=paragraph_index,
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
            paragraph_index += 1
    return blocks

