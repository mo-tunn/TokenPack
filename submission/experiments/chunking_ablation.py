from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenpack.chunking import SemanticThresholdChunker, StructureAwareChunker
from tokenpack.embeddings import EmbeddingCache, make_embedder
from tokenpack.index import ChunkIndex, save_index
from tokenpack.loaders import iter_supported_files, load_text_blocks
from tokenpack.models import Chunk, TextBlock
from tokenpack.scoring import SCORING_PROFILES, score_chunks
from tokenpack.selectors import select_chunks
from tokenpack.tokenization import TokenCounter


DEFAULT_SOURCE = ROOT / "resources"
DEFAULT_OUTPUT_DIR = ROOT / "submission" / "results"
DEFAULT_WORK_DIR = ROOT / ".tokenpack" / "chunking-ablation"
CHUNKERS = ["semantic-threshold", "structure-aware"]
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{3,}")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "between",
    "context",
    "from",
    "have",
    "into",
    "large",
    "language",
    "model",
    "models",
    "that",
    "their",
    "there",
    "these",
    "this",
    "using",
    "with",
}


@dataclass(slots=True)
class EvidenceTemplate:
    query: str
    answer: str
    evidence_terms: set[str]
    source_path: str
    paragraph_index: int


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare chunking strategies under the same knapsack solver.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--budget-ratios", default="0.01,0.03,0.05")
    parser.add_argument("--budgets", help="Optional comma-separated absolute budgets; overrides --budget-ratios.")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--candidate-pool", type=int, default=300)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--base-block-tokens", type=int, default=90)
    parser.add_argument("--min-evidence-terms", type=int, default=12)
    parser.add_argument("--max-documents", type=int, default=8)
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    parser.add_argument(
        "--scoring",
        choices=list(SCORING_PROFILES),
        default="cosine",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    token_counter = TokenCounter()
    embedder = make_embedder(model_name=args.model, local_files_only=True)
    blocks = _load_experiment_blocks(
        Path(args.source),
        max_documents=args.max_documents,
        base_block_tokens=args.base_block_tokens,
        token_counter=token_counter,
    )
    if not blocks:
        raise SystemExit("No supported source blocks found for chunking ablation.")

    total_source_tokens = sum(token_counter.count(block.text) for block in blocks)
    budgets = _parse_budgets(args.budgets, args.budget_ratios, total_source_tokens)
    templates = _templates_from_blocks(
        blocks,
        sample_size=args.sample_size,
        min_evidence_terms=args.min_evidence_terms,
    )
    if not templates:
        raise SystemExit("No evidence templates could be derived from the source blocks.")

    rows: list[dict[str, object]] = []
    for chunker_name in CHUNKERS:
        index = _build_index(
            chunker_name=chunker_name,
            blocks=blocks,
            embedder=embedder,
            work_dir=work_dir,
            target_tokens=args.target_tokens,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            semantic_threshold=args.semantic_threshold,
            token_counter=token_counter,
        )
        for budget in budgets:
            metrics = _evaluate_index(
                index=index,
                templates=templates,
                embedder=embedder,
                budget=budget,
                candidate_pool=args.candidate_pool,
                scoring=args.scoring,
            )
            rows.append(
                {
                    "chunker": chunker_name,
                    "scoring": args.scoring,
                    "budget_ratio": budget / max(1, total_source_tokens),
                    "budget": budget,
                    "document_count": len({block.source_path for block in blocks}),
                    "evidence_count": len(templates),
                    "chunk_count": len(index.chunks),
                    "avg_chunk_tokens": _avg(chunk.token_count for chunk in index.chunks),
                    **metrics,
                }
            )

    csv_path = output_dir / "chunking_ablation.csv"
    table_path = output_dir / "chunking_ablation_table.tex"
    paper_table_path = ROOT / "submission" / "paper" / "tables" / "chunking_ablation_table.tex"
    _write_csv(rows, csv_path)
    _write_latex(rows, table_path)
    _write_latex(rows, paper_table_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")
    return 0


def _load_experiment_blocks(
    source: Path,
    max_documents: int,
    base_block_tokens: int,
    token_counter: TokenCounter,
) -> list[TextBlock]:
    files = iter_supported_files(source)[:max_documents]
    blocks: list[TextBlock] = []
    for document_index, file_path in enumerate(files):
        if file_path.suffix.lower() == ".pdf":
            blocks.extend(_load_pdf_sentence_blocks(file_path, document_index, base_block_tokens, token_counter))
        else:
            blocks.extend(load_text_blocks(file_path, document_index))
    return blocks


def _load_pdf_sentence_blocks(
    path: Path,
    document_index: int,
    base_block_tokens: int,
    token_counter: TokenCounter,
) -> list[TextBlock]:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    blocks: list[TextBlock] = []
    paragraph_index = 0
    for page_index, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if not text:
            continue
        current: list[str] = []
        current_tokens = 0
        for sentence in SENTENCE_RE.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = max(1, token_counter.count(sentence))
            if current and current_tokens + sentence_tokens > base_block_tokens:
                blocks.append(
                    _make_block(path, document_index, page_index, paragraph_index, " ".join(current))
                )
                paragraph_index += 1
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sentence_tokens
        if current:
            blocks.append(_make_block(path, document_index, page_index, paragraph_index, " ".join(current)))
            paragraph_index += 1
    return blocks


def _make_block(path: Path, document_index: int, page: int, paragraph_index: int, text: str) -> TextBlock:
    return TextBlock(
        text=text,
        source_path=str(path),
        document_index=document_index,
        page=page,
        paragraph_index=paragraph_index,
        char_start=0,
        char_end=len(text),
    )


def _build_index(
    chunker_name: str,
    blocks: list[TextBlock],
    embedder: object,
    work_dir: Path,
    target_tokens: int,
    min_tokens: int,
    max_tokens: int,
    semantic_threshold: float,
    token_counter: TokenCounter,
) -> ChunkIndex:
    cache = EmbeddingCache(work_dir / f"{chunker_name}.embeddings.json")
    if chunker_name == "structure-aware":
        block_embeddings = cache.get_or_embed([block.text for block in blocks], embedder)
        chunker = StructureAwareChunker(
            target_tokens,
            min_tokens,
            max_tokens,
            token_counter=token_counter,
            block_embeddings=block_embeddings,
            semantic_threshold=semantic_threshold,
        )
    elif chunker_name == "semantic-threshold":
        block_embeddings = cache.get_or_embed([block.text for block in blocks], embedder)
        chunker = SemanticThresholdChunker(
            block_embeddings,
            similarity_threshold=semantic_threshold,
            target_tokens=target_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    else:
        raise ValueError(f"Unknown chunker: {chunker_name}")
    chunks = chunker.chunk(blocks)
    embeddings = cache.get_or_embed([chunk.text for chunk in chunks], embedder)
    index = ChunkIndex(chunks=chunks, embeddings=embeddings, model_name=embedder.model_name)
    save_index(index, work_dir / f"{chunker_name}.json")
    return index


def _templates_from_blocks(
    blocks: list[TextBlock],
    sample_size: int,
    min_evidence_terms: int,
) -> list[EvidenceTemplate]:
    candidates = [
        (index, block, _content_terms(block.text))
        for index, block in enumerate(blocks)
    ]
    candidates = [
        (index, block, terms)
        for index, block, terms in candidates
        if len(set(terms)) >= min_evidence_terms
    ]
    templates: list[EvidenceTemplate] = []
    for _, block, terms in _spaced_items(candidates, sample_size):
        keywords = _keywords(terms, limit=7)
        evidence_terms = set(_keywords(terms, limit=32))
        templates.append(
            EvidenceTemplate(
                query=" ".join(keywords),
                answer=_first_sentence(block.text),
                evidence_terms=evidence_terms,
                source_path=block.source_path,
                paragraph_index=block.paragraph_index,
            )
        )
    return templates


def _evaluate_index(
    index: ChunkIndex,
    templates: list[EvidenceTemplate],
    embedder: object,
    budget: int,
    candidate_pool: int,
    scoring: str,
    scorer=score_chunks,
) -> dict[str, float]:
    totals = {
        "evidence_term_recall": 0.0,
        "complete_evidence_rate": 0.0,
        "avg_used_tokens": 0.0,
        "budget_utilization": 0.0,
        "redundancy_score": 0.0,
        "latency_seconds": 0.0,
    }
    for template in templates:
        query_embedding = embedder.embed([template.query])[0]
        scored = scorer(
            query_embedding,
            index.chunks,
            index.embeddings,
            scoring=scoring,
            query_text=template.query,
        )
        started = time.perf_counter()
        result = select_chunks(
            scored,
            strategy="knapsack",
            budget=budget,
            candidate_pool=candidate_pool,
            embeddings=index.embeddings,
        )
        elapsed = time.perf_counter() - started
        selected_terms = set(_content_terms(" ".join(item.chunk.text for item in result.selected)))
        recall = len(template.evidence_terms & selected_terms) / max(1, len(template.evidence_terms))
        totals["evidence_term_recall"] += recall
        totals["complete_evidence_rate"] += 1.0 if recall >= 0.80 else 0.0
        totals["avg_used_tokens"] += result.used_tokens
        totals["budget_utilization"] += result.used_tokens / max(1, budget)
        totals["redundancy_score"] += _redundancy_score([item.chunk for item in result.selected])
        totals["latency_seconds"] += elapsed

    runs = max(1, len(templates))
    return {key: value / runs for key, value in totals.items()}


def _redundancy_score(chunks: list[Chunk]) -> float:
    if len(chunks) < 2:
        return 0.0
    sources = [set(_content_terms(chunk.text)) for chunk in chunks]
    total = 0.0
    comparisons = 0
    for left_index, left in enumerate(sources):
        for right in sources[left_index + 1 :]:
            total += len(left & right) / max(1, len(left | right))
            comparisons += 1
    return total / max(1, comparisons)


def _content_terms(text: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in TOKEN_RE.finditer(text)
        if match.group(0).lower() not in STOPWORDS
    ]


def _keywords(terms: Iterable[str], limit: int) -> list[str]:
    return [term for term, _ in Counter(terms).most_common(limit)]


def _first_sentence(text: str) -> str:
    return SENTENCE_RE.split(text.strip(), maxsplit=1)[0][:500]


def _spaced_items(items: list[object], sample_size: int) -> list[object]:
    if len(items) <= sample_size:
        return items
    step = len(items) / sample_size
    return [items[min(len(items) - 1, int(index * step))] for index in range(sample_size)]


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "chunker",
        "scoring",
        "budget_ratio",
        "budget",
        "document_count",
        "evidence_count",
        "chunk_count",
        "avg_chunk_tokens",
        "evidence_term_recall",
        "complete_evidence_rate",
        "avg_used_tokens",
        "budget_utilization",
        "redundancy_score",
        "latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Chunking Ablation With the Same Knapsack Solver}",
        r"\label{tab:chunking-ablation}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Chunker & Budget & Chunks & Avg. Tok. & Term Recall & Complete & Avg. Used & Util. \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_text(str(row["chunker"])),
                    f"{float(row['budget_ratio']) * 100:.0f}\\%",
                    str(row["chunk_count"]),
                    _fmt(row["avg_chunk_tokens"]),
                    _fmt(row["evidence_term_recall"]),
                    _fmt(row["complete_evidence_rate"]),
                    _fmt(row["avg_used_tokens"]),
                    _fmt(row["budget_utilization"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_budgets(raw_budgets: str | None, raw_ratios: str, total_tokens: int) -> list[int]:
    if raw_budgets:
        budgets = [int(item.strip()) for item in raw_budgets.split(",") if item.strip()]
    else:
        ratios = [float(item.strip()) for item in raw_ratios.split(",") if item.strip()]
        budgets = [max(1, int(total_tokens * ratio)) for ratio in ratios]
    if not budgets:
        raise ValueError("At least one budget or budget ratio is required.")
    return budgets


def _avg(values: Iterable[int]) -> float:
    items = list(values)
    return sum(items) / max(1, len(items))


def _fmt(value: object) -> str:
    return f"{float(value):.3f}"


def _latex_text(value: str) -> str:
    return value.replace("_", r"\_")


if __name__ == "__main__":
    raise SystemExit(main())
