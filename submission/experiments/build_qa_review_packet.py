from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chunking_ablation import (  # type: ignore
    DEFAULT_SOURCE,
    _content_terms,
    _first_sentence,
    _keywords,
    _load_experiment_blocks,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tokenpack.tokenization import TokenCounter


DEFAULT_OUTPUT = ROOT / "submission" / "gold" / "resources_candidate_qa.jsonl"
DEFAULT_PACKET = ROOT / "submission" / "gold" / "resources_qa_review_packet.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reviewable multi-evidence QA candidates from resources PDFs.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--review-packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--max-documents", type=int, default=8)
    parser.add_argument("--base-block-tokens", type=int, default=90)
    parser.add_argument("--min-evidence-terms", type=int, default=12)
    args = parser.parse_args()

    token_counter = TokenCounter()
    blocks = _load_experiment_blocks(
        Path(args.source),
        max_documents=args.max_documents,
        base_block_tokens=args.base_block_tokens,
        token_counter=token_counter,
    )
    candidates = _build_candidates(
        blocks=blocks,
        sample_size=args.sample_size,
        min_evidence_terms=args.min_evidence_terms,
    )
    _write_jsonl(candidates, Path(args.output))
    _write_packet(candidates, Path(args.review_packet))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.review_packet}")
    return 0


def _build_candidates(blocks, sample_size: int, min_evidence_terms: int) -> list[dict]:
    by_source: dict[str, list] = {}
    for block in blocks:
        terms = set(_content_terms(block.text))
        if len(terms) >= min_evidence_terms:
            by_source.setdefault(block.source_path, []).append(block)

    records: list[dict] = []
    for source_path, source_blocks in by_source.items():
        for left, right in zip(source_blocks, source_blocks[1:]):
            left_terms = _content_terms(left.text)
            right_terms = _content_terms(right.text)
            if len(set(left_terms)) < min_evidence_terms or len(set(right_terms)) < min_evidence_terms:
                continue
            left_keywords = _keywords(left_terms, limit=4)
            right_keywords = _keywords(right_terms, limit=4)
            query = _question(left_keywords, right_keywords)
            answer = f"{_first_sentence(left.text)} {_first_sentence(right.text)}"
            records.append(
                {
                    "query": query,
                    "answer": answer[:900],
                    "source_path": source_path,
                    "evidence": [
                        _evidence_block(left, left_terms),
                        _evidence_block(right, right_terms),
                    ],
                    "notes": "Candidate multi-evidence QA; needs human review before use.",
                    "metadata": {
                        "dataset": "resources_multi_evidence_candidate",
                        "reviewed": False,
                        "difficulty": "medium",
                    },
                }
            )

    if len(records) <= sample_size:
        return records
    step = len(records) / sample_size
    return [records[min(len(records) - 1, int(index * step))] for index in range(sample_size)]


def _question(left_keywords: list[str], right_keywords: list[str]) -> str:
    left = ", ".join(left_keywords[:3]) or "the first passage"
    right = ", ".join(right_keywords[:3]) or "the following passage"
    return f"What does the document say about {left}, and how is it connected to {right}?"


def _evidence_block(block, terms: list[str]) -> dict:
    return {
        "source_path": block.source_path,
        "page": block.page,
        "paragraph_index": block.paragraph_index,
        "text": block.text,
        "evidence_terms": _keywords(terms, limit=32),
    }


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_packet(records: list[dict], path: Path) -> None:
    sections = [
        "# Resources QA Review Packet",
        "",
        "Accept only records where the query is meaningful and the answer is supported by both evidence passages.",
        "Edit awkward queries/answers before accepting. Reject records that are just bibliography noise, broken PDF text, or unsupported.",
        "",
    ]
    for index, record in enumerate(records, start=1):
        sections.extend(
            [
                f"## Record {index}/{len(records)}",
                "",
                f"Decision: [ ] Accept  [ ] Edit  [ ] Reject",
                "",
                f"Query: {record['query']}",
                "",
                f"Answer: {record['answer']}",
                "",
                f"Source: `{record['source_path']}`",
                "",
            ]
        )
        for evidence_index, evidence in enumerate(record["evidence"], start=1):
            sections.extend(
                [
                    f"Evidence {evidence_index} page {evidence.get('page')}, paragraph {evidence.get('paragraph_index')}:",
                    "",
                    "```text",
                    textwrap.fill(evidence["text"], width=100),
                    "```",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
