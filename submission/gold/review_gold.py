from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {chunk["id"]: chunk for chunk in payload.get("chunks", [])}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_number}: {exc}") from exc
            records.append(record)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evidence_blocks(record: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for chunk_id in record.get("evidence_chunk_ids", []):
        chunk = chunks.get(chunk_id)
        if chunk is None:
            blocks.append({"id": chunk_id, "missing": True})
        else:
            blocks.append(chunk)
    return blocks


def render_record(
    index: int,
    total: int,
    record: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    lines = [
        "=" * 88,
        f"Record {index}/{total}",
        "-" * 88,
        f"QUERY: {record.get('query', '')}",
        "",
        f"ANSWER: {record.get('answer', '')}",
        "",
        "RAG EVIDENCE DATA USED BY THIS RECORD:",
    ]

    for block in evidence_blocks(record, chunks):
        if block.get("missing"):
            lines.append(f"- Missing chunk id: {block['id']}")
            continue

        text = str(block.get("text", ""))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        wrapped = textwrap.indent(textwrap.fill(text, width=96), "  ")
        lines.extend(
            [
                f"- chunk_id: {block.get('id')}",
                f"  source_path: {block.get('source_path')}",
                f"  pages: {block.get('start_page')} - {block.get('end_page')}",
                f"  token_count: {block.get('token_count')}",
                "  text:",
                wrapped,
                "",
            ]
        )
    return "\n".join(lines)


def export_markdown(
    records: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    path: Path,
    *,
    max_chars: int,
) -> None:
    sections = [
        "# Gold Evidence Review Packet",
        "",
        "Use this file to check whether each candidate query and answer is supported by the listed RAG evidence chunk.",
        "",
        "Review decision guide:",
        "",
        "- Accept if the answer is directly supported by the evidence text.",
        "- Edit if the evidence is useful but the query or answer is awkward.",
        "- Reject if the query is meaningless or the evidence does not support the answer.",
        "",
    ]
    for idx, record in enumerate(records, start=1):
        sections.append("```text")
        sections.append(render_record(idx, len(records), record, chunks, max_chars=max_chars))
        sections.append("```")
        sections.append("")
        sections.append("Decision: [ ] Accept  [ ] Edit  [ ] Reject")
        sections.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections), encoding="utf-8")


def interactive_review(
    records: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
    output: Path,
    rejected_output: Path,
    *,
    max_chars: int,
) -> None:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        print(render_record(idx, len(records), record, chunks, max_chars=max_chars))
        print("Choose: [a]ccept, [e]dit, [r]eject, [s]kip, [q]uit")
        choice = input("> ").strip().lower() or "a"

        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "r":
            rejected.append(record)
            continue
        if choice == "e":
            current_query = str(record.get("query", ""))
            current_answer = str(record.get("answer", ""))
            print("New query. Leave empty to keep current.")
            new_query = input(f"query [{current_query}] > ").strip()
            print("New answer. Leave empty to keep current.")
            new_answer = input(f"answer [{current_answer}] > ").strip()
            if new_query:
                record["query"] = new_query
            if new_answer:
                record["answer"] = new_answer

        record["notes"] = "Human-reviewed gold evidence."
        metadata = dict(record.get("metadata") or {})
        metadata["reviewed"] = True
        metadata["review_status"] = "accepted"
        record["metadata"] = metadata
        accepted.append(record)

    write_jsonl(accepted, output)
    write_jsonl(rejected, rejected_output)
    print(f"\nAccepted records written to: {output}")
    print(f"Rejected records written to: {rejected_output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Review candidate gold evidence with its RAG source chunks.")
    parser.add_argument("--index", default=".tokenpack/late-pdf-index.json", help="TokenPack index JSON.")
    parser.add_argument("--input", default="submission/gold/candidate_gold.jsonl", help="Candidate gold JSONL.")
    parser.add_argument("--output", default="submission/gold/gold_reviewed.jsonl", help="Accepted reviewed JSONL.")
    parser.add_argument("--rejected-output", default="submission/gold/gold_rejected.jsonl")
    parser.add_argument("--export-markdown", default="", help="Write a human-readable review packet and exit.")
    parser.add_argument("--max-chars", type=int, default=1800, help="Max evidence characters shown per chunk.")
    args = parser.parse_args()

    chunks = load_index(Path(args.index))
    records = load_jsonl(Path(args.input))

    if args.export_markdown:
        export_markdown(records, chunks, Path(args.export_markdown), max_chars=args.max_chars)
        print(f"Review packet written to: {args.export_markdown}")
        return 0

    interactive_review(
        records,
        chunks,
        Path(args.output),
        Path(args.rejected_output),
        max_chars=args.max_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
