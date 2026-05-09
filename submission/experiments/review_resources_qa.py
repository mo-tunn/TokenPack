from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

DEFAULT_INPUT = Path("submission/gold/resources_candidate_qa.jsonl")
DEFAULT_OUTPUT = Path("submission/gold/resources_reviewed_qa.jsonl")
DEFAULT_REJECTED = Path("submission/gold/resources_rejected_qa.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactively accept/edit/reject resources QA candidates.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rejected-output", default=str(DEFAULT_REJECTED))
    parser.add_argument("--max-chars", type=int, default=1400)
    args = parser.parse_args()

    accepted: list[dict] = []
    rejected: list[dict] = []
    records = _load_jsonl(Path(args.input))
    for index, record in enumerate(records, start=1):
        print(_render_record(index, len(records), record, max_chars=args.max_chars))
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
            query = input(f"query [{record.get('query', '')}] > ").strip()
            answer = input(f"answer [{record.get('answer', '')}] > ").strip()
            if query:
                record["query"] = query
            if answer:
                record["answer"] = answer
        metadata = dict(record.get("metadata") or {})
        metadata["reviewed"] = True
        metadata["review_status"] = "accepted"
        record["metadata"] = metadata
        record["notes"] = "Human-reviewed resources QA."
        accepted.append(record)

    _write_jsonl(accepted, Path(args.output))
    _write_jsonl(rejected, Path(args.rejected_output))
    print(f"Accepted records written to: {args.output}")
    print(f"Rejected records written to: {args.rejected_output}")
    return 0


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _render_record(index: int, total: int, record: dict, max_chars: int) -> str:
    lines = [
        "=" * 90,
        f"Record {index}/{total}",
        "-" * 90,
        f"QUERY: {record.get('query', '')}",
        "",
        f"ANSWER: {record.get('answer', '')}",
        "",
        f"SOURCE: {record.get('source_path', '')}",
        "",
    ]
    for evidence_index, evidence in enumerate(record.get("evidence", []), start=1):
        text = str(evidence.get("text", ""))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        lines.extend(
            [
                f"EVIDENCE {evidence_index}: page={evidence.get('page')} paragraph={evidence.get('paragraph_index')}",
                textwrap.fill(text, width=100),
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
