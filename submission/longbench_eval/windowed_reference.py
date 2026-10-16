"""Make an honest, context-window-safe reference baseline for the 60-case pilot.

The five-method design becomes:
- windowed-reference-210k (full when fits; deterministic head+tail otherwise)
- production-rag-50
- tokenpack-50
- only-longllmlingua-rate050
- tokenpack-50+longllmlingua-rate050

Other methods are never trimmed silently. Keep this transformed run separate
from earlier, true full-context results.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from tokenpack.tokenization import TokenCounter

REFERENCE_NAME = "windowed-reference-210k"
ORIGINAL_NAME = "full-context"
EXPECTED_METHODS = {
    "full-context", "production-rag-50", "tokenpack-50",
    "only-longllmlingua-rate050", "tokenpack-50+longllmlingua-rate050",
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def adapt(rows: list[dict], *, token_limit: int, counter: TokenCounter) -> tuple[list[dict], dict]:
    if token_limit < 100:
        raise ValueError("Token guard must be at least 100")
    if counter._encoding is None:
        raise RuntimeError("tiktoken must be available")
    groups: dict[str,list[dict]] = collections.defaultdict(list)
    for x in rows:
        groups[str(x["case_id"])].append(x)
    if len(groups) != 60:
        raise ValueError(f"Expected exactly 60 distinct cases, found {len(groups)}")
    for case, group in groups.items():
        if {x["method"] for x in group} != EXPECTED_METHODS:
            raise ValueError(f"Expected 5 matching methods for case {case}")
    transformed = []
    window_events = []
    long_others = []
    for row in rows:
        item = dict(row)
        text = str(item["context"])
        count = counter.count(text)
        if item["method"] == ORIGINAL_NAME:
            kind = "full"
            if count > token_limit:
                original = counter._encoding.encode(text, disallowed_special=())
                # Equal front/back. No gold-dependent evidence selection.
                front = token_limit//2
                rear = token_limit-front
                text = counter._encoding.decode(original[:front]) + (
                    "\n\n[REFERENCE WINDOW BOUNDARY — SOURCE MIDDLE OMITTED]\n\n"
                ) + counter._encoding.decode(original[-rear:])
                kind = "head_tail"
            item["method"] = REFERENCE_NAME
            item["task_id"] = f"{item['case_id']}::{REFERENCE_NAME}"
            item["context"] = text
            item["reference_window_mode"] = kind
            item["reference_window_limit_est"] = token_limit
            item["context_tokens"] = counter.count(text)
            item["selected_content_tokens"] = item["context_tokens"]
            item["token_saving_vs_full"] = 1 - item["context_tokens"]/max(1,int(item["source_tokens"]))
            window_events.append({"length":str(item.get("length","")).lower(),"mode":kind})
        elif count > token_limit:
            long_others.append((str(item["case_id"]),str(item["method"]),count))
        transformed.append(item)
    if long_others:
        counts=collections.Counter(method for _,method,_ in long_others)
        raise ValueError(
            f"Other methods also exceed API budget; refusing silent truncation: {dict(counts)}; "
            f"max={max(v for _,_,v in long_others)}. Adjust compression or source eligibility."
        )
    report={
        "reference_method": REFERENCE_NAME,
        "reference_window_limit_est":token_limit,
        "reference_mode_counts": dict(collections.Counter(v["mode"] for v in window_events)),
        "reference_mode_by_length": {
            length: dict(collections.Counter(x["mode"] for x in window_events if x["length"]==length))
            for length in ("short","medium","long")
        },
        "total_cases":len(groups),
        "total_tasks":len(transformed),
        "note":"Do not call the renamed baseline full context. Some source middles were omitted. Other methods are unchanged.",
    }
    return transformed,report


def main()->None:
    p=argparse.ArgumentParser(description="Explicitly window the reference baseline only")
    p.add_argument("--tasks-jsonl",required=True,type=Path)
    p.add_argument("--output-jsonl",required=True,type=Path)
    p.add_argument("--reference-token-limit",type=int,default=210000)
    args=p.parse_args()
    if args.output_jsonl.exists():
        raise FileExistsError("Refusing overwrite; create a fresh run")
    counter=TokenCounter()
    ready,report=adapt(list(read_jsonl(args.tasks_jsonl)),token_limit=args.reference_token_limit,counter=counter)
    args.output_jsonl.parent.mkdir(parents=True,exist_ok=True)
    with args.output_jsonl.open("x",encoding="utf-8") as fp:
        for row in ready: fp.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    report["output_sha256"]=hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest()
    args.output_jsonl.with_name("windowed_reference_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
