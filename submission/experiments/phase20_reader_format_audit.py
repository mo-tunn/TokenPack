"""Faz20B format failure analysis, only AFTER reader log frozen/scored.
Never rewrite originally saved answers, no selective rescoring or model calls.
"""
from __future__ import annotations
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from submission.experiments.phase20_reader_register import OUT
from submission.experiments.phase12_register import sha_file,new_json
def main():
    scored=json.loads((OUT/"READER_POST_GOLD_RESULTS.json").read_text(encoding="utf8"))
    saved=OUT/"reader_responses.jsonl"
    assert sha_file(saved)==scored["responses_sha256"] and scored["n"]==32
    rows=[json.loads(s) for s in saved.open(encoding="utf8") if s.strip()]
    cases=[r for r in rows if r["arm"]=="verified-evidence-first"]
    assert len(cases)==32
    summary=collections.Counter()
    detail=[]
    for r in cases:
        raw=r.get("raw_reader_output") or ""
        status=r["parse_status"]
        isjson=raw.lstrip().startswith("{")
        code=raw.lstrip().startswith("```")
        finish=r.get("finish_reason")
        quote_count=len(r.get("source_citation_steps") or [])
        incomplete=raw.count("{")>raw.count("}")
        try:
            parsed=json.loads(raw)
            json_syntax_valid=True
            keys=sorted(parsed) if isinstance(parsed,dict) else []
        except Exception:
            json_syntax_valid=False;keys=[]
        row={"id":r["id"],"parse_status":status,
            "char_count":len(raw),"output_tokens":r.get("output_tokens"),
            "finish_reason":finish,
            "starts_json_brace":isjson,"starts_markdown_fence":code,
            "appears_unbalanced":incomplete,
            "json_syntax_valid_without_preprocessing":json_syntax_valid,
            "keys_if_json":keys,
            "quotes_checked":quote_count,
            "strict_exact_quotes":sum(x["quote_exact_in_context"] for x in r.get("source_citation_steps") or []),
            "answer_len_scored":len(r["answer"]),
            # For a reproducible failure audit, snippets are model-only source
            # outputs, no model inputs or gold answer labels.
            "first_120_chars":raw[:120],"last_100_chars":raw[-100:]}
        detail.append(row)
        summary.update({
            "n":1,"valid_json":int(status=="valid_json"),
            "invalid_json":int(status!="valid_json"),
            "starts_json":int(isjson),"fenced":int(code),
            "unbalanced":int(incomplete),"syntax_valid_raw":int(json_syntax_valid),
            "empty_answer":int(not r["answer"]),
            "quoted_steps":quote_count,
            "exact_quote_steps":row["strict_exact_quotes"],
        })
        summary["finish_reason_"+str(finish)]+=1
    out=OUT/"STRUCTURED_READER_FORMAT_AUDIT.json"
    if out.exists():raise FileExistsError(out)
    report={"exploratory_after_response_freeze":True,
       "no_reader_answers_rescored":True,"no_remotely_retried_failed_JSON":True,
       "source_log_sha256":sha_file(saved),
       "stats":dict(summary),"per_case":detail}
    new_json(out,report)
    print("PHASE20B_READER_FORMAT_AUDIT",json.dumps(report["stats"]),flush=True)
    print("PHASE20B_FAILURE_EXCERPTS",
      json.dumps([{k:x[k] for k in ("id","parse_status","output_tokens","finish_reason","first_120_chars","last_100_chars")}
           for x in detail if x["parse_status"]!="valid_json"][:5],ensure_ascii=False),flush=True)
if __name__=="__main__":main()
