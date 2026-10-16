"""Offline canonical reanalysis: python -m submission.evaluation.reanalyze."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from . import METRIC_VERSION, SCHEMA_VERSION
from .core import compare, paired_statistics, prepare
from .evidence import rename_legacy_metrics
from .reporting import mean, summarize_longbench, write_readout

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "submission/results"
PILOT = RESULTS / "longbench_v2_modal_hybrid_greedy_83_latency"
QASPER = [RESULTS / name / "qasper_compression_eval_raw.csv" for name in (
    "qasper_compression_eval_200q_hybrid_greedy_rate050_full",
    "qasper_compression_eval_200q_hybrid_greedy_rates075_085")]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def qasper_analysis(output):
    migrated = []
    groups = defaultdict(dict)
    for path in QASPER:
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row = rename_legacy_metrics(raw)
                key = (row["paper_id"], row["question_id"])
                if not all(key):
                    raise ValueError("Invalid QASPER identifier")
                method = row["setting_label"]
                if key in groups[method]:
                    raise ValueError(f"Duplicate QASPER case: {method} {key}")
                row.update(schema_version=SCHEMA_VERSION, metric_version=METRIC_VERSION,
                           cluster_id=row["paper_id"], source_artifact=path.relative_to(ROOT).as_posix(),
                           evidence_status="unavailable_missing_text", verbatim_evidence_recall=None,
                           complete_evidence_set_retained=None, source_span_coverage=None)
                groups[method][key] = row
                migrated.append(row)
    summaries = []
    for method, cases in sorted(groups.items()):
        rows = list(cases.values())
        summary = {"schema_version": SCHEMA_VERSION, "metric_version": METRIC_VERSION,
                   "setting_label": method, "questions": len(rows), "papers": len({key[0] for key in cases}),
                   "denominator_source": "observed_only_exploratory", "evidence_status": "unavailable_missing_text"}
        for key in ("total_saving_vs_full", "final_lexical_evidence_coverage", "final_lexical_coverage_ge_080",
                    "final_answer_context_token_overlap_f1", "selection_seconds", "compression_seconds", "recorded_selection_compression_seconds"):
            summary[key] = mean(row[key] for row in rows)
        summaries.append(summary)
    baseline = next(name for name in groups if name.startswith("only-llmlingua2"))
    comparisons = []
    for method, cases in sorted(groups.items()):
        if method == baseline:
            continue
        if cases.keys() != groups[baseline].keys():
            raise ValueError("QASPER comparison case sets differ")
        ids = sorted(cases)
        for metric in ("final_lexical_evidence_coverage", "final_lexical_coverage_ge_080", "total_saving_vs_full"):
            differences = [float(cases[key][metric]) - float(groups[baseline][key][metric]) for key in ids]
            comparisons.append({"method": method, "baseline": baseline, "metric": metric, "compared": len(ids),
                                **paired_statistics(differences, [key[0] for key in ids], binary=False)})
    write_csv(output / "qasper_legacy_metrics_v2.csv", migrated)
    write_json(output / "qasper_summary.json", summaries)
    write_csv(output / "qasper_summary.csv", summaries)
    write_json(output / "qasper_pairwise.json", comparisons)
    write_csv(output / "qasper_pairwise.csv", comparisons)
    return summaries, comparisons, migrated


def table(label, caption, header, rows):
    return "\n".join([r"\begin{table}[htbp]", r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\centering\scriptsize", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
                      r"\toprule", " & ".join(header) + r" \\", r"\midrule",
                      *[" & ".join(row) + r" \\" for row in rows], r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def write_tables(output, summary, qasper):
    labels = {"full-context": "Full context", "production-rag-50": "Prod-RAG-50", "tokenpack-50": "TP-HG-50",
              "only-longllmlingua-rate050": "LLL-0.50", "tokenpack-50+longllmlingua-rate050": "TP-HG-50 + LLL-0.50"}
    by_method = {row["method"]: row for row in summary}
    generation, latency = [], []
    for method, label in labels.items():
        row = by_method[method]
        generation.append([label, str(row["evaluated_denominator"]), f"{row['correct']}/{row['evaluated_denominator']}",
                           f"{row['accuracy_all_planned']:.3f}", f"{row['avg_context_tokens']:.0f}", f"{row['avg_token_saving_vs_full']:.3f}"])
        latency.append([label, f"{row['avg_recorded_preprocessing_seconds']:.3f}", f"{row['avg_answer_latency_seconds']:.3f}",
                        f"{row['avg_recorded_stage_total_seconds']:.3f}", f"{row['recorded_stage_speedup_vs_full']:.2f}"])
    tables = {
        "longbench_generation_table.tex": table("tab:longbench-generation", "Metric-v2 exploratory 83-case pilot. Parse failures count as incorrect. TP-HG versus full context: +6.02 percentage points; 37/83 versus 32/83. Equal totals do not establish equivalence.",
                                                ["Method", "Planned", "Correct", "Accuracy", "Context", "Saving"], generation),
        "longbench_latency_table.tex": table("tab:longbench-latency", "Recorded selection + compression + generation only, batch size one. Excludes chunking, embeddings, scoring, model loading and cold start; not end-to-end product latency.",
                                             ["Method", "Recorded prep s", "Generation s", "Stage sum s", "Stage speedup"], latency),
    }
    names = {"only-tokenpack / budget-top-k / budget=50%": "Only TokenPack", "only-llmlingua2 / rate=0.50": "Only LLM2-0.50"}
    comparison, frontier = [], []
    for row in qasper:
        method = row["setting_label"]
        label = names.get(method, "TokenPack + LLM2-" + method.rsplit("=", 1)[-1])
        values = [label, f"{100 * row['total_saving_vs_full']:.1f}" + r"\%", f"{row['final_lexical_evidence_coverage']:.3f}",
                  f"{row['final_lexical_coverage_ge_080']:.3f}", f"{row['final_answer_context_token_overlap_f1']:.3f}"]
        frontier.append(values)
        if "rate=0.75" not in method and "rate=0.85" not in method:
            comparison.append(values)
    for filename, label, rows in (("qasper_compression_comparison_table.tex", "tab:qasper-compression-comparison", comparison),
                                  ("qasper_compression_frontier_table.tex", "tab:qasper-compression-frontier", frontier)):
        tables[filename] = table(label, "Metric-v2 lexical proxies on 200 QASPER questions (77 papers). Coverage threshold is not complete semantic evidence. Historical text unavailable for span reanalysis.",
                                 ["Method", "Saving", "Lexical coverage", r"Coverage $\geq .80$", "Answer-context F1"], rows)
    for filename, contents in tables.items():
        (output / filename).write_text(contents, encoding="utf-8")
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "metric_v2/canonical")
    parser.add_argument("--update-paper-tables", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == PILOT.resolve() or any(output == path.parent.resolve() for path in QASPER):
        raise ValueError("Cannot overwrite historical runs")
    output.mkdir(parents=True, exist_ok=True)
    paths = [PILOT / name for name in ("longbench_generation_results.jsonl", "longbench_generation_tasks.jsonl", "task_report.json")] + QASPER
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    rows, tasks = read_jsonl(paths[0]), read_jsonl(paths[1])
    observed, planned = prepare(rows, tasks)
    summary = summarize_longbench(rows, tasks)
    pairs = [compare(rows, method, baseline, tasks) for method, baseline in (
        ("tokenpack-50", "full-context"), ("tokenpack-50", "production-rag-50"),
        ("tokenpack-50+longllmlingua-rate050", "tokenpack-50"))]
    write_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", summary)
    write_json(output / "pairwise.json", pairs)
    write_csv(output / "pairwise.csv", pairs)
    write_readout(summary, pairs, output / "readout.md")
    qasper, qasper_pairs, migrated = qasper_analysis(output)
    tables = write_tables(output, summary, qasper)
    if args.update_paper_tables:
        for name, contents in tables.items():
            (ROOT / "submission/paper/tables" / name).write_text(contents, encoding="utf-8")
    report = json.loads(paths[2].read_text(encoding="utf-8"))
    manifest = {"schema_version": SCHEMA_VERSION, "metric_version": METRIC_VERSION, "run_id": "canonical_metric_v2",
                "sources": hashes, "evaluator_commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--short"),
                "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
                "evaluator_source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "submission/evaluation").glob("*.py"))},
                "protocol_sha256": hashlib.sha256((ROOT / "docs/research/evaluation-protocol-v1.md").read_bytes()).hexdigest(),
                "python": platform.python_version(), "seed": 20260916, "bootstrap_replicates": 20000,
                "model_ids": sorted({row.get("model_id", "unknown") for row in rows}), "model_revision": "unknown",
                "tokenizer_revision": "unknown", "context_tokenizer": "unknown_historical_counter_backend_not_recorded",
                "hardware": "unknown", "resolved_historical_report": report,
                "expected_case_methods": [{"case_id": task["case_id"], "method": task["method"], "cluster_id": task.get("cluster_id")} for task in tasks],
                "interpretation": "exploratory_development_data_not_independent_validation",
                "reproduce": "python -m submission.evaluation.reanalyze"}
    write_json(output / "manifest.json", manifest)
    write_json(output / "integrity.json", {"longbench_results": len(rows), "longbench_tasks": len(tasks),
               "duplicates": 0, "unexpected_results": 0, "missing_results": len(planned.keys() - observed.keys()),
               "qasper_scalar_rows": len(migrated), "qasper_pairwise_case_sets_equal": True,
               "source_hashes_unchanged": all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value for path, value in hashes.items()),
               "length_labels": dict(Counter(task["length"] for task in tasks if task["method"] == "full-context"))})
    write_json(output / "unavailable.json", {
        "qasper_verbatim_and_spans": "unavailable_missing_text; historical scalar CSV cannot recover evidence spans",
        "longbench_end_to_end_latency": "unavailable_missing_chunking_embedding_scoring_timings",
        "longbench_cluster_bootstrap": "unavailable_no_verified_document_cluster_ids",
        "human_evaluation": "not_performed", "new_model_runs": "not_performed",
        "historical_prompt_tokens": "unavailable_not_recorded"})
    with (output / "readout.md").open("a", encoding="utf-8") as handle:
        handle.write("\n## QASPER and interpretation\n\nLexical proxies were migrated from scalar CSVs on 200 questions / 77 papers; matched comparisons use paper-cluster bootstrap. Verbatim and source-span metrics cannot be reconstructed. See qasper_summary.csv and qasper_pairwise.json.\n\nThe cascade gains eight answers and loses eight relative to TokenPack: equal aggregate accuracy does not establish equivalence. All 83 LongBench cases are labelled short. Reported speedup omits chunking, embeddings and scoring.\n\nReproduce: `python -m submission.evaluation.reanalyze`. Historical input files are unchanged; unavailable measurements are listed in unavailable.json.\n")
    print(f"Wrote metric-v2 canonical analysis to {output}")


if __name__ == "__main__":
    main()
