"""Evidence retention diagnostics, not semantic faithfulness judgments."""
from __future__ import annotations

import re
import unicodedata


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def evidence_metrics(annotation_groups, segments, *, source_spans=None, selected_spans=None):
    groups = [[normalize(text) for text in group if normalize(text)] for group in annotation_groups]
    groups = [group for group in groups if group]
    empty = {"verbatim_evidence_recall": None, "complete_evidence_set_retained": None,
             "source_span_coverage": None, "annotation_metrics": []}
    if not groups:
        return {**empty, "evidence_status": "excluded_no_valid_evidence"}
    if segments is None:
        return {**empty, "evidence_status": "unavailable_missing_text"}
    normalized = [normalize(segment) for segment in segments]
    annotations = []
    for group in groups:
        hits = []
        for evidence in group:
            pattern = (r"(?<!\w)" if evidence[0].isalnum() else "") + re.escape(evidence)
            pattern += r"(?!\w)" if evidence[-1].isalnum() else ""
            hits.append(any(re.search(pattern, segment) is not None for segment in normalized))
        annotations.append({"verbatim_evidence_recall": sum(hits) / len(hits),
                            "complete_evidence_set_retained": all(hits), "hits": hits})
    return {"verbatim_evidence_recall": max(x["verbatim_evidence_recall"] for x in annotations),
            "complete_evidence_set_retained": any(x["complete_evidence_set_retained"] for x in annotations),
            "annotation_metrics": annotations, "evidence_status": "available_verbatim_not_semantic",
            "source_span_coverage": span_coverage(source_spans, selected_spans)}


def span_coverage(gold_groups, selected):
    """Half-open character spans [source_id, start, end]; best alternative.

    None means no reliable mapping. Empty selected spans means nothing selected.
    """
    if not gold_groups or selected is None:
        return None
    def validate(span):
        if len(span) != 3 or not isinstance(span[0], str) or not all(isinstance(v, int) for v in span[1:]) or not 0 <= span[1] < span[2]:
            raise ValueError(f"Invalid source span: {span}")
    def union_length(intervals):
        total, end = 0, -1
        for start, stop in sorted(intervals):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
        return total
    for span in selected:
        validate(span)
    scores = []
    for group in gold_groups:
        sources = {}
        for span in group:
            validate(span)
            sources.setdefault(span[0], []).append(span[1:])
        covered = total = 0
        for source, intervals in sources.items():
            total += union_length(intervals)
            overlaps = [(max(a, c), min(b, d)) for a, b in intervals
                        for key, c, d in selected if key == source and max(a, c) < min(b, d)]
            covered += union_length(overlaps)
        if total:
            scores.append(covered / total)
    return max(scores) if scores else None


def rename_legacy_metrics(row):
    replacements = (("complete_evidence", "lexical_coverage_ge_080"),
                    ("evidence_recall", "lexical_evidence_coverage"),
                    ("answer_token_f1", "answer_context_token_overlap_f1"),
                    ("helped_evidence", "helped_lexical_coverage"),
                    ("hurt_evidence", "hurt_lexical_coverage"))
    result = {}
    for key, value in row.items():
        if key.endswith(("verbatim_evidence_recall", "complete_evidence_set_retained")):
            result[key] = value
            continue
        for old, new in replacements:
            key = key.replace(old, new)
        key = {"total_seconds": "recorded_selection_compression_seconds",
               "avg_total_seconds": "avg_recorded_selection_compression_seconds"}.get(key, key)
        result[key] = value
    return result
