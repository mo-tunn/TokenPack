from __future__ import annotations

import math
import re
from collections import Counter

from tokenpack.embeddings import cosine
from tokenpack.models import Chunk, ScoredChunk

HYBRID_WEIGHTS = {
    "cosine": 0.55,
    "bm25": 0.25,
    "position": 0.15,
    "neighbor_coherence": 0.05,
}
EVIDENCE_HYBRID_WEIGHTS = {
    "cosine": 0.35,
    "bm25": 0.25,
    "query_coverage": 0.20,
    "structural_prior": 0.15,
    "neighbor_coherence": 0.05,
}
KNAPSACK_AWARE_WEIGHTS = {
    "evidence_base": 0.70,
    "value_density_prior": 0.15,
    "length_utility": 0.10,
    "term_specificity": 0.05,
}
QUERY_SUPPORT_WEIGHTS = {
    "evidence_base": 0.50,
    "support_likelihood": 0.25,
    "phrase_overlap": 0.10,
    "term_proximity": 0.05,
    "length_utility": 0.05,
    "term_specificity": 0.05,
}
DECISION_AWARE_WEIGHTS = {
    "evidence_base": 0.45,
    "question_support": 0.20,
    "candidate_support": 0.15,
    "candidate_contrast": 0.10,
    "term_proximity": 0.05,
    "length_utility": 0.05,
}
BUDGETMEM_STYLE_WEIGHTS = {
    "bm25": 0.30,
    "query_coverage": 0.20,
    "position": 0.15,
    "term_specificity": 0.15,
    "entity_density": 0.08,
    "numerical_density": 0.05,
    "discourse_marker": 0.05,
    "length_utility": 0.02,
}
BASELINE_SCORING_PROFILES = ("cosine", "hybrid")
DEFAULT_SCORING_PROFILE = "evidence-hybrid"
BUDGET_AWARE_SCORING_PROFILES = ("knapsack-aware",)
RELATED_WORK_BASELINE_SCORING_PROFILES = ("budgetmem-style",)
EXPERIMENTAL_SCORING_PROFILES = ("query-support", "decision-aware")
SCORING_PROFILES = (
    *BASELINE_SCORING_PROFILES,
    DEFAULT_SCORING_PROFILE,
    *BUDGET_AWARE_SCORING_PROFILES,
    *RELATED_WORK_BASELINE_SCORING_PROFILES,
    *EXPERIMENTAL_SCORING_PROFILES,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", flags=re.UNICODE)
_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}", flags=re.UNICODE)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,}))*\b",
    flags=re.UNICODE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\d+/\d+)(?![A-Za-z])", flags=re.UNICODE)
_DISCOURSE_MARKERS = (
    "therefore",
    "however",
    "because",
    "although",
    "whereas",
    "in contrast",
    "as a result",
    "for example",
    "we show",
    "we find",
    "we propose",
    "our results",
    "experiments show",
    "results indicate",
    "in conclusion",
)


def score_experimental_chunks(
    query_embedding: list[float],
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    redundancy_penalty: float = 0.0,
    scoring: str = "cosine",
    query_text: str | None = None,
    redundancy_candidate_pool: int | None = 250,
) -> list[ScoredChunk]:
    similarities = [cosine(query_embedding, embedding) for embedding in chunk_embeddings]
    if scoring == "cosine":
        values = _minmax(similarities)
        components = [{"cosine": value} for value in values]
    elif scoring == "hybrid":
        components = _hybrid_components(query_text or "", chunks, chunk_embeddings, similarities)
        values = [
            sum(HYBRID_WEIGHTS[name] * item[name] for name in HYBRID_WEIGHTS)
            for item in components
        ]
    elif scoring == "evidence-hybrid":
        components = _evidence_hybrid_components(query_text or "", chunks, chunk_embeddings, similarities)
        values = [
            sum(EVIDENCE_HYBRID_WEIGHTS[name] * item[name] for name in EVIDENCE_HYBRID_WEIGHTS)
            for item in components
        ]
    elif scoring == "knapsack-aware":
        components = _knapsack_aware_components(query_text or "", chunks, chunk_embeddings, similarities)
        values = [
            sum(KNAPSACK_AWARE_WEIGHTS[name] * item[name] for name in KNAPSACK_AWARE_WEIGHTS)
            for item in components
        ]
    elif scoring == "budgetmem-style":
        components = _budgetmem_style_components(query_text or "", chunks)
        values = [
            sum(BUDGETMEM_STYLE_WEIGHTS[name] * item[name] for name in BUDGETMEM_STYLE_WEIGHTS)
            for item in components
        ]
    elif scoring == "query-support":
        components = _query_support_components(query_text or "", chunks, chunk_embeddings, similarities)
        values = [
            sum(QUERY_SUPPORT_WEIGHTS[name] * item[name] for name in QUERY_SUPPORT_WEIGHTS)
            for item in components
        ]
    elif scoring == "decision-aware":
        components = _decision_aware_components(query_text or "", chunks, chunk_embeddings, similarities)
        values = [
            sum(DECISION_AWARE_WEIGHTS[name] * item[name] for name in DECISION_AWARE_WEIGHTS)
            for item in components
        ]
    else:
        raise ValueError(f"Unknown scoring profile: {scoring}")
    scored = [
        ScoredChunk(
            chunk=chunk,
            value=value,
            raw_similarity=similarity,
            weight=chunk.token_count,
            embedding=embedding,
            score_components=component,
        )
        for chunk, value, similarity, embedding, component in zip(
            chunks,
            values,
            similarities,
            chunk_embeddings,
            components,
            strict=True,
        )
    ]
    if redundancy_penalty > 0:
        _apply_redundancy_penalty(
            scored,
            chunk_embeddings,
            redundancy_penalty,
            candidate_pool=redundancy_candidate_pool,
        )
    return scored


def _hybrid_components(
    query_text: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    similarities: list[float],
) -> list[dict[str, float]]:
    cosine_values = _minmax(similarities)
    bm25_values = _normalized_signal(_bm25_scores(query_text, chunks))
    position_values = _position_bias(chunks)
    neighbor_values = _normalized_signal(_neighbor_coherence(chunks, chunk_embeddings))
    return [
        {
            "cosine": cosine_value,
            "bm25": bm25_value,
            "position": position_value,
            "neighbor_coherence": neighbor_value,
        }
        for cosine_value, bm25_value, position_value, neighbor_value in zip(
            cosine_values,
            bm25_values,
            position_values,
            neighbor_values,
            strict=True,
        )
    ]


def _evidence_hybrid_components(
    query_text: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    similarities: list[float],
) -> list[dict[str, float]]:
    cosine_values = _minmax(similarities)
    bm25_values = _normalized_signal(_bm25_scores(query_text, chunks))
    coverage_values = _query_coverage(query_text, chunks)
    structural_values = _structural_prior(query_text, chunks)
    neighbor_values = _normalized_signal(_neighbor_coherence(chunks, chunk_embeddings))
    return [
        {
            "cosine": cosine_value,
            "bm25": bm25_value,
            "query_coverage": coverage_value,
            "structural_prior": structural_value,
            "neighbor_coherence": neighbor_value,
        }
        for cosine_value, bm25_value, coverage_value, structural_value, neighbor_value in zip(
            cosine_values,
            bm25_values,
            coverage_values,
            structural_values,
            neighbor_values,
            strict=True,
        )
    ]


def _knapsack_aware_components(
    query_text: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    similarities: list[float],
) -> list[dict[str, float]]:
    evidence_components = _evidence_hybrid_components(query_text, chunks, chunk_embeddings, similarities)
    evidence_values = [
        sum(EVIDENCE_HYBRID_WEIGHTS[name] * item[name] for name in EVIDENCE_HYBRID_WEIGHTS)
        for item in evidence_components
    ]
    density_values = _normalized_signal(
        [value / math.sqrt(max(1, chunk.token_count)) for value, chunk in zip(evidence_values, chunks, strict=True)]
    )
    length_values = _length_utility(chunks)
    specificity_values = _term_specificity(chunks)
    return [
        {
            **component,
            "evidence_base": evidence_value,
            "value_density_prior": density_value,
            "length_utility": length_value,
            "term_specificity": specificity_value,
        }
        for component, evidence_value, density_value, length_value, specificity_value in zip(
            evidence_components,
            evidence_values,
            density_values,
            length_values,
            specificity_values,
            strict=True,
        )
    ]


def _query_support_components(
    query_text: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    similarities: list[float],
) -> list[dict[str, float]]:
    evidence_components = _evidence_hybrid_components(query_text, chunks, chunk_embeddings, similarities)
    evidence_values = [
        sum(EVIDENCE_HYBRID_WEIGHTS[name] * item[name] for name in EVIDENCE_HYBRID_WEIGHTS)
        for item in evidence_components
    ]
    support_values = _support_likelihood(query_text, chunks)
    phrase_values = _phrase_overlap(query_text, chunks)
    proximity_values = _term_proximity(query_text, chunks)
    length_values = _length_utility(chunks)
    specificity_values = _term_specificity(chunks)
    return [
        {
            **component,
            "evidence_base": evidence_value,
            "support_likelihood": support_value,
            "phrase_overlap": phrase_value,
            "term_proximity": proximity_value,
            "length_utility": length_value,
            "term_specificity": specificity_value,
        }
        for component, evidence_value, support_value, phrase_value, proximity_value, length_value, specificity_value in zip(
            evidence_components,
            evidence_values,
            support_values,
            phrase_values,
            proximity_values,
            length_values,
            specificity_values,
            strict=True,
        )
    ]


def _decision_aware_components(
    query_text: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    similarities: list[float],
) -> list[dict[str, float]]:
    evidence_components = _evidence_hybrid_components(query_text, chunks, chunk_embeddings, similarities)
    evidence_values = [
        sum(EVIDENCE_HYBRID_WEIGHTS[name] * item[name] for name in EVIDENCE_HYBRID_WEIGHTS)
        for item in evidence_components
    ]
    question_text, candidates = _split_decision_query(query_text)
    question_values = _support_likelihood(question_text or query_text, chunks)
    candidate_values, contrast_values = _candidate_decision_signals(candidates, chunks)
    proximity_values = _term_proximity(question_text or query_text, chunks)
    length_values = _length_utility(chunks)
    return [
        {
            **component,
            "evidence_base": evidence_value,
            "question_support": question_value,
            "candidate_support": candidate_value,
            "candidate_contrast": contrast_value,
            "term_proximity": proximity_value,
            "length_utility": length_value,
        }
        for component, evidence_value, question_value, candidate_value, contrast_value, proximity_value, length_value in zip(
            evidence_components,
            evidence_values,
            question_values,
            candidate_values,
            contrast_values,
            proximity_values,
            length_values,
            strict=True,
        )
    ]


def _budgetmem_style_components(
    query_text: str,
    chunks: list[Chunk],
) -> list[dict[str, float]]:
    """Hand-designed proxy for BudgetMem-like feature salience.

    This is intentionally not a reproduction of BudgetMem's learned policy.
    It exposes the same broad feature family for an artifact-local baseline.
    """
    bm25_values = _normalized_signal(_bm25_scores(query_text, chunks))
    coverage_values = _query_coverage(query_text, chunks)
    position_values = _position_bias(chunks)
    specificity_values = _term_specificity(chunks)
    entity_values = _entity_density(chunks)
    numerical_values = _numerical_density(chunks)
    discourse_values = _discourse_marker_density(chunks)
    length_values = _length_utility(chunks)
    return [
        {
            "bm25": bm25_value,
            "query_coverage": coverage_value,
            "position": position_value,
            "term_specificity": specificity_value,
            "entity_density": entity_value,
            "numerical_density": numerical_value,
            "discourse_marker": discourse_value,
            "length_utility": length_value,
        }
        for (
            bm25_value,
            coverage_value,
            position_value,
            specificity_value,
            entity_value,
            numerical_value,
            discourse_value,
            length_value,
        ) in zip(
            bm25_values,
            coverage_values,
            position_values,
            specificity_values,
            entity_values,
            numerical_values,
            discourse_values,
            length_values,
            strict=True,
        )
    ]


def _bm25_scores(query_text: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = _tokenize(query_text)
    if not query_terms or not chunks:
        return [0.0 for _ in chunks]

    documents = [_tokenize(chunk.text) for chunk in chunks]
    frequencies = [Counter(document) for document in documents]
    lengths = [len(document) for document in documents]
    avg_length = sum(lengths) / max(1, len(lengths))
    document_frequency: Counter[str] = Counter()
    for frequency in frequencies:
        document_frequency.update(frequency.keys())

    scores: list[float] = []
    total_documents = len(documents)
    for frequency, length in zip(frequencies, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            term_frequency = frequency.get(term, 0)
            if term_frequency == 0:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
            denominator = term_frequency + k1 * (1.0 - b + b * length / max(1.0, avg_length))
            score += idf * (term_frequency * (k1 + 1.0)) / denominator
        scores.append(score)
    return scores


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _query_coverage(query_text: str, chunks: list[Chunk]) -> list[float]:
    query_terms = set(_tokenize(query_text))
    if not query_terms:
        return [0.0 for _ in chunks]
    scores: list[float] = []
    for chunk in chunks:
        chunk_terms = set(_tokenize(chunk.text))
        scores.append(len(query_terms & chunk_terms) / max(1, len(query_terms)))
    return scores


def _support_likelihood(query_text: str, chunks: list[Chunk]) -> list[float]:
    query_terms = _important_query_terms(query_text)
    if not query_terms:
        return [0.0 for _ in chunks]
    query_counter = Counter(query_terms)
    query_weight = sum(query_counter.values())
    scores: list[float] = []
    for chunk in chunks:
        chunk_terms = set(_content_terms(chunk.text))
        weighted_coverage = sum(weight for term, weight in query_counter.items() if term in chunk_terms) / max(1, query_weight)
        breadth = len(set(query_terms) & chunk_terms) / max(1, len(set(query_terms)))
        scores.append(min(1.0, 0.65 * weighted_coverage + 0.35 * breadth))
    return scores


def _phrase_overlap(query_text: str, chunks: list[Chunk]) -> list[float]:
    phrases = _query_phrases(query_text)
    if not phrases:
        return [0.0 for _ in chunks]
    scores: list[float] = []
    for chunk in chunks:
        normalized = _normalize_text(chunk.text)
        hits = sum(1 for phrase in phrases if phrase in normalized)
        scores.append(hits / max(1, len(phrases)))
    return scores


def _term_proximity(query_text: str, chunks: list[Chunk]) -> list[float]:
    query_terms = set(_important_query_terms(query_text))
    if len(query_terms) < 2:
        return [0.0 for _ in chunks]
    scores: list[float] = []
    for chunk in chunks:
        tokens = _tokenize(chunk.text)
        positions_by_term: dict[str, list[int]] = {}
        for index, token in enumerate(tokens):
            if token in query_terms:
                positions_by_term.setdefault(token, []).append(index)
        if len(positions_by_term) < 2:
            scores.append(0.0)
            continue
        ordered = sorted((position, term) for term, positions in positions_by_term.items() for position in positions)
        best_span = min(
            right_position - left_position
            for (left_position, left_term), (right_position, right_term) in zip(ordered, ordered[1:], strict=False)
            if left_term != right_term
        )
        scores.append(1.0 / (1.0 + best_span / 12.0))
    return scores


def _split_decision_query(query_text: str) -> tuple[str, list[str]]:
    question_lines: list[str] = []
    candidates: list[str] = []
    for line in query_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(?:[A-Z]|[0-9]{1,2})[\.\)]\s+(.+)$", stripped)
        if match:
            candidates.append(match.group(1).strip())
        else:
            question_lines.append(stripped)
    return " ".join(question_lines).strip(), candidates


def _candidate_decision_signals(candidates: list[str], chunks: list[Chunk]) -> tuple[list[float], list[float]]:
    candidate_terms = [set(_important_query_terms(candidate)) for candidate in candidates]
    candidate_terms = [terms for terms in candidate_terms if terms]
    if not candidate_terms:
        return [0.0 for _ in chunks], [0.0 for _ in chunks]

    support_scores: list[float] = []
    contrast_scores: list[float] = []
    for chunk in chunks:
        chunk_terms = set(_content_terms(chunk.text))
        scores = [
            len(terms & chunk_terms) / max(1, len(terms))
            for terms in candidate_terms
        ]
        ranked = sorted(scores, reverse=True)
        best = ranked[0] if ranked else 0.0
        runner_up = ranked[1] if len(ranked) > 1 else 0.0
        support_scores.append(best)
        contrast_scores.append(max(0.0, best - runner_up))
    return support_scores, contrast_scores


def _structural_prior(query_text: str, chunks: list[Chunk]) -> list[float]:
    query_terms = set(_tokenize(query_text))
    scores: list[float] = []
    for chunk in chunks:
        metadata = chunk.metadata
        score = 0.0
        content_type = metadata.get("content_type")
        if content_type == "code":
            score += 0.10
            symbol_text = " ".join(
                str(metadata.get(key, ""))
                for key in ("symbol_name", "symbol_kind", "language")
            )
            symbol_terms = set(_tokenize(symbol_text))
            if query_terms and query_terms & symbol_terms:
                score += 0.35
            header = "\n".join(chunk.text.splitlines()[:6])
            header_terms = set(_tokenize(header))
            if query_terms and query_terms & header_terms:
                score += 0.20
            if _has_dependency_hint(chunk.text, query_terms):
                score += 0.15
        section_hint = str(metadata.get("section_hint", ""))
        if section_hint and query_terms & set(_tokenize(section_hint)):
            score += 0.25
        scores.append(min(1.0, score))
    return scores


def _has_dependency_hint(text: str, query_terms: set[str]) -> bool:
    if not query_terms:
        return False
    dependency_lines = [
        line
        for line in text.splitlines()[:20]
        if re.match(r"\s*(import|from|#include|package|use)\b", line)
    ]
    if not dependency_lines:
        return False
    dependency_terms = set(_tokenize("\n".join(dependency_lines)))
    return bool(query_terms & dependency_terms)


def _position_bias(chunks: list[Chunk]) -> list[float]:
    scores = [0.0 for _ in chunks]
    groups: dict[tuple[int, str], list[int]] = {}
    for index, chunk in enumerate(chunks):
        groups.setdefault((chunk.document_index, chunk.source_path), []).append(index)

    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: chunks[index].order_key)
        if len(ordered) == 1:
            scores[ordered[0]] = 1.0
            continue
        last = len(ordered) - 1
        for position, index in enumerate(ordered):
            scores[index] = abs((2.0 * position / last) - 1.0)
    return scores


def _neighbor_coherence(chunks: list[Chunk], embeddings: list[list[float]]) -> list[float]:
    scores: list[float] = []
    for index, chunk in enumerate(chunks):
        neighbors: list[float] = []
        if index > 0 and _same_document(chunk, chunks[index - 1]):
            neighbors.append(max(0.0, cosine(embeddings[index], embeddings[index - 1])))
        if index + 1 < len(chunks) and _same_document(chunk, chunks[index + 1]):
            neighbors.append(max(0.0, cosine(embeddings[index], embeddings[index + 1])))
        scores.append(sum(neighbors) / len(neighbors) if neighbors else 0.0)
    return scores


def _length_utility(chunks: list[Chunk]) -> list[float]:
    if not chunks:
        return []
    weights = sorted(max(1, chunk.token_count) for chunk in chunks)
    ideal = weights[len(weights) // 2]
    ideal = max(40, min(320, ideal))
    scores: list[float] = []
    for chunk in chunks:
        weight = max(1, chunk.token_count)
        if weight <= ideal:
            score = 0.60 + 0.40 * (weight / ideal)
        else:
            score = max(0.25, 1.0 - 0.60 * ((weight - ideal) / max(1, ideal)))
        scores.append(min(1.0, max(0.0, score)))
    return scores


def _term_specificity(chunks: list[Chunk]) -> list[float]:
    if not chunks:
        return []
    term_sets = [_content_terms(chunk.text) for chunk in chunks]
    document_frequency: Counter[str] = Counter()
    for terms in term_sets:
        document_frequency.update(terms)
    total = len(chunks)
    raw_scores: list[float] = []
    for terms in term_sets:
        if not terms:
            raw_scores.append(0.0)
            continue
        idf_sum = sum(math.log(1.0 + total / max(1, document_frequency[term])) for term in terms)
        raw_scores.append(idf_sum / len(terms))
    return _normalized_signal(raw_scores)


def _entity_density(chunks: list[Chunk]) -> list[float]:
    raw_scores: list[float] = []
    for chunk in chunks:
        terms = _content_terms(chunk.text)
        entity_count = sum(1 for match in _ENTITY_RE.finditer(chunk.text) if len(match.group(0)) >= 3)
        raw_scores.append(entity_count / max(1, len(terms)))
    return _normalized_signal(raw_scores)


def _numerical_density(chunks: list[Chunk]) -> list[float]:
    raw_scores: list[float] = []
    for chunk in chunks:
        token_count = max(1, len(_tokenize(chunk.text)))
        raw_scores.append(len(_NUMBER_RE.findall(chunk.text)) / token_count)
    return _normalized_signal(raw_scores)


def _discourse_marker_density(chunks: list[Chunk]) -> list[float]:
    raw_scores: list[float] = []
    for chunk in chunks:
        normalized = _normalize_text(chunk.text)
        marker_count = sum(normalized.count(marker) for marker in _DISCOURSE_MARKERS)
        raw_scores.append(marker_count / max(1, len(_tokenize(normalized))))
    return _normalized_signal(raw_scores)


def _same_document(left: Chunk, right: Chunk) -> bool:
    return left.document_index == right.document_index and left.source_path == right.source_path


def _normalized_signal(values: list[float]) -> list[float]:
    if not values:
        return []
    if max(values) <= 0.0:
        return [0.0 for _ in values]
    return _minmax(values)


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _apply_redundancy_penalty(
    scored: list[ScoredChunk],
    embeddings: list[list[float]],
    penalty_strength: float,
    candidate_pool: int | None = 250,
) -> None:
    ranked_indices = sorted(range(len(scored)), key=lambda index: scored[index].value, reverse=True)
    if candidate_pool is not None and candidate_pool > 0:
        ranked_indices = ranked_indices[:candidate_pool]
    stronger: list[int] = []
    for index in ranked_indices:
        best_overlap = {
            "embedding_overlap": 0.0,
            "lexical_overlap": 0.0,
            "structural_overlap": 0.0,
            "redundancy_overlap": 0.0,
        }
        for previous in stronger:
            overlap = _redundancy_overlap(
                scored[index].chunk,
                scored[previous].chunk,
                embeddings[index],
                embeddings[previous],
            )
            if overlap["redundancy_overlap"] > best_overlap["redundancy_overlap"]:
                best_overlap = overlap
        penalty = penalty_strength * best_overlap["redundancy_overlap"]
        scored[index].redundancy_penalty = penalty
        scored[index].value = max(0.0, scored[index].value * (1.0 - penalty))
        scored[index].score_components.update(best_overlap)
        scored[index].score_components["novelty"] = 1.0 - best_overlap["redundancy_overlap"]
        stronger.append(index)


def _redundancy_overlap(
    left: Chunk,
    right: Chunk,
    left_embedding: list[float],
    right_embedding: list[float],
) -> dict[str, float]:
    embedding_overlap = max(0.0, cosine(left_embedding, right_embedding))
    lexical_overlap = _lexical_overlap(left.text, right.text)
    structural_overlap = _structural_overlap(left, right)
    combined = min(
        1.0,
        0.50 * embedding_overlap + 0.35 * lexical_overlap + 0.15 * structural_overlap,
    )
    return {
        "embedding_overlap": embedding_overlap,
        "lexical_overlap": lexical_overlap,
        "structural_overlap": structural_overlap,
        "redundancy_overlap": combined,
    }


def _lexical_overlap(left: str, right: str) -> float:
    left_terms = _content_terms(left)
    right_terms = _content_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))


def _content_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _CONTENT_TOKEN_RE.finditer(text)}


def _important_query_terms(query_text: str) -> list[str]:
    return [
        token
        for token in _tokenize(query_text)
        if len(token) >= 3 and token not in _STOPWORDS
    ]


def _query_phrases(query_text: str) -> list[str]:
    tokens = _important_query_terms(query_text)
    phrases: list[str] = []
    for size in (2, 3):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[index : index + size])
            if len(phrase) >= 8:
                phrases.append(phrase)
    return phrases[:24]


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _structural_overlap(left: Chunk, right: Chunk) -> float:
    if left.source_path != right.source_path:
        return 0.0
    left_meta = left.metadata
    right_meta = right.metadata
    if left_meta.get("symbol_name") and left_meta.get("symbol_name") == right_meta.get("symbol_name"):
        return 1.0
    if left_meta.get("section_hint") and left_meta.get("section_hint") == right_meta.get("section_hint"):
        return 0.75
    if left_meta.get("content_type") == "code" and right_meta.get("content_type") == "code":
        line_overlap = _line_overlap(left_meta, right_meta)
        if line_overlap > 0:
            return line_overlap
        if left_meta.get("language") == right_meta.get("language"):
            return 0.25
    return 0.0


def _line_overlap(left_meta: dict, right_meta: dict) -> float:
    left_start = left_meta.get("start_line")
    left_end = left_meta.get("end_line")
    right_start = right_meta.get("start_line")
    right_end = right_meta.get("end_line")
    if not all(isinstance(value, int) for value in (left_start, left_end, right_start, right_end)):
        return 0.0
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    span = max(left_end, right_end) - min(left_start, right_start) + 1
    return overlap / max(1, span)


_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "also",
    "among",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "doing",
    "during",
    "each",
    "following",
    "from",
    "have",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "same",
    "should",
    "such",
    "than",
    "that",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "would",
}
