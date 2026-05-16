"""Adaptive risk-informed evidence localization.

ARIEL classifies each summary sentence into a coarse risk type, uses that type
to choose a dynamic BM25/embedding fusion weight, and reranks candidate
evidence with lightweight coverage features.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.retrieval_hybrid import (
    index_by_query_id,
    merge_chunk_metadata,
    score_map,
    top_k_by_chunk,
)
from src.utils import (
    first_present,
    iter_jsonl,
    normalize_whitespace,
    simple_tokenize,
    write_jsonl,
)


RISK_NUMBER_DATE = "number_date"
RISK_ENTITY = "entity"
RISK_RELATION = "relation"
RISK_GENERAL = "general"
RISK_TYPES = (RISK_NUMBER_DATE, RISK_ENTITY, RISK_RELATION, RISK_GENERAL)

DEFAULT_ALPHA_BY_RISK = {
    RISK_NUMBER_DATE: 0.75,
    RISK_ENTITY: 0.65,
    RISK_RELATION: 0.35,
    RISK_GENERAL: 0.50,
}
DEFAULT_COVERAGE_WEIGHTS_BY_RISK = {
    RISK_NUMBER_DATE: {
        "keyword": 0.05,
        "number_date": 0.20,
        "entity": 0.05,
    },
    RISK_ENTITY: {
        "keyword": 0.05,
        "number_date": 0.05,
        "entity": 0.20,
    },
    RISK_RELATION: {
        "keyword": 0.12,
        "number_date": 0.04,
        "entity": 0.04,
    },
    RISK_GENERAL: {
        "keyword": 0.10,
        "number_date": 0.05,
        "entity": 0.05,
    },
}
DEFAULT_EXPANSION_RADIUS = 1
DEFAULT_EXPANSION_THRESHOLDS = {
    RISK_NUMBER_DATE: {
        "number_date": 1.0,
        "entity": 0.5,
        "keyword": 0.2,
    },
    RISK_ENTITY: {
        "entity": 1.0,
        "keyword": 0.2,
    },
    RISK_RELATION: {
        "keyword": 0.45,
        "entity": 0.3,
    },
    RISK_GENERAL: {
        "keyword": 0.35,
    },
}

QUERY_TEXT_KEYS = ("text", "query", "sentence")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b|\$\s*\d+(?:,\d{3})*(?:\.\d+)?")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\b",
    re.IGNORECASE,
)
DATE_CUE_PATTERN = re.compile(r"\b(?:fiscal year|quarter|month|week|day|date)\b", re.IGNORECASE)
ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")
MULTIWORD_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z&.-]+(?:\s+[A-Z][A-Za-z&.-]+)+\b")
RELATION_CUE_PATTERN = re.compile(
    r"\b(?:because|caused|due to|led to|resulted|attributed|according to|"
    r"compared with|compared to|increase|increased|decrease|decreased|rose|fell|"
    r"requires|required|allows|allowed|amended|extended|restrict|restricted|"
    r"created|established|funded|produced|partnered|found|reported|estimated)\b",
    re.IGNORECASE,
)


@dataclass
class ArielStats:
    input_bm25_results: int = 0
    input_embedding_results: int = 0
    written_results: int = 0
    missing_embedding_results: int = 0
    empty_candidate_sets: int = 0
    number_date_queries: int = 0
    entity_queries: int = 0
    relation_queries: int = 0
    general_queries: int = 0
    expanded_evidence_items: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def record_risk_type(self, risk_type: str) -> None:
        if risk_type == RISK_NUMBER_DATE:
            self.number_date_queries += 1
        elif risk_type == RISK_ENTITY:
            self.entity_queries += 1
        elif risk_type == RISK_RELATION:
            self.relation_queries += 1
        else:
            self.general_queries += 1

    def record_expansion(self, count: int) -> None:
        self.expanded_evidence_items += count


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def safe_ratio(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return matched / total


def canonical_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def canonical_number_date(value: str) -> str:
    value = value.lower().strip()
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = value.rstrip(".")
    return value


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in simple_tokenize(text)
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    }


def extract_number_date_terms(text: str) -> set[str]:
    terms = {
        canonical_number_date(match.group(0))
        for match in NUMBER_PATTERN.finditer(text)
    }
    terms.update(canonical_number_date(match.group(0)) for match in YEAR_PATTERN.finditer(text))
    terms.update(canonical_text(match.group(0)) for match in MONTH_PATTERN.finditer(text))
    return {term for term in terms if term}


def trim_entity(entity: str) -> str:
    entity = entity.strip(" ,.;:()[]{}")
    for prefix in ("The ", "A ", "An "):
        if entity.startswith(prefix):
            entity = entity[len(prefix) :]
    return entity.strip(" ,.;:()[]{}")


def extract_entity_terms(text: str) -> set[str]:
    terms = {canonical_text(match.group(0)) for match in ACRONYM_PATTERN.finditer(text)}
    for match in MULTIWORD_ENTITY_PATTERN.finditer(text):
        entity = trim_entity(match.group(0))
        if entity and len(entity.split()) <= 6:
            terms.add(canonical_text(entity))
    return {term for term in terms if term}


def coverage_ratio(query_terms: set[str], chunk_terms: set[str]) -> float:
    return safe_ratio(len(query_terms & chunk_terms), len(query_terms))


def substring_coverage_ratio(query_terms: set[str], chunk_text: str) -> tuple[float, list[str]]:
    chunk_text = canonical_text(chunk_text)
    matched = sorted(term for term in query_terms if term and term in chunk_text)
    return safe_ratio(len(matched), len(query_terms)), matched


def keyword_coverage(query: str, chunk_text: str) -> tuple[float, list[str]]:
    query_terms = content_tokens(query)
    chunk_terms = content_tokens(chunk_text)
    matched = sorted(query_terms & chunk_terms)
    return coverage_ratio(query_terms, chunk_terms), matched


def number_date_coverage(query: str, chunk_text: str) -> tuple[float, list[str]]:
    query_terms = extract_number_date_terms(query)
    chunk_terms = extract_number_date_terms(chunk_text)
    matched = sorted(query_terms & chunk_terms)
    return coverage_ratio(query_terms, chunk_terms), matched


def entity_coverage(query: str, chunk_text: str) -> tuple[float, list[str]]:
    query_terms = extract_entity_terms(query)
    return substring_coverage_ratio(query_terms, chunk_text)


def coverage_weights_for_risk(
    risk_type: str,
    coverage_weights_by_risk: dict[str, dict[str, float]],
) -> dict[str, float]:
    return coverage_weights_by_risk.get(
        risk_type,
        coverage_weights_by_risk[RISK_GENERAL],
    )


def normalize_coverage_weights(
    coverage_weights_by_risk: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    source = coverage_weights_by_risk or DEFAULT_COVERAGE_WEIGHTS_BY_RISK
    normalized: dict[str, dict[str, float]] = {}
    for risk_type in RISK_TYPES:
        source_weights = source.get(risk_type, DEFAULT_COVERAGE_WEIGHTS_BY_RISK[risk_type])
        normalized[risk_type] = {}
        for key in ("keyword", "number_date", "entity"):
            value = float(source_weights.get(key, DEFAULT_COVERAGE_WEIGHTS_BY_RISK[risk_type][key]))
            if value < 0:
                raise ValueError(f"coverage weight {key} for {risk_type} must be non-negative.")
            normalized[risk_type][key] = value
    return normalized


def compute_coverage_features(
    query: str,
    chunk_text: str,
    risk_type: str,
    coverage_weights_by_risk: dict[str, dict[str, float]],
    enable_coverage: bool = True,
) -> dict[str, Any]:
    if not enable_coverage:
        return {
            "keyword_coverage": 0.0,
            "number_date_coverage": 0.0,
            "entity_coverage": 0.0,
            "coverage_bonus": 0.0,
            "matched_keywords": [],
            "matched_number_date_terms": [],
            "matched_entities": [],
        }

    keyword_score, matched_keywords = keyword_coverage(query, chunk_text)
    number_date_score, matched_number_date_terms = number_date_coverage(query, chunk_text)
    entity_score, matched_entities = entity_coverage(query, chunk_text)
    weights = coverage_weights_for_risk(risk_type, coverage_weights_by_risk)
    coverage_bonus = (
        weights["keyword"] * keyword_score
        + weights["number_date"] * number_date_score
        + weights["entity"] * entity_score
    )
    return {
        "keyword_coverage": round(keyword_score, 6),
        "number_date_coverage": round(number_date_score, 6),
        "entity_coverage": round(entity_score, 6),
        "coverage_bonus": round(coverage_bonus, 6),
        "matched_keywords": matched_keywords,
        "matched_number_date_terms": matched_number_date_terms,
        "matched_entities": matched_entities,
    }


def has_query_terms_for_feature(query: str, feature_name: str) -> bool:
    if feature_name == "keyword":
        return bool(content_tokens(query))
    if feature_name == "number_date":
        return bool(extract_number_date_terms(query))
    if feature_name == "entity":
        return bool(extract_entity_terms(query))
    return False


def coverage_is_sufficient(
    query: str,
    risk_type: str,
    coverage_features: dict[str, Any],
) -> bool:
    thresholds = DEFAULT_EXPANSION_THRESHOLDS.get(
        risk_type,
        DEFAULT_EXPANSION_THRESHOLDS[RISK_GENERAL],
    )
    checked_any = False
    for feature_name, threshold in thresholds.items():
        if not has_query_terms_for_feature(query, feature_name):
            continue
        checked_any = True
        score = float(coverage_features.get(f"{feature_name}_coverage", 0.0))
        if score < threshold:
            return False
    return True if checked_any else True


def build_chunk_lookup(
    chunks: Iterable[dict[str, Any]] | None,
) -> dict[str, dict[int, dict[str, Any]]]:
    lookup: dict[str, dict[int, dict[str, Any]]] = {}
    if chunks is None:
        return lookup
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "").strip()
        chunk_index = chunk.get("chunk_index")
        if not doc_id or chunk_index is None:
            continue
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            continue
        lookup.setdefault(doc_id, {})[chunk_index] = chunk
    return lookup


def adjacent_chunk_records(
    doc_id: str,
    chunk_index: int | None,
    chunk_lookup: dict[str, dict[int, dict[str, Any]]],
    radius: int,
) -> list[dict[str, Any]]:
    if chunk_index is None or radius <= 0:
        return []
    doc_chunks = chunk_lookup.get(doc_id, {})
    records: list[dict[str, Any]] = []
    for index in range(chunk_index - radius, chunk_index + radius + 1):
        record = doc_chunks.get(index)
        if record is not None:
            records.append(record)
    return records


def expanded_context(
    chunk: dict[str, Any],
    doc_id: str,
    chunk_lookup: dict[str, dict[int, dict[str, Any]]],
    radius: int,
) -> tuple[str, list[str]]:
    chunk_index = chunk.get("chunk_index")
    try:
        chunk_index = int(chunk_index)
    except (TypeError, ValueError):
        chunk_index = None
    records = adjacent_chunk_records(
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_lookup=chunk_lookup,
        radius=radius,
    )
    if not records:
        text = normalize_whitespace(chunk.get("text"))
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        return text, [chunk_id] if chunk_id else []
    texts = [normalize_whitespace(record.get("text")) for record in records]
    chunk_ids = [str(record.get("chunk_id") or "").strip() for record in records]
    return "\n\n".join(text for text in texts if text), [chunk_id for chunk_id in chunk_ids if chunk_id]


def maybe_expand_evidence(
    query: str,
    risk_type: str,
    chunk: dict[str, Any],
    doc_id: str,
    coverage_features: dict[str, Any],
    coverage_weights_by_risk: dict[str, dict[str, float]],
    chunk_lookup: dict[str, dict[int, dict[str, Any]]],
    expansion_radius: int,
    enable_expansion: bool,
) -> tuple[str, dict[str, Any], bool, list[str], str]:
    original_text = normalize_whitespace(chunk.get("text"))
    if (
        not enable_expansion
        or not chunk_lookup
        or coverage_is_sufficient(query, risk_type, coverage_features)
    ):
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        return original_text, coverage_features, False, [chunk_id] if chunk_id else [], ""

    expanded_text, expanded_chunk_ids = expanded_context(
        chunk=chunk,
        doc_id=doc_id,
        chunk_lookup=chunk_lookup,
        radius=expansion_radius,
    )
    if not expanded_text or expanded_text == original_text:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        return original_text, coverage_features, False, [chunk_id] if chunk_id else [], ""

    expanded_features = compute_coverage_features(
        query=query,
        chunk_text=expanded_text,
        risk_type=risk_type,
        coverage_weights_by_risk=coverage_weights_by_risk,
        enable_coverage=True,
    )
    if float(expanded_features["coverage_bonus"]) < float(coverage_features["coverage_bonus"]):
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        return original_text, coverage_features, False, [chunk_id] if chunk_id else [], ""

    reason = "coverage below risk-specific threshold"
    return expanded_text, expanded_features, True, expanded_chunk_ids, reason


def has_number_or_date(text: str) -> bool:
    return bool(
        NUMBER_PATTERN.search(text)
        or YEAR_PATTERN.search(text)
        or MONTH_PATTERN.search(text)
        or DATE_CUE_PATTERN.search(text)
    )


def has_entity_signal(text: str) -> bool:
    return bool(ACRONYM_PATTERN.search(text) or MULTIWORD_ENTITY_PATTERN.search(text))


def has_relation_signal(text: str) -> bool:
    return bool(RELATION_CUE_PATTERN.search(text))


def classify_risk_type(query: str) -> str:
    query = normalize_whitespace(query)
    if not query:
        return RISK_GENERAL
    if has_number_or_date(query):
        return RISK_NUMBER_DATE
    if has_relation_signal(query):
        return RISK_RELATION
    if has_entity_signal(query):
        return RISK_ENTITY
    return RISK_GENERAL


def alpha_for_risk(risk_type: str, alpha_by_risk: dict[str, float]) -> float:
    return alpha_by_risk.get(risk_type, alpha_by_risk[RISK_GENERAL])


def normalize_alpha_values(alpha_by_risk: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for risk_type in RISK_TYPES:
        alpha = alpha_by_risk.get(risk_type, DEFAULT_ALPHA_BY_RISK[risk_type])
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha for {risk_type} must be between 0 and 1.")
        normalized[risk_type] = alpha
    return normalized


def query_text_from_record(
    bm25_record: dict[str, Any],
    embedding_record: dict[str, Any],
    query_by_id: dict[str, str],
) -> str:
    query_id = str(bm25_record.get("query_id") or embedding_record.get("query_id") or "")
    query_text = str(
        first_present(bm25_record, QUERY_TEXT_KEYS)
        or first_present(embedding_record, QUERY_TEXT_KEYS)
        or query_by_id.get(query_id, "")
        or ""
    )
    return normalize_whitespace(query_text)


def index_queries_by_query_id(queries: Iterable[dict[str, Any]]) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for item in queries:
        query_id = str(item.get("query_id") or "").strip()
        query_text = normalize_whitespace(first_present(item, QUERY_TEXT_KEYS))
        if query_id and query_text:
            indexed[query_id] = query_text
    return indexed


def fuse_one_ariel_result(
    bm25_record: dict[str, Any],
    embedding_record: dict[str, Any],
    query_by_id: dict[str, str],
    alpha_by_risk: dict[str, float],
    coverage_weights_by_risk: dict[str, dict[str, float]],
    chunk_lookup: dict[str, dict[int, dict[str, Any]]],
    top_k: int = 3,
    enable_coverage: bool = True,
    enable_expansion: bool = True,
    expansion_radius: int = DEFAULT_EXPANSION_RADIUS,
) -> dict[str, Any] | None:
    bm25_chunks = top_k_by_chunk(bm25_record)
    embedding_chunks = top_k_by_chunk(embedding_record)
    candidate_ids = set(bm25_chunks) | set(embedding_chunks)
    if not candidate_ids:
        return None

    query_text = query_text_from_record(bm25_record, embedding_record, query_by_id)
    risk_type = classify_risk_type(query_text)
    alpha = alpha_for_risk(risk_type, alpha_by_risk)
    doc_id = str(bm25_record.get("doc_id") or embedding_record.get("doc_id") or "")

    bm25_scores = minmax_normalize(score_map(bm25_chunks))
    embedding_scores = minmax_normalize(score_map(embedding_chunks))

    fused: list[dict[str, Any]] = []
    for chunk_id in candidate_ids:
        bm25_norm = bm25_scores.get(chunk_id, 0.0)
        embedding_norm = embedding_scores.get(chunk_id, 0.0)
        base_score = alpha * bm25_norm + (1 - alpha) * embedding_norm
        chunk = merge_chunk_metadata(chunk_id, bm25_chunks, embedding_chunks)
        coverage_features = compute_coverage_features(
            query=query_text,
            chunk_text=str(chunk.get("text") or ""),
            risk_type=risk_type,
            coverage_weights_by_risk=coverage_weights_by_risk,
            enable_coverage=enable_coverage,
        )
        evidence_text, coverage_features, expanded, expanded_chunk_ids, expansion_reason = (
            maybe_expand_evidence(
                query=query_text,
                risk_type=risk_type,
                chunk=chunk,
                doc_id=doc_id,
                coverage_features=coverage_features,
                coverage_weights_by_risk=coverage_weights_by_risk,
                chunk_lookup=chunk_lookup,
                expansion_radius=expansion_radius,
                enable_expansion=enable_expansion and enable_coverage,
            )
        )
        ariel_score = base_score + float(coverage_features["coverage_bonus"])
        chunk["text"] = evidence_text
        chunk.update(
            {
                "score": round(ariel_score, 6),
                "base_score": round(base_score, 6),
                "bm25_score": round(float(bm25_chunks.get(chunk_id, {}).get("score", 0.0)), 6),
                "embedding_score": round(
                    float(embedding_chunks.get(chunk_id, {}).get("score", 0.0)),
                    6,
                ),
                "bm25_norm": round(bm25_norm, 6),
                "embedding_norm": round(embedding_norm, 6),
                "risk_type": risk_type,
                "alpha": alpha,
                "coverage_rerank": enable_coverage,
                "adaptive_expansion": enable_expansion,
                "expanded_context": expanded,
                "expanded_chunk_ids": expanded_chunk_ids,
                "expansion_reason": expansion_reason,
                **coverage_features,
            }
        )
        fused.append(chunk)

    fused.sort(key=lambda item: item["score"], reverse=True)
    top_chunks = []
    for rank, item in enumerate(fused[:top_k], start=1):
        item = dict(item)
        item["rank"] = rank
        top_chunks.append(item)
    expanded_count = sum(1 for item in top_chunks if item.get("expanded_context"))

    return {
        "query_id": bm25_record["query_id"],
        "doc_id": doc_id,
        "query": query_text,
        "retrieval_method": "ariel",
        "risk_type": risk_type,
        "alpha": alpha,
        "coverage_rerank": enable_coverage,
        "adaptive_expansion": enable_expansion,
        "expansion_radius": expansion_radius,
        "expanded_evidence_count": expanded_count,
        "coverage_weights": coverage_weights_for_risk(risk_type, coverage_weights_by_risk),
        "top_k": top_chunks,
    }


def retrieve_ariel(
    bm25_results: Iterable[dict[str, Any]],
    embedding_results: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]] | None = None,
    chunks: Iterable[dict[str, Any]] | None = None,
    alpha_by_risk: dict[str, float] | None = None,
    coverage_weights_by_risk: dict[str, dict[str, float]] | None = None,
    top_k: int = 3,
    enable_coverage: bool = True,
    enable_expansion: bool = True,
    expansion_radius: int = DEFAULT_EXPANSION_RADIUS,
) -> tuple[list[dict[str, Any]], ArielStats]:
    if expansion_radius < 0:
        raise ValueError("expansion_radius must be non-negative.")
    alpha_by_risk = normalize_alpha_values(alpha_by_risk or DEFAULT_ALPHA_BY_RISK)
    coverage_weights_by_risk = normalize_coverage_weights(coverage_weights_by_risk)
    query_by_id = index_queries_by_query_id(queries or [])
    chunk_lookup = build_chunk_lookup(chunks)
    if not chunk_lookup:
        enable_expansion = False

    bm25_records = list(bm25_results)
    embedding_records = list(embedding_results)
    embedding_by_query_id = index_by_query_id(embedding_records)

    stats = ArielStats(
        input_bm25_results=len(bm25_records),
        input_embedding_results=len(embedding_records),
    )
    output: list[dict[str, Any]] = []

    for bm25_record in bm25_records:
        query_id = str(bm25_record.get("query_id") or "").strip()
        embedding_record = embedding_by_query_id.get(query_id)
        if embedding_record is None:
            stats.missing_embedding_results += 1
            continue

        fused_record = fuse_one_ariel_result(
            bm25_record=bm25_record,
            embedding_record=embedding_record,
            query_by_id=query_by_id,
            alpha_by_risk=alpha_by_risk,
            coverage_weights_by_risk=coverage_weights_by_risk,
            chunk_lookup=chunk_lookup,
            top_k=top_k,
            enable_coverage=enable_coverage,
            enable_expansion=enable_expansion,
            expansion_radius=expansion_radius,
        )
        if fused_record is None:
            stats.empty_candidate_sets += 1
            continue
        stats.record_risk_type(str(fused_record.get("risk_type") or RISK_GENERAL))
        stats.record_expansion(int(fused_record.get("expanded_evidence_count") or 0))
        output.append(fused_record)
        stats.written_results += 1

    return output, stats


def retrieve_ariel_jsonl(
    bm25_path: str | Path,
    embedding_path: str | Path,
    output_path: str | Path,
    queries_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
    alpha_by_risk: dict[str, float] | None = None,
    coverage_weights_by_risk: dict[str, dict[str, float]] | None = None,
    top_k: int = 3,
    enable_coverage: bool = True,
    enable_expansion: bool = True,
    expansion_radius: int = DEFAULT_EXPANSION_RADIUS,
) -> ArielStats:
    queries = iter_jsonl(queries_path) if queries_path else None
    chunks = iter_jsonl(chunks_path) if chunks_path else None
    results, stats = retrieve_ariel(
        bm25_results=iter_jsonl(bm25_path),
        embedding_results=iter_jsonl(embedding_path),
        queries=queries,
        chunks=chunks,
        alpha_by_risk=alpha_by_risk,
        coverage_weights_by_risk=coverage_weights_by_risk,
        top_k=top_k,
        enable_coverage=enable_coverage,
        enable_expansion=enable_expansion,
        expansion_radius=expansion_radius,
    )
    write_jsonl(results, output_path)
    return stats


def alpha_config_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        RISK_NUMBER_DATE: args.number_date_alpha,
        RISK_ENTITY: args.entity_alpha,
        RISK_RELATION: args.relation_alpha,
        RISK_GENERAL: args.general_alpha,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ARIEL retrieval fusion with coverage reranking."
    )
    parser.add_argument("--bm25", required=True, help="BM25 retrieval JSONL path.")
    parser.add_argument("--embedding", required=True, help="Embedding retrieval JSONL path.")
    parser.add_argument("--output", required=True, help="ARIEL retrieval JSONL output path.")
    parser.add_argument(
        "--queries",
        default=None,
        help="Optional summary sentence query JSONL path for risk classification.",
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help="Optional full chunk JSONL path for adaptive adjacent-context expansion.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to keep.")
    parser.add_argument(
        "--disable-coverage-rerank",
        action="store_true",
        help="Disable keyword, number/date, and entity coverage reranking.",
    )
    parser.add_argument(
        "--disable-adaptive-expansion",
        action="store_true",
        help="Disable adjacent chunk expansion when coverage is insufficient.",
    )
    parser.add_argument(
        "--expansion-radius",
        type=int,
        default=DEFAULT_EXPANSION_RADIUS,
        help="Number of neighboring chunks to include on each side when expanding.",
    )
    parser.add_argument(
        "--number-date-alpha",
        type=float,
        default=DEFAULT_ALPHA_BY_RISK[RISK_NUMBER_DATE],
        help="BM25 weight for number/date risk sentences.",
    )
    parser.add_argument(
        "--entity-alpha",
        type=float,
        default=DEFAULT_ALPHA_BY_RISK[RISK_ENTITY],
        help="BM25 weight for entity risk sentences.",
    )
    parser.add_argument(
        "--relation-alpha",
        type=float,
        default=DEFAULT_ALPHA_BY_RISK[RISK_RELATION],
        help="BM25 weight for relation risk sentences.",
    )
    parser.add_argument(
        "--general-alpha",
        type=float,
        default=DEFAULT_ALPHA_BY_RISK[RISK_GENERAL],
        help="BM25 weight for general sentences.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = retrieve_ariel_jsonl(
        bm25_path=args.bm25,
        embedding_path=args.embedding,
        output_path=args.output,
        queries_path=args.queries,
        chunks_path=args.chunks,
        alpha_by_risk=alpha_config_from_args(args),
        top_k=args.top_k,
        enable_coverage=not args.disable_coverage_rerank,
        enable_expansion=not args.disable_adaptive_expansion,
        expansion_radius=args.expansion_radius,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
