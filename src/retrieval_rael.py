"""Risk-aware adaptive evidence localization.

The basic RAEL version classifies each summary sentence into a coarse risk
type and uses that type to choose a dynamic BM25/embedding fusion weight.
Coverage reranking and adaptive context expansion are intentionally left for
later extensions.
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
from src.utils import first_present, iter_jsonl, normalize_whitespace, write_jsonl


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

QUERY_TEXT_KEYS = ("text", "query", "sentence")
NUMBER_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b|\$\s*\d")
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
class RaelStats:
    input_bm25_results: int = 0
    input_embedding_results: int = 0
    written_results: int = 0
    missing_embedding_results: int = 0
    empty_candidate_sets: int = 0
    number_date_queries: int = 0
    entity_queries: int = 0
    relation_queries: int = 0
    general_queries: int = 0

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


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


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


def fuse_one_rael_result(
    bm25_record: dict[str, Any],
    embedding_record: dict[str, Any],
    query_by_id: dict[str, str],
    alpha_by_risk: dict[str, float],
    top_k: int = 3,
) -> dict[str, Any] | None:
    bm25_chunks = top_k_by_chunk(bm25_record)
    embedding_chunks = top_k_by_chunk(embedding_record)
    candidate_ids = set(bm25_chunks) | set(embedding_chunks)
    if not candidate_ids:
        return None

    query_text = query_text_from_record(bm25_record, embedding_record, query_by_id)
    risk_type = classify_risk_type(query_text)
    alpha = alpha_for_risk(risk_type, alpha_by_risk)

    bm25_scores = minmax_normalize(score_map(bm25_chunks))
    embedding_scores = minmax_normalize(score_map(embedding_chunks))

    fused: list[dict[str, Any]] = []
    for chunk_id in candidate_ids:
        bm25_norm = bm25_scores.get(chunk_id, 0.0)
        embedding_norm = embedding_scores.get(chunk_id, 0.0)
        rael_score = alpha * bm25_norm + (1 - alpha) * embedding_norm
        chunk = merge_chunk_metadata(chunk_id, bm25_chunks, embedding_chunks)
        chunk.update(
            {
                "score": round(rael_score, 6),
                "bm25_score": round(float(bm25_chunks.get(chunk_id, {}).get("score", 0.0)), 6),
                "embedding_score": round(
                    float(embedding_chunks.get(chunk_id, {}).get("score", 0.0)),
                    6,
                ),
                "bm25_norm": round(bm25_norm, 6),
                "embedding_norm": round(embedding_norm, 6),
                "risk_type": risk_type,
                "alpha": alpha,
            }
        )
        fused.append(chunk)

    fused.sort(key=lambda item: item["score"], reverse=True)
    top_chunks = []
    for rank, item in enumerate(fused[:top_k], start=1):
        item = dict(item)
        item["rank"] = rank
        top_chunks.append(item)

    return {
        "query_id": bm25_record["query_id"],
        "doc_id": bm25_record.get("doc_id"),
        "query": query_text,
        "retrieval_method": "rael",
        "risk_type": risk_type,
        "alpha": alpha,
        "top_k": top_chunks,
    }


def retrieve_rael(
    bm25_results: Iterable[dict[str, Any]],
    embedding_results: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]] | None = None,
    alpha_by_risk: dict[str, float] | None = None,
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], RaelStats]:
    alpha_by_risk = normalize_alpha_values(alpha_by_risk or DEFAULT_ALPHA_BY_RISK)
    query_by_id = index_queries_by_query_id(queries or [])

    bm25_records = list(bm25_results)
    embedding_records = list(embedding_results)
    embedding_by_query_id = index_by_query_id(embedding_records)

    stats = RaelStats(
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

        fused_record = fuse_one_rael_result(
            bm25_record=bm25_record,
            embedding_record=embedding_record,
            query_by_id=query_by_id,
            alpha_by_risk=alpha_by_risk,
            top_k=top_k,
        )
        if fused_record is None:
            stats.empty_candidate_sets += 1
            continue
        stats.record_risk_type(str(fused_record.get("risk_type") or RISK_GENERAL))
        output.append(fused_record)
        stats.written_results += 1

    return output, stats


def retrieve_rael_jsonl(
    bm25_path: str | Path,
    embedding_path: str | Path,
    output_path: str | Path,
    queries_path: str | Path | None = None,
    alpha_by_risk: dict[str, float] | None = None,
    top_k: int = 3,
) -> RaelStats:
    queries = iter_jsonl(queries_path) if queries_path else None
    results, stats = retrieve_rael(
        bm25_results=iter_jsonl(bm25_path),
        embedding_results=iter_jsonl(embedding_path),
        queries=queries,
        alpha_by_risk=alpha_by_risk,
        top_k=top_k,
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
    parser = argparse.ArgumentParser(description="Run basic RAEL retrieval fusion.")
    parser.add_argument("--bm25", required=True, help="BM25 retrieval JSONL path.")
    parser.add_argument("--embedding", required=True, help="Embedding retrieval JSONL path.")
    parser.add_argument("--output", required=True, help="RAEL retrieval JSONL output path.")
    parser.add_argument(
        "--queries",
        default=None,
        help="Optional summary sentence query JSONL path for risk classification.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to keep.")
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
    stats = retrieve_rael_jsonl(
        bm25_path=args.bm25,
        embedding_path=args.embedding,
        output_path=args.output,
        queries_path=args.queries,
        alpha_by_risk=alpha_config_from_args(args),
        top_k=args.top_k,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
