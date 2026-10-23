"""Hybrid evidence retrieval by fusing BM25 and embedding scores."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import iter_jsonl, write_jsonl


@dataclass
class HybridStats:
    input_bm25_results: int = 0
    input_embedding_results: int = 0
    written_results: int = 0
    missing_embedding_results: int = 0
    empty_candidate_sets: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {key: 0.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def index_by_query_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def top_k_by_chunk(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for item in record.get("top_k", []):
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id:
            chunks[chunk_id] = item
    return chunks


def score_map(chunks: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        chunk_id: float(item.get("score", 0.0))
        for chunk_id, item in chunks.items()
    }


def merge_chunk_metadata(
    chunk_id: str,
    bm25_chunks: dict[str, dict[str, Any]],
    embedding_chunks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = bm25_chunks.get(chunk_id) or embedding_chunks[chunk_id]
    return {
        "chunk_id": source["chunk_id"],
        "chunk_index": source.get("chunk_index"),
        "text": source.get("text", ""),
    }


def fuse_one_result(
    bm25_record: dict[str, Any],
    embedding_record: dict[str, Any],
    alpha: float = 0.5,
    top_k: int = 3,
) -> dict[str, Any] | None:
    bm25_chunks = top_k_by_chunk(bm25_record)
    embedding_chunks = top_k_by_chunk(embedding_record)
    candidate_ids = set(bm25_chunks) | set(embedding_chunks)
    if not candidate_ids:
        return None

    bm25_scores = minmax_normalize(score_map(bm25_chunks))
    embedding_scores = minmax_normalize(score_map(embedding_chunks))

    fused: list[dict[str, Any]] = []
    for chunk_id in candidate_ids:
        bm25_norm = bm25_scores.get(chunk_id, 0.0)
        embedding_norm = embedding_scores.get(chunk_id, 0.0)
        hybrid_score = alpha * bm25_norm + (1 - alpha) * embedding_norm
        chunk = merge_chunk_metadata(chunk_id, bm25_chunks, embedding_chunks)
        chunk.update(
            {
                "score": round(hybrid_score, 6),
                "bm25_score": round(float(bm25_chunks.get(chunk_id, {}).get("score", 0.0)), 6),
                "embedding_score": round(
                    float(embedding_chunks.get(chunk_id, {}).get("score", 0.0)),
                    6,
                ),
                "bm25_norm": round(bm25_norm, 6),
                "embedding_norm": round(embedding_norm, 6),
            }
        )
        fused.append(chunk)

    fused.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    top_chunks = []
    for rank, item in enumerate(fused[:top_k], start=1):
        item = dict(item)
        item["rank"] = rank
        top_chunks.append(item)

    return {
        "query_id": bm25_record["query_id"],
        "doc_id": bm25_record.get("doc_id"),
        "query": bm25_record.get("query", embedding_record.get("query", "")),
        "retrieval_method": "hybrid",
        "alpha": alpha,
        "top_k": top_chunks,
    }


def retrieve_hybrid(
    bm25_results: Iterable[dict[str, Any]],
    embedding_results: Iterable[dict[str, Any]],
    alpha: float = 0.5,
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], HybridStats]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    bm25_records = list(bm25_results)
    embedding_records = list(embedding_results)
    embedding_by_query_id = index_by_query_id(embedding_records)

    stats = HybridStats(
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
        fused_record = fuse_one_result(
            bm25_record=bm25_record,
            embedding_record=embedding_record,
            alpha=alpha,
            top_k=top_k,
        )
        if fused_record is None:
            stats.empty_candidate_sets += 1
            continue
        output.append(fused_record)
        stats.written_results += 1

    return output, stats


def retrieve_hybrid_jsonl(
    bm25_path: str | Path,
    embedding_path: str | Path,
    output_path: str | Path,
    alpha: float = 0.5,
    top_k: int = 3,
) -> HybridStats:
    results, stats = retrieve_hybrid(
        bm25_results=iter_jsonl(bm25_path),
        embedding_results=iter_jsonl(embedding_path),
        alpha=alpha,
        top_k=top_k,
    )
    write_jsonl(results, output_path)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse BM25 and embedding retrieval.")
    parser.add_argument("--bm25", required=True, help="BM25 retrieval JSONL path.")
    parser.add_argument(
        "--embedding",
        required=True,
        help="Embedding retrieval JSONL path.",
    )
    parser.add_argument("--output", required=True, help="Hybrid retrieval JSONL path.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for normalized BM25 score. Embedding weight is 1-alpha.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to keep.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = retrieve_hybrid_jsonl(
        bm25_path=args.bm25,
        embedding_path=args.embedding,
        output_path=args.output,
        alpha=args.alpha,
        top_k=args.top_k,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
