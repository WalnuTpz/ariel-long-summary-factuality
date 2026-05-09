"""Dependency-free BM25 evidence retrieval."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import first_present, iter_jsonl, simple_tokenize, write_jsonl


QUERY_ID_KEYS = ("fact_id", "query_id", "id")
QUERY_TEXT_KEYS = ("fact", "query", "claim", "text")


@dataclass
class RetrievalStats:
    input_queries: int = 0
    written_results: int = 0
    queries_without_doc_chunks: int = 0
    queries_without_text: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class BM25Index:
    """Small BM25 implementation for one document's chunks."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Index requires at least one chunk.")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_chunks = [simple_tokenize(chunk.get("text", "")) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_chunks]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_chunks]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized_chunks:
            doc_freq.update(set(tokens))

        total_docs = len(self.tokenized_chunks)
        return {
            term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        query_terms = simple_tokenize(query)
        if not query_terms or self.avg_doc_length == 0:
            return [0.0 for _ in self.chunks]

        scores: list[float] = []
        for term_freq, doc_length in zip(self.term_frequencies, self.doc_lengths):
            score = 0.0
            for term in query_terms:
                if term not in term_freq:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = term_freq[term]
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avg_doc_length
                )
                score += idf * (freq * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        scores = self.score(query)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]
        results: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked_indices, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"],
                    "score": round(scores[index], 6),
                    "text": chunk["text"],
                }
            )
        return results


def group_chunks_by_doc(chunks: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        doc_id = str(chunk.get("doc_id") or "").strip()
        if doc_id:
            grouped[doc_id].append(chunk)
    return dict(grouped)


def build_doc_indices(
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, BM25Index]:
    return {
        doc_id: BM25Index(chunks, k1=k1, b=b)
        for doc_id, chunks in chunks_by_doc.items()
        if chunks
    }


def normalize_query(item: dict[str, Any], index: int) -> tuple[str, str, str]:
    query_id = str(first_present(item, QUERY_ID_KEYS) or f"query_{index:05d}")
    doc_id = str(item.get("doc_id") or "").strip()
    query_text = str(first_present(item, QUERY_TEXT_KEYS) or "").strip()
    return query_id, doc_id, query_text


def retrieve_bm25(
    chunks: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]],
    top_k: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[list[dict[str, Any]], RetrievalStats]:
    chunks_by_doc = group_chunks_by_doc(chunks)
    indices = build_doc_indices(chunks_by_doc, k1=k1, b=b)

    stats = RetrievalStats()
    output: list[dict[str, Any]] = []
    for index, query_item in enumerate(queries, start=1):
        stats.input_queries += 1
        query_id, doc_id, query_text = normalize_query(query_item, index=index)
        if not query_text:
            stats.queries_without_text += 1
            continue
        if doc_id not in indices:
            stats.queries_without_doc_chunks += 1
            continue

        top_chunks = indices[doc_id].retrieve(query_text, top_k=top_k)
        output.append(
            {
                "fact_id": query_id,
                "doc_id": doc_id,
                "query": query_text,
                "retrieval_method": "bm25",
                "top_k": top_chunks,
            }
        )
        stats.written_results += 1
    return output, stats


def retrieve_bm25_jsonl(
    chunks_path: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    top_k: int = 3,
    k1: float = 1.5,
    b: float = 0.75,
) -> RetrievalStats:
    results, stats = retrieve_bm25(
        chunks=iter_jsonl(chunks_path),
        queries=iter_jsonl(queries_path),
        top_k=top_k,
        k1=k1,
        b=b,
    )
    write_jsonl(results, output_path)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BM25 retrieval over chunks.")
    parser.add_argument("--chunks", required=True, help="Chunk JSONL input path.")
    parser.add_argument(
        "--queries",
        required=True,
        help="Query/fact JSONL input path. Requires doc_id and fact/query text.",
    )
    parser.add_argument("--output", required=True, help="Retrieval JSONL output path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to keep.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = retrieve_bm25_jsonl(
        chunks_path=args.chunks,
        queries_path=args.queries,
        output_path=args.output,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
