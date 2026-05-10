"""Embedding evidence retrieval with sentence-transformers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from src.retrieval_bm25 import (
    RetrievalStats,
    group_chunks_by_doc,
    normalize_query,
)
from src.utils import iter_jsonl, stringify_text, write_jsonl


DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    backend_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into vectors."""


@dataclass
class EmbeddingIndex:
    chunks: list[dict[str, Any]]
    embedder: Embedder

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("EmbeddingIndex requires at least one chunk.")
        self.chunk_vectors = self.embedder.encode(
            [stringify_text(chunk.get("text", "")) for chunk in self.chunks]
        )

    def score(self, query: str) -> list[float]:
        query_vector = self.embedder.encode([query])[0]
        return [
            cosine_similarity(query_vector, chunk_vector)
            for chunk_vector in self.chunk_vectors
        ]

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


class SentenceTransformerEmbedder:
    backend_name = "sentence-transformers"

    def __init__(
        self,
        model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        device: str = "cpu",
        local_files_only: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [to_float_list(vector) for vector in vectors]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(l_value * r_value for l_value, r_value in zip(left, right))


def to_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def build_embedder(
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    device: str = "cpu",
    local_files_only: bool = False,
) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
    )


def build_embedding_indices(
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    embedder: Embedder,
) -> dict[str, EmbeddingIndex]:
    return {
        doc_id: EmbeddingIndex(chunks=chunks, embedder=embedder)
        for doc_id, chunks in chunks_by_doc.items()
        if chunks
    }


def retrieve_embedding(
    chunks: Iterable[dict[str, Any]],
    queries: Iterable[dict[str, Any]],
    top_k: int = 3,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    device: str = "cpu",
    local_files_only: bool = False,
) -> tuple[list[dict[str, Any]], RetrievalStats]:
    embedder = build_embedder(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
    )
    chunks_by_doc = group_chunks_by_doc(chunks)
    indices = build_embedding_indices(chunks_by_doc, embedder=embedder)

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
                "query_id": query_id,
                "doc_id": doc_id,
                "query": query_text,
                "retrieval_method": "embedding",
                "embedding_backend": embedder.backend_name,
                "top_k": top_chunks,
            }
        )
        stats.written_results += 1
    return output, stats


def retrieve_embedding_jsonl(
    chunks_path: str | Path,
    queries_path: str | Path,
    output_path: str | Path,
    top_k: int = 3,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    device: str = "cpu",
    local_files_only: bool = False,
) -> RetrievalStats:
    results, stats = retrieve_embedding(
        chunks=iter_jsonl(chunks_path),
        queries=iter_jsonl(queries_path),
        top_k=top_k,
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
    )
    write_jsonl(results, output_path)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run embedding retrieval over chunks.")
    parser.add_argument("--chunks", required=True, help="Chunk JSONL input path.")
    parser.add_argument(
        "--queries",
        required=True,
        help="Query JSONL input path. Requires query_id, doc_id, and text/query.",
    )
    parser.add_argument("--output", required=True, help="Retrieval JSONL output path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to keep.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for sentence-transformers, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the embedding model from local Hugging Face cache only.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = retrieve_embedding_jsonl(
        chunks_path=args.chunks,
        queries_path=args.queries,
        output_path=args.output,
        top_k=args.top_k,
        model_name=args.model_name,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
