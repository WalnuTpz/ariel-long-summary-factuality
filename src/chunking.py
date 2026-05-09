"""Chunk normalized long documents into retrievable evidence units."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import iter_jsonl, normalize_whitespace, simple_tokenize, write_jsonl


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    text: str
    chunk_type: str
    token_count: int
    char_start: int
    char_end: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkStats:
    input_documents: int = 0
    written_chunks: int = 0
    skipped_empty_documents: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def paragraph_spans(document: str) -> list[tuple[str, int, int]]:
    """Return non-empty paragraph text with approximate character spans."""
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for raw_paragraph in document.split("\n\n"):
        start = document.find(raw_paragraph, cursor)
        end = start + len(raw_paragraph)
        cursor = end
        text = normalize_whitespace(raw_paragraph)
        if text:
            spans.append((text, start, end))
    return spans


def whitespace_token_spans(document: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for token in document.split():
        start = document.find(token, cursor)
        end = start + len(token)
        cursor = end
        spans.append((token, start, end))
    return spans


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}_chunk_{chunk_index:04d}"


def make_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    chunk_type: str,
    char_start: int,
    char_end: int,
    meta: dict[str, Any] | None = None,
) -> ChunkRecord:
    text = normalize_whitespace(text)
    return ChunkRecord(
        doc_id=doc_id,
        chunk_id=make_chunk_id(doc_id, chunk_index),
        chunk_index=chunk_index,
        text=text,
        chunk_type=chunk_type,
        token_count=len(simple_tokenize(text)),
        char_start=char_start,
        char_end=char_end,
        meta=meta or {},
    )


def paragraph_chunks(
    doc_id: str,
    document: str,
    max_tokens: int | None = None,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for paragraph_index, (text, start, end) in enumerate(paragraph_spans(document)):
        token_count = len(simple_tokenize(text))
        if max_tokens and token_count > max_tokens:
            for split_chunk in window_chunks(
                doc_id=doc_id,
                document=text,
                max_tokens=max_tokens,
                stride=max_tokens,
                chunk_type="paragraph_window",
                chunk_index_offset=len(chunks),
                char_offset=start,
            ):
                chunks.append(split_chunk)
            continue
        chunks.append(
            make_chunk(
                doc_id=doc_id,
                chunk_index=len(chunks),
                text=text,
                chunk_type="paragraph",
                char_start=start,
                char_end=end,
                meta={"paragraph_index": paragraph_index},
            )
        )
    return chunks


def window_chunks(
    doc_id: str,
    document: str,
    max_tokens: int = 256,
    stride: int | None = None,
    chunk_type: str = "token_window",
    chunk_index_offset: int = 0,
    char_offset: int = 0,
) -> list[ChunkRecord]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    stride = stride or max_tokens
    if stride <= 0:
        raise ValueError("stride must be positive.")

    token_spans = whitespace_token_spans(document)
    chunks: list[ChunkRecord] = []
    for start_index in range(0, len(token_spans), stride):
        window = token_spans[start_index : start_index + max_tokens]
        if not window:
            continue
        text = " ".join(token for token, _, _ in window)
        char_start = char_offset + window[0][1]
        char_end = char_offset + window[-1][2]
        chunks.append(
            make_chunk(
                doc_id=doc_id,
                chunk_index=chunk_index_offset + len(chunks),
                text=text,
                chunk_type=chunk_type,
                char_start=char_start,
                char_end=char_end,
                meta={
                    "token_start": start_index,
                    "token_end": start_index + len(window),
                },
            )
        )
        if start_index + max_tokens >= len(token_spans):
            break
    return chunks


def chunk_document(
    item: dict[str, Any],
    chunk_type: str = "paragraph",
    max_tokens: int = 256,
    stride: int | None = None,
) -> list[ChunkRecord]:
    doc_id = str(item.get("doc_id") or item.get("id") or "").strip()
    if not doc_id:
        raise ValueError("Document record is missing doc_id.")
    document = normalize_whitespace(item.get("document"))
    if not document:
        return []
    if chunk_type == "paragraph":
        return paragraph_chunks(doc_id=doc_id, document=document, max_tokens=max_tokens)
    if chunk_type == "token_window":
        return window_chunks(
            doc_id=doc_id,
            document=document,
            max_tokens=max_tokens,
            stride=stride,
        )
    raise ValueError(f"Unsupported chunk_type: {chunk_type}")


def chunk_documents(
    documents: Iterable[dict[str, Any]],
    chunk_type: str = "paragraph",
    max_tokens: int = 256,
    stride: int | None = None,
) -> tuple[list[dict[str, Any]], ChunkStats]:
    stats = ChunkStats()
    output: list[dict[str, Any]] = []
    for item in documents:
        stats.input_documents += 1
        chunks = chunk_document(
            item,
            chunk_type=chunk_type,
            max_tokens=max_tokens,
            stride=stride,
        )
        if not chunks:
            stats.skipped_empty_documents += 1
            continue
        output.extend(chunk.to_dict() for chunk in chunks)
        stats.written_chunks += len(chunks)
    return output, stats


def chunk_jsonl_documents(
    input_path: str | Path,
    output_path: str | Path,
    chunk_type: str = "paragraph",
    max_tokens: int = 256,
    stride: int | None = None,
) -> ChunkStats:
    chunks, stats = chunk_documents(
        iter_jsonl(input_path),
        chunk_type=chunk_type,
        max_tokens=max_tokens,
        stride=stride,
    )
    write_jsonl(chunks, output_path)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk normalized document JSONL.")
    parser.add_argument("--input", required=True, help="Normalized document JSONL path.")
    parser.add_argument("--output", required=True, help="Chunk JSONL output path.")
    parser.add_argument(
        "--chunk-type",
        choices=["paragraph", "token_window"],
        default="paragraph",
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per chunk or window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for token_window chunking. Defaults to max_tokens.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = chunk_jsonl_documents(
        input_path=args.input,
        output_path=args.output,
        chunk_type=args.chunk_type,
        max_tokens=args.max_tokens,
        stride=args.stride,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
