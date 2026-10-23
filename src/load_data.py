"""Load and normalize raw long-document summarization data.

Batch 1 supports a generic JSONL input format and converts common field names
into the project-level document schema:

{
  "doc_id": "...",
  "title": "...",
  "document": "...",
  "reference_summary": "...",
  "meta": {...}
}
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils import first_present, iter_jsonl, normalize_whitespace, safe_str, write_jsonl


DOC_ID_KEYS = ("doc_id", "id", "paper_id", "document_id", "sample_id")
TITLE_KEYS = ("title", "heading")
DOCUMENT_KEYS = ("document", "article", "text", "source", "source_text", "body")
SUMMARY_KEYS = (
    "reference_summary",
    "summary",
    "abstract",
    "highlights",
    "target",
)


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    document: str
    reference_summary: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadStats:
    input_records: int = 0
    written_records: int = 0
    skipped_empty_document: int = 0
    skipped_short_document: int = 0
    skipped_duplicate_id: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_record(
    item: dict[str, Any],
    index: int,
    min_document_chars: int = 200,
) -> tuple[DocumentRecord | None, str | None]:
    """Convert one raw JSON object to DocumentRecord.

    Returns a pair of (record, skip_reason). Exactly one element in the pair is
    non-null.
    """
    raw_document = first_present(item, DOCUMENT_KEYS)
    document = normalize_whitespace(raw_document)
    if not document:
        return None, "empty_document"
    if len(document) < min_document_chars:
        return None, "short_document"

    raw_doc_id = first_present(item, DOC_ID_KEYS)
    doc_id = safe_str(raw_doc_id).strip() or make_stable_doc_id(document, index)
    title = normalize_whitespace(first_present(item, TITLE_KEYS))
    reference_summary = normalize_whitespace(first_present(item, SUMMARY_KEYS))

    source_meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    used_keys = set(DOC_ID_KEYS + TITLE_KEYS + DOCUMENT_KEYS + SUMMARY_KEYS + ("meta",))
    meta = dict(source_meta)
    meta.update(
        {
            key: value
            for key, value in item.items()
            if key not in used_keys and value not in (None, "")
        }
    )
    meta.setdefault("source_index", index)

    return (
        DocumentRecord(
            doc_id=doc_id,
            title=title,
            document=document,
            reference_summary=reference_summary,
            meta=meta,
        ),
        None,
    )


def make_stable_doc_id(document: str, index: int) -> str:
    digest = hashlib.sha1(document.encode("utf-8")).hexdigest()[:10]
    return f"doc_{index:05d}_{digest}"


def load_jsonl_documents(
    input_path: str | Path,
    output_path: str | Path,
    min_document_chars: int = 200,
    limit: int | None = None,
) -> LoadStats:
    stats = LoadStats()
    seen_doc_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    for index, item in enumerate(iter_jsonl(input_path), start=1):
        stats.input_records += 1
        record, skip_reason = normalize_record(
            item,
            index=index,
            min_document_chars=min_document_chars,
        )
        if skip_reason == "empty_document":
            stats.skipped_empty_document += 1
            continue
        if skip_reason == "short_document":
            stats.skipped_short_document += 1
            continue
        if record is None:
            continue
        if record.doc_id in seen_doc_ids:
            stats.skipped_duplicate_id += 1
            continue

        seen_doc_ids.add(record.doc_id)
        records.append(record.to_dict())
        stats.written_records += 1
        if limit is not None and stats.written_records >= limit:
            break

    write_jsonl(records, output_path)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize raw JSONL data into project document JSONL."
    )
    parser.add_argument("--input", required=True, help="Raw JSONL input path.")
    parser.add_argument("--output", required=True, help="Normalized JSONL output path.")
    parser.add_argument(
        "--min-document-chars",
        type=int,
        default=200,
        help="Skip documents shorter than this many characters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of normalized records to write.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = load_jsonl_documents(
        input_path=args.input,
        output_path=args.output,
        min_document_chars=args.min_document_chars,
        limit=args.limit,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
