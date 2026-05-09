"""Download a small GovReport sample into project raw JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a GovReport JSONL sample.")
    parser.add_argument(
        "--output",
        default="data/raw/govreport_30.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--split", default="validation", help="Dataset split to sample.")
    parser.add_argument("--limit", type=int, default=30, help="Number of records to save.")
    parser.add_argument(
        "--min-document-chars",
        type=int,
        default=2000,
        help="Skip documents shorter than this many characters.",
    )
    return parser


def normalize_item(
    item: dict[str, Any],
    source_index: int,
    split: str,
) -> dict[str, Any] | None:
    document = str(item.get("report") or "").strip()
    reference_summary = str(item.get("summary") or "").strip()
    if not document or not reference_summary:
        return None

    return {
        "doc_id": f"govreport_{source_index:05d}",
        "title": "",
        "document": document,
        "reference_summary": reference_summary,
        "meta": {
            "dataset": "ccdv/govreport-summarization",
            "split": split,
            "source_index": source_index,
        },
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "ccdv/govreport-summarization",
        split=args.split,
        streaming=True,
    )

    written = 0
    scanned = 0
    with output_path.open("w", encoding="utf-8") as file:
        for source_index, item in enumerate(dataset):
            scanned += 1
            record = normalize_item(item, source_index=source_index, split=args.split)
            if record is None:
                continue
            if len(record["document"]) < args.min_document_chars:
                continue
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.limit:
                break

    print(f"scanned_records: {scanned}")
    print(f"written_records: {written}")
    print(f"output_path: {output_path}")


if __name__ == "__main__":
    main()
