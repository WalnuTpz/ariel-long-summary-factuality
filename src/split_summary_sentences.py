"""Split generated summaries into sentence-level retrieval queries."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import first_present, iter_jsonl, normalize_whitespace, simple_tokenize, write_jsonl


SUMMARY_TEXT_KEYS = (
    "generated_summary",
    "summary",
    "clean_summary",
    "reference_summary",
    "abstract",
)

ABBREVIATIONS = (
    "U.S.",
    "U.K.",
    "U.N.",
    "D.C.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Prof.",
    "Sen.",
    "Rep.",
    "Gov.",
    "No.",
    "Fig.",
    "fig.",
    "e.g.",
    "i.e.",
)

SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
DECIMAL_DOT_PATTERN = re.compile(r"(?<=\d)\.(?=\d)")
DOT_PLACEHOLDER = "<DOT>"


@dataclass
class SplitStats:
    input_summaries: int = 0
    written_sentences: int = 0
    skipped_missing_doc_id: int = 0
    skipped_missing_summary: int = 0
    skipped_short_sentences: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def protect_sentence_text(text: str) -> str:
    protected = text
    for abbreviation in ABBREVIATIONS:
        protected = protected.replace(abbreviation, abbreviation.replace(".", DOT_PLACEHOLDER))
    return DECIMAL_DOT_PATTERN.sub(DOT_PLACEHOLDER, protected)


def restore_sentence_text(text: str) -> str:
    return text.replace(DOT_PLACEHOLDER, ".")


def split_sentences(text: Any) -> list[str]:
    text = normalize_whitespace(text).replace("\n\n", " ")
    if not text:
        return []
    protected = protect_sentence_text(text)
    parts = SENTENCE_BOUNDARY_PATTERN.split(protected)
    return [restore_sentence_text(part).strip() for part in parts if part.strip()]


def resolve_summary_text(
    item: dict[str, Any],
    summary_field: str | None = None,
) -> tuple[str, str]:
    if summary_field:
        return normalize_whitespace(item.get(summary_field)), summary_field
    used_field = next(
        (key for key in SUMMARY_TEXT_KEYS if item.get(key) not in (None, "")),
        "",
    )
    if not used_field:
        return "", ""
    return normalize_whitespace(item.get(used_field)), used_field


def make_summary_id(item: dict[str, Any], doc_id: str) -> str:
    raw_summary_id = str(item.get("summary_id") or "").strip()
    return raw_summary_id or f"{doc_id}_summary"


def make_sentence_record(
    item: dict[str, Any],
    doc_id: str,
    summary_id: str,
    sentence: str,
    sentence_id: int,
    summary_word_count: int,
    source_summary_field: str,
) -> dict[str, Any]:
    query_id = f"{doc_id}_s{sentence_id:03d}"
    record = {
        "query_id": query_id,
        "doc_id": doc_id,
        "summary_id": summary_id,
        "sentence_id": sentence_id,
        "text": sentence,
        "sentence_word_count": len(simple_tokenize(sentence)),
        "summary_word_count": summary_word_count,
        "source_summary_field": source_summary_field,
    }
    model = item.get("model")
    if model not in (None, ""):
        record["model"] = model
    return record


def split_summary_records(
    summaries: Iterable[dict[str, Any]],
    summary_field: str | None = None,
    min_sentence_chars: int = 1,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], SplitStats]:
    stats = SplitStats()
    output: list[dict[str, Any]] = []

    for item in summaries:
        stats.input_summaries += 1
        doc_id = str(first_present(item, ("doc_id", "document_id", "id")) or "").strip()
        if not doc_id:
            stats.skipped_missing_doc_id += 1
            continue

        summary_text, used_field = resolve_summary_text(item, summary_field=summary_field)
        if not summary_text:
            stats.skipped_missing_summary += 1
            continue

        summary_id = make_summary_id(item, doc_id=doc_id)
        summary_word_count = len(simple_tokenize(summary_text))
        sentence_id = 0
        for sentence in split_sentences(summary_text):
            if len(sentence) < min_sentence_chars:
                stats.skipped_short_sentences += 1
                continue
            sentence_id += 1
            output.append(
                make_sentence_record(
                    item=item,
                    doc_id=doc_id,
                    summary_id=summary_id,
                    sentence=sentence,
                    sentence_id=sentence_id,
                    summary_word_count=summary_word_count,
                    source_summary_field=used_field,
                )
            )
            stats.written_sentences += 1

        if limit is not None and stats.input_summaries >= limit:
            break

    return output, stats


def split_summary_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    summary_field: str | None = None,
    min_sentence_chars: int = 1,
    limit: int | None = None,
) -> SplitStats:
    records, stats = split_summary_records(
        summaries=iter_jsonl(input_path),
        summary_field=summary_field,
        min_sentence_chars=min_sentence_chars,
        limit=limit,
    )
    write_jsonl(records, output_path)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split summary JSONL into sentence queries.")
    parser.add_argument("--input", required=True, help="Summary JSONL input path.")
    parser.add_argument("--output", required=True, help="Sentence query JSONL output path.")
    parser.add_argument(
        "--summary-field",
        default=None,
        help=(
            "Summary text field to read. Defaults to the first available field among "
            f"{', '.join(SUMMARY_TEXT_KEYS)}."
        ),
    )
    parser.add_argument(
        "--min-sentence-chars",
        type=int,
        default=1,
        help="Skip sentence fragments shorter than this many characters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of summary records to read.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = split_summary_jsonl(
        input_path=args.input,
        output_path=args.output,
        summary_field=args.summary_field,
        min_sentence_chars=args.min_sentence_chars,
        limit=args.limit,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
