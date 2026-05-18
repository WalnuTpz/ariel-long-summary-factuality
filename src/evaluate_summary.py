"""Evaluate generated and corrected summaries.

The script computes dependency-free ROUGE-1/2/L against GovReport reference
summaries and aggregates sentence-level factuality labels from NLI verification
outputs. It supports before-correction records alone, or before/after records
when corrected summaries and after-correction verification are available.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import (
    ensure_parent_dir,
    first_present,
    iter_jsonl,
    normalize_whitespace,
    simple_tokenize,
    write_jsonl,
)
from src.verification_schema import NOT_ENOUGH_INFO, SUPPORTED, UNSUPPORTED


SUMMARY_TEXT_KEYS = (
    "generated_summary",
    "summary",
    "clean_summary",
    "corrected_summary",
    "reference_summary",
)
REFERENCE_TEXT_KEYS = ("reference_summary", "abstract", "target", "summary")
BEFORE_VERSION = "before_correction"
AFTER_VERSION = "after_correction"
SHORT_MAX_WORDS = 150
MEDIUM_MAX_WORDS = 225
LENGTH_BUCKETS = ("short", "medium", "long")
DEFAULT_LENGTH_BUCKET_STRATEGY = "fixed"


@dataclass
class EvaluateStats:
    input_summaries: int = 0
    input_corrections: int = 0
    input_documents: int = 0
    verification_before_records: int = 0
    verification_after_records: int = 0
    written_records: int = 0
    skipped_missing_doc_id: int = 0
    skipped_missing_summary: int = 0
    records_without_reference: int = 0
    records_without_verification: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def summary_key(doc_id: str, summary_id: str) -> tuple[str, str]:
    return doc_id, summary_id


def normalize_summary_id(item: dict[str, Any], doc_id: str) -> str:
    return str(item.get("summary_id") or f"{doc_id}_summary").strip()


def resolve_text(item: dict[str, Any], keys: Iterable[str], forced_key: str | None = None) -> tuple[str, str]:
    if forced_key:
        return normalize_whitespace(item.get(forced_key)), forced_key
    used_key = next((key for key in keys if item.get(key) not in (None, "")), "")
    if not used_key:
        return "", ""
    return normalize_whitespace(item.get(used_key)), used_key


def index_documents(path: str | Path | None) -> tuple[dict[str, dict[str, Any]], int]:
    if path is None:
        return {}, 0
    indexed: dict[str, dict[str, Any]] = {}
    count = 0
    for item in iter_jsonl(path):
        count += 1
        doc_id = str(first_present(item, ("doc_id", "document_id", "id")) or "").strip()
        if doc_id:
            indexed[doc_id] = item
    return indexed, count


def load_summaries(
    summaries_path: str | Path,
    documents_by_doc_id: dict[str, dict[str, Any]],
    summary_field: str | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], EvaluateStats]:
    stats = EvaluateStats()
    summaries: dict[tuple[str, str], dict[str, Any]] = {}

    for item in iter_jsonl(summaries_path):
        stats.input_summaries += 1
        doc_id = str(first_present(item, ("doc_id", "document_id", "id")) or "").strip()
        if not doc_id:
            stats.skipped_missing_doc_id += 1
            continue
        summary_text, summary_source = resolve_text(
            item,
            keys=SUMMARY_TEXT_KEYS,
            forced_key=summary_field,
        )
        if not summary_text:
            stats.skipped_missing_summary += 1
            continue

        reference_text, reference_source = resolve_text(item, keys=REFERENCE_TEXT_KEYS)
        if not reference_text:
            reference_text, reference_source = resolve_text(
                documents_by_doc_id.get(doc_id, {}),
                keys=REFERENCE_TEXT_KEYS,
            )

        summary_id = normalize_summary_id(item, doc_id=doc_id)
        summaries[summary_key(doc_id, summary_id)] = {
            "doc_id": doc_id,
            "summary_id": summary_id,
            "summary": summary_text,
            "summary_source": summary_source,
            "reference_summary": reference_text,
            "reference_source": reference_source,
            "model": item.get("model", ""),
            "generation_method": item.get("generation_method", ""),
        }

    return summaries, stats


def load_corrections(path: str | Path | None) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    if path is None:
        return {}, 0
    corrections: dict[tuple[str, str], dict[str, Any]] = {}
    count = 0
    for item in iter_jsonl(path):
        count += 1
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        summary_id = normalize_summary_id(item, doc_id=doc_id)
        corrected_summary = normalize_whitespace(item.get("corrected_summary"))
        if not corrected_summary:
            continue
        corrections[summary_key(doc_id, summary_id)] = dict(item)
    return corrections, count


def f1_score(overlap: int, candidate_total: int, reference_total: int) -> float:
    if overlap <= 0 or candidate_total <= 0 or reference_total <= 0:
        return 0.0
    precision = overlap / candidate_total
    recall = overlap / reference_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def rouge_n(candidate_tokens: list[str], reference_tokens: list[str], n: int) -> float:
    candidate_ngrams = Counter(ngrams(candidate_tokens, n))
    reference_ngrams = Counter(ngrams(reference_tokens, n))
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    overlap = sum((candidate_ngrams & reference_ngrams).values())
    return f1_score(overlap, sum(candidate_ngrams.values()), sum(reference_ngrams.values()))


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate_tokens: list[str], reference_tokens: list[str]) -> float:
    overlap = lcs_length(candidate_tokens, reference_tokens)
    return f1_score(overlap, len(candidate_tokens), len(reference_tokens))


def compute_rouge(candidate: str, reference: str) -> dict[str, float | None]:
    if not reference:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    candidate_tokens = simple_tokenize(candidate)
    reference_tokens = simple_tokenize(reference)
    return {
        "rouge1": round(rouge_n(candidate_tokens, reference_tokens, 1), 6),
        "rouge2": round(rouge_n(candidate_tokens, reference_tokens, 2), 6),
        "rougeL": round(rouge_l(candidate_tokens, reference_tokens), 6),
    }


def fixed_length_bucket(word_count: int, short_max_words: int, medium_max_words: int) -> str:
    if word_count <= short_max_words:
        return "short"
    if word_count <= medium_max_words:
        return "medium"
    return "long"


def tertile_length_buckets(
    summaries: dict[tuple[str, str], dict[str, Any]]
) -> dict[tuple[str, str], str]:
    keyed_lengths = [
        (key, len(simple_tokenize(summary.get("summary", ""))))
        for key, summary in summaries.items()
    ]
    keyed_lengths.sort(key=lambda item: (item[1], item[0][0], item[0][1]))
    total = len(keyed_lengths)
    if total == 0:
        return {}

    bucket_by_key: dict[tuple[str, str], str] = {}
    for rank, (key, _) in enumerate(keyed_lengths):
        bucket_index = min(len(LENGTH_BUCKETS) - 1, rank * len(LENGTH_BUCKETS) // total)
        bucket_by_key[key] = LENGTH_BUCKETS[bucket_index]
    return bucket_by_key


def length_buckets_by_key(
    summaries: dict[tuple[str, str], dict[str, Any]],
    strategy: str,
    short_max_words: int,
    medium_max_words: int,
) -> dict[tuple[str, str], str]:
    if strategy == "tertile":
        return tertile_length_buckets(summaries)
    return {
        key: fixed_length_bucket(
            len(simple_tokenize(summary.get("summary", ""))),
            short_max_words=short_max_words,
            medium_max_words=medium_max_words,
        )
        for key, summary in summaries.items()
    }


def load_verification_metrics(
    path: str | Path | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], int]:
    if path is None:
        return {}, {}, 0

    counts_by_key: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    counts_by_doc_id: dict[str, Counter[str]] = defaultdict(Counter)
    count = 0
    for item in iter_jsonl(path):
        count += 1
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        summary_id = str(item.get("summary_id") or "").strip()
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        if summary_id:
            counts_by_key[summary_key(doc_id, summary_id)][label] += 1
        counts_by_doc_id[doc_id][label] += 1

    return (
        {key: factuality_metrics(counter) for key, counter in counts_by_key.items()},
        {doc_id: factuality_metrics(counter) for doc_id, counter in counts_by_doc_id.items()},
        count,
    )


def factuality_metrics(counter: Counter[str]) -> dict[str, Any]:
    supported = counter.get(SUPPORTED, 0)
    unsupported = counter.get(UNSUPPORTED, 0)
    nei = counter.get(NOT_ENOUGH_INFO, 0)
    total = sum(counter.values())
    risk = unsupported + nei
    supported_ratio = supported / total if total else None
    risk_ratio = risk / total if total else None
    return {
        "total_sentences": total,
        "supported_count": supported,
        "unsupported_count": unsupported,
        "not_enough_info_count": nei,
        "risk_count": risk,
        "supported_ratio": round(supported_ratio, 6) if supported_ratio is not None else None,
        "unsupported_or_nei_ratio": round(risk_ratio, 6) if risk_ratio is not None else None,
        "label_counts": dict(counter),
    }


def lookup_factuality(
    key: tuple[str, str],
    by_key: dict[tuple[str, str], dict[str, Any]],
    by_doc_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    doc_id, _ = key
    return by_key.get(key) or by_doc_id.get(doc_id)


def make_eval_record(
    base_summary: dict[str, Any],
    summary_text: str,
    summary_version: str,
    factuality: dict[str, Any] | None,
    length_bucket_name: str,
    length_bucket_strategy: str,
    correction_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = base_summary.get("reference_summary", "")
    word_count = len(simple_tokenize(summary_text))
    record: dict[str, Any] = {
        "doc_id": base_summary["doc_id"],
        "summary_id": base_summary["summary_id"],
        "summary_version": summary_version,
        **compute_rouge(summary_text, reference),
        "supported_ratio": None,
        "unsupported_or_nei_ratio": None,
        "summary_word_count": word_count,
        "reference_word_count": len(simple_tokenize(reference)) if reference else None,
        "length_bucket": length_bucket_name,
        "length_bucket_strategy": length_bucket_strategy,
        "model": base_summary.get("model", ""),
        "generation_method": base_summary.get("generation_method", ""),
        "reference_available": bool(reference),
    }
    if factuality:
        record.update(factuality)
    if correction_record is not None:
        original_word_count = int(correction_record.get("original_summary_word_count") or 0)
        corrected_word_count = int(correction_record.get("corrected_summary_word_count") or word_count)
        if original_word_count <= 0:
            original_word_count = len(simple_tokenize(correction_record.get("original_summary")))
        record["length_ratio"] = (
            round(corrected_word_count / original_word_count, 6)
            if original_word_count
            else None
        )
        record["num_sentence_edits"] = correction_record.get("num_sentence_edits")
    return record


def evaluate_records(
    summaries: dict[tuple[str, str], dict[str, Any]],
    corrections: dict[tuple[str, str], dict[str, Any]],
    verification_before: tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]],
    verification_after: tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]],
    short_max_words: int,
    medium_max_words: int,
    length_bucket_strategy: str,
) -> tuple[list[dict[str, Any]], int, int]:
    before_by_key, before_by_doc = verification_before
    after_by_key, after_by_doc = verification_after
    records: list[dict[str, Any]] = []
    without_reference = 0
    without_verification = 0
    bucket_by_key = length_buckets_by_key(
        summaries=summaries,
        strategy=length_bucket_strategy,
        short_max_words=short_max_words,
        medium_max_words=medium_max_words,
    )

    for key, base_summary in summaries.items():
        length_bucket_name = bucket_by_key.get(key, "medium")
        before_factuality = lookup_factuality(key, before_by_key, before_by_doc)
        if before_factuality is None:
            without_verification += 1
        if not base_summary.get("reference_summary"):
            without_reference += 1
        records.append(
            make_eval_record(
                base_summary=base_summary,
                summary_text=base_summary["summary"],
                summary_version=BEFORE_VERSION,
                factuality=before_factuality,
                length_bucket_name=length_bucket_name,
                length_bucket_strategy=length_bucket_strategy,
            )
        )

        correction_record = corrections.get(key)
        if correction_record is None:
            continue
        after_factuality = lookup_factuality(key, after_by_key, after_by_doc)
        if after_factuality is None:
            without_verification += 1
        corrected_summary = normalize_whitespace(correction_record.get("corrected_summary"))
        records.append(
            make_eval_record(
                base_summary=base_summary,
                summary_text=corrected_summary,
                summary_version=AFTER_VERSION,
                factuality=after_factuality,
                length_bucket_name=length_bucket_name,
                length_bucket_strategy=length_bucket_strategy,
                correction_record=correction_record,
            )
        )

    return records, without_reference, without_verification


def mean(values: Iterable[float | int | None]) -> float | None:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    return round(sum(clean_values) / len(clean_values), 6)


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "rouge1": mean(record.get("rouge1") for record in records),
        "rouge2": mean(record.get("rouge2") for record in records),
        "rougeL": mean(record.get("rougeL") for record in records),
        "supported_ratio": mean(record.get("supported_ratio") for record in records),
        "unsupported_or_nei_ratio": mean(
            record.get("unsupported_or_nei_ratio") for record in records
        ),
        "summary_word_count": mean(record.get("summary_word_count") for record in records),
        "length_ratio": mean(record.get("length_ratio") for record in records),
        "risk_count": mean(record.get("risk_count") for record in records),
        "total_sentences": mean(record.get("total_sentences") for record in records),
    }


def paired_error_reduction(records: list[dict[str, Any]]) -> float | None:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = summary_key(record["doc_id"], record["summary_id"])
        by_key[key][record["summary_version"]] = record

    reductions: list[float] = []
    for versions in by_key.values():
        before = versions.get(BEFORE_VERSION)
        after = versions.get(AFTER_VERSION)
        if not before or not after:
            continue
        before_risk = before.get("risk_count")
        after_risk = after.get("risk_count")
        if before_risk is None or after_risk is None or before_risk <= 0:
            continue
        reductions.append((before_risk - after_risk) / before_risk)
    return mean(reductions)


def aggregate_records(records: list[dict[str, Any]], stats: EvaluateStats) -> dict[str, Any]:
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for record in records:
        version = str(record.get("summary_version") or "")
        bucket = str(record.get("length_bucket") or "")
        by_version[version].append(record)
        by_bucket[bucket][version].append(record)

    return {
        "stats": stats.to_dict(),
        "by_version": {
            version: summarize_group(group)
            for version, group in sorted(by_version.items())
        },
        "by_length_bucket": {
            bucket: {
                version: summarize_group(group)
                for version, group in sorted(version_groups.items())
            }
            for bucket, version_groups in sorted(by_bucket.items())
        },
        "paired_error_reduction_rate": paired_error_reduction(records),
    }


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = ensure_parent_dir(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_summary_jsonl(
    summaries_path: str | Path,
    output_records_path: str | Path,
    documents_path: str | Path | None = None,
    corrections_path: str | Path | None = None,
    verification_before_path: str | Path | None = None,
    verification_after_path: str | Path | None = None,
    output_summary_path: str | Path | None = None,
    summary_field: str | None = None,
    short_max_words: int = SHORT_MAX_WORDS,
    medium_max_words: int = MEDIUM_MAX_WORDS,
    length_bucket_strategy: str = DEFAULT_LENGTH_BUCKET_STRATEGY,
) -> EvaluateStats:
    documents_by_doc_id, document_count = index_documents(documents_path)
    summaries, stats = load_summaries(
        summaries_path=summaries_path,
        documents_by_doc_id=documents_by_doc_id,
        summary_field=summary_field,
    )
    stats.input_documents = document_count

    corrections, correction_count = load_corrections(corrections_path)
    stats.input_corrections = correction_count

    before_by_key, before_by_doc, before_count = load_verification_metrics(
        verification_before_path
    )
    after_by_key, after_by_doc, after_count = load_verification_metrics(
        verification_after_path
    )
    stats.verification_before_records = before_count
    stats.verification_after_records = after_count

    records, missing_reference, missing_verification = evaluate_records(
        summaries=summaries,
        corrections=corrections,
        verification_before=(before_by_key, before_by_doc),
        verification_after=(after_by_key, after_by_doc),
        short_max_words=short_max_words,
        medium_max_words=medium_max_words,
        length_bucket_strategy=length_bucket_strategy,
    )
    stats.records_without_reference = missing_reference
    stats.records_without_verification = missing_verification
    stats.written_records = write_jsonl(records, output_records_path)

    if output_summary_path is not None:
        write_json(aggregate_records(records, stats), output_summary_path)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated and corrected summaries with ROUGE and NLI metrics."
    )
    parser.add_argument("--summaries", required=True, help="Generated summary JSONL path.")
    parser.add_argument(
        "--documents",
        default=None,
        help="Optional normalized document JSONL path for reference summaries.",
    )
    parser.add_argument(
        "--corrections",
        default=None,
        help="Optional correction JSONL path for after-correction evaluation.",
    )
    parser.add_argument(
        "--verification-before",
        default=None,
        help="Optional before-correction NLI verification JSONL path.",
    )
    parser.add_argument(
        "--verification-after",
        default=None,
        help="Optional after-correction NLI verification JSONL path.",
    )
    parser.add_argument("--output-records", required=True, help="Evaluation JSONL output path.")
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Optional aggregate JSON output path.",
    )
    parser.add_argument(
        "--summary-field",
        default=None,
        help="Summary field to evaluate. Defaults to generated_summary first.",
    )
    parser.add_argument(
        "--length-bucket-strategy",
        choices=["tertile", "fixed"],
        default=DEFAULT_LENGTH_BUCKET_STRATEGY,
        help="How to assign short/medium/long buckets. Defaults to fixed thresholds.",
    )
    parser.add_argument(
        "--short-max-words",
        type=int,
        default=SHORT_MAX_WORDS,
        help="Maximum word count for the short summary bucket when using fixed buckets.",
    )
    parser.add_argument(
        "--medium-max-words",
        type=int,
        default=MEDIUM_MAX_WORDS,
        help="Maximum word count for the medium summary bucket when using fixed buckets.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = evaluate_summary_jsonl(
        summaries_path=args.summaries,
        documents_path=args.documents,
        corrections_path=args.corrections,
        verification_before_path=args.verification_before,
        verification_after_path=args.verification_after,
        output_records_path=args.output_records,
        output_summary_path=args.output_summary,
        summary_field=args.summary_field,
        short_max_words=args.short_max_words,
        medium_max_words=args.medium_max_words,
        length_bucket_strategy=args.length_bucket_strategy,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
