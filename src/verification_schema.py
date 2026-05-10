"""Shared schema helpers for sentence-level factuality verification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import first_present, iter_jsonl, normalize_whitespace, write_jsonl


SUPPORTED = "Supported"
UNSUPPORTED = "Unsupported"
NOT_ENOUGH_INFO = "Not Enough Information"
FACTUAL_RISK_LABELS = {UNSUPPORTED, NOT_ENOUGH_INFO}

SENTENCE_TEXT_KEYS = ("text", "sentence", "query")


@dataclass
class VerificationInputStats:
    input_sentences: int = 0
    input_retrieval_results: int = 0
    written_inputs: int = 0
    sentences_without_text: int = 0
    sentences_without_retrieval: int = 0
    inputs_without_evidence: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def normalize_sentence_record(item: dict[str, Any], index: int) -> dict[str, Any]:
    doc_id = str(item.get("doc_id") or "").strip()
    query_id = str(item.get("query_id") or "").strip()
    if not query_id and doc_id:
        query_id = f"{doc_id}_s{index:03d}"

    sentence = normalize_whitespace(first_present(item, SENTENCE_TEXT_KEYS))
    summary_id = str(item.get("summary_id") or "").strip()
    sentence_id = item.get("sentence_id")

    return {
        "query_id": query_id,
        "doc_id": doc_id,
        "summary_id": summary_id,
        "sentence_id": sentence_id,
        "sentence": sentence,
        "sentence_word_count": item.get("sentence_word_count"),
        "summary_word_count": item.get("summary_word_count"),
        "source_summary_field": item.get("source_summary_field"),
    }


def index_retrieval_by_query_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = record
    return indexed


def evidence_from_retrieval(
    retrieval_record: dict[str, Any] | None,
    top_k: int = 3,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    if not retrieval_record:
        return [], [], []

    evidence_texts: list[str] = []
    evidence_ids: list[str] = []
    evidence_items: list[dict[str, Any]] = []

    for item in retrieval_record.get("top_k", [])[:top_k]:
        text = normalize_whitespace(item.get("text"))
        if not text:
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        evidence_texts.append(text)
        evidence_ids.append(chunk_id)
        evidence_items.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": item.get("chunk_index"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "text": text,
            }
        )

    return evidence_texts, evidence_ids, evidence_items


def build_premise(evidence: list[str], separator: str = "\n\n") -> str:
    return separator.join(text for text in evidence if text)


def build_verification_input(
    sentence_record: dict[str, Any],
    retrieval_record: dict[str, Any] | None,
    method: str,
    verifier: str,
    retrieval_method: str | None,
    top_k: int = 3,
) -> dict[str, Any]:
    evidence, evidence_ids, evidence_items = evidence_from_retrieval(
        retrieval_record,
        top_k=top_k,
    )
    resolved_retrieval_method = (
        retrieval_method
        or (retrieval_record or {}).get("retrieval_method")
        or ""
    )
    return {
        "query_id": sentence_record["query_id"],
        "doc_id": sentence_record["doc_id"],
        "summary_id": sentence_record.get("summary_id", ""),
        "sentence_id": sentence_record.get("sentence_id"),
        "sentence": sentence_record["sentence"],
        "premise": build_premise(evidence),
        "evidence": evidence,
        "evidence_ids": evidence_ids,
        "evidence_items": evidence_items,
        "retrieval_method": resolved_retrieval_method,
        "verifier": verifier,
        "method": method,
        "top_k": top_k,
        "sentence_word_count": sentence_record.get("sentence_word_count"),
        "summary_word_count": sentence_record.get("summary_word_count"),
    }


def build_verification_inputs(
    sentences: Iterable[dict[str, Any]],
    retrieval_results: Iterable[dict[str, Any]],
    method: str,
    verifier: str,
    retrieval_method: str | None = None,
    top_k: int = 3,
    allow_missing_retrieval: bool = False,
) -> tuple[list[dict[str, Any]], VerificationInputStats]:
    retrieval_records = list(retrieval_results)
    retrieval_by_query_id = index_retrieval_by_query_id(retrieval_records)
    stats = VerificationInputStats(input_retrieval_results=len(retrieval_records))
    output: list[dict[str, Any]] = []

    for index, item in enumerate(sentences, start=1):
        stats.input_sentences += 1
        sentence_record = normalize_sentence_record(item, index=index)
        if not sentence_record["sentence"]:
            stats.sentences_without_text += 1
            continue

        retrieval_record = retrieval_by_query_id.get(sentence_record["query_id"])
        if retrieval_record is None:
            stats.sentences_without_retrieval += 1
            if not allow_missing_retrieval:
                continue

        verification_input = build_verification_input(
            sentence_record=sentence_record,
            retrieval_record=retrieval_record,
            method=method,
            verifier=verifier,
            retrieval_method=retrieval_method,
            top_k=top_k,
        )
        if not verification_input["evidence"]:
            stats.inputs_without_evidence += 1

        output.append(verification_input)
        stats.written_inputs += 1

    return output, stats


def build_verification_result(
    verification_input: dict[str, Any],
    label: str,
    reason: str,
    model: str,
    scores: dict[str, float] | None = None,
    raw_label: str = "",
) -> dict[str, Any]:
    return {
        "query_id": verification_input["query_id"],
        "doc_id": verification_input["doc_id"],
        "summary_id": verification_input.get("summary_id", ""),
        "sentence_id": verification_input.get("sentence_id"),
        "sentence": verification_input["sentence"],
        "retrieval_method": verification_input.get("retrieval_method", ""),
        "verifier": verification_input.get("verifier", "nli"),
        "method": verification_input.get("method", ""),
        "evidence": verification_input.get("evidence", []),
        "evidence_ids": verification_input.get("evidence_ids", []),
        "label": label,
        "reason": reason,
        "model": model,
        "scores": scores or {},
        "raw_label": raw_label,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build unified sentence-level verification inputs."
    )
    parser.add_argument("--sentences", required=True, help="Summary sentence JSONL path.")
    parser.add_argument("--retrieval", required=True, help="Retrieval JSONL path.")
    parser.add_argument("--output", required=True, help="Verification input JSONL path.")
    parser.add_argument(
        "--method",
        default="Hybrid + NLI",
        help="Experiment method name saved in each record.",
    )
    parser.add_argument("--verifier", default="nli", help="Verifier name.")
    parser.add_argument(
        "--retrieval-method",
        default=None,
        help="Optional retrieval method override.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Evidence chunks to keep.")
    parser.add_argument(
        "--allow-missing-retrieval",
        action="store_true",
        help="Keep sentence records even when retrieval is missing.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records, stats = build_verification_inputs(
        sentences=iter_jsonl(args.sentences),
        retrieval_results=iter_jsonl(args.retrieval),
        method=args.method,
        verifier=args.verifier,
        retrieval_method=args.retrieval_method,
        top_k=args.top_k,
        allow_missing_retrieval=args.allow_missing_retrieval,
    )
    write_jsonl(records, args.output)
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
