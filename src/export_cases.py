"""Export representative factuality and correction cases for reports.

The exporter joins sentence-level verification results with optional local
correction outputs, optional after-correction verification, and optional
evaluation records. It writes compact JSONL cases plus a Markdown file that can
be used directly when drafting the report or presentation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils import ensure_parent_dir, iter_jsonl, normalize_whitespace, write_jsonl
from src.verification_schema import FACTUAL_RISK_LABELS, NOT_ENOUGH_INFO, SUPPORTED


CASE_SUCCESSFUL_CORRECTION = "successful_correction"
CASE_CORRECTION_FAILED = "correction_failed"
CASE_CORRECTED_UNVERIFIED = "detected_and_corrected_unverified"
CASE_EVIDENCE_INSUFFICIENT = "retrieval_evidence_insufficient"
CASE_RISK_UNCORRECTED = "detected_risk_uncorrected"
CASE_SUPPORTED = "supported_sentence"
BEFORE_VERSION = "before_correction"
AFTER_VERSION = "after_correction"


@dataclass
class ExportStats:
    input_verification_before: int = 0
    input_verification_after: int = 0
    input_corrections: int = 0
    input_evaluation_records: int = 0
    candidate_cases: int = 0
    written_cases: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def read_records_by_query_id(path: str | Path | None) -> tuple[dict[str, dict[str, Any]], int]:
    if path is None:
        return {}, 0
    records: dict[str, dict[str, Any]] = {}
    count = 0
    for item in iter_jsonl(path):
        count += 1
        query_id = str(item.get("query_id") or "").strip()
        if query_id:
            records[query_id] = item
    return records, count


def read_correction_edits(path: str | Path | None) -> tuple[dict[str, dict[str, Any]], int]:
    if path is None:
        return {}, 0
    edits: dict[str, dict[str, Any]] = {}
    count = 0
    for correction in iter_jsonl(path):
        count += 1
        for edit in correction.get("sentence_edits", []) or []:
            query_id = str(edit.get("query_id") or "").strip()
            if not query_id:
                continue
            enriched = dict(edit)
            enriched["doc_id"] = correction.get("doc_id", "")
            enriched["summary_id"] = correction.get("summary_id", "")
            enriched["correction_model"] = correction.get("model", "")
            edits[query_id] = enriched
    return edits, count


def evaluation_key(doc_id: str, summary_id: str, version: str) -> tuple[str, str, str]:
    return doc_id, summary_id, version


def read_evaluation(path: str | Path | None) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
    if path is None:
        return {}, 0
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    count = 0
    for item in iter_jsonl(path):
        count += 1
        doc_id = str(item.get("doc_id") or "").strip()
        summary_id = str(item.get("summary_id") or "").strip()
        version = str(item.get("summary_version") or "").strip()
        if doc_id and summary_id and version:
            records[evaluation_key(doc_id, summary_id, version)] = item
    return records, count


def clip_text(text: Any, max_chars: int) -> str:
    text = normalize_whitespace(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def evidence_items(record: dict[str, Any], max_evidence: int, max_evidence_chars: int) -> list[dict[str, str]]:
    evidence_texts = record.get("evidence") or []
    evidence_ids = record.get("evidence_ids") or []
    output: list[dict[str, str]] = []
    for index, text in enumerate(evidence_texts[:max_evidence]):
        chunk_id = ""
        if index < len(evidence_ids):
            chunk_id = str(evidence_ids[index] or "").strip()
        output.append(
            {
                "chunk_id": chunk_id,
                "text": clip_text(text, max_evidence_chars),
            }
        )
    return output


def top_score(record: dict[str, Any]) -> float | None:
    raw_label = str(record.get("raw_label") or "").strip()
    scores = record.get("scores") or {}
    if raw_label and raw_label in scores:
        return float(scores[raw_label])
    numeric_scores = [float(value) for value in scores.values() if isinstance(value, (int, float))]
    if numeric_scores:
        return max(numeric_scores)
    return None


def is_risk_label(label: str) -> bool:
    return label in FACTUAL_RISK_LABELS


def has_no_evidence(record: dict[str, Any]) -> bool:
    evidence = record.get("evidence") or []
    return not any(normalize_whitespace(text) for text in evidence)


def classify_case(
    before: dict[str, Any],
    edit: dict[str, Any] | None,
    after: dict[str, Any] | None,
    include_supported: bool,
) -> tuple[str | None, int]:
    before_label = str(before.get("label") or "").strip()
    after_label = str((after or {}).get("label") or "").strip()
    action = str((edit or {}).get("action") or "").strip()

    if before_label == NOT_ENOUGH_INFO and has_no_evidence(before):
        return CASE_EVIDENCE_INSUFFICIENT, 95
    if is_risk_label(before_label):
        if after_label == SUPPORTED:
            if action == "replace":
                return CASE_SUCCESSFUL_CORRECTION, 100
            return CASE_RISK_UNCORRECTED, 60
        if after_label and is_risk_label(after_label):
            return CASE_CORRECTION_FAILED, 90
        if action in {"replace", "remove"}:
            return CASE_CORRECTED_UNVERIFIED, 80
        if action == "keep":
            return CASE_CORRECTION_FAILED, 70
        return CASE_RISK_UNCORRECTED, 60
    if include_supported and before_label == SUPPORTED:
        return CASE_SUPPORTED, 10
    return None, 0


def metric_subset(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    keys = (
        "summary_version",
        "rouge1",
        "rouge2",
        "rougeL",
        "supported_ratio",
        "unsupported_or_nei_ratio",
        "risk_count",
        "total_sentences",
        "summary_word_count",
        "length_ratio",
        "length_bucket",
    )
    return {key: record.get(key) for key in keys if key in record}


def make_case(
    case_index: int,
    case_type: str,
    priority: int,
    before: dict[str, Any],
    edit: dict[str, Any] | None,
    after: dict[str, Any] | None,
    evaluation: dict[tuple[str, str, str], dict[str, Any]],
    max_evidence: int,
    max_evidence_chars: int,
) -> dict[str, Any]:
    doc_id = str(before.get("doc_id") or "").strip()
    summary_id = str(before.get("summary_id") or "").strip()
    query_id = str(before.get("query_id") or "").strip()
    original_sentence = normalize_whitespace(
        (edit or {}).get("original_sentence") or before.get("sentence")
    )
    corrected_sentence = normalize_whitespace((edit or {}).get("corrected_sentence"))
    before_eval = evaluation.get(evaluation_key(doc_id, summary_id, BEFORE_VERSION))
    after_eval = evaluation.get(evaluation_key(doc_id, summary_id, AFTER_VERSION))

    return {
        "case_id": f"case_{case_index:03d}",
        "case_type": case_type,
        "priority": priority,
        "doc_id": doc_id,
        "summary_id": summary_id,
        "query_id": query_id,
        "sentence_id": before.get("sentence_id"),
        "original_sentence": original_sentence,
        "corrected_sentence": corrected_sentence,
        "before_label": before.get("label", ""),
        "before_raw_label": before.get("raw_label", ""),
        "before_score": top_score(before),
        "after_label": (after or {}).get("label", ""),
        "after_raw_label": (after or {}).get("raw_label", ""),
        "after_score": top_score(after or {}),
        "action": (edit or {}).get("action", ""),
        "correction_reason": (edit or {}).get("reason", ""),
        "verification_reason": before.get("reason", ""),
        "after_verification_reason": (after or {}).get("reason", ""),
        "retrieval_method": before.get("retrieval_method", ""),
        "method": before.get("method", ""),
        "evidence": evidence_items(before, max_evidence, max_evidence_chars),
        "evidence_chunk_ids": before.get("evidence_ids", []),
        "selected_evidence_id": before.get("selected_evidence_id", ""),
        "evaluation_before": metric_subset(before_eval),
        "evaluation_after": metric_subset(after_eval),
    }


def build_candidates(
    verification_before: dict[str, dict[str, Any]],
    correction_edits: dict[str, dict[str, Any]],
    verification_after: dict[str, dict[str, Any]],
    evaluation: dict[tuple[str, str, str], dict[str, Any]],
    include_supported: bool,
    max_evidence: int,
    max_evidence_chars: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for before in verification_before.values():
        query_id = str(before.get("query_id") or "").strip()
        case_type, priority = classify_case(
            before=before,
            edit=correction_edits.get(query_id),
            after=verification_after.get(query_id),
            include_supported=include_supported,
        )
        if case_type is None:
            continue
        candidates.append(
            make_case(
                case_index=len(candidates) + 1,
                case_type=case_type,
                priority=priority,
                before=before,
                edit=correction_edits.get(query_id),
                after=verification_after.get(query_id),
                evaluation=evaluation,
                max_evidence=max_evidence,
                max_evidence_chars=max_evidence_chars,
            )
        )
    return candidates


def select_cases(
    candidates: list[dict[str, Any]],
    max_cases: int,
    max_per_type: int,
) -> list[dict[str, Any]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            -int(item.get("priority") or 0),
            str(item.get("case_type") or ""),
            str(item.get("query_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()

    for item in sorted_candidates:
        case_type = str(item.get("case_type") or "")
        if max_per_type > 0 and type_counts[case_type] >= max_per_type:
            continue
        selected.append(dict(item))
        type_counts[case_type] += 1
        if len(selected) >= max_cases:
            break

    if len(selected) < max_cases and max_per_type > 0:
        selected_ids = {item["query_id"] for item in selected}
        for item in sorted_candidates:
            if item["query_id"] in selected_ids:
                continue
            selected.append(dict(item))
            selected_ids.add(item["query_id"])
            if len(selected) >= max_cases:
                break

    for index, item in enumerate(selected, start=1):
        item["case_id"] = f"case_{index:03d}"
        item.pop("priority", None)
    return selected


def markdown_quote(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return "> N/A"
    return "\n".join(f"> {line}" for line in text.splitlines())


def format_metric_line(case: dict[str, Any]) -> str:
    before = case.get("evaluation_before") or {}
    after = case.get("evaluation_after") or {}
    if not before and not after:
        return ""
    parts = []
    if before:
        parts.append(
            "Before: "
            f"R1={before.get('rouge1')}, "
            f"R2={before.get('rouge2')}, "
            f"RL={before.get('rougeL')}, "
            f"Risk={before.get('unsupported_or_nei_ratio')}"
        )
    if after:
        parts.append(
            "After: "
            f"R1={after.get('rouge1')}, "
            f"R2={after.get('rouge2')}, "
            f"RL={after.get('rougeL')}, "
            f"Risk={after.get('unsupported_or_nei_ratio')}, "
            f"LenRatio={after.get('length_ratio')}"
        )
    return " | ".join(parts)


def render_markdown(cases: list[dict[str, Any]], stats: ExportStats) -> str:
    lines = [
        "# Exported Cases",
        "",
        "## Summary",
        "",
        f"- Written cases: {stats.written_cases}",
        f"- Candidate cases: {stats.candidate_cases}",
        "",
    ]
    type_counts = Counter(case["case_type"] for case in cases)
    if type_counts:
        lines.extend(["## Case Types", ""])
        for case_type, count in sorted(type_counts.items()):
            lines.append(f"- {case_type}: {count}")
        lines.append("")

    for case in cases:
        lines.extend(
            [
                f"## {case['case_id']}: {case['case_type']}",
                "",
                f"- Document: `{case['doc_id']}`",
                f"- Summary: `{case['summary_id']}`",
                f"- Query: `{case['query_id']}`",
                f"- Labels: {case.get('before_label') or 'N/A'} -> {case.get('after_label') or 'N/A'}",
                f"- Action: {case.get('action') or 'N/A'}",
                f"- Retrieval: {case.get('retrieval_method') or 'N/A'}",
            ]
        )
        metric_line = format_metric_line(case)
        if metric_line:
            lines.append(f"- Metrics: {metric_line}")

        lines.extend(["", "Original sentence:", "", markdown_quote(case["original_sentence"]), ""])
        if case.get("corrected_sentence"):
            lines.extend(
                [
                    "Corrected sentence:",
                    "",
                    markdown_quote(case["corrected_sentence"]),
                    "",
                ]
            )
        if case.get("correction_reason"):
            lines.extend(["Correction reason:", "", markdown_quote(case["correction_reason"]), ""])
        if case.get("verification_reason"):
            lines.extend(["Verification reason:", "", markdown_quote(case["verification_reason"]), ""])

        evidence = case.get("evidence") or []
        if evidence:
            lines.extend(["Evidence:", ""])
            for item in evidence:
                chunk_id = item.get("chunk_id") or "unknown"
                lines.append(f"- `{chunk_id}`")
                lines.append("")
                lines.append(markdown_quote(item.get("text", "")))
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_text(text: str, path: str | Path) -> None:
    path = ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def export_cases(
    verification_before_path: str | Path,
    output_jsonl_path: str | Path,
    corrections_path: str | Path | None = None,
    verification_after_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    output_markdown_path: str | Path | None = None,
    max_cases: int = 10,
    max_per_type: int = 3,
    max_evidence: int = 2,
    max_evidence_chars: int = 800,
    include_supported: bool = False,
) -> ExportStats:
    before, before_count = read_records_by_query_id(verification_before_path)
    after, after_count = read_records_by_query_id(verification_after_path)
    edits, correction_count = read_correction_edits(corrections_path)
    evaluation, evaluation_count = read_evaluation(evaluation_path)

    stats = ExportStats(
        input_verification_before=before_count,
        input_verification_after=after_count,
        input_corrections=correction_count,
        input_evaluation_records=evaluation_count,
    )

    candidates = build_candidates(
        verification_before=before,
        correction_edits=edits,
        verification_after=after,
        evaluation=evaluation,
        include_supported=include_supported,
        max_evidence=max_evidence,
        max_evidence_chars=max_evidence_chars,
    )
    stats.candidate_cases = len(candidates)
    cases = select_cases(candidates, max_cases=max_cases, max_per_type=max_per_type)
    stats.written_cases = write_jsonl(cases, output_jsonl_path)

    if output_markdown_path is not None:
        write_text(render_markdown(cases, stats), output_markdown_path)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export representative factuality and correction cases."
    )
    parser.add_argument(
        "--verification-before",
        required=True,
        help="Before-correction NLI verification JSONL path.",
    )
    parser.add_argument(
        "--corrections",
        default=None,
        help="Optional correction JSONL path.",
    )
    parser.add_argument(
        "--verification-after",
        default=None,
        help="Optional after-correction NLI verification JSONL path.",
    )
    parser.add_argument(
        "--evaluation",
        default=None,
        help="Optional evaluation JSONL path.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Case JSONL output path.")
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional Markdown output path.",
    )
    parser.add_argument("--max-cases", type=int, default=10, help="Maximum cases to export.")
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=3,
        help="Maximum cases per type before filling remaining slots. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=2,
        help="Maximum evidence chunks per case.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=800,
        help="Maximum characters per evidence chunk.",
    )
    parser.add_argument(
        "--include-supported",
        action="store_true",
        help="Also export supported sentence examples when there are open slots.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = export_cases(
        verification_before_path=args.verification_before,
        corrections_path=args.corrections,
        verification_after_path=args.verification_after,
        evaluation_path=args.evaluation,
        output_jsonl_path=args.output_jsonl,
        output_markdown_path=args.output_markdown,
        max_cases=args.max_cases,
        max_per_type=args.max_per_type,
        max_evidence=args.max_evidence,
        max_evidence_chars=args.max_evidence_chars,
        include_supported=args.include_supported,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
