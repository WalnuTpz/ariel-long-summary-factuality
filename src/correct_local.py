"""Evidence-conditioned local rewrite for generated summaries.

This module rewrites only summary sentences flagged as factual risks by the
sentence-level verifier. Supported sentences are copied unchanged, and risky
sentences are conservatively rewritten with retrieved evidence instead of freely
rewriting the whole summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.summarize import DEFAULT_MODEL_NAME, HuggingFaceSummarizer
from src.utils import (
    ensure_parent_dir,
    first_present,
    iter_jsonl,
    normalize_whitespace,
    simple_tokenize,
)
from src.verification_schema import (
    FACTUAL_RISK_LABELS,
    NOT_ENOUGH_INFO,
    SUPPORTED,
    UNSUPPORTED,
)


SUMMARY_TEXT_KEYS = (
    "generated_summary",
    "corrected_summary",
    "summary",
    "clean_summary",
    "reference_summary",
)
ACTION_REPLACE = "replace"
ACTION_REMOVE = "remove"
ACTION_KEEP = "keep"
ACTION_ERROR = "error"
RAW_CONTRADICTION = "contradiction"
RAW_NEUTRAL = "neutral"
DEFAULT_NEUTRAL_REWRITE_THRESHOLD = 0.7
FORMAT_ARTIFACT_PATTERN = re.compile(
    r"\b(?:action|corrected\s+sentence|reason|original\s+sentence|evidence)\s*:",
    flags=re.I,
)
RELATIVE_LENGTH_BUFFER_WORDS = 8
RELATIVE_LENGTH_MULTIPLIER = 1.8


@dataclass
class CorrectionStats:
    input_summaries: int = 0
    input_sentences: int = 0
    input_verification_results: int = 0
    written_records: int = 0
    skipped_missing_summary_text: int = 0
    skipped_summaries_without_sentences: int = 0
    supported_sentences: int = 0
    risk_sentences: int = 0
    sentences_without_verification: int = 0
    edits_replaced: int = 0
    edits_removed: int = 0
    edits_kept: int = 0
    failed_corrections: int = 0
    parse_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def resolve_summary_text(item: dict[str, Any], summary_field: str | None = None) -> tuple[str, str]:
    if summary_field:
        return normalize_whitespace(item.get(summary_field)), summary_field
    used_field = next(
        (key for key in SUMMARY_TEXT_KEYS if item.get(key) not in (None, "")),
        "",
    )
    if not used_field:
        return "", ""
    return normalize_whitespace(item.get(used_field)), used_field


def normalize_summary_id(item: dict[str, Any], doc_id: str) -> str:
    return str(item.get("summary_id") or f"{doc_id}_summary").strip()


def summary_key(doc_id: str, summary_id: str) -> tuple[str, str]:
    return doc_id, summary_id


def index_summaries(
    summaries_path: str | Path,
    summary_field: str | None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], CorrectionStats]:
    stats = CorrectionStats()
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_doc_id: dict[str, dict[str, Any]] = {}

    for item in iter_jsonl(summaries_path):
        stats.input_summaries += 1
        doc_id = str(first_present(item, ("doc_id", "document_id", "id")) or "").strip()
        if not doc_id:
            continue
        summary_text, used_field = resolve_summary_text(item, summary_field=summary_field)
        if not summary_text:
            stats.skipped_missing_summary_text += 1
            continue
        summary_id = normalize_summary_id(item, doc_id=doc_id)
        record = dict(item)
        record["doc_id"] = doc_id
        record["summary_id"] = summary_id
        record["original_summary"] = summary_text
        record["source_summary_field"] = used_field
        by_key[summary_key(doc_id, summary_id)] = record
        by_doc_id.setdefault(doc_id, record)

    return by_key, by_doc_id, stats


def sentence_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    sentence_id = item.get("sentence_id")
    try:
        return int(sentence_id), str(item.get("query_id") or "")
    except (TypeError, ValueError):
        return 10**9, str(item.get("query_id") or "")


def group_sentences(sentences_path: str | Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    count = 0
    for item in iter_jsonl(sentences_path):
        count += 1
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        summary_id = str(item.get("summary_id") or f"{doc_id}_summary").strip()
        sentence = normalize_whitespace(first_present(item, ("text", "sentence", "query")))
        if not sentence:
            continue
        record = dict(item)
        record["text"] = sentence
        grouped.setdefault(summary_key(doc_id, summary_id), []).append(record)

    for records in grouped.values():
        records.sort(key=sentence_sort_key)
    return grouped, count


def index_verification(verification_path: str | Path) -> tuple[dict[str, dict[str, Any]], int]:
    indexed: dict[str, dict[str, Any]] = {}
    count = 0
    for item in iter_jsonl(verification_path):
        count += 1
        query_id = str(item.get("query_id") or "").strip()
        if query_id:
            indexed[query_id] = item
    return indexed, count


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_whitespace(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def selected_first_evidence(
    verification: dict[str, Any],
    max_evidence_chunks: int,
    max_evidence_chars: int,
) -> list[dict[str, str]]:
    evidence_texts = verification.get("evidence") or []
    evidence_ids = verification.get("evidence_ids") or []
    evidence: list[dict[str, str]] = []

    for index, text in enumerate(evidence_texts):
        text = normalize_whitespace(text)
        if not text:
            continue
        chunk_id = ""
        if index < len(evidence_ids):
            chunk_id = str(evidence_ids[index] or "").strip()
        evidence.append(
            {
                "chunk_id": chunk_id,
                "text": clip_text(text, max_evidence_chars),
            }
        )

    selected_id = str(verification.get("selected_evidence_id") or "").strip()
    if selected_id:
        evidence.sort(key=lambda item: 0 if item["chunk_id"] == selected_id else 1)
    return evidence[:max_evidence_chunks]


def format_evidence(evidence: list[dict[str, str]]) -> str:
    if not evidence:
        return "No retrieved evidence is available."
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        chunk_id = item["chunk_id"] or f"evidence_{index}"
        blocks.append(f"Evidence {index} [{chunk_id}]:\n{item['text']}")
    return "\n\n".join(blocks)


def correction_prompt(
    sentence: str,
    label: str,
    evidence: list[dict[str, str]],
    max_sentence_words: int,
) -> str:
    return (
        "You are a factual consistency correction model. Given several evidence "
        "passages from the source document and one summary sentence, rewrite "
        "the summary sentence to make it factually consistent with the evidence.\n\n"
        f"Verification label: {label}\n"
        f"Evidence passages:\n{format_evidence(evidence)}\n\n"
        f"Original summary sentence:\n{sentence}\n\n"
        "Instructions:\n"
        "- Only correct factual errors in the original sentence.\n"
        "- Keep the sentence as close as possible to the original wording and structure.\n"
        "- Do not add information that is not explicitly supported by the evidence.\n"
        "- If a number, date, entity, or relation conflicts with the evidence, "
        "replace it with the supported one.\n"
        "- If some content is not supported by the evidence, remove only that "
        "unsupported part, not the whole sentence.\n"
        f"- Keep the corrected sentence under {max_sentence_words} words.\n\n"
        "Return JSON only with this schema:\n"
        "{"
        '"corrected_sentence": "..."'
        "}\n\n"
        "Output exactly one JSON object. Do not include markdown, labels, "
        "explanations, or any text outside the JSON object."
    )


def strip_code_fence(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_correction_response(text: str) -> tuple[str, str, str, bool]:
    cleaned = strip_code_fence(text)
    json_text = cleaned
    if not json_text.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        json_text = match.group(0) if match else ""

    if json_text:
        try:
            item = json.loads(json_text)
            if isinstance(item, dict):
                corrected_sentence = normalize_whitespace(
                    item.get("corrected_sentence") or item.get("sentence") or ""
                )
                reason = normalize_whitespace(item.get("reason"))
                if corrected_sentence:
                    return ACTION_REPLACE, corrected_sentence, reason, True
        except json.JSONDecodeError:
            pass

    return ACTION_ERROR, "", "Could not parse correction model output.", False


def validate_corrected_sentence(
    sentence: str,
    original_sentence: str,
    max_sentence_words: int,
) -> tuple[str, str | None]:
    sentence = normalize_whitespace(sentence)
    if not sentence:
        return "", "Corrected sentence is empty."
    if FORMAT_ARTIFACT_PATTERN.search(sentence):
        return "", "Corrected sentence contains response-format artifacts."
    if "```" in sentence or "{" in sentence or "}" in sentence:
        return "", "Corrected sentence contains markup or JSON artifacts."
    word_count = len(simple_tokenize(sentence))
    if max_sentence_words > 0 and word_count > max_sentence_words:
        return "", (
            f"Corrected sentence has {word_count} words, "
            f"exceeding the {max_sentence_words}-word limit."
        )
    original_word_count = len(simple_tokenize(original_sentence))
    if original_word_count > 0:
        max_relative_words = max(
            original_word_count + RELATIVE_LENGTH_BUFFER_WORDS,
            int(original_word_count * RELATIVE_LENGTH_MULTIPLIER),
        )
        if word_count > max_relative_words:
            return "", (
                f"Corrected sentence has {word_count} words, exceeding the "
                f"relative length limit of {max_relative_words} words."
            )
    return sentence, None


def apply_fallback_action(
    fallback_action: str,
    original_sentence: str,
) -> tuple[str, str, str]:
    return (
        ACTION_KEEP,
        original_sentence,
        "Correction failed; kept original sentence as fallback.",
    )


def score_for_raw_label(verification: dict[str, Any], raw_label: str) -> float:
    scores = verification.get("scores")
    if not isinstance(scores, dict):
        return 0.0
    try:
        return float(scores.get(raw_label, 0.0))
    except (TypeError, ValueError):
        return 0.0


def should_rewrite_sentence(
    verification: dict[str, Any],
    neutral_rewrite_threshold: float,
) -> tuple[bool, str]:
    label = str(verification.get("label") or "").strip()
    contradiction_score = score_for_raw_label(verification, RAW_CONTRADICTION)
    neutral_score = score_for_raw_label(verification, RAW_NEUTRAL)

    if label == UNSUPPORTED:
        return True, (
            "Rewritten because verification predicted contradiction "
            f"(score={contradiction_score:.3f})."
        )

    if label == NOT_ENOUGH_INFO:
        if neutral_score >= neutral_rewrite_threshold:
            return True, (
                "Rewritten because verification predicted high-confidence neutral "
                f"(score={neutral_score:.3f}, threshold={neutral_rewrite_threshold:.3f})."
            )
        return False, (
            "Kept because neutral verification confidence was below the rewrite "
            f"threshold (score={neutral_score:.3f}, threshold={neutral_rewrite_threshold:.3f})."
        )

    return False, "Kept because the sentence was not selected for local rewrite."


def make_sentence_edit(
    sentence_record: dict[str, Any],
    verification: dict[str, Any],
    corrected_sentence: str,
    action: str,
    reason: str,
    model_output: str | None = None,
) -> dict[str, Any]:
    edit = {
        "query_id": sentence_record.get("query_id", ""),
        "sentence_id": sentence_record.get("sentence_id"),
        "original_sentence": sentence_record.get("text", ""),
        "corrected_sentence": corrected_sentence,
        "verification_label": verification.get("label", ""),
        "evidence_chunk_ids": verification.get("evidence_ids", []),
        "selected_evidence_id": verification.get("selected_evidence_id", ""),
        "action": action,
        "reason": reason,
    }
    if model_output is not None:
        edit["model_output"] = model_output
    return edit


def count_action(stats: CorrectionStats, action: str) -> None:
    if action == ACTION_REPLACE:
        stats.edits_replaced += 1
    elif action == ACTION_REMOVE:
        stats.edits_removed += 1
    elif action == ACTION_KEEP:
        stats.edits_kept += 1
    else:
        stats.failed_corrections += 1


def correct_sentence(
    corrector: HuggingFaceSummarizer,
    sentence: str,
    label: str,
    evidence: list[dict[str, str]],
    max_sentence_words: int,
    max_new_tokens: int,
) -> tuple[str, str, str, str, bool]:
    prompt = correction_prompt(
        sentence=sentence,
        label=label,
        evidence=evidence,
        max_sentence_words=max_sentence_words,
    )
    model_output = corrector.generate(prompt, max_new_tokens=max_new_tokens)
    action, corrected_sentence, reason, parsed = parse_correction_response(model_output)
    if action == ACTION_REPLACE:
        corrected_sentence, validation_error = validate_corrected_sentence(
            corrected_sentence,
            original_sentence=sentence,
            max_sentence_words=max_sentence_words,
        )
        if validation_error:
            action = ACTION_ERROR
            reason = validation_error
    return action, corrected_sentence, reason, model_output, parsed


def join_summary_sentences(sentences: list[str]) -> str:
    return normalize_whitespace(" ".join(sentence for sentence in sentences if sentence))


def build_corrected_record(
    summary_record: dict[str, Any],
    sentence_records: list[dict[str, Any]],
    verification_by_query_id: dict[str, dict[str, Any]],
    corrector: HuggingFaceSummarizer,
    model_name: str,
    max_evidence_chunks: int,
    max_evidence_chars: int,
    max_sentence_words: int,
    max_new_tokens: int,
    fallback_action: str,
    neutral_rewrite_threshold: float,
    stats: CorrectionStats,
    continue_on_error: bool,
    save_model_outputs: bool,
) -> dict[str, Any]:
    corrected_sentences: list[str] = []
    sentence_edits: list[dict[str, Any]] = []

    for index, sentence_record in enumerate(sentence_records):
        sentence = str(sentence_record.get("text") or "")
        query_id = str(sentence_record.get("query_id") or "").strip()
        verification = verification_by_query_id.get(query_id)

        if verification is None:
            stats.sentences_without_verification += 1
            corrected_sentences.append(sentence)
            continue

        label = str(verification.get("label") or "").strip()
        if label == SUPPORTED or label not in FACTUAL_RISK_LABELS:
            stats.supported_sentences += 1
            corrected_sentences.append(sentence)
            continue

        stats.risk_sentences += 1
        should_rewrite, decision_reason = should_rewrite_sentence(
            verification=verification,
            neutral_rewrite_threshold=neutral_rewrite_threshold,
        )
        if not should_rewrite:
            action = ACTION_KEEP
            corrected_sentence = sentence
            count_action(stats, action)
            corrected_sentences.append(sentence)
            sentence_edits.append(
                make_sentence_edit(
                    sentence_record=sentence_record,
                    verification=verification,
                    corrected_sentence=corrected_sentence,
                    action=action,
                    reason=decision_reason,
                    model_output=None,
                )
            )
            continue

        evidence = selected_first_evidence(
            verification,
            max_evidence_chunks=max_evidence_chunks,
            max_evidence_chars=max_evidence_chars,
        )

        try:
            action, corrected_sentence, reason, model_output, parsed = correct_sentence(
                corrector=corrector,
                sentence=sentence,
                label=label,
                evidence=evidence,
                max_sentence_words=max_sentence_words,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            stats.failed_corrections += 1
            if not continue_on_error:
                raise
            action, corrected_sentence, reason = apply_fallback_action(
                fallback_action=fallback_action,
                original_sentence=sentence,
            )
            model_output = f"correction_error: {exc}"
            parsed = True

        if not parsed:
            stats.parse_failures += 1

        if action == ACTION_ERROR:
            stats.failed_corrections += 1
            action, corrected_sentence, reason = apply_fallback_action(
                fallback_action=fallback_action,
                original_sentence=sentence,
            )
        elif not reason:
            reason = decision_reason

        count_action(stats, action)
        corrected_sentences.append(corrected_sentence or sentence)

        sentence_edits.append(
            make_sentence_edit(
                sentence_record=sentence_record,
                verification=verification,
                corrected_sentence=corrected_sentence,
                action=action,
                reason=reason,
                model_output=model_output if save_model_outputs else None,
            )
        )

    original_summary = summary_record["original_summary"]
    corrected_summary = join_summary_sentences(corrected_sentences)
    return {
        "doc_id": summary_record["doc_id"],
        "summary_id": summary_record["summary_id"],
        "method": "evidence_constrained_local_correction",
        "original_summary": original_summary,
        "corrected_summary": corrected_summary,
        "sentence_edits": sentence_edits,
        "model": model_name,
        "original_summary_word_count": len(simple_tokenize(original_summary)),
        "corrected_summary_word_count": len(simple_tokenize(corrected_summary)),
        "num_sentences": len(sentence_records),
        "num_sentence_edits": len(sentence_edits),
        "source_summary_field": summary_record.get("source_summary_field", ""),
    }


def correct_local_jsonl(
    summaries_path: str | Path,
    sentences_path: str | Path,
    verification_path: str | Path,
    output_path: str | Path,
    summary_field: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    dtype: str = "auto",
    max_prompt_tokens: int = 4096,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 128,
    max_evidence_chunks: int = 3,
    max_evidence_chars: int = 1200,
    max_sentence_words: int = 45,
    fallback_action: str = ACTION_KEEP,
    neutral_rewrite_threshold: float = DEFAULT_NEUTRAL_REWRITE_THRESHOLD,
    limit: int | None = None,
    continue_on_error: bool = False,
    save_model_outputs: bool = False,
) -> CorrectionStats:
    summaries_by_key, summaries_by_doc_id, stats = index_summaries(
        summaries_path=summaries_path,
        summary_field=summary_field,
    )
    grouped_sentences, sentence_count = group_sentences(sentences_path)
    verification_by_query_id, verification_count = index_verification(verification_path)
    stats.input_sentences = sentence_count
    stats.input_verification_results = verification_count

    corrector = HuggingFaceSummarizer(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_prompt_tokens=max_prompt_tokens,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    output_path = ensure_parent_dir(output_path)
    written = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for key, sentence_records in grouped_sentences.items():
            if limit is not None and written >= limit:
                break
            doc_id, _ = key
            summary_record = summaries_by_key.get(key) or summaries_by_doc_id.get(doc_id)
            if summary_record is None:
                stats.skipped_missing_summary_text += 1
                continue
            if not sentence_records:
                stats.skipped_summaries_without_sentences += 1
                continue

            record = build_corrected_record(
                summary_record=summary_record,
                sentence_records=sentence_records,
                verification_by_query_id=verification_by_query_id,
                corrector=corrector,
                model_name=model_name,
                max_evidence_chunks=max_evidence_chunks,
                max_evidence_chars=max_evidence_chars,
                max_sentence_words=max_sentence_words,
                max_new_tokens=max_new_tokens,
                fallback_action=fallback_action,
                neutral_rewrite_threshold=neutral_rewrite_threshold,
                stats=stats,
                continue_on_error=continue_on_error,
                save_model_outputs=save_model_outputs,
            )
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            written += 1
            stats.written_records += 1
            print(f"written_correction: {record['doc_id']}", file=sys.stderr, flush=True)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evidence-constrained local correction for risky summary sentences."
    )
    parser.add_argument("--summaries", required=True, help="Generated summary JSONL path.")
    parser.add_argument("--sentences", required=True, help="Summary sentence JSONL path.")
    parser.add_argument("--verification", required=True, help="NLI verification JSONL path.")
    parser.add_argument("--output", required=True, help="Correction JSONL output path.")
    parser.add_argument(
        "--summary-field",
        default=None,
        help="Summary text field to read. Defaults to generated_summary first.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace causal language model name or local path.",
    )
    parser.add_argument("--device", default="auto", help="Generation device.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4096,
        help="Maximum prompt tokens passed to the correction model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens for each corrected sentence.",
    )
    parser.add_argument(
        "--max-evidence-chunks",
        type=int,
        default=3,
        help="Maximum evidence chunks shown to the correction model.",
    )
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=1200,
        help="Maximum characters kept for each evidence chunk.",
    )
    parser.add_argument(
        "--max-sentence-words",
        type=int,
        default=45,
        help="Soft word limit requested for replacement sentences.",
    )
    parser.add_argument(
        "--fallback-action",
        choices=[ACTION_KEEP],
        default=ACTION_KEEP,
        help=(
            "Deprecated compatibility option. Failed rewrites are kept to avoid "
            "sentence-level deletion."
        ),
    )
    parser.add_argument(
        "--neutral-rewrite-threshold",
        type=float,
        default=DEFAULT_NEUTRAL_REWRITE_THRESHOLD,
        help=(
            "Only rewrite Not Enough Information / neutral sentences when the "
            "neutral NLI score is at least this value."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local HuggingFace cache or local path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repositories with custom code.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of deterministic greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of summaries to correct.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a document-level correction failure.",
    )
    parser.add_argument(
        "--save-model-outputs",
        action="store_true",
        help="Store raw correction model outputs for debugging.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = correct_local_jsonl(
        summaries_path=args.summaries,
        sentences_path=args.sentences,
        verification_path=args.verification,
        output_path=args.output,
        summary_field=args.summary_field,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_prompt_tokens=args.max_prompt_tokens,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_evidence_chunks=args.max_evidence_chunks,
        max_evidence_chars=args.max_evidence_chars,
        max_sentence_words=args.max_sentence_words,
        fallback_action=args.fallback_action,
        neutral_rewrite_threshold=args.neutral_rewrite_threshold,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
        save_model_outputs=args.save_model_outputs,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
