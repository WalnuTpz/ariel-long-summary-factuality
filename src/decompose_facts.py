"""LLM-based atomic fact decomposition with structured outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import ensure_parent_dir, iter_jsonl, stringify_text


DEFAULT_MODEL = "gpt-4o-mini"
SUMMARY_TYPES = ("clean", "perturbed")
FACT_TYPES = (
    "entity",
    "number",
    "date",
    "relation",
    "event",
    "attribute",
    "comparison",
    "causal",
    "policy",
    "other",
)
ERROR_TYPES = ("none", "entity", "number", "date", "relation", "unsupported", "other")


FACT_DECOMPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_id": {"type": "integer"},
                    "sentence_text": {"type": "string"},
                    "fact_index": {"type": "integer"},
                    "fact": {"type": "string"},
                    "fact_type": {"type": "string", "enum": list(FACT_TYPES)},
                    "modification_index": {"type": "integer"},
                    "controlled_error_type": {
                        "type": "string",
                        "enum": list(ERROR_TYPES),
                    },
                },
                "required": [
                    "sentence_id",
                    "sentence_text",
                    "fact_index",
                    "fact",
                    "fact_type",
                    "modification_index",
                    "controlled_error_type",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You decompose summaries into atomic factual claims for a long-document factuality experiment.

Rules:
- Output only facts explicitly stated by the summary. Do not use outside knowledge.
- Do not verify the fact against the source document; this step only decomposes the summary.
- One atomic fact must express one independently checkable claim.
- Split conjunctions when they contain multiple checkable claims.
- Keep numbers, dates, entities, agencies, quantities, and comparisons exact.
- Make each fact self-contained. Replace pronouns with the nearest explicit entity when the reference is clear.
- Preserve the summary sentence order. sentence_id is 1-based.
- sentence_text should be the original summary sentence that the fact comes from.
- fact_index is 1-based within each sentence.
- For clean summaries, set modification_index to 0 and controlled_error_type to "none".
- For perturbed summaries, use the modifications list only to mark controlled errors:
  if a fact expresses a modified_fact, set modification_index to its 1-based index
  and controlled_error_type to its error_type. Otherwise use 0 and "none".
- If a whole added sentence is unsupported, mark each unsupported fact from that added content
  with the corresponding unsupported modification.
"""


@dataclass
class DecompositionStats:
    input_records: int = 0
    summary_tasks: int = 0
    written_facts: int = 0
    skipped_empty_summary: int = 0
    skipped_existing_summary: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs without overriding the current environment."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def resolve_model(cli_model: str | None) -> str:
    return (
        cli_model
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL_NAME")
        or DEFAULT_MODEL
    )


def resolve_base_url(cli_base_url: str | None) -> str | None:
    value = cli_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or ""
    return value.strip() or None


def normalize_summary_types(values: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(values)
    invalid = [value for value in selected if value not in SUMMARY_TYPES]
    if invalid:
        raise ValueError(f"Invalid summary type(s): {', '.join(invalid)}")
    return selected


def build_payload(
    record: dict[str, Any],
    summary_type: str,
    summary_text: str,
) -> dict[str, Any]:
    modifications = record.get("modifications")
    if not isinstance(modifications, list):
        modifications = []

    indexed_modifications = []
    for index, modification in enumerate(modifications, start=1):
        if not isinstance(modification, dict):
            continue
        indexed_modifications.append(
            {
                "modification_index": index,
                "original_fact": stringify_text(modification.get("original_fact")),
                "modified_fact": stringify_text(modification.get("modified_fact")),
                "error_type": stringify_text(modification.get("error_type")),
            }
        )

    return {
        "doc_id": stringify_text(record.get("doc_id")),
        "summary_type": summary_type,
        "summary_text": summary_text,
        "modifications": indexed_modifications if summary_type == "perturbed" else [],
    }


def call_openai_decomposer(
    payload: dict[str, Any],
    model: str,
    base_url: str | None,
    temperature: float,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai is not installed. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    client_kwargs: dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "atomic_fact_decomposition",
                "strict": True,
                "schema": FACT_DECOMPOSITION_SCHEMA,
            }
        },
        temperature=temperature,
    )
    return json.loads(extract_response_text(response))


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if isinstance(response, dict):
        for output_item in response.get("output", []):
            for content_item in output_item.get("content", []):
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if text:
                        return str(text)
    raise ValueError("Could not extract text from OpenAI response.")


def normalize_fact_type(value: Any) -> str:
    text = stringify_text(value).strip().lower()
    return text if text in FACT_TYPES else "other"


def normalize_error_type(value: Any) -> str:
    text = stringify_text(value).strip().lower()
    return text if text in ERROR_TYPES else "other"


def make_fact_id(
    doc_id: str,
    summary_type: str,
    sentence_id: int,
    fact_index: int,
) -> str:
    safe_doc_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc_id).strip("_")
    return f"{safe_doc_id}_{summary_type}_s{sentence_id:03d}_f{fact_index:02d}"


def normalize_model_facts(
    raw_response: dict[str, Any],
    record: dict[str, Any],
    summary_type: str,
    model: str,
) -> list[dict[str, Any]]:
    doc_id = stringify_text(record.get("doc_id")).strip()
    if not doc_id:
        raise ValueError("Summary record is missing doc_id.")

    raw_facts = raw_response.get("facts")
    if not isinstance(raw_facts, list):
        raise ValueError("Model response must contain a facts list.")

    output: list[dict[str, Any]] = []
    for raw_fact in raw_facts:
        if not isinstance(raw_fact, dict):
            continue
        fact = stringify_text(raw_fact.get("fact")).strip()
        if not fact:
            continue

        sentence_id = int(raw_fact.get("sentence_id") or 0)
        fact_index = int(raw_fact.get("fact_index") or 0)
        if sentence_id <= 0 or fact_index <= 0:
            continue

        modification_index = int(raw_fact.get("modification_index") or 0)
        controlled_error_type = normalize_error_type(
            raw_fact.get("controlled_error_type")
        )
        if summary_type == "clean":
            modification_index = 0
            controlled_error_type = "none"
        if modification_index <= 0:
            modification_index = 0
            controlled_error_type = "none"

        output.append(
            {
                "fact_id": make_fact_id(
                    doc_id=doc_id,
                    summary_type=summary_type,
                    sentence_id=sentence_id,
                    fact_index=fact_index,
                ),
                "doc_id": doc_id,
                "summary_type": summary_type,
                "sentence_id": sentence_id,
                "fact_index": fact_index,
                "fact": fact,
                "fact_type": normalize_fact_type(raw_fact.get("fact_type")),
                "source_sentence": stringify_text(raw_fact.get("sentence_text")).strip(),
                "controlled_error_type": controlled_error_type,
                "modification_index": modification_index,
                "decomposition_method": "openai_structured",
                "model": model,
            }
        )

    return sorted(output, key=lambda item: (item["sentence_id"], item["fact_index"]))


def completed_summary_keys(output_path: str | Path) -> set[tuple[str, str]]:
    path = Path(output_path)
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for item in iter_jsonl(path):
        doc_id = stringify_text(item.get("doc_id")).strip()
        summary_type = stringify_text(item.get("summary_type")).strip()
        if doc_id and summary_type:
            keys.add((doc_id, summary_type))
    return keys


def decompose_summaries_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    summary_types: tuple[str, ...],
    model: str,
    base_url: str | None = None,
    temperature: float = 0.0,
    limit: int | None = None,
    resume: bool = False,
) -> DecompositionStats:
    output_path = ensure_parent_dir(output_path)
    completed = completed_summary_keys(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"
    stats = DecompositionStats()

    with Path(output_path).open(mode, encoding="utf-8") as output_file:
        for record in iter_jsonl(input_path):
            stats.input_records += 1
            if limit is not None and stats.summary_tasks >= limit:
                break

            doc_id = stringify_text(record.get("doc_id")).strip()
            for summary_type in summary_types:
                if limit is not None and stats.summary_tasks >= limit:
                    break
                summary_key = f"{summary_type}_summary"
                summary_text = stringify_text(record.get(summary_key)).strip()
                if not summary_text:
                    stats.skipped_empty_summary += 1
                    continue
                if (doc_id, summary_type) in completed:
                    stats.skipped_existing_summary += 1
                    continue

                stats.summary_tasks += 1
                payload = build_payload(
                    record=record,
                    summary_type=summary_type,
                    summary_text=summary_text,
                )
                raw_response = call_openai_decomposer(
                    payload=payload,
                    model=model,
                    base_url=base_url,
                    temperature=temperature,
                )
                facts = normalize_model_facts(
                    raw_response=raw_response,
                    record=record,
                    summary_type=summary_type,
                    model=model,
                )
                for fact in facts:
                    output_file.write(json.dumps(fact, ensure_ascii=False) + "\n")
                    stats.written_facts += 1
                output_file.flush()

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompose clean/perturbed summaries into atomic facts with an LLM."
    )
    parser.add_argument(
        "--input",
        default="data/processed/summaries.jsonl",
        help="Summary JSONL input path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/facts.jsonl",
        help="Atomic fact JSONL output path.",
    )
    parser.add_argument(
        "--summary-types",
        nargs="+",
        default=list(SUMMARY_TYPES),
        help="Summary types to decompose: clean perturbed.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenAI model name. Defaults to OPENAI_MODEL, MODEL_NAME, or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of summary texts to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output and skip doc_id/summary_type pairs already present.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional local env file containing OPENAI_API_KEY.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    load_env_file(args.env_file)
    summary_types = normalize_summary_types(args.summary_types)
    stats = decompose_summaries_jsonl(
        input_path=args.input,
        output_path=args.output,
        summary_types=summary_types,
        model=resolve_model(args.model),
        base_url=resolve_base_url(args.base_url),
        temperature=args.temperature,
        limit=args.limit,
        resume=args.resume,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
