"""Run sentence-level NLI verification over retrieved evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.utils import iter_jsonl, write_jsonl
from src.verification_schema import (
    NOT_ENOUGH_INFO,
    SUPPORTED,
    UNSUPPORTED,
    build_verification_inputs,
    build_verification_result,
)


DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
DEFAULT_LABEL_ORDER = ("contradiction", "neutral", "entailment")
NLI_TO_PROJECT_LABEL = {
    "entailment": SUPPORTED,
    "contradiction": UNSUPPORTED,
    "neutral": NOT_ENOUGH_INFO,
}


@dataclass
class NliStats:
    input_records: int = 0
    written_results: int = 0
    records_without_premise: int = 0
    records_without_hypothesis: int = 0
    truncated_or_empty_outputs: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    import math

    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in values]
    return [value / total for value in exps]


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    if "entail" in label:
        return "entailment"
    if "contrad" in label:
        return "contradiction"
    if "neutral" in label:
        return "neutral"
    return label


class NliVerifier:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str = "cpu",
        max_length: int = 1024,
        local_files_only: bool = False,
        label_order: tuple[str, ...] = DEFAULT_LABEL_ORDER,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=local_files_only,
            )
        except OSError as exc:
            cache_hint = "local cache" if local_files_only else "Hugging Face"
            raise RuntimeError(
                f"Could not load NLI model `{model_name}` from {cache_hint}. "
                "Download the model first or rerun without --local-files-only when "
                "network access is available."
            ) from exc
        self.model.to(self.device)
        self.model.eval()
        self.label_order = self._resolve_label_order(label_order)

    def _resolve_label_order(self, fallback: tuple[str, ...]) -> tuple[str, ...]:
        id2label = getattr(self.model.config, "id2label", None) or {}
        if not id2label:
            return fallback
        labels: list[str] = []
        for index in range(len(id2label)):
            labels.append(normalize_label(str(id2label.get(index, ""))))
        if all(label in NLI_TO_PROJECT_LABEL for label in labels):
            return tuple(labels)
        return fallback

    def predict(self, premise: str, hypothesis: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
        logits = outputs.logits[0].detach().cpu().tolist()
        probabilities = softmax([float(value) for value in logits])
        scores = {
            label: round(float(probability), 6)
            for label, probability in zip(self.label_order, probabilities)
        }
        raw_label = max(scores, key=scores.get) if scores else ""
        project_label = NLI_TO_PROJECT_LABEL.get(raw_label, NOT_ENOUGH_INFO)
        return {
            "label": project_label,
            "raw_label": raw_label,
            "scores": scores,
        }


def make_reason(raw_label: str, scores: dict[str, float]) -> str:
    if not raw_label:
        return "NLI did not return a valid label."
    score = scores.get(raw_label, 0.0)
    return f"NLI predicted {raw_label} with score {score:.3f}."


def verify_inputs(
    inputs: Iterable[dict[str, Any]],
    verifier: NliVerifier,
) -> tuple[list[dict[str, Any]], NliStats]:
    stats = NliStats()
    output: list[dict[str, Any]] = []

    for item in inputs:
        stats.input_records += 1
        premise = str(item.get("premise") or "").strip()
        hypothesis = str(item.get("sentence") or "").strip()
        if not premise:
            stats.records_without_premise += 1
            if hypothesis:
                output.append(
                    build_verification_result(
                        verification_input=item,
                        label=NOT_ENOUGH_INFO,
                        reason="No retrieved evidence was available for this sentence.",
                        model=verifier.model_name,
                        scores={},
                        raw_label="",
                    )
                )
                stats.written_results += 1
                continue
        if not hypothesis:
            stats.records_without_hypothesis += 1
            continue

        prediction = verifier.predict(premise=premise, hypothesis=hypothesis)
        if not prediction["raw_label"]:
            stats.truncated_or_empty_outputs += 1
        output.append(
            build_verification_result(
                verification_input=item,
                label=prediction["label"],
                reason=make_reason(prediction["raw_label"], prediction["scores"]),
                model=verifier.model_name,
                scores=prediction["scores"],
                raw_label=prediction["raw_label"],
            )
        )
        stats.written_results += 1

    return output, stats


def verify_nli_jsonl(
    output_path: str | Path,
    model_name: str = DEFAULT_NLI_MODEL,
    device: str = "cpu",
    max_length: int = 1024,
    local_files_only: bool = False,
    label_order: tuple[str, ...] = DEFAULT_LABEL_ORDER,
    inputs_path: str | Path | None = None,
    sentences_path: str | Path | None = None,
    retrieval_path: str | Path | None = None,
    method: str = "Hybrid + NLI",
    retrieval_method: str | None = None,
    top_k: int = 3,
    limit: int | None = None,
) -> NliStats:
    if inputs_path is None and (sentences_path is None or retrieval_path is None):
        raise ValueError("Provide either --inputs or both --sentences and --retrieval.")

    if inputs_path is not None:
        inputs = list(iter_jsonl(inputs_path))
    else:
        inputs, input_stats = build_verification_inputs(
            sentences=iter_jsonl(sentences_path),  # type: ignore[arg-type]
            retrieval_results=iter_jsonl(retrieval_path),  # type: ignore[arg-type]
            method=method,
            verifier="nli",
            retrieval_method=retrieval_method,
            top_k=top_k,
        )
        for key, value in input_stats.to_dict().items():
            print(f"{key}: {value}")

    if limit is not None:
        inputs = inputs[:limit]

    verifier = NliVerifier(
        model_name=model_name,
        device=device,
        max_length=max_length,
        local_files_only=local_files_only,
        label_order=label_order,
    )
    results, stats = verify_inputs(inputs, verifier=verifier)
    write_jsonl(results, output_path)
    return stats


def parse_label_order(value: str) -> tuple[str, ...]:
    labels = tuple(normalize_label(part) for part in value.split(",") if part.strip())
    if len(labels) != 3:
        raise argparse.ArgumentTypeError("label order must contain exactly three labels")
    unknown = [label for label in labels if label not in NLI_TO_PROJECT_LABEL]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown NLI labels: {', '.join(unknown)}")
    return labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sentence-level NLI verification.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--inputs",
        default=None,
        help="Prepared verification input JSONL path.",
    )
    input_group.add_argument(
        "--sentences",
        default=None,
        help="Summary sentence JSONL path. Requires --retrieval.",
    )
    parser.add_argument(
        "--retrieval",
        default=None,
        help="Retrieval JSONL path when using --sentences.",
    )
    parser.add_argument("--output", required=True, help="Verification result JSONL path.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_NLI_MODEL,
        help="NLI model name.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--max-length", type=int, default=1024, help="NLI max tokens.")
    parser.add_argument("--top-k", type=int, default=3, help="Evidence chunks to use.")
    parser.add_argument("--method", default="Hybrid + NLI", help="Experiment method name.")
    parser.add_argument(
        "--retrieval-method",
        default=None,
        help="Optional retrieval method override.",
    )
    parser.add_argument(
        "--label-order",
        type=parse_label_order,
        default=DEFAULT_LABEL_ORDER,
        help="Fallback model logit label order, e.g. contradiction,neutral,entailment.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the NLI model from local Hugging Face cache only.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = verify_nli_jsonl(
        inputs_path=args.inputs,
        sentences_path=args.sentences,
        retrieval_path=args.retrieval,
        output_path=args.output,
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        label_order=args.label_order,
        method=args.method,
        retrieval_method=args.retrieval_method,
        top_k=args.top_k,
        limit=args.limit,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
