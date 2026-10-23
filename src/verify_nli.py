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
AGGREGATION_CONCAT = "concat"
AGGREGATION_PER_CHUNK = "per_chunk"
AGGREGATION_METHODS = (AGGREGATION_CONCAT, AGGREGATION_PER_CHUNK)
DEFAULT_AGGREGATION = AGGREGATION_PER_CHUNK
DEFAULT_ENTAILMENT_THRESHOLD = 0.5
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
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


def make_concat_result(
    item: dict[str, Any],
    verifier: NliVerifier,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    result = build_verification_result(
        verification_input=item,
        label=prediction["label"],
        reason=make_reason(prediction["raw_label"], prediction["scores"]),
        model=verifier.model_name,
        scores=prediction["scores"],
        raw_label=prediction["raw_label"],
    )
    result["aggregation"] = AGGREGATION_CONCAT
    return result


def evidence_items_for_prediction(item: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_items = item.get("evidence_items")
    if isinstance(evidence_items, list) and evidence_items:
        return [
            evidence_item
            for evidence_item in evidence_items
            if str(evidence_item.get("text") or "").strip()
        ]

    evidence_texts = item.get("evidence") or []
    evidence_ids = item.get("evidence_ids") or []
    fallback_items: list[dict[str, Any]] = []
    for index, text in enumerate(evidence_texts, start=1):
        text = str(text or "").strip()
        if not text:
            continue
        chunk_id = ""
        if index - 1 < len(evidence_ids):
            chunk_id = str(evidence_ids[index - 1] or "").strip()
        fallback_items.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": None,
                "rank": index,
                "score": None,
                "text": text,
            }
        )
    return fallback_items


def make_chunk_prediction(
    evidence_item: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    return {
        "chunk_id": evidence_item.get("chunk_id", ""),
        "chunk_index": evidence_item.get("chunk_index"),
        "rank": evidence_item.get("rank"),
        "retrieval_score": evidence_item.get("score"),
        "label": prediction["label"],
        "raw_label": prediction["raw_label"],
        "scores": prediction["scores"],
    }


def max_scores(chunk_predictions: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for prediction in chunk_predictions:
        for label, score in prediction.get("scores", {}).items():
            scores[label] = max(scores.get(label, 0.0), float(score))
    return {label: round(score, 6) for label, score in scores.items()}


def label_score(prediction: dict[str, Any], label: str) -> float:
    return float(prediction.get("scores", {}).get(label, 0.0))


def choose_chunk_prediction(
    chunk_predictions: list[dict[str, Any]],
    label: str,
    threshold: float,
) -> dict[str, Any] | None:
    candidates = [
        prediction
        for prediction in chunk_predictions
        if prediction.get("raw_label") == label
        and label_score(prediction, label) >= threshold
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda prediction: label_score(prediction, label))


def make_per_chunk_reason(
    selected_prediction: dict[str, Any] | None,
    final_label: str,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> str:
    if selected_prediction:
        raw_label = str(selected_prediction.get("raw_label") or "")
        score = label_score(selected_prediction, raw_label)
        chunk_id = selected_prediction.get("chunk_id") or "unknown chunk"
        return (
            f"Per-chunk NLI selected {raw_label} from {chunk_id} "
            f"with score {score:.3f}."
        )
    return (
        "Per-chunk NLI found no evidence chunk above thresholds "
        f"(entailment >= {entailment_threshold:.2f}, "
        f"contradiction >= {contradiction_threshold:.2f}); "
        f"assigned {final_label}."
    )


def aggregate_chunk_predictions(
    chunk_predictions: list[dict[str, Any]],
    entailment_threshold: float,
    contradiction_threshold: float,
) -> tuple[str, str, dict[str, float], str, dict[str, Any] | None]:
    selected_entailment = choose_chunk_prediction(
        chunk_predictions,
        label="entailment",
        threshold=entailment_threshold,
    )
    if selected_entailment:
        return (
            SUPPORTED,
            "entailment",
            selected_entailment["scores"],
            make_per_chunk_reason(
                selected_entailment,
                final_label=SUPPORTED,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold,
            ),
            selected_entailment,
        )

    selected_contradiction = choose_chunk_prediction(
        chunk_predictions,
        label="contradiction",
        threshold=contradiction_threshold,
    )
    if selected_contradiction:
        return (
            UNSUPPORTED,
            "contradiction",
            selected_contradiction["scores"],
            make_per_chunk_reason(
                selected_contradiction,
                final_label=UNSUPPORTED,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold,
            ),
            selected_contradiction,
        )

    scores = max_scores(chunk_predictions)
    raw_label = max(scores, key=scores.get) if scores else "neutral"
    return (
        NOT_ENOUGH_INFO,
        raw_label,
        scores,
        make_per_chunk_reason(
            selected_prediction=None,
            final_label=NOT_ENOUGH_INFO,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        ),
        None,
    )


def make_per_chunk_result(
    item: dict[str, Any],
    verifier: NliVerifier,
    hypothesis: str,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> tuple[dict[str, Any], bool]:
    chunk_predictions: list[dict[str, Any]] = []
    had_empty_output = False
    for evidence_item in evidence_items_for_prediction(item):
        prediction = verifier.predict(
            premise=str(evidence_item.get("text") or ""),
            hypothesis=hypothesis,
        )
        if not prediction["raw_label"]:
            had_empty_output = True
        chunk_predictions.append(make_chunk_prediction(evidence_item, prediction))

    label, raw_label, scores, reason, selected_prediction = aggregate_chunk_predictions(
        chunk_predictions=chunk_predictions,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )
    result = build_verification_result(
        verification_input=item,
        label=label,
        reason=reason,
        model=verifier.model_name,
        scores=scores,
        raw_label=raw_label,
    )
    result["aggregation"] = AGGREGATION_PER_CHUNK
    result["chunk_predictions"] = chunk_predictions
    result["selected_evidence_id"] = (
        selected_prediction.get("chunk_id", "") if selected_prediction else ""
    )
    result["entailment_threshold"] = entailment_threshold
    result["contradiction_threshold"] = contradiction_threshold
    return result, had_empty_output


def make_no_evidence_result(item: dict[str, Any], verifier: NliVerifier) -> dict[str, Any]:
    result = build_verification_result(
        verification_input=item,
        label=NOT_ENOUGH_INFO,
        reason="No retrieved evidence was available for this sentence.",
        model=verifier.model_name,
        scores={},
        raw_label="",
    )
    result["aggregation"] = ""
    return result


def verify_inputs(
    inputs: Iterable[dict[str, Any]],
    verifier: NliVerifier,
    aggregation: str = DEFAULT_AGGREGATION,
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
) -> tuple[list[dict[str, Any]], NliStats]:
    if aggregation not in AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")

    stats = NliStats()
    output: list[dict[str, Any]] = []

    for item in inputs:
        stats.input_records += 1
        premise = str(item.get("premise") or "").strip()
        hypothesis = str(item.get("sentence") or "").strip()
        if not premise:
            stats.records_without_premise += 1
            if hypothesis:
                output.append(make_no_evidence_result(item, verifier=verifier))
                stats.written_results += 1
                continue
        if not hypothesis:
            stats.records_without_hypothesis += 1
            continue

        if aggregation == AGGREGATION_CONCAT:
            prediction = verifier.predict(premise=premise, hypothesis=hypothesis)
            if not prediction["raw_label"]:
                stats.truncated_or_empty_outputs += 1
            output.append(make_concat_result(item, verifier=verifier, prediction=prediction))
        else:
            result, had_empty_output = make_per_chunk_result(
                item=item,
                verifier=verifier,
                hypothesis=hypothesis,
                entailment_threshold=entailment_threshold,
                contradiction_threshold=contradiction_threshold,
            )
            if had_empty_output:
                stats.truncated_or_empty_outputs += 1
            output.append(result)
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
    aggregation: str = DEFAULT_AGGREGATION,
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
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
            allow_missing_retrieval=True,
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
    results, stats = verify_inputs(
        inputs,
        verifier=verifier,
        aggregation=aggregation,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )
    write_jsonl(results, output_path)
    return stats


def parse_label_order(value: str) -> tuple[str, ...]:
    labels = tuple(normalize_label(part) for part in value.split(",") if part.strip())
    if len(labels) != 3 or len(set(labels)) != 3:
        raise argparse.ArgumentTypeError("label order must contain three distinct labels")
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
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_METHODS,
        default=DEFAULT_AGGREGATION,
        help=(
            "Evidence aggregation strategy. concat runs one NLI call over joined "
            "evidence; per_chunk runs NLI on each evidence chunk and aggregates labels."
        ),
    )
    parser.add_argument(
        "--entailment-threshold",
        type=float,
        default=DEFAULT_ENTAILMENT_THRESHOLD,
        help="Minimum per-chunk entailment score required for Supported.",
    )
    parser.add_argument(
        "--contradiction-threshold",
        type=float,
        default=DEFAULT_CONTRADICTION_THRESHOLD,
        help="Minimum per-chunk contradiction score required for Unsupported.",
    )
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
        aggregation=args.aggregation,
        entailment_threshold=args.entailment_threshold,
        contradiction_threshold=args.contradiction_threshold,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
