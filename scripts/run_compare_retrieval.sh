#!/usr/bin/env bash
set -euo pipefail

# Compare retrieval methods for sentence-level NLI factuality detection.
#
# Typical debug run:
#   PYTHON=.venv/bin/python RUN_NAME=debug_compare bash scripts/run_compare_retrieval.sh
#
# Typical main run:
#   PYTHON=.venv/bin/python \
#   RUN_NAME=main_500_compare \
#   DOCUMENTS=data/processed/documents_main_500.jsonl \
#   SUMMARIES=data/processed/generated_summaries.qwen2p5_1p5b.jsonl \
#   bash scripts/run_compare_retrieval.sh

CONFIG="${CONFIG:-}"
if [[ -n "${CONFIG}" ]]; then
  if [[ ! -f "${CONFIG}" ]]; then
    echo "missing config file: ${CONFIG}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONFIG}"
fi

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  echo "+ $*"
  "$@"
}

method_label() {
  case "$1" in
    bm25) echo "BM25 + NLI" ;;
    embedding) echo "Embedding + NLI" ;;
    hybrid) echo "Hybrid + NLI" ;;
    rael) echo "RAEL + NLI" ;;
    rael_no_coverage) echo "RAEL w/o coverage + NLI" ;;
    rael_no_expansion) echo "RAEL w/o adaptive expansion + NLI" ;;
    rael_no_risk_weighting) echo "RAEL w/o risk-adaptive weighting + NLI" ;;
    *) echo "$1 + NLI" ;;
  esac
}

retrieval_method_name() {
  case "$1" in
    rael_no_coverage|rael_no_expansion|rael_no_risk_weighting) echo "rael" ;;
    *) echo "$1" ;;
  esac
}

PYTHON_BIN="${PYTHON:-python3}"
RUN_NAME="${RUN_NAME:-debug_retrieval_compare}"
RUN_DIR="${RUN_DIR:-data/results/${RUN_NAME}}"
DOCUMENTS="${DOCUMENTS:-data/processed/documents_debug_5.jsonl}"
SUMMARIES="${SUMMARIES:-data/processed/summaries_debug_5.jsonl}"
SUMMARY_FIELD="${SUMMARY_FIELD:-}"
METHODS="${METHODS:-bm25 embedding hybrid rael rael_no_coverage rael_no_expansion rael_no_risk_weighting}"

FORCE="${FORCE:-1}"
TOP_K="${TOP_K:-3}"
CHUNK_TYPE="${CHUNK_TYPE:-paragraph}"
CHUNK_MAX_TOKENS="${CHUNK_MAX_TOKENS:-256}"
CHUNK_STRIDE="${CHUNK_STRIDE:-}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"

HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
EMBEDDING_LOCAL_FILES_ONLY="${EMBEDDING_LOCAL_FILES_ONLY:-${HF_LOCAL_FILES_ONLY}}"

NLI_MODEL="${NLI_MODEL:-facebook/bart-large-mnli}"
NLI_DEVICE="${NLI_DEVICE:-cpu}"
NLI_MAX_LENGTH="${NLI_MAX_LENGTH:-1024}"
NLI_AGGREGATION="${NLI_AGGREGATION:-per_chunk}"
NLI_ENTAILMENT_THRESHOLD="${NLI_ENTAILMENT_THRESHOLD:-0.5}"
NLI_CONTRADICTION_THRESHOLD="${NLI_CONTRADICTION_THRESHOLD:-0.5}"
NLI_LOCAL_FILES_ONLY="${NLI_LOCAL_FILES_ONLY:-${HF_LOCAL_FILES_ONLY}}"
NLI_LIMIT="${NLI_LIMIT:-}"

EXPORT_CASES="${EXPORT_CASES:-1}"

mkdir -p "${RUN_DIR}"
require_file "${DOCUMENTS}"
require_file "${SUMMARIES}"

summary_field_args=()
if [[ -n "${SUMMARY_FIELD}" ]]; then
  summary_field_args+=(--summary-field "${SUMMARY_FIELD}")
fi

chunk_stride_args=()
if [[ -n "${CHUNK_STRIDE}" ]]; then
  chunk_stride_args+=(--stride "${CHUNK_STRIDE}")
fi

embedding_local_args=()
if is_true "${EMBEDDING_LOCAL_FILES_ONLY}"; then
  embedding_local_args+=(--local-files-only)
fi

nli_local_args=()
if is_true "${NLI_LOCAL_FILES_ONLY}"; then
  nli_local_args+=(--local-files-only)
fi

nli_limit_args=()
if [[ -n "${NLI_LIMIT}" ]]; then
  nli_limit_args+=(--limit "${NLI_LIMIT}")
fi

CHUNKS="${CHUNKS:-${RUN_DIR}/chunks.jsonl}"
SENTENCES="${RUN_DIR}/summary_sentences.jsonl"
BM25_RETRIEVAL="${RUN_DIR}/retrieval.bm25.jsonl"
EMBEDDING_RETRIEVAL="${RUN_DIR}/retrieval.embedding.jsonl"
HYBRID_RETRIEVAL="${RUN_DIR}/retrieval.hybrid.jsonl"
RAEL_RETRIEVAL="${RUN_DIR}/retrieval.rael.jsonl"
RAEL_NO_COVERAGE_RETRIEVAL="${RUN_DIR}/retrieval.rael_no_coverage.jsonl"
RAEL_NO_EXPANSION_RETRIEVAL="${RUN_DIR}/retrieval.rael_no_expansion.jsonl"
RAEL_NO_RISK_WEIGHTING_RETRIEVAL="${RUN_DIR}/retrieval.rael_no_risk_weighting.jsonl"
COMPARISON_TSV="${RUN_DIR}/comparison.tsv"
COMPARISON_MD="${RUN_DIR}/comparison.md"

if is_true "${FORCE}" || [[ ! -s "${CHUNKS}" ]]; then
  run_cmd "${PYTHON_BIN}" -m src.chunking \
    --input "${DOCUMENTS}" \
    --output "${CHUNKS}" \
    --chunk-type "${CHUNK_TYPE}" \
    --max-tokens "${CHUNK_MAX_TOKENS}" \
    "${chunk_stride_args[@]}"
fi

run_cmd "${PYTHON_BIN}" -m src.split_summary_sentences \
  --input "${SUMMARIES}" \
  --output "${SENTENCES}" \
  "${summary_field_args[@]}"

run_cmd "${PYTHON_BIN}" -m src.retrieval_bm25 \
  --chunks "${CHUNKS}" \
  --queries "${SENTENCES}" \
  --output "${BM25_RETRIEVAL}" \
  --top-k "${TOP_K}"

run_cmd "${PYTHON_BIN}" -m src.retrieval_embedding \
  --chunks "${CHUNKS}" \
  --queries "${SENTENCES}" \
  --output "${EMBEDDING_RETRIEVAL}" \
  --top-k "${TOP_K}" \
  --model-name "${EMBEDDING_MODEL}" \
  --device "${EMBEDDING_DEVICE}" \
  "${embedding_local_args[@]}"

run_cmd "${PYTHON_BIN}" -m src.retrieval_hybrid \
  --bm25 "${BM25_RETRIEVAL}" \
  --embedding "${EMBEDDING_RETRIEVAL}" \
  --output "${HYBRID_RETRIEVAL}" \
  --alpha "${HYBRID_ALPHA}" \
  --top-k "${TOP_K}"

run_cmd "${PYTHON_BIN}" -m src.retrieval_rael \
  --bm25 "${BM25_RETRIEVAL}" \
  --embedding "${EMBEDDING_RETRIEVAL}" \
  --output "${RAEL_RETRIEVAL}" \
  --queries "${SENTENCES}" \
  --chunks "${CHUNKS}" \
  --top-k "${TOP_K}"

run_cmd "${PYTHON_BIN}" -m src.retrieval_rael \
  --bm25 "${BM25_RETRIEVAL}" \
  --embedding "${EMBEDDING_RETRIEVAL}" \
  --output "${RAEL_NO_COVERAGE_RETRIEVAL}" \
  --queries "${SENTENCES}" \
  --chunks "${CHUNKS}" \
  --top-k "${TOP_K}" \
  --disable-coverage-rerank

run_cmd "${PYTHON_BIN}" -m src.retrieval_rael \
  --bm25 "${BM25_RETRIEVAL}" \
  --embedding "${EMBEDDING_RETRIEVAL}" \
  --output "${RAEL_NO_EXPANSION_RETRIEVAL}" \
  --queries "${SENTENCES}" \
  --chunks "${CHUNKS}" \
  --top-k "${TOP_K}" \
  --disable-adaptive-expansion

run_cmd "${PYTHON_BIN}" -m src.retrieval_rael \
  --bm25 "${BM25_RETRIEVAL}" \
  --embedding "${EMBEDDING_RETRIEVAL}" \
  --output "${RAEL_NO_RISK_WEIGHTING_RETRIEVAL}" \
  --queries "${SENTENCES}" \
  --chunks "${CHUNKS}" \
  --top-k "${TOP_K}" \
  --number-date-alpha "${HYBRID_ALPHA}" \
  --entity-alpha "${HYBRID_ALPHA}" \
  --relation-alpha "${HYBRID_ALPHA}" \
  --general-alpha "${HYBRID_ALPHA}"

retrieval_path_for_method() {
  case "$1" in
    bm25) echo "${BM25_RETRIEVAL}" ;;
    embedding) echo "${EMBEDDING_RETRIEVAL}" ;;
    hybrid) echo "${HYBRID_RETRIEVAL}" ;;
    rael) echo "${RAEL_RETRIEVAL}" ;;
    rael_no_coverage) echo "${RAEL_NO_COVERAGE_RETRIEVAL}" ;;
    rael_no_expansion) echo "${RAEL_NO_EXPANSION_RETRIEVAL}" ;;
    rael_no_risk_weighting) echo "${RAEL_NO_RISK_WEIGHTING_RETRIEVAL}" ;;
    *)
      echo "unsupported method: $1" >&2
      exit 1
      ;;
  esac
}

IFS=' ' read -r -a method_names <<< "${METHODS}"

for method in "${method_names[@]}"; do
  retrieval_path="$(retrieval_path_for_method "${method}")"
  verification_path="${RUN_DIR}/verification.${method}.jsonl"
  evaluation_records_path="${RUN_DIR}/evaluation.${method}.jsonl"
  evaluation_summary_path="${RUN_DIR}/evaluation.${method}.summary.json"

  run_cmd "${PYTHON_BIN}" -m src.verify_nli \
    --sentences "${SENTENCES}" \
    --retrieval "${retrieval_path}" \
    --output "${verification_path}" \
    --model-name "${NLI_MODEL}" \
    --device "${NLI_DEVICE}" \
    --max-length "${NLI_MAX_LENGTH}" \
    --top-k "${TOP_K}" \
    --aggregation "${NLI_AGGREGATION}" \
    --entailment-threshold "${NLI_ENTAILMENT_THRESHOLD}" \
    --contradiction-threshold "${NLI_CONTRADICTION_THRESHOLD}" \
    --method "$(method_label "${method}")" \
    --retrieval-method "$(retrieval_method_name "${method}")" \
    "${nli_local_args[@]}" \
    "${nli_limit_args[@]}"

  run_cmd "${PYTHON_BIN}" -m src.evaluate_summary \
    --summaries "${SUMMARIES}" \
    --documents "${DOCUMENTS}" \
    --verification-before "${verification_path}" \
    --output-records "${evaluation_records_path}" \
    --output-summary "${evaluation_summary_path}" \
    "${summary_field_args[@]}"

  if is_true "${EXPORT_CASES}"; then
    run_cmd "${PYTHON_BIN}" -m src.export_cases \
      --verification-before "${verification_path}" \
      --evaluation "${evaluation_records_path}" \
      --output-jsonl "${RUN_DIR}/cases.${method}.jsonl" \
      --output-markdown "${RUN_DIR}/cases.${method}.md" \
      --max-cases 10
  fi
done

"${PYTHON_BIN}" - "${RUN_DIR}" "${METHODS}" "${COMPARISON_TSV}" "${COMPARISON_MD}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
methods = sys.argv[2].split()
tsv_path = Path(sys.argv[3])
md_path = Path(sys.argv[4])

columns = [
    "method",
    "count",
    "rouge1",
    "rouge2",
    "rougeL",
    "supported_ratio",
    "unsupported_or_nei_ratio",
    "risk_count",
    "total_sentences",
]

rows = []
for method in methods:
    path = run_dir / f"evaluation.{method}.summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("by_version", {}).get("before_correction", {})
    row = {"method": method}
    row.update({column: metrics.get(column) for column in columns if column != "method"})
    rows.append(row)

tsv_lines = ["\t".join(columns)]
for row in rows:
    tsv_lines.append("\t".join("" if row.get(column) is None else str(row.get(column)) for column in columns))
tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

md_lines = [
    "| Method | Count | ROUGE-1 | ROUGE-2 | ROUGE-L | Supported Ratio | Unsupported/NEI Ratio | Risk Count | Total Sentences |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    md_lines.append(
        "| "
        + " | ".join(
            "" if row.get(column) is None else str(row.get(column))
            for column in columns
        )
        + " |"
    )
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
PY

echo "retrieval comparison complete: ${RUN_DIR}"
echo "comparison table: ${COMPARISON_MD}"
