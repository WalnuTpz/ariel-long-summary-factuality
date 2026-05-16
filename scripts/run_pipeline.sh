#!/usr/bin/env bash
set -euo pipefail

# End-to-end pipeline for one retrieval method.
#
# Typical debug run:
#   PYTHON=.venv/bin/python RUN_NAME=debug_5 bash scripts/run_pipeline.sh
#
# Typical main run:
#   PYTHON=.venv/bin/python \
#   RUN_NAME=main_500 \
#   DOCUMENTS=data/processed/documents_main_500.jsonl \
#   SUMMARIES=data/processed/generated_summaries.qwen2p5_1p5b.jsonl \
#   RETRIEVAL_METHOD=ariel \
#   bash scripts/run_pipeline.sh

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
    ariel) echo "ARIEL + NLI" ;;
    *) echo "$1 + NLI" ;;
  esac
}

PYTHON_BIN="${PYTHON:-python3}"
RUN_NAME="${RUN_NAME:-debug_pipeline}"
RUN_DIR="${RUN_DIR:-data/results/${RUN_NAME}}"
DOCUMENTS="${DOCUMENTS:-data/processed/documents_debug_5.jsonl}"
SUMMARIES="${SUMMARIES:-data/processed/summaries_debug_5.jsonl}"
SUMMARY_FIELD="${SUMMARY_FIELD:-}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-ariel}"

RUN_SUMMARIZE="${RUN_SUMMARIZE:-0}"
RUN_CORRECTION="${RUN_CORRECTION:-1}"
RUN_AFTER_VERIFY="${RUN_AFTER_VERIFY:-1}"
RUN_CASE_EXPORT="${RUN_CASE_EXPORT:-1}"
FORCE="${FORCE:-1}"

TOP_K="${TOP_K:-3}"
CHUNK_TYPE="${CHUNK_TYPE:-paragraph}"
CHUNK_MAX_TOKENS="${CHUNK_MAX_TOKENS:-256}"
CHUNK_STRIDE="${CHUNK_STRIDE:-}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"

HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

SUMMARY_MODEL="${SUMMARY_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SUMMARY_DEVICE="${SUMMARY_DEVICE:-auto}"
SUMMARY_DTYPE="${SUMMARY_DTYPE:-auto}"
SUMMARY_TARGET_WORDS="${SUMMARY_TARGET_WORDS:-240}"
SUMMARY_CHUNK_INPUT_TOKENS="${SUMMARY_CHUNK_INPUT_TOKENS:-6000}"
SUMMARY_CHUNK_OVERLAP_TOKENS="${SUMMARY_CHUNK_OVERLAP_TOKENS:-256}"
SUMMARY_CHUNK_MAX_NEW_TOKENS="${SUMMARY_CHUNK_MAX_NEW_TOKENS:-256}"
SUMMARY_FINAL_MAX_NEW_TOKENS="${SUMMARY_FINAL_MAX_NEW_TOKENS:-384}"

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

CORRECTION_MODEL="${CORRECTION_MODEL:-${SUMMARY_MODEL}}"
CORRECTION_DEVICE="${CORRECTION_DEVICE:-${SUMMARY_DEVICE}}"
CORRECTION_DTYPE="${CORRECTION_DTYPE:-${SUMMARY_DTYPE}}"
CORRECTION_LOCAL_FILES_ONLY="${CORRECTION_LOCAL_FILES_ONLY:-${HF_LOCAL_FILES_ONLY}}"
CORRECTION_MAX_PROMPT_TOKENS="${CORRECTION_MAX_PROMPT_TOKENS:-4096}"
CORRECTION_MAX_NEW_TOKENS="${CORRECTION_MAX_NEW_TOKENS:-128}"
CORRECTION_NEUTRAL_REWRITE_THRESHOLD="${CORRECTION_NEUTRAL_REWRITE_THRESHOLD:-0.7}"
CORRECTION_MAX_EVIDENCE_SENTENCES="${CORRECTION_MAX_EVIDENCE_SENTENCES:-5}"
CORRECTION_FALLBACK_ACTION="${CORRECTION_FALLBACK_ACTION:-keep}"
CORRECTION_CONTINUE_ON_ERROR="${CORRECTION_CONTINUE_ON_ERROR:-1}"

mkdir -p "${RUN_DIR}"

summary_field_args=()
if [[ -n "${SUMMARY_FIELD}" ]]; then
  summary_field_args+=(--summary-field "${SUMMARY_FIELD}")
fi

chunk_stride_args=()
if [[ -n "${CHUNK_STRIDE}" ]]; then
  chunk_stride_args+=(--stride "${CHUNK_STRIDE}")
fi

hf_local_args=()
if is_true "${HF_LOCAL_FILES_ONLY}"; then
  hf_local_args+=(--local-files-only)
fi

trust_remote_code_args=()
if is_true "${TRUST_REMOTE_CODE}"; then
  trust_remote_code_args+=(--trust-remote-code)
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

correction_local_args=()
if is_true "${CORRECTION_LOCAL_FILES_ONLY}"; then
  correction_local_args+=(--local-files-only)
fi

correction_error_args=()
if is_true "${CORRECTION_CONTINUE_ON_ERROR}"; then
  correction_error_args+=(--continue-on-error)
fi

ACTIVE_SUMMARIES="${SUMMARIES}"
if is_true "${RUN_SUMMARIZE}"; then
  ACTIVE_SUMMARIES="${RUN_DIR}/generated_summaries.jsonl"
  run_cmd "${PYTHON_BIN}" -m src.summarize \
    --input "${DOCUMENTS}" \
    --output "${ACTIVE_SUMMARIES}" \
    --model-name "${SUMMARY_MODEL}" \
    --device "${SUMMARY_DEVICE}" \
    --dtype "${SUMMARY_DTYPE}" \
    --target-words "${SUMMARY_TARGET_WORDS}" \
    --chunk-input-tokens "${SUMMARY_CHUNK_INPUT_TOKENS}" \
    --chunk-overlap-tokens "${SUMMARY_CHUNK_OVERLAP_TOKENS}" \
    --chunk-max-new-tokens "${SUMMARY_CHUNK_MAX_NEW_TOKENS}" \
    --final-max-new-tokens "${SUMMARY_FINAL_MAX_NEW_TOKENS}" \
    --resume \
    --continue-on-error \
    "${hf_local_args[@]}" \
    "${trust_remote_code_args[@]}"
else
  require_file "${ACTIVE_SUMMARIES}"
fi
require_file "${DOCUMENTS}"

CHUNKS="${CHUNKS:-${RUN_DIR}/chunks.jsonl}"
SENTENCES_BEFORE="${RUN_DIR}/summary_sentences.before.jsonl"
RETRIEVAL_BEFORE="${RUN_DIR}/retrieval.before.${RETRIEVAL_METHOD}.jsonl"
VERIFY_BEFORE="${RUN_DIR}/verification.before.${RETRIEVAL_METHOD}.jsonl"
CORRECTIONS="${RUN_DIR}/corrections.${RETRIEVAL_METHOD}.jsonl"
SENTENCES_AFTER="${RUN_DIR}/summary_sentences.after.jsonl"
RETRIEVAL_AFTER="${RUN_DIR}/retrieval.after.${RETRIEVAL_METHOD}.jsonl"
VERIFY_AFTER="${RUN_DIR}/verification.after.${RETRIEVAL_METHOD}.jsonl"
EVAL_RECORDS="${RUN_DIR}/evaluation.${RETRIEVAL_METHOD}.jsonl"
EVAL_SUMMARY="${RUN_DIR}/evaluation.${RETRIEVAL_METHOD}.summary.json"
CASES_JSONL="${RUN_DIR}/cases.${RETRIEVAL_METHOD}.jsonl"
CASES_MD="${RUN_DIR}/cases.${RETRIEVAL_METHOD}.md"

if is_true "${FORCE}" || [[ ! -s "${CHUNKS}" ]]; then
  run_cmd "${PYTHON_BIN}" -m src.chunking \
    --input "${DOCUMENTS}" \
    --output "${CHUNKS}" \
    --chunk-type "${CHUNK_TYPE}" \
    --max-tokens "${CHUNK_MAX_TOKENS}" \
    "${chunk_stride_args[@]}"
fi

run_cmd "${PYTHON_BIN}" -m src.split_summary_sentences \
  --input "${ACTIVE_SUMMARIES}" \
  --output "${SENTENCES_BEFORE}" \
  "${summary_field_args[@]}"

run_retrieval() {
  local stage="$1"
  local method="$2"
  local queries="$3"
  local output="$4"
  local bm25_path="${RUN_DIR}/retrieval.${stage}.bm25.jsonl"
  local embedding_path="${RUN_DIR}/retrieval.${stage}.embedding.jsonl"

  case "${method}" in
    bm25)
      run_cmd "${PYTHON_BIN}" -m src.retrieval_bm25 \
        --chunks "${CHUNKS}" \
        --queries "${queries}" \
        --output "${output}" \
        --top-k "${TOP_K}"
      ;;
    embedding)
      run_cmd "${PYTHON_BIN}" -m src.retrieval_embedding \
        --chunks "${CHUNKS}" \
        --queries "${queries}" \
        --output "${output}" \
        --top-k "${TOP_K}" \
        --model-name "${EMBEDDING_MODEL}" \
        --device "${EMBEDDING_DEVICE}" \
        "${embedding_local_args[@]}"
      ;;
    hybrid|ariel)
      run_cmd "${PYTHON_BIN}" -m src.retrieval_bm25 \
        --chunks "${CHUNKS}" \
        --queries "${queries}" \
        --output "${bm25_path}" \
        --top-k "${TOP_K}"
      run_cmd "${PYTHON_BIN}" -m src.retrieval_embedding \
        --chunks "${CHUNKS}" \
        --queries "${queries}" \
        --output "${embedding_path}" \
        --top-k "${TOP_K}" \
        --model-name "${EMBEDDING_MODEL}" \
        --device "${EMBEDDING_DEVICE}" \
        "${embedding_local_args[@]}"
      if [[ "${method}" == "hybrid" ]]; then
        run_cmd "${PYTHON_BIN}" -m src.retrieval_hybrid \
          --bm25 "${bm25_path}" \
          --embedding "${embedding_path}" \
          --output "${output}" \
          --alpha "${HYBRID_ALPHA}" \
          --top-k "${TOP_K}"
      else
        run_cmd "${PYTHON_BIN}" -m src.retrieval_ariel \
          --bm25 "${bm25_path}" \
          --embedding "${embedding_path}" \
          --output "${output}" \
          --queries "${queries}" \
          --chunks "${CHUNKS}" \
          --top-k "${TOP_K}"
      fi
      ;;
    *)
      echo "unsupported RETRIEVAL_METHOD: ${method}" >&2
      exit 1
      ;;
  esac
}

run_verification() {
  local stage="$1"
  local queries="$2"
  local retrieval="$3"
  local output="$4"
  local method="$5"

  run_cmd "${PYTHON_BIN}" -m src.verify_nli \
    --sentences "${queries}" \
    --retrieval "${retrieval}" \
    --output "${output}" \
    --model-name "${NLI_MODEL}" \
    --device "${NLI_DEVICE}" \
    --max-length "${NLI_MAX_LENGTH}" \
    --top-k "${TOP_K}" \
    --aggregation "${NLI_AGGREGATION}" \
    --entailment-threshold "${NLI_ENTAILMENT_THRESHOLD}" \
    --contradiction-threshold "${NLI_CONTRADICTION_THRESHOLD}" \
    --method "$(method_label "${method}")" \
    --retrieval-method "${method}" \
    "${nli_local_args[@]}" \
    "${nli_limit_args[@]}"
}

run_retrieval before "${RETRIEVAL_METHOD}" "${SENTENCES_BEFORE}" "${RETRIEVAL_BEFORE}"
run_verification before "${SENTENCES_BEFORE}" "${RETRIEVAL_BEFORE}" "${VERIFY_BEFORE}" "${RETRIEVAL_METHOD}"

eval_args=(
  --summaries "${ACTIVE_SUMMARIES}"
  --documents "${DOCUMENTS}"
  --verification-before "${VERIFY_BEFORE}"
  --output-records "${EVAL_RECORDS}"
  --output-summary "${EVAL_SUMMARY}"
  "${summary_field_args[@]}"
)

if is_true "${RUN_CORRECTION}"; then
  run_cmd "${PYTHON_BIN}" -m src.correct_local \
    --summaries "${ACTIVE_SUMMARIES}" \
    --sentences "${SENTENCES_BEFORE}" \
    --verification "${VERIFY_BEFORE}" \
    --output "${CORRECTIONS}" \
    --model-name "${CORRECTION_MODEL}" \
    --device "${CORRECTION_DEVICE}" \
    --dtype "${CORRECTION_DTYPE}" \
    --max-prompt-tokens "${CORRECTION_MAX_PROMPT_TOKENS}" \
    --max-new-tokens "${CORRECTION_MAX_NEW_TOKENS}" \
    --neutral-rewrite-threshold "${CORRECTION_NEUTRAL_REWRITE_THRESHOLD}" \
    --max-evidence-sentences "${CORRECTION_MAX_EVIDENCE_SENTENCES}" \
    --fallback-action "${CORRECTION_FALLBACK_ACTION}" \
    "${summary_field_args[@]}" \
    "${correction_local_args[@]}" \
    "${trust_remote_code_args[@]}" \
    "${correction_error_args[@]}"

  eval_args+=(--corrections "${CORRECTIONS}")

  if is_true "${RUN_AFTER_VERIFY}"; then
    run_cmd "${PYTHON_BIN}" -m src.split_summary_sentences \
      --input "${CORRECTIONS}" \
      --output "${SENTENCES_AFTER}" \
      --summary-field corrected_summary

    run_retrieval after "${RETRIEVAL_METHOD}" "${SENTENCES_AFTER}" "${RETRIEVAL_AFTER}"
    run_verification after "${SENTENCES_AFTER}" "${RETRIEVAL_AFTER}" "${VERIFY_AFTER}" "${RETRIEVAL_METHOD}"
    eval_args+=(--verification-after "${VERIFY_AFTER}")
  fi
fi

run_cmd "${PYTHON_BIN}" -m src.evaluate_summary "${eval_args[@]}"

if is_true "${RUN_CASE_EXPORT}"; then
  case_args=(
    --verification-before "${VERIFY_BEFORE}"
    --evaluation "${EVAL_RECORDS}"
    --output-jsonl "${CASES_JSONL}"
    --output-markdown "${CASES_MD}"
    --max-cases 10
  )
  if [[ -s "${CORRECTIONS}" ]]; then
    case_args+=(--corrections "${CORRECTIONS}")
  fi
  if [[ -s "${VERIFY_AFTER}" ]]; then
    case_args+=(--verification-after "${VERIFY_AFTER}")
  fi
  run_cmd "${PYTHON_BIN}" -m src.export_cases "${case_args[@]}"
fi

echo "pipeline complete: ${RUN_DIR}"
