#!/usr/bin/env bash
set -euo pipefail

# Controlled summary-length experiment.
#
# The same GovReport documents are summarized at three target lengths and then
# evaluated with the same ARIEL + NLI verification pipeline. Correction is not
# run because this experiment isolates the effect of summary length.
#
# Typical run:
#   CONFIG=configs/controlled_length_100_gpu.env bash scripts/run_controlled_length_experiment.sh

CONFIG="${CONFIG:-}"
if [[ -n "${CONFIG}" ]]; then
  if [[ ! -f "${CONFIG}" ]]; then
    echo "missing config file: ${CONFIG}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONFIG}"
fi

run_cmd() {
  echo "+ $*" >&2
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    exit 1
  fi
}

PYTHON_BIN="${PYTHON:-python3}"
CONTROLLED_RUN_NAME="${CONTROLLED_RUN_NAME:-controlled_length_100_wide_gpu}"
RUN_ROOT="${RUN_ROOT:-data/results/${CONTROLLED_RUN_NAME}}"
SOURCE_DOCUMENTS="${SOURCE_DOCUMENTS:-data/processed/documents_main_600.jsonl}"
DOCUMENTS="${DOCUMENTS:-data/processed/documents_controlled_length_100.jsonl}"
CONTROLLED_LIMIT="${CONTROLLED_LIMIT:-100}"
LENGTH_SPECS="${LENGTH_SPECS:-short:80:128:160 medium:240:256:384 long:520:640:900}"
SUMMARY_TARGET_WORD_TOLERANCE="${SUMMARY_TARGET_WORD_TOLERANCE:-0.15}"
FORCE_DOCUMENT_SUBSET="${FORCE_DOCUMENT_SUBSET:-0}"
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
SUMMARY_CHUNK_INPUT_TOKENS="${SUMMARY_CHUNK_INPUT_TOKENS:-6000}"
SUMMARY_CHUNK_OVERLAP_TOKENS="${SUMMARY_CHUNK_OVERLAP_TOKENS:-256}"

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

OUTPUT_DIR="${OUTPUT_DIR:-outputs/tables/${CONTROLLED_RUN_NAME}}"

require_file "${SOURCE_DOCUMENTS}"
mkdir -p "$(dirname "${DOCUMENTS}")" "${RUN_ROOT}" "${OUTPUT_DIR}"

if [[ "${SOURCE_DOCUMENTS}" != "${DOCUMENTS}" ]]; then
  if [[ ! -s "${DOCUMENTS}" || "${FORCE_DOCUMENT_SUBSET}" == "1" ]]; then
    run_cmd head -n "${CONTROLLED_LIMIT}" "${SOURCE_DOCUMENTS}" > "${DOCUMENTS}"
  fi
fi
require_file "${DOCUMENTS}"

for spec in ${LENGTH_SPECS}; do
  IFS=':' read -r label target_words chunk_max_new final_max_new <<< "${spec}"
  if [[ -z "${label}" || -z "${target_words}" || -z "${chunk_max_new}" || -z "${final_max_new}" ]]; then
    echo "invalid length spec: ${spec}" >&2
    exit 1
  fi

  run_dir="${RUN_ROOT}/${label}"
  run_cmd env \
    PYTHON="${PYTHON_BIN}" \
    RUN_NAME="${CONTROLLED_RUN_NAME}_${label}" \
    RUN_DIR="${run_dir}" \
    DOCUMENTS="${DOCUMENTS}" \
    SUMMARY_FIELD="generated_summary" \
    RETRIEVAL_METHOD="ariel" \
    RUN_SUMMARIZE="1" \
    RUN_CORRECTION="0" \
    RUN_AFTER_VERIFY="0" \
    RUN_CASE_EXPORT="0" \
    FORCE="${FORCE}" \
    TOP_K="${TOP_K}" \
    CHUNK_TYPE="${CHUNK_TYPE}" \
    CHUNK_MAX_TOKENS="${CHUNK_MAX_TOKENS}" \
    CHUNK_STRIDE="${CHUNK_STRIDE}" \
    HYBRID_ALPHA="${HYBRID_ALPHA}" \
    HF_LOCAL_FILES_ONLY="${HF_LOCAL_FILES_ONLY}" \
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
    SUMMARY_MODEL="${SUMMARY_MODEL}" \
    SUMMARY_DEVICE="${SUMMARY_DEVICE}" \
    SUMMARY_DTYPE="${SUMMARY_DTYPE}" \
    SUMMARY_TARGET_WORDS="${target_words}" \
    SUMMARY_TARGET_WORD_TOLERANCE="${SUMMARY_TARGET_WORD_TOLERANCE}" \
    SUMMARY_CHUNK_INPUT_TOKENS="${SUMMARY_CHUNK_INPUT_TOKENS}" \
    SUMMARY_CHUNK_OVERLAP_TOKENS="${SUMMARY_CHUNK_OVERLAP_TOKENS}" \
    SUMMARY_CHUNK_MAX_NEW_TOKENS="${chunk_max_new}" \
    SUMMARY_FINAL_MAX_NEW_TOKENS="${final_max_new}" \
    EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    EMBEDDING_DEVICE="${EMBEDDING_DEVICE}" \
    EMBEDDING_LOCAL_FILES_ONLY="${EMBEDDING_LOCAL_FILES_ONLY}" \
    NLI_MODEL="${NLI_MODEL}" \
    NLI_DEVICE="${NLI_DEVICE}" \
    NLI_MAX_LENGTH="${NLI_MAX_LENGTH}" \
    NLI_AGGREGATION="${NLI_AGGREGATION}" \
    NLI_ENTAILMENT_THRESHOLD="${NLI_ENTAILMENT_THRESHOLD}" \
    NLI_CONTRADICTION_THRESHOLD="${NLI_CONTRADICTION_THRESHOLD}" \
    NLI_LOCAL_FILES_ONLY="${NLI_LOCAL_FILES_ONLY}" \
    bash scripts/run_pipeline.sh
done

run_cmd "${PYTHON_BIN}" scripts/summarize_controlled_length_results.py \
  --run-root "${RUN_ROOT}" \
  --output-markdown "${OUTPUT_DIR}/controlled_length.md" \
  --output-json "${OUTPUT_DIR}/controlled_length.json"

echo "controlled length experiment complete: ${RUN_ROOT}"
echo "table: ${OUTPUT_DIR}/controlled_length.md"
