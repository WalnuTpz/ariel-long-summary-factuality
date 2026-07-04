#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

required_dirs=(
  "src"
  "scripts"
)

for dir in "${required_dirs[@]}"; do
  test -d "${dir}"
done

raw_path="/tmp/long_summary_factuality_debug_raw.jsonl"
documents_path="/tmp/long_summary_factuality_debug_documents.jsonl"
sentences_path="/tmp/long_summary_factuality_debug_sentences.jsonl"
chunks_path="/tmp/long_summary_factuality_debug_chunks.jsonl"
retrieval_path="/tmp/long_summary_factuality_debug_bm25.jsonl"
embedding_path="/tmp/long_summary_factuality_debug_embedding.jsonl"
hybrid_path="/tmp/long_summary_factuality_debug_hybrid.jsonl"
verification_inputs_path="/tmp/long_summary_factuality_debug_verification_inputs.jsonl"

"${PYTHON_BIN}" scripts/download_govreport_sample.py \
  --output "${raw_path}" \
  --split validation \
  --limit 5 \
  --min-document-chars 100

"${PYTHON_BIN}" -m src.load_data \
  --input "${raw_path}" \
  --output "${documents_path}" \
  --min-document-chars 100

"${PYTHON_BIN}" -m src.split_summary_sentences \
  --input "${documents_path}" \
  --output "${sentences_path}" \
  --summary-field reference_summary

"${PYTHON_BIN}" -m src.chunking \
  --input "${documents_path}" \
  --output "${chunks_path}" \
  --chunk-type paragraph \
  --max-tokens 128

"${PYTHON_BIN}" -m src.retrieval_bm25 \
  --chunks "${chunks_path}" \
  --queries "${sentences_path}" \
  --output "${retrieval_path}" \
  --top-k 2

"${PYTHON_BIN}" -m src.retrieval_embedding \
  --chunks "${chunks_path}" \
  --queries "${sentences_path}" \
  --output "${embedding_path}" \
  --top-k 2 \
  --device cpu

"${PYTHON_BIN}" -m src.retrieval_hybrid \
  --bm25 "${retrieval_path}" \
  --embedding "${embedding_path}" \
  --output "${hybrid_path}" \
  --alpha 0.5 \
  --top-k 2

"${PYTHON_BIN}" -m src.verification_schema \
  --sentences "${sentences_path}" \
  --retrieval "${hybrid_path}" \
  --output "${verification_inputs_path}" \
  --method "Hybrid + NLI" \
  --verifier nli \
  --retrieval-method hybrid \
  --top-k 2

test -s "${documents_path}"
test -s "${sentences_path}"
test -s "${chunks_path}"
test -s "${retrieval_path}"
test -s "${embedding_path}"
test -s "${hybrid_path}"
test -s "${verification_inputs_path}"
echo "data loading, sentence splitting, chunking, retrieval, and verification input prep ok"
