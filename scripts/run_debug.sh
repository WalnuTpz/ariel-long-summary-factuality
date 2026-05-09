#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

required_dirs=(
  "data/raw"
  "data/processed"
  "data/annotations"
  "data/results"
  "prompts"
  "src"
  "scripts"
  "outputs/tables"
  "outputs/figures"
  "report"
)

for dir in "${required_dirs[@]}"; do
  test -d "${dir}"
done

documents_path="/tmp/long_summary_factuality_debug_documents.jsonl"
chunks_path="/tmp/long_summary_factuality_debug_chunks.jsonl"
retrieval_path="/tmp/long_summary_factuality_debug_bm25.jsonl"

"${PYTHON_BIN}" -m src.load_data \
  --input examples/debug_documents.jsonl \
  --output "${documents_path}" \
  --min-document-chars 100

"${PYTHON_BIN}" -m src.chunking \
  --input "${documents_path}" \
  --output "${chunks_path}" \
  --chunk-type paragraph \
  --max-tokens 128

"${PYTHON_BIN}" -m src.retrieval_bm25 \
  --chunks "${chunks_path}" \
  --queries examples/debug_facts.jsonl \
  --output "${retrieval_path}" \
  --top-k 2

test -s "${documents_path}"
test -s "${chunks_path}"
test -s "${retrieval_path}"
echo "project skeleton, data loading, chunking, and BM25 retrieval ok"
