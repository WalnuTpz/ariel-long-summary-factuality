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
embedding_path="/tmp/long_summary_factuality_debug_embedding.jsonl"
hybrid_path="/tmp/long_summary_factuality_debug_hybrid.jsonl"

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

"${PYTHON_BIN}" -m src.retrieval_embedding \
  --chunks "${chunks_path}" \
  --queries examples/debug_facts.jsonl \
  --output "${embedding_path}" \
  --top-k 2 \
  --device cpu \
  --local-files-only

"${PYTHON_BIN}" -m src.retrieval_hybrid \
  --bm25 "${retrieval_path}" \
  --embedding "${embedding_path}" \
  --output "${hybrid_path}" \
  --alpha 0.5 \
  --top-k 2

test -s "${documents_path}"
test -s "${chunks_path}"
test -s "${retrieval_path}"
test -s "${embedding_path}"
test -s "${hybrid_path}"
echo "project skeleton, data loading, chunking, BM25, embedding, and hybrid retrieval ok"
