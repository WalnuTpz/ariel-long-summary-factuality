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

output_path="/tmp/long_summary_factuality_debug_documents.jsonl"
"${PYTHON_BIN}" -m src.load_data \
  --input examples/debug_documents.jsonl \
  --output "${output_path}" \
  --min-document-chars 100

test -s "${output_path}"
echo "project skeleton and data loading ok"
