#!/usr/bin/env bash
set -euo pipefail

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

echo "project skeleton ok"
