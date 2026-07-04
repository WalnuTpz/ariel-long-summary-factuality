# ARIEL for Long-Document Summary Factuality

This repository contains a lightweight pipeline for evidence-grounded factuality
verification and local correction in long-document summarization.

The project was originally developed as an NLP course final project and is
released here as a cleaned public implementation.

Given a source document and a generated summary, the system splits the summary
into sentence-level claims, retrieves source evidence for each claim, verifies
the claim with an NLI model, and optionally performs evidence-conditioned local
correction for high-risk sentences.

## Highlights

- Long-document summarization pipeline for GovReport-style inputs.
- Sentence-level evidence retrieval with BM25, dense embedding retrieval, fixed
  hybrid retrieval, and ARIEL.
- ARIEL evidence localization with coverage-aware reranking and adaptive
  evidence expansion.
- NLI-based factuality estimation using retrieved evidence as premise and each
  summary sentence as hypothesis.
- Conservative local correction for model-predicted high-risk sentences.
- Evaluation scripts for ROUGE, automatic factuality estimates, summary-length
  analysis, and before/after correction comparison.

## Repository Structure

```text
configs/       Environment-style configs for debug and main experiments
data/          Data format documentation; large data files are not committed
scripts/       Pipeline and experiment entry points
src/           Core implementation
requirements.txt
```

No full dataset, generated summaries, experiment outputs, model weights, paper
drafts, or presentation files are included in this public repository.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run may download HuggingFace datasets and models. GPU execution is
recommended for full summarization, NLI verification, and correction runs.

## Quick Check

The repository does not commit real debug data. The debug script downloads a
small temporary GovReport sample into `/tmp`, uses the reference summaries as
sentence-level queries, and checks data loading, chunking, retrieval, and
verification-input formatting:

```bash
bash scripts/run_debug.sh
```

## Data Preparation

Download a GovReport sample:

```bash
python scripts/download_govreport_sample.py \
  --split validation \
  --limit 600 \
  --output data/raw/govreport_main_600.jsonl
```

Normalize the downloaded JSONL file:

```bash
python -m src.load_data \
  --input data/raw/govreport_main_600.jsonl \
  --output data/processed/documents_main_600.jsonl \
  --min-document-chars 2000
```

Full experiments also require generated summaries. They can be produced with
`src.summarize` or by running `scripts/run_pipeline.sh` with `RUN_SUMMARIZE=1`
in the selected config.

## Main Commands

Compare retrieval methods and ARIEL ablations:

```bash
CONFIG=configs/main_600_compare_gpu.env bash scripts/run_compare_retrieval.sh
```

Run ARIEL verification and local correction:

```bash
CONFIG=configs/main_600_pipeline_gpu.env bash scripts/run_pipeline.sh
```

Generate tables from experiment outputs:

```bash
CONFIG=configs/main_600_tables_gpu.env bash scripts/make_tables.sh
```

Run the controlled summary-length experiment:

```bash
CONFIG=configs/controlled_length_100_gpu.env \
bash scripts/run_controlled_length_experiment.sh
```

CPU variants are also provided in `configs/`, but full runs are substantially
slower without GPU acceleration.

## Notes

- Automatic factuality labels are NLI-based estimates, not human gold labels.
- ROUGE measures reference overlap and should not be interpreted as factuality.
- Generated data and experiment outputs are intentionally ignored by git.
- Config paths can be edited to match local dataset and model locations.
