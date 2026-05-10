# Source Code

This directory contains the project implementation.

Current batch:

- `utils.py`: dependency-free JSONL, text, and normalization helpers.
- `load_data.py`: normalize raw JSONL data into the project document schema.
- `chunking.py`: split normalized documents into paragraph or token-window chunks.
- `retrieval_bm25.py`: run dependency-free BM25 retrieval over document chunks.
- `retrieval_embedding.py`: run sentence-transformers embedding retrieval over
  document chunks.
- `retrieval_hybrid.py`: fuse BM25 and embedding retrieval with min-max
  normalized scores.
- `retrieval_rael.py`: risk-aware adaptive evidence localization with
  sentence risk classification, dynamic BM25/embedding fusion, and coverage
  reranking plus adaptive adjacent-context expansion.
- `split_summary_sentences.py`: split generated summaries into sentence-level
  query records for retrieval and NLI.
- `verification_schema.py`: prepare unified verification inputs and result
  records from summary sentence queries.
- `verify_nli.py`: run sentence-level NLI verification with retrieved
  evidence. It supports concatenated evidence and per-chunk aggregation; the
  final experiment should use a course-recommended NLI model such as
  `facebook/bart-large-mnli`.

Planned modules:

- `summarize.py`: generate long-document summaries.
- `correct_local.py`: perform evidence-constrained local correction.
- `evaluate_summary.py`: compute ROUGE and factuality metrics.
- `export_cases.py`: export success/failure cases for the report.
