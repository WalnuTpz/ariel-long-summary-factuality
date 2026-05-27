# Source Code

This directory contains the project implementation.

Current modules:

- `utils.py`: dependency-free JSONL, text, and normalization helpers.
- `load_data.py`: normalize raw JSONL data into the project document schema.
- `chunking.py`: split normalized documents into paragraph or token-window chunks.
- `retrieval_bm25.py`: run dependency-free BM25 retrieval over document chunks.
- `retrieval_embedding.py`: run sentence-transformers embedding retrieval over
  document chunks.
- `retrieval_hybrid.py`: fuse BM25 and embedding retrieval with min-max
  normalized scores.
- `retrieval_ariel.py`: adaptive risk-informed evidence localization with
  sentence risk classification, lightweight risk-guided BM25/embedding
  fusion, coverage-aware reranking, and adaptive adjacent-context expansion.
  The final analysis treats coverage reranking and expansion as the primary
  contributors, with risk weighting as an auxiliary signal.
- `summarize.py`: generate long-document summaries with a local HuggingFace
  causal language model using section summaries plus final summary merging.
- `split_summary_sentences.py`: split generated summaries into sentence-level
  query records for retrieval and NLI.
- `verification_schema.py`: prepare unified verification inputs and result
  records from summary sentence queries.
- `verify_nli.py`: run sentence-level NLI verification with retrieved
  evidence. It supports concatenated evidence and per-chunk aggregation; the
  final experiment should use a course-recommended NLI model such as
  `facebook/bart-large-mnli`.
- `correct_local.py`: perform evidence-conditioned local rewrite for
  Unsupported summary sentences and high-confidence Not Enough Information
  sentences, then reconstruct the corrected summary. It records `replace`
  only when the corrected sentence actually changes; otherwise it records
  `keep` with a reason.
- `evaluate_summary.py`: compute dependency-free ROUGE-1/2/L, sentence-level
  automatic NLI factuality estimates, length ratios, error reduction, and
  length-bucket aggregate statistics.
- `export_cases.py`: export representative success/failure factuality and
  correction cases as JSONL and Markdown for the report and slides.
