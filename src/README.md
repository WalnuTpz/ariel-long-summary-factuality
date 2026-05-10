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
- `retrieval_rael.py` (planned): risk-aware adaptive evidence localization
  with dynamic fusion, coverage reranking, and evidence expansion.
- `verification_schema.py`: prepare unified verification inputs and result
  records from summary sentence queries.
- `verify_nli.py`: run sentence-level NLI verification with retrieved
  evidence. The final experiment should use a course-recommended NLI model
  such as `facebook/bart-large-mnli`.

Later batches will add summarization, summary sentence splitting, local
correction, ROUGE/factuality evaluation, and case export modules.
