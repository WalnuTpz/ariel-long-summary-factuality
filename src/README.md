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

Later batches will add LLM calls, NLI/LLM verification, correction, and
evaluation modules.
