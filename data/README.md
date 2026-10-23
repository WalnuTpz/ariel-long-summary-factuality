# Data Directory

Large raw data files and generated experiment outputs should not be committed.

Recommended layout:

```text
raw/          Original GovReport files or downloaded samples
processed/    Normalized documents, generated summaries, sentence units, chunks
results/      Retrieval, NLI verification, ROUGE, correction, and analysis outputs
```

## Document JSONL

```json
{
  "doc_id": "govreport_00001",
  "title": "Optional document title",
  "document": "Full source document text...",
  "reference_summary": "Dataset reference summary...",
  "meta": {
    "dataset": "ccdv/govreport-summarization",
    "split": "validation"
  }
}
```

## Generated Summary JSONL

Generated summaries are produced by a long-document summarization model and
used as the input to sentence-level factuality evaluation.

```json
{
  "doc_id": "govreport_00001",
  "summary_id": "govreport_00001_qwen",
  "generated_summary": "Model-generated summary...",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "generation_method": "chunk_summarize_then_merge",
  "summary_word_count": 230,
  "meta": {}
}
```

## Summary Sentence JSONL

Sentence-level NLI uses generated summary sentences as hypotheses. Labels in
these files are automatic verifier predictions rather than human factuality
annotations.

```json
{
  "query_id": "govreport_00001::govreport_00001_qwen::s001",
  "doc_id": "govreport_00001",
  "summary_id": "govreport_00001_qwen",
  "sentence_id": 1,
  "text": "The report found that program costs increased in 2021.",
  "sentence_word_count": 11
}
```

The retrieval modules use `query_id` as the query identifier and `text` as the
retrieval query. IDs include both `doc_id` and `summary_id`:
`<doc_id>::<summary_id>::s<sentence_id>`. The sentence number is its position in
the original summary, so filtered short sentences can leave gaps. Corrected
records retain those positions through `original_summary` and `sentence_edits`;
a short replacement is still verified under its original query ID.

Regenerate sentence, retrieval, verification, and evaluation files together when
switching from the older document-only query IDs. Existing generated summaries
can be reused.

## Chunk JSONL

```json
{
  "doc_id": "govreport_00001",
  "chunk_id": "govreport_00001_chunk_0000",
  "chunk_index": 0,
  "text": "Evidence chunk text...",
  "chunk_type": "paragraph",
  "token_count": 128,
  "char_start": 0,
  "char_end": 500,
  "meta": {}
}
```

## Retrieval JSONL

BM25, embedding, fixed hybrid, and ARIEL retrieval share the same output shape.
The query is a generated summary sentence.

```json
{
  "query_id": "govreport_00001::govreport_00001_qwen::s001",
  "doc_id": "govreport_00001",
  "query": "The report found that program costs increased in 2021.",
  "retrieval_method": "bm25|embedding|hybrid|ariel",
  "top_k": [
    {
      "rank": 1,
      "chunk_id": "govreport_00001_chunk_0000",
      "chunk_index": 0,
      "score": 3.42,
      "text": "Evidence chunk text..."
    }
  ]
}
```

For ARIEL outputs, each top-k item may additionally include coverage features:

```json
{
  "entity_coverage": 0.0,
  "number_date_coverage": 0.0,
  "risk_type": "number_date|entity|relation|general",
  "expanded_context": false
}
```

## Sentence-Level Verification JSONL

```json
{
  "query_id": "govreport_00001::govreport_00001_qwen::s001",
  "doc_id": "govreport_00001",
  "summary_id": "govreport_00001_qwen",
  "sentence_id": 1,
  "sentence": "The report found that program costs increased in 2021.",
  "retrieval_method": "hybrid",
  "verifier": "nli",
  "method": "Hybrid + NLI",
  "evidence": ["chunk text"],
  "evidence_ids": ["govreport_00001_chunk_0000"],
  "label": "Supported|Unsupported|Not Enough Information",
  "reason": "NLI predicted entailment with score 0.921.",
  "model": "facebook/bart-large-mnli",
  "scores": {}
}
```

`Unsupported` and `Not Enough Information` are both treated as model-predicted
factual-risk sentences in before/after correction analysis.

## Correction JSONL

Correction rewrites only Unsupported sentences and high-confidence Not Enough
Information sentences. It should not rewrite the whole summary freely. A rewrite
must contain one sentence; invalid output keeps the original sentence. Unselected text, including short
sentences excluded from verification, remains in the corrected summary.

```json
{
  "doc_id": "govreport_00001",
  "summary_id": "govreport_00001_qwen",
  "method": "evidence_constrained_local_correction",
  "original_summary": "Summary before correction...",
  "corrected_summary": "Summary after correction...",
  "sentence_edits": [
    {
      "sentence_id": 1,
      "original_sentence": "Original sentence...",
      "corrected_sentence": "Corrected sentence...",
      "verification_label": "Unsupported",
      "evidence_chunk_ids": ["govreport_00001_chunk_0000"],
      "selected_evidence_id": "govreport_00001_chunk_0000",
      "action": "replace|keep",
      "fallback_reason": "optional reason when the original sentence is kept"
    }
  ],
  "model": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

Current correction uses evidence-conditioned sentence rewriting only. The model
is asked to output one corrected sentence; the program records `replace` only
when the text actually changes and records `keep` for low-confidence NEI
sentences, unchanged rewrites, or fallback cases. Supported sentences are
copied unchanged without adding a `sentence_edits` entry.

## Evaluation JSONL

```json
{
  "doc_id": "govreport_00001",
  "summary_id": "govreport_00001_qwen",
  "summary_version": "before_correction|after_correction",
  "rouge1": 0.0,
  "rouge2": 0.0,
  "rougeL": 0.0,
  "supported_ratio": 0.0,
  "unsupported_or_nei_ratio": 0.0,
  "summary_word_count": 230,
  "length_bucket": "short|medium|long",
  "length_bucket_strategy": "fixed"
}
```

The default length bucket strategy uses fixed thresholds on original generated
summary length: short <= 150 words, medium <= 225 words, and long > 225 words.
