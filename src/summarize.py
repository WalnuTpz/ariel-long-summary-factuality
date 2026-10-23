"""Generate long-document summaries for the factuality pipeline.

The final project route uses a local HuggingFace causal language model. Long
GovReport documents are summarized by first summarizing token windows and then
merging the window summaries into one document-level generated summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils import ensure_parent_dir, first_present, iter_jsonl, normalize_whitespace, simple_tokenize


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DOC_ID_KEYS = ("doc_id", "document_id", "id", "sample_id")
SYSTEM_PROMPT = (
    "You are a faithful government-report summarizer. Output only plain "
    "summary text. No Markdown, bullets, headings, word counts, explanations, "
    "or chatty text."
)
PLAIN_OUTPUT_RULES = (
    "- Write plain paragraphs only.\n"
    "- Do not use Markdown, headings, bullet points, numbered lists, tables, "
    "or word counts.\n"
    "- Do not include explanations, notes, greetings, disclaimers, or offers "
    "to help."
)


@dataclass
class SummarizeStats:
    input_documents: int = 0
    written_summaries: int = 0
    skipped_missing_doc_id: int = 0
    skipped_empty_document: int = 0
    skipped_existing_doc_id: int = 0
    failed_documents: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def target_word_range(target_words: int, tolerance: float) -> tuple[int, int] | None:
    if tolerance <= 0:
        return None
    lower = max(1, int(round(target_words * (1 - tolerance))))
    upper = max(lower, int(round(target_words * (1 + tolerance))))
    return lower, upper


def length_instruction(target_words: int, tolerance: float) -> str:
    word_range = target_word_range(target_words, tolerance)
    if word_range is None:
        return f"about {target_words} words"
    lower, upper = word_range
    return f"between {lower} and {upper} words"


def length_rule(target_words: int, tolerance: float) -> str:
    word_range = target_word_range(target_words, tolerance)
    if word_range is None:
        return ""
    lower, upper = word_range
    return f"- Keep the output between {lower} and {upper} words.\n"


def clean_generated_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"^\s*(#+\s*)?(final\s+)?summary\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*\*\*(final\s+)?summary\*\*\s*[:\-]?\s*", "", text, flags=re.I)
    return normalize_whitespace(text)


def make_model_tag(model_name: str) -> str:
    tag = model_name.rstrip("/").split("/")[-1].lower()
    tag = tag.replace(".", "p")
    tag = re.sub(r"[^a-z0-9]+", "_", tag).strip("_")
    return tag or "model"


def make_summary_id(doc_id: str, model_name: str, summary_id_suffix: str | None = None) -> str:
    suffix = summary_id_suffix or make_model_tag(model_name)
    return f"{doc_id}_{suffix}"


def resolve_doc_id(item: dict[str, Any]) -> str:
    return str(first_present(item, DOC_ID_KEYS) or "").strip()


def chunk_prompt(
    title: str,
    section_text: str,
    target_words: int,
    target_word_tolerance: float,
) -> str:
    title_line = f"Title: {title}\n\n" if title else ""
    if target_word_tolerance > 0:
        section_words = max(50, min(300, target_words // 2))
    else:
        section_words = max(120, min(180, target_words // 2))
    return (
        "Summarize this report excerpt in "
        f"{length_instruction(section_words, target_word_tolerance)}.\n\n"
        "Rules:\n"
        "- Use only facts stated in the excerpt.\n"
        "- Keep key entities, dates, numbers, findings, and recommendations.\n"
        f"{length_rule(section_words, target_word_tolerance)}"
        f"{PLAIN_OUTPUT_RULES}\n\n"
        f"{title_line}"
        "Excerpt:\n"
        f"{section_text}"
    )


def single_pass_prompt(
    title: str,
    document: str,
    target_words: int,
    target_word_tolerance: float,
) -> str:
    title_line = f"Title: {title}\n\n" if title else ""
    return (
        "Summarize the government report in "
        f"{length_instruction(target_words, target_word_tolerance)}.\n\n"
        "Rules:\n"
        "- Use only facts stated in the report.\n"
        "- Keep key entities, dates, numbers, findings, and recommendations.\n"
        f"{length_rule(target_words, target_word_tolerance)}"
        f"{PLAIN_OUTPUT_RULES}\n\n"
        f"{title_line}"
        "Report:\n"
        f"{document}"
    )


def final_prompt(
    title: str,
    summary_source: str,
    target_words: int,
    target_word_tolerance: float,
) -> str:
    title_line = f"Title: {title}\n\n" if title else ""
    return (
        "Merge the intermediate summaries into one factual summary in "
        f"{length_instruction(target_words, target_word_tolerance)}.\n\n"
        "Rules:\n"
        "- Remove repetition.\n"
        "- Use only facts supported by the intermediate summaries.\n"
        "- Do not mention sections or intermediate summaries.\n"
        f"{length_rule(target_words, target_word_tolerance)}"
        f"{PLAIN_OUTPUT_RULES}\n\n"
        f"{title_line}"
        "Intermediate summaries:\n"
        f"{summary_source}"
    )


class HuggingFaceSummarizer:
    def __init__(
        self,
        model_name: str,
        device: str,
        dtype: str,
        max_prompt_tokens: int,
        local_files_only: bool,
        trust_remote_code: bool,
        do_sample: bool,
        temperature: float,
        top_p: float,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing summarization dependencies. Install transformers and torch first."
            ) from exc

        self.torch = torch
        self.device = self.resolve_device(torch, device)
        torch_dtype = self.resolve_dtype(torch, dtype, self.device)
        self.max_prompt_tokens = max_prompt_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def resolve_device(torch: Any, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def resolve_dtype(torch: Any, dtype: str, device: str) -> Any:
        if dtype == "float32":
            return torch.float32
        if dtype == "float16":
            return torch.float16
        if dtype == "bfloat16":
            return torch.bfloat16
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_text(self, text: str, chunk_input_tokens: int, chunk_overlap_tokens: int) -> list[str]:
        if chunk_input_tokens <= 0:
            raise ValueError("chunk_input_tokens must be positive.")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens cannot be negative.")
        if chunk_overlap_tokens >= chunk_input_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_input_tokens.")

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return []
        chunks: list[str] = []
        step = chunk_input_tokens - chunk_overlap_tokens
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + chunk_input_tokens]
            if not window:
                break
            chunks.append(self.tokenizer.decode(window, skip_special_tokens=True))
            if start + chunk_input_tokens >= len(token_ids):
                break
        return chunks

    def format_chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.tokenizer.chat_template is None:
            return f"{self.system_prompt}\n\n{user_prompt}\n\nAssistant:"
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, user_prompt: str, max_new_tokens: int) -> str:
        prompt_text = self.format_chat_prompt(user_prompt)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=self.tokenizer.chat_template is None,
            truncation=True,
            max_length=self.max_prompt_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return clean_generated_text(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        )


def summarize_document(
    item: dict[str, Any],
    summarizer: HuggingFaceSummarizer,
    model_name: str,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
    chunk_max_new_tokens: int,
    final_max_new_tokens: int,
    target_words: int,
    target_word_tolerance: float,
    summary_id_suffix: str | None,
    save_chunk_summaries: bool,
) -> dict[str, Any]:
    doc_id = resolve_doc_id(item)
    title = normalize_whitespace(item.get("title"))
    document = normalize_whitespace(item.get("document"))
    chunks = summarizer.split_text(
        document,
        chunk_input_tokens=chunk_input_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    chunk_summaries: list[str] = []
    if not chunks:
        raise RuntimeError(f"Document has no model-token chunks: {doc_id}.")

    if len(chunks) == 1:
        generated_summary = summarizer.generate(
            single_pass_prompt(
                title=title,
                document=chunks[0],
                target_words=target_words,
                target_word_tolerance=target_word_tolerance,
            ),
            max_new_tokens=final_max_new_tokens,
        )
        generation_method = "single_pass"
    else:
        for chunk_index, chunk_text in enumerate(chunks, start=1):
            prompt = chunk_prompt(
                title=title,
                section_text=chunk_text,
                target_words=target_words,
                target_word_tolerance=target_word_tolerance,
            )
            chunk_summary = summarizer.generate(prompt, max_new_tokens=chunk_max_new_tokens)
            if chunk_summary:
                chunk_summaries.append(f"Intermediate summary {chunk_index}: {chunk_summary}")
        if not chunk_summaries:
            raise RuntimeError(f"Model generated no section summaries for {doc_id}.")
        generated_summary = summarizer.generate(
            final_prompt(
                title=title,
                summary_source="\n\n".join(chunk_summaries),
                target_words=target_words,
                target_word_tolerance=target_word_tolerance,
            ),
            max_new_tokens=final_max_new_tokens,
        )
        generation_method = "chunk_summarize_then_merge"

    if not generated_summary:
        raise RuntimeError(f"Model generated an empty summary for {doc_id}.")

    meta: dict[str, Any] = {
        "chunk_count": len(chunks),
        "input_model_tokens": summarizer.count_tokens(document),
        "input_document_word_count": len(simple_tokenize(document)),
        "target_words": target_words,
        "target_word_tolerance": target_word_tolerance,
        "chunk_input_tokens": chunk_input_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
    }
    source_meta = item.get("meta")
    if isinstance(source_meta, dict) and source_meta:
        meta["source_meta"] = source_meta
    reference_summary = normalize_whitespace(item.get("reference_summary"))
    if reference_summary:
        meta["reference_summary_word_count"] = len(simple_tokenize(reference_summary))
    if save_chunk_summaries and chunk_summaries:
        meta["chunk_summaries"] = chunk_summaries

    record = {
        "doc_id": doc_id,
        "summary_id": make_summary_id(
            doc_id=doc_id,
            model_name=model_name,
            summary_id_suffix=summary_id_suffix,
        ),
        "generated_summary": generated_summary,
        "model": model_name,
        "generation_method": generation_method,
        "summary_word_count": len(simple_tokenize(generated_summary)),
        "meta": meta,
    }
    if title:
        record["title"] = title
    return record


def read_existing_doc_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(item.get("doc_id") or "").strip()
        for item in iter_jsonl(path)
        if str(item.get("doc_id") or "").strip()
    }


def summarize_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str,
    device: str,
    dtype: str,
    max_prompt_tokens: int,
    local_files_only: bool,
    trust_remote_code: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    chunk_input_tokens: int,
    chunk_overlap_tokens: int,
    chunk_max_new_tokens: int,
    final_max_new_tokens: int,
    target_words: int,
    target_word_tolerance: float,
    summary_id_suffix: str | None,
    save_chunk_summaries: bool,
    limit: int | None,
    resume: bool,
    continue_on_error: bool,
) -> SummarizeStats:
    stats = SummarizeStats()
    output_path = ensure_parent_dir(output_path)
    existing_doc_ids = read_existing_doc_ids(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"

    summarizer = HuggingFaceSummarizer(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_prompt_tokens=max_prompt_tokens,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    with output_path.open(mode, encoding="utf-8") as output_file:
        for item in iter_jsonl(input_path):
            stats.input_documents += 1
            if limit is not None and stats.input_documents > limit:
                break

            doc_id = resolve_doc_id(item)
            if not doc_id:
                stats.skipped_missing_doc_id += 1
                continue
            if resume and doc_id in existing_doc_ids:
                stats.skipped_existing_doc_id += 1
                continue
            if not normalize_whitespace(item.get("document")):
                stats.skipped_empty_document += 1
                continue

            try:
                record = summarize_document(
                    item=item,
                    summarizer=summarizer,
                    model_name=model_name,
                    chunk_input_tokens=chunk_input_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                    chunk_max_new_tokens=chunk_max_new_tokens,
                    final_max_new_tokens=final_max_new_tokens,
                    target_words=target_words,
                    target_word_tolerance=target_word_tolerance,
                    summary_id_suffix=summary_id_suffix,
                    save_chunk_summaries=save_chunk_summaries,
                )
            except Exception as exc:
                stats.failed_documents += 1
                if not continue_on_error:
                    raise
                print(f"failed_document: {doc_id}: {exc}", file=sys.stderr, flush=True)
                continue

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            stats.written_summaries += 1
            existing_doc_ids.add(doc_id)
            print(f"written_summary: {doc_id}", file=sys.stderr, flush=True)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate long-document summaries.")
    parser.add_argument("--input", required=True, help="Normalized document JSONL path.")
    parser.add_argument("--output", required=True, help="Generated summary JSONL path.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace causal language model name or local path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Generation device: auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype. Defaults to float16 on CUDA and float32 otherwise.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=12000,
        help="Maximum prompt tokens passed to the model.",
    )
    parser.add_argument(
        "--chunk-input-tokens",
        type=int,
        default=6000,
        help="Maximum document tokens per section summary prompt.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=256,
        help="Token overlap between adjacent document sections.",
    )
    parser.add_argument(
        "--chunk-max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for each section summary.",
    )
    parser.add_argument(
        "--final-max-new-tokens",
        type=int,
        default=384,
        help="Maximum new tokens for the final merged summary.",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=240,
        help="Approximate target length of the final summary.",
    )
    parser.add_argument(
        "--target-word-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional relative word-count tolerance used to write an explicit "
            "target range in prompts. For example, 0.15 requests 85%%-115%% of "
            "the target word count. The default keeps the original about-X "
            "prompt wording."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from local HuggingFace cache or local path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repositories with custom code.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling instead of deterministic greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature, only used with --do-sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p, only used with --do-sample.",
    )
    parser.add_argument(
        "--summary-id-suffix",
        default=None,
        help="Optional summary_id suffix. Defaults to a normalized model name.",
    )
    parser.add_argument(
        "--save-chunk-summaries",
        action="store_true",
        help="Store intermediate section summaries in meta for audit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of input documents to read.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to output and skip doc_ids already present there.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a document-level generation failure.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = summarize_jsonl(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        max_prompt_tokens=args.max_prompt_tokens,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        chunk_input_tokens=args.chunk_input_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        chunk_max_new_tokens=args.chunk_max_new_tokens,
        final_max_new_tokens=args.final_max_new_tokens,
        target_words=args.target_words,
        target_word_tolerance=args.target_word_tolerance,
        summary_id_suffix=args.summary_id_suffix,
        save_chunk_summaries=args.save_chunk_summaries,
        limit=args.limit,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
    )
    for key, value in stats.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
