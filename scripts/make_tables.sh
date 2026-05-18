#!/usr/bin/env bash
set -euo pipefail

# Generate paper-ready Markdown tables from evaluation outputs.
#
# Typical usage after running scripts/run_pipeline.sh:
#   CONFIG=configs/main_600_tables_gpu.env bash scripts/make_tables.sh
#
# Typical usage after running scripts/run_compare_retrieval.sh:
#   PYTHON=.venv/bin/python \
#   RUN_DIR=data/results/main_600_compare_gpu \
#   CORRECTION_RUN_DIR=data/results/main_600_ariel_gpu \
#   PRIMARY_METHOD=ariel \
#   bash scripts/make_tables.sh

CONFIG="${CONFIG:-}"
if [[ -n "${CONFIG}" ]]; then
  if [[ ! -f "${CONFIG}" ]]; then
    echo "missing config file: ${CONFIG}" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONFIG}"
fi

PYTHON_BIN="${PYTHON:-python3}"
RUN_DIR="${RUN_DIR:-data/results/debug_retrieval_compare}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/tables/$(basename "${RUN_DIR}")}"
PRIMARY_METHOD="${PRIMARY_METHOD:-ariel}"
CORRECTION_RUN_DIR="${CORRECTION_RUN_DIR:-}"
CORRECTION_METHOD="${CORRECTION_METHOD:-${PRIMARY_METHOD}}"
DOCUMENTS="${DOCUMENTS:-}"
TITLE="${TITLE:-实验结果表格}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" - "${RUN_DIR}" "${OUTPUT_DIR}" "${PRIMARY_METHOD}" "${DOCUMENTS}" "${TITLE}" "${CORRECTION_RUN_DIR}" "${CORRECTION_METHOD}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


RUN_DIR = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])
PRIMARY_METHOD = sys.argv[3]
DOCUMENTS = Path(sys.argv[4]) if sys.argv[4] else None
TITLE = sys.argv[5]
CORRECTION_RUN_DIR = Path(sys.argv[6]) if sys.argv[6] else None
CORRECTION_METHOD = sys.argv[7]

METHOD_LABELS = {
    "bm25": "BM25 + NLI",
    "embedding": "Embedding + NLI",
    "hybrid": "Fixed Hybrid + NLI",
    "ariel": "ARIEL + NLI",
    "ariel_no_coverage": "ARIEL w/o coverage rerank",
    "ariel_no_expansion": "ARIEL w/o adaptive expansion",
    "ariel_no_risk_weighting": "ARIEL w/o risk-adaptive weighting",
}

VERSION_LABELS = {
    "before_correction": "Before correction",
    "after_correction": "After local correction",
}

BUCKET_LABELS = {
    "short": "Short",
    "medium": "Medium",
    "long": "Long",
}
SHORT_MAX_WORDS = 150
MEDIUM_MAX_WORDS = 225


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def simple_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text.lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def method_from_summary_path(path: Path) -> str:
    name = path.name
    prefix = "evaluation."
    suffix = ".summary.json"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def load_summaries(run_dir: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("evaluation.*.summary.json")):
        method = method_from_summary_path(path)
        summaries[method] = read_json(path)
    return summaries


def choose_primary_method(summaries: dict[str, dict[str, Any]], preferred: str) -> str | None:
    if preferred in summaries:
        return preferred
    if "ariel" in summaries:
        return "ariel"
    if summaries:
        return sorted(summaries)[0]
    return None


def get_version(summary: dict[str, Any], version: str) -> dict[str, Any]:
    return summary.get("by_version", {}).get(version, {}) or {}


def format_int(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value)


def format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def mean(values: list[Any]) -> float | None:
    clean_values = []
    for value in values:
        if value is None:
            continue
        try:
            clean_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def metric(summary: dict[str, Any], version: str, key: str) -> Any:
    return get_version(summary, version).get(key)


def markdown_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["left"] * len(headers)
    align_marks = {
        "left": "---",
        "right": "---:",
        "center": ":---:",
    }
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align_marks.get(align, "---") for align in aligns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def write_table(name: str, title: str, headers: list[str], rows: list[list[str]], aligns: list[str]) -> str:
    content = f"### {title}\n\n"
    if rows:
        content += markdown_table(headers, rows, aligns)
    else:
        content += "_暂无可用结果。_\n"
    path = OUTPUT_DIR / name
    path.write_text(content + "\n", encoding="utf-8")
    return content


def average_document_words(documents: Path | None) -> str:
    if documents is None or not documents.exists():
        return "-"
    counts = []
    for item in read_jsonl(documents):
        text = str(item.get("document") or "")
        if text:
            counts.append(len(simple_tokens(text)))
    if not counts:
        return "-"
    return format_float(sum(counts) / len(counts), digits=1)


def main_metrics_rows(summary: dict[str, Any], primary_method: str, documents: Path | None) -> list[list[str]]:
    before = get_version(summary, "before_correction")
    stats = summary.get("stats", {})
    return [
        ["样本数", format_int(before.get("count") or stats.get("input_summaries"))],
        ["平均文档长度（词）", average_document_words(documents)],
        ["平均生成摘要长度（词）", format_float(before.get("summary_word_count"), digits=1)],
        ["ROUGE-1", format_float(before.get("rouge1"))],
        ["ROUGE-2", format_float(before.get("rouge2"))],
        ["ROUGE-L", format_float(before.get("rougeL"))],
        ["Supported Ratio", format_percent(before.get("supported_ratio"))],
        ["Unsupported/NEI Ratio", format_percent(before.get("unsupported_or_nei_ratio"))],
        ["证据定位方法", METHOD_LABELS.get(primary_method, primary_method)],
    ]


def retrieval_rows(summaries: dict[str, dict[str, Any]]) -> list[list[str]]:
    preferred = [
        "bm25",
        "embedding",
        "hybrid",
        "ariel",
        "ariel_no_coverage",
        "ariel_no_expansion",
        "ariel_no_risk_weighting",
    ]
    methods = [method for method in preferred if method in summaries]
    methods.extend(method for method in sorted(summaries) if method not in methods)
    rows = []
    for method in methods:
        before = get_version(summaries[method], "before_correction")
        if not before:
            continue
        rows.append(
            [
                METHOD_LABELS.get(method, method),
                format_int(before.get("count")),
                format_float(before.get("rouge1")),
                format_float(before.get("rouge2")),
                format_float(before.get("rougeL")),
                format_percent(before.get("supported_ratio")),
                format_percent(before.get("unsupported_or_nei_ratio")),
                format_float(before.get("risk_count"), digits=2),
            ]
        )
    return rows


def ariel_ablation_rows(summaries: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for method in ["ariel_no_risk_weighting", "ariel_no_coverage", "ariel_no_expansion", "ariel"]:
        if method not in summaries:
            continue
        before = get_version(summaries[method], "before_correction")
        rows.append(
            [
                METHOD_LABELS.get(method, method),
                format_percent(before.get("supported_ratio")),
                format_percent(before.get("unsupported_or_nei_ratio")),
                format_float(before.get("risk_count"), digits=2),
                format_float(before.get("rougeL")),
            ]
        )
    return rows


def summarize_eval_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "summary_word_count": mean([record.get("summary_word_count") for record in records]),
        "rouge1": mean([record.get("rouge1") for record in records]),
        "rouge2": mean([record.get("rouge2") for record in records]),
        "rougeL": mean([record.get("rougeL") for record in records]),
        "supported_ratio": mean([record.get("supported_ratio") for record in records]),
        "unsupported_or_nei_ratio": mean(
            [record.get("unsupported_or_nei_ratio") for record in records]
        ),
    }


def threshold_bucket(word_count: float) -> str:
    if word_count <= SHORT_MAX_WORDS:
        return "short"
    if word_count <= MEDIUM_MAX_WORDS:
        return "medium"
    return "long"


def threshold_length_bucket_rows(records: list[dict[str, Any]]) -> list[list[str]]:
    before_records = [
        record
        for record in records
        if record.get("summary_version") == "before_correction"
        and record.get("summary_word_count") is not None
    ]
    if not before_records:
        return []

    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in ["short", "medium", "long"]}
    for record in before_records:
        bucket = threshold_bucket(float(record.get("summary_word_count") or 0))
        grouped[bucket].append(record)

    rows = []
    for bucket in ["short", "medium", "long"]:
        summary = summarize_eval_records(grouped[bucket])
        rows.append(
            [
                BUCKET_LABELS.get(bucket, bucket),
                format_int(summary.get("count")),
                format_float(summary.get("summary_word_count"), digits=1),
                format_float(summary.get("rouge1")),
                format_float(summary.get("rouge2")),
                format_float(summary.get("rougeL")),
                format_percent(summary.get("supported_ratio")),
                format_percent(summary.get("unsupported_or_nei_ratio")),
            ]
        )
    return rows


def length_bucket_rows(summary: dict[str, Any], records: list[dict[str, Any]] | None = None) -> list[list[str]]:
    if records:
        threshold_rows = threshold_length_bucket_rows(records)
        if threshold_rows:
            return threshold_rows

    rows = []
    buckets = summary.get("by_length_bucket", {})
    for bucket in ["short", "medium", "long"]:
        if bucket not in buckets:
            continue
        before = buckets[bucket].get("before_correction", {})
        if not before:
            continue
        rows.append(
            [
                BUCKET_LABELS.get(bucket, bucket),
                format_int(before.get("count")),
                format_float(before.get("summary_word_count"), digits=1),
                format_float(before.get("rouge1")),
                format_float(before.get("rouge2")),
                format_float(before.get("rougeL")),
                format_percent(before.get("supported_ratio")),
                format_percent(before.get("unsupported_or_nei_ratio")),
            ]
        )
    return rows


def correction_rows(summary: dict[str, Any]) -> list[list[str]]:
    rows = []
    for version in ["before_correction", "after_correction"]:
        values = get_version(summary, version)
        if not values:
            continue
        error_reduction = "-"
        if version == "after_correction":
            error_reduction = format_percent(summary.get("paired_error_reduction_rate"))
        length_ratio = "1.000" if version == "before_correction" else format_float(values.get("length_ratio"))
        rows.append(
            [
                VERSION_LABELS.get(version, version),
                format_float(values.get("rouge1")),
                format_float(values.get("rouge2")),
                format_float(values.get("rougeL")),
                format_percent(values.get("supported_ratio")),
                format_percent(values.get("unsupported_or_nei_ratio")),
                error_reduction,
                length_ratio,
            ]
        )
    return rows


def notes(
    summaries: dict[str, dict[str, Any]],
    primary_method: str | None,
    correction_run_dir: Path | None,
    correction_method: str | None,
) -> str:
    lines = [
        "### 表格说明",
        "",
        "- ROUGE-1/2/L 为 F1 形式，脚本按项目内置实现计算。",
        "- Supported Ratio 和 Unsupported/NEI Ratio 基于句子级 NLI 结果统计。",
        "- Error Reduction Rate = (Risk_before - Risk_after) / Risk_before。",
        "- Length Ratio = corrected summary word count / original summary word count。",
        f"- 表 2 按原始生成摘要长度分桶：Short <= {SHORT_MAX_WORDS} 词，Medium <= {MEDIUM_MAX_WORDS} 词，Long > {MEDIUM_MAX_WORDS} 词。",
    ]
    if primary_method:
        lines.append(f"- 主方法表默认使用 `{METHOD_LABELS.get(primary_method, primary_method)}`。")
    if correction_run_dir and correction_method:
        lines.append(
            f"- 表 5 使用 `{correction_run_dir}` 中的 `{METHOD_LABELS.get(correction_method, correction_method)}` 纠错结果。"
        )
    if not summaries:
        lines.append("- 未在 RUN_DIR 中找到 `evaluation.*.summary.json`。")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(RUN_DIR)
    primary_method = choose_primary_method(summaries, PRIMARY_METHOD)
    primary_summary = summaries.get(primary_method, {}) if primary_method else {}
    primary_records = (
        read_jsonl(RUN_DIR / f"evaluation.{primary_method}.jsonl")
        if primary_method
        else []
    )
    correction_summaries = load_summaries(CORRECTION_RUN_DIR) if CORRECTION_RUN_DIR else {}
    correction_method = (
        choose_primary_method(correction_summaries, CORRECTION_METHOD)
        if correction_summaries
        else primary_method
    )
    correction_summary = (
        correction_summaries.get(correction_method, {})
        if correction_summaries and correction_method
        else primary_summary
    )
    if correction_summary and not get_version(correction_summary, "after_correction"):
        print(
            "warning: table 5 has no after_correction data; "
            "set CORRECTION_RUN_DIR to a run_pipeline output directory.",
            file=sys.stderr,
        )

    sections = [
        f"# {TITLE}\n\n",
        notes(summaries, primary_method, CORRECTION_RUN_DIR, correction_method),
        "\n",
    ]

    sections.append(
        write_table(
            "table_1_main_metrics.md",
            "表 1：基础流程总体结果",
            ["项目", "数值"],
            main_metrics_rows(primary_summary, primary_method or "", DOCUMENTS) if primary_summary else [],
            ["left", "right"],
        )
    )

    sections.append(
        write_table(
            "table_2_length_buckets.md",
            "表 2：摘要长度分桶分析",
            [
                "长度区间",
                "样本数",
                "平均摘要长度",
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
                "Supported Ratio",
                "Unsupported/NEI Ratio",
            ],
            length_bucket_rows(primary_summary, primary_records) if primary_summary else [],
            ["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )

    sections.append(
        write_table(
            "table_3_retrieval_comparison.md",
            "表 3：证据定位方法对比",
            [
                "证据定位方法",
                "样本数",
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
                "Supported Ratio",
                "Unsupported/NEI Ratio",
                "平均风险句数",
            ],
            retrieval_rows(summaries),
            ["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )

    sections.append(
        write_table(
            "table_4_ariel_ablation.md",
            "表 4：ARIEL 消融实验",
            [
                "方法",
                "Supported Ratio",
                "Unsupported/NEI Ratio",
                "平均风险句数",
                "ROUGE-L",
            ],
            ariel_ablation_rows(summaries),
            ["left", "right", "right", "right", "right"],
        )
    )

    sections.append(
        write_table(
            "table_5_correction.md",
            "表 5：局部纠错前后对比",
            [
                "版本",
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
                "Supported Ratio ↑",
                "Unsupported/NEI Ratio ↓",
                "Error Reduction Rate ↑",
                "Length Ratio",
            ],
            correction_rows(correction_summary) if correction_summary else [],
            ["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )

    combined_path = OUTPUT_DIR / "paper_tables.md"
    combined_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")

    print(f"tables_written: {OUTPUT_DIR}")
    print(f"combined_tables: {combined_path}")
    if primary_method:
        print(f"primary_method: {primary_method}")
    if CORRECTION_RUN_DIR:
        print(f"correction_run_dir: {CORRECTION_RUN_DIR}")
    if correction_method:
        print(f"correction_method: {correction_method}")
    print(f"summary_files: {len(summaries)}")


if __name__ == "__main__":
    main()
PY
