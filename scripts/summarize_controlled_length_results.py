"""Build a Markdown table for the controlled summary-length experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import ensure_parent_dir, iter_jsonl


DEFAULT_LABELS = ("short", "medium", "long")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_summary_target_words(path: Path) -> int | None:
    if not path.exists():
        return None
    for item in iter_jsonl(path):
        meta = item.get("meta")
        if isinstance(meta, dict) and meta.get("target_words") is not None:
            return int(meta["target_words"])
        if item.get("target_words") is not None:
            return int(item["target_words"])
        break
    return None


def pct(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def count(value: float | int | None) -> str:
    if value is None:
        return "-"
    return str(int(round(float(value))))


def load_row(run_root: Path, label: str) -> dict[str, Any] | None:
    run_dir = run_root / label
    summary_path = run_dir / "evaluation.ariel.summary.json"
    if not summary_path.exists():
        return None
    data = load_json(summary_path)
    metrics = data.get("by_version", {}).get("before_correction")
    if not isinstance(metrics, dict):
        return None
    return {
        "label": label,
        "target_words": first_summary_target_words(run_dir / "generated_summaries.jsonl"),
        **metrics,
    }


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Length setting | Target words | Count | Avg. words | Avg. sentences | ROUGE-1 | ROUGE-2 | ROUGE-L | Supported Ratio | Risk Ratio | Avg. risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    count(row.get("target_words")),
                    count(row.get("count")),
                    num(row.get("summary_word_count"), digits=1),
                    num(row.get("total_sentences"), digits=2),
                    num(row.get("rouge1")),
                    num(row.get("rouge2")),
                    num(row.get("rougeL")),
                    pct(row.get("supported_ratio")),
                    pct(row.get("unsupported_or_nei_ratio")),
                    num(row.get("risk_count"), digits=2),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize controlled short/medium/long experiment results."
    )
    parser.add_argument(
        "--run-root",
        default="data/results/controlled_length_100_wide_gpu",
        help="Directory containing one subdirectory per length setting.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Length labels to collect, in table order.",
    )
    parser.add_argument(
        "--output-markdown",
        default="outputs/tables/controlled_length_100_wide_gpu/controlled_length.md",
        help="Markdown table output path.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/tables/controlled_length_100_wide_gpu/controlled_length.json",
        help="JSON rows output path.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = [row for label in args.labels if (row := load_row(run_root, label)) is not None]
    if not rows:
        raise SystemExit(f"no controlled-length results found under {run_root}")

    markdown_path = ensure_parent_dir(args.output_markdown)
    json_path = ensure_parent_dir(args.output_json)
    markdown_path.write_text(build_markdown(rows), encoding="utf-8")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written markdown: {markdown_path}")
    print(f"written json: {json_path}")


if __name__ == "__main__":
    main()
