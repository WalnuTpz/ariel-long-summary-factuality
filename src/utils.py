"""Shared infrastructure utilities for batch 1.

This module intentionally stays small and dependency-free. It provides the
basic file and JSONL helpers needed before the project adds retrieval, LLM, or
evaluation code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list of dictionaries."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dictionaries from a UTF-8 JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield item


def write_jsonl(items: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write dictionaries to a UTF-8 JSONL file and return the item count."""
    path = ensure_parent_dir(path)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(text: str, path: str | Path) -> None:
    path = ensure_parent_dir(path)
    path.write_text(text, encoding="utf-8")


def stringify_text(value: Any) -> str:
    """Convert common dataset text fields into a string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(stringify_text(part) for part in value if part not in (None, ""))
    if isinstance(value, tuple):
        return "\n".join(stringify_text(part) for part in value if part not in (None, ""))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_whitespace(text: Any) -> str:
    """Collapse repeated whitespace while preserving paragraph breaks."""
    if not text:
        return ""
    text = stringify_text(text)
    paragraphs = re.split(r"\n\s*\n", text.strip())
    cleaned = [" ".join(paragraph.split()) for paragraph in paragraphs]
    return "\n\n".join(paragraph for paragraph in cleaned if paragraph)


def first_present(item: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-empty value for a list of possible field names."""
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def safe_str(value: Any) -> str:
    return stringify_text(value)
