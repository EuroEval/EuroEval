"""JSONL parsing utilities for benchmark results."""

from __future__ import annotations

import json


def parse_jsonl_lines(lines: list[str], source: str) -> list[dict[str, object]]:
    """Parse JSONL lines into record dictionaries with robust `}{` handling.

    Blank lines are skipped and concatenated objects (``}{``) on a single line
    are split apart before parsing. This handles malformed JSONL where multiple
    JSON objects appear on the same line without a newline separator.

    Args:
        lines:
            The raw lines to parse.
        source:
            A human-readable label for the input, used in error messages.

    Returns:
        The parsed records, one dictionary per JSON object.

    Raises:
        ValueError:
            If a line contains invalid JSON or a non-object value.
    """
    records: list[dict[str, object]] = []
    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        # Split concatenated JSON objects on the same line (e.g., `}{` -> `}\n{`)
        for sub_line in line.replace("}{", "}\n{").splitlines():
            if not sub_line.strip():
                continue
            try:
                value = json.loads(sub_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source} line {line_idx:,}: {sub_line}."
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Invalid result in {source} line {line_idx:,}: expected object."
                )
            records.append(value)
    return records
