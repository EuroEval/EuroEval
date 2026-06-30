"""JSONL parsing and loading utilities for result records."""

from __future__ import annotations

from pathlib import Path

from euroeval.jsonl_io import parse_jsonl_lines


def load_records_from_jsonl_files(paths: list[Path]) -> list[dict[str, object]]:
    """Load JSONL result records from files.

    Args:
        paths:
            JSONL files to read.

    Returns:
        Parsed records.
    """
    records: list[dict[str, object]] = []
    for path in paths:
        records.extend(
            parse_jsonl_lines(
                lines=path.read_text(encoding="utf-8").splitlines(), source=str(path)
            )
        )
    return records
