"""JSONL parsing and loading utilities for result records."""

from __future__ import annotations

import json
from pathlib import Path

from euroeval.jsonl_io import parse_jsonl_lines


def load_records_from_result_tree(results_dir: Path) -> list[dict[str, object]]:
    """Load result records from the JSON tree structure.

    Walks ``results_dir/<model>/<dataset>__<split>__<shot>.json`` files,
    reading each as a single JSON dict. Fails loudly on malformed files.

    Args:
        results_dir:
            The root directory containing model subdirectories with JSON
            result files.

    Returns:
        Parsed records from all JSON files in the tree.

    Raises:
        json.JSONDecodeError:
            If any JSON file contains malformed JSON.
        ValueError:
            If any JSON file does not contain a dict object.
    """
    records: list[dict[str, object]] = []
    json_files = sorted(results_dir.glob("*/*.json"))

    for path in json_files:
        content = path.read_text(encoding="utf-8")
        try:
            record = json.loads(content)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"Malformed JSON in {path}", exc.doc, exc.pos
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(f"Expected dict in {path}, got {type(record).__name__}")

        records.append(record)

    return records


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
