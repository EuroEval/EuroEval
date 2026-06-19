"""Validation helpers for leaderboard result records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .constants import REQUIRED_METADATA_FIELDS

logger = logging.getLogger(__name__)


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
            _load_jsonl_lines(
                lines=path.read_text(encoding="utf-8").splitlines(), source=str(path)
            )
        )
    return records


def validate_eee_record(record: dict[str, object], context: str = "record") -> None:
    """Validate that a result record is acceptable leaderboard input/output.

    Args:
        record:
            Record to validate.
        context (optional):
            Human-readable location used in error messages. Defaults to
            ``"record"``.

    Raises:
        ValueError:
            If the record is not EEE format or lacks required metadata.
    """
    if not is_eee_record(record=record):
        raise ValueError(f"{context} is not an EEE-format record.")

    additional: dict = record.get("model_info", {}).get("additional_details", {})
    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in additional]
    if missing:
        raise ValueError(
            f"{context} is missing precious metadata field(s): " + ", ".join(missing)
        )

    non_bool = [
        field
        for field in REQUIRED_METADATA_FIELDS
        if not isinstance(additional.get(field), bool)
    ]
    if non_bool:
        raise ValueError(
            f"{context} has non-boolean precious metadata field(s): "
            + ", ".join(non_bool)
        )


def validate_eee_records(records: list[dict[str, object]], context: str) -> None:
    """Validate a list of EEE result records.

    Args:
        records:
            Records to validate.
        context:
            Human-readable source name used in error messages.

    """
    for idx, record in enumerate(records, start=1):
        validate_eee_record(record=record, context=f"{context} record {idx:,}")


def is_eee_record(record: dict[str, object]) -> bool:
    """Return whether a record uses the EEE envelope.

    Args:
        record:
            Record to inspect.

    Returns:
        True if the record has the required EEE top-level structures.
    """
    return (
        "schema_version" in record
        and isinstance(record.get("model_info"), dict)
        and isinstance(record.get("eval_library"), dict)
        and isinstance(record.get("evaluation_results"), list)
    )


def dump_jsonl_records(records: list[dict[str, object]]) -> str:
    """Serialise EEE records as JSONL.

    Args:
        records:
            Records to serialise. All records must already be EEE-valid.

    Returns:
        JSONL text with a trailing newline if records are present.

    """
    validate_eee_records(records=records, context="JSONL output")
    if not records:
        return ""
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    return "\n".join(lines) + "\n"


def _load_jsonl_lines(lines: list[str], source: str) -> list[dict[str, object]]:
    """Parse JSONL lines into record dictionaries.

    Blank lines are skipped and concatenated objects (``}{``) on a single line
    are split apart before parsing.

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
