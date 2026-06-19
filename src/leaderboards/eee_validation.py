"""Validation helpers for leaderboard result records."""

from __future__ import annotations

import json
import logging

from .constants import REQUIRED_METADATA_FIELDS

logger = logging.getLogger(__name__)


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
