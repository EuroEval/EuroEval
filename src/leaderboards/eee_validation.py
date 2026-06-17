"""Validation helpers for leaderboard result records."""

from __future__ import annotations

import collections.abc as c
import json
import logging
import tarfile
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

from .constants import REQUIRED_METADATA_FIELDS
from .records import plain_model_id

logger = logging.getLogger(__name__)


@dataclass
class PreciousMetadataCache:
    """Precious model metadata recovered from existing result records."""

    commercially_licensed: dict[str, bool] = field(default_factory=dict)
    open: dict[str, bool] = field(default_factory=dict)
    trained_from_scratch: dict[str, bool] = field(default_factory=dict)


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


def load_records_from_results_archive(path: Path) -> list[dict[str, object]]:
    """Load result records from a ``results.tar.gz`` archive.

    Args:
        path:
            Archive to read.

    Returns:
        Parsed records.

    Raises:
        FileNotFoundError:
            If the archive or inner JSONL file is missing.
    """
    with tarfile.open(path, "r:gz") as tar:
        results_file = tar.extractfile(member="results/results.jsonl")
        if results_file is None:
            raise FileNotFoundError("results/results.jsonl not found in archive.")
        lines = results_file.read().decode(encoding="utf-8").splitlines()
    return _load_jsonl_lines(lines=lines, source=str(path))


def build_precious_metadata_cache(
    records: list[dict[str, object]],
) -> PreciousMetadataCache:
    """Build a cache from precious metadata already present in records.

    Args:
        records:
            Existing result records. Both EEE and old EuroEval migration records
            may be used as metadata sources.

    Returns:
        Metadata recovered from the records themselves.
    """
    cache = PreciousMetadataCache()
    for record in records:
        model_id = get_record_model_id(record=record)
        if model_id is None:
            continue
        if isinstance(record.get("commercially_licensed"), bool):
            cache.commercially_licensed[model_id] = t.cast(
                bool, record["commercially_licensed"]
            )
        if isinstance(record.get("open"), bool):
            cache.open[model_id] = t.cast(bool, record["open"])
        if isinstance(record.get("trained_from_scratch"), bool):
            cache.trained_from_scratch[model_id] = t.cast(
                bool, record["trained_from_scratch"]
            )
    return cache


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


def get_record_model_id(record: dict[str, object]) -> str | None:
    """Extract a plain model id from an EEE or old migration record.

    Args:
        record:
            Record to inspect.

    Returns:
        The model id, if present.
    """
    model_id: object | None
    model_info = record.get("model_info")
    if isinstance(model_info, c.Mapping):
        typed_model_info = t.cast(c.Mapping[str, object], model_info)
        model_id = typed_model_info.get("id") or typed_model_info.get("name")
    else:
        model_id = record.get("model")
    if not isinstance(model_id, str) or not model_id:
        return None
    return plain_model_id(model_id=model_id)


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


def _cached_metadata_value(
    cache: PreciousMetadataCache, field_name: str, model_id: str
) -> bool | None:
    if field_name == "commercially_licensed":
        return cache.commercially_licensed.get(model_id)
    if field_name == "open":
        return cache.open.get(model_id)
    if field_name == "trained_from_scratch":
        return cache.trained_from_scratch.get(model_id)
    return None


def _load_jsonl_lines(lines: list[str], source: str) -> list[dict[str, object]]:
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
