"""Canonical result identity and path helpers for EuroEval results storage.

This module provides a single canonical implementation of record identity
used by writers, readers, deduplication, and sync operations. The identity
tuple is ``(model_id, dataset, validation_split, few_shot)`` where ``model_id``
is the full model identifier including ``@revision`` and ``#param`` variants.
"""

from __future__ import annotations

import re
from pathlib import Path

from euroeval.data_models import BenchmarkResult

# Type alias for the identity tuple
ResultIdentity = tuple[str, str, bool | None, bool | None]


def sanitise_model_dir_name(model_id: str) -> str:
    """Produce a sanitised model directory name from a model id.

    Replaces forward slashes with underscores while preserving ``@`` and ``#``
    characters as used in per-model filenames (e.g.,
    ``Qwen_Qwen3-30B-A3B#no-thinking``, ``PleIAs_Baguettotron@refs_pr_6``).

    Args:
        model_id:
            The full model identifier, potentially containing slashes, ``@``,
            and ``#`` characters.

    Returns:
        A sanitised string suitable for use as a directory name.
    """
    return model_id.replace("/", "_")


def sanitise_dataset_name(dataset: str) -> str:
    """Produce a sanitised dataset name for use in filenames.

    Replaces forward slashes with underscores.

    Args:
        dataset:
            The dataset name, potentially containing slashes.

    Returns:
        A sanitised string suitable for use in filenames.
    """
    return dataset.replace("/", "_")


def normalise_bool_value(value: bool | str | None) -> bool | None:
    """Normalise a boolean value that may arrive as bool, string, or None.

    Handles Python bool, None, or lowercase strings 'true'/'false'/'none'.

    Args:
        value:
            The value to normalise. May be a bool, a string ('true', 'false',
            'none', case-insensitive), or None.

    Returns:
        A normalised bool or None.

    Raises:
        ValueError:
            If the string value is not one of 'true', 'false', or 'none'.
        TypeError:
            If the value is not a bool, str, or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower == "none":
            return None
        raise ValueError(f"Invalid boolean string value: {value!r}")
    raise TypeError(f"Unexpected type for boolean value: {type(value)}")


def split_label(validation_split: bool | None) -> str:
    """Convert a validation_split boolean to its label string.

    Args:
        validation_split:
            The validation split flag, or None.

    Returns:
        Label string: 'val' for True, 'test' for False, 'none' for None.
    """
    if validation_split is True:
        return "val"
    if validation_split is False:
        return "test"
    return "none"


def shot_label(few_shot: bool | None) -> str:
    """Convert a few_shot boolean to its label string.

    Args:
        few_shot:
            The few-shot flag, or None.

    Returns:
        Label string: 'fewshot' for True, 'zeroshot' for False, 'none' for None.
    """
    if few_shot is True:
        return "fewshot"
    if few_shot is False:
        return "zeroshot"
    return "none"


def identity_from_eee_record(record: dict) -> ResultIdentity:
    """Extract the identity tuple from an EEE-format record dict.

    Identity is ``(model_id, dataset, validation_split, few_shot)`` where
    ``model_id`` is the full model identifier including ``@revision`` and
    ``#param`` (NOT the display name).

    Model id comes from ``model_info.id`` with fallback to ``model_info.name``.
    Dataset, few_shot, and validation_split come from
    ``eval_library.additional_details`` (keys: ``dataset``, ``few_shot``,
    ``validation_split``).

    Args:
        record:
            A result record in EEE format.

    Returns:
        Identity tuple ``(model_id, dataset, validation_split, few_shot)``.

    Raises:
        ValueError:
            If required fields are missing or invalid.
    """
    model_info = record.get("model_info", {})
    model_id = model_info.get("id") or model_info.get("name")
    if not model_id:
        raise ValueError(
            "model_info.id and model_info.name are both missing or empty in record"
        )

    additional = record.get("eval_library", {}).get("additional_details", {})

    dataset = additional.get("dataset")
    if not dataset:
        raise ValueError("Missing eval_library.additional_details.dataset in record")

    few_shot_raw = additional.get("few_shot")
    validation_split_raw = additional.get("validation_split")

    few_shot = normalise_bool_value(few_shot_raw)
    validation_split = normalise_bool_value(validation_split_raw)

    return (model_id, dataset, validation_split, few_shot)


def identity_from_benchmark_result(result: BenchmarkResult) -> ResultIdentity:
    """Extract the identity tuple from a BenchmarkResult.

    Identity is ``(model_id, dataset, validation_split, few_shot)`` where
    ``model_id`` is the full model identifier.

    Args:
        result:
            A BenchmarkResult instance.

    Returns:
        Identity tuple ``(model_id, dataset, validation_split, few_shot)``.
    """
    return (result.model, result.dataset, result.validation_split, result.few_shot)


def record_filename(
    dataset: str, validation_split: bool | None, few_shot: bool | None
) -> str:
    """Produce the per-record filename for a given identity.

    Format: ``<sanitise(dataset)>__<split_label>__<shot_label>.json``

    Args:
        dataset:
            The dataset name.
        validation_split:
            The validation split flag. Maps to: True -> 'val', False -> 'test',
            None -> 'none'.
        few_shot:
            The few-shot flag. Maps to: True -> 'fewshot', False -> 'zeroshot',
            None -> 'none'.

    Returns:
        Filename string suitable for storing the result record.
    """
    sanitised_dataset = sanitise_dataset_name(dataset)
    split_lbl = split_label(validation_split)
    shot_lbl = shot_label(few_shot)
    return f"{sanitised_dataset}__{split_lbl}__{shot_lbl}.json"


def record_relative_path(
    model_id: str, dataset: str, validation_split: bool | None, few_shot: bool | None
) -> Path:
    """Produce the full relative path for a result record.

    Path format: ``<sanitise(model_id)>/<sanitise(dataset)>__<split>__<shot>.json``

    Args:
        model_id:
            The full model identifier.
        dataset:
            The dataset name.
        validation_split:
            The validation split flag.
        few_shot:
            The few-shot flag.

    Returns:
        Path object representing the relative path.
    """
    model_dir = sanitise_model_dir_name(model_id)
    filename = record_filename(dataset, validation_split, few_shot)
    return Path(model_dir) / filename


def identity_to_path(identity: ResultIdentity) -> Path:
    """Produce the full relative path for a result identity.

    Args:
        identity:
            Identity tuple ``(model_id, dataset, validation_split, few_shot)``.

    Returns:
        Path object representing the relative path.
    """
    model_id, dataset, validation_split, few_shot = identity
    return record_relative_path(model_id, dataset, validation_split, few_shot)


def dedup_newer_record(record_a: dict, record_b: dict) -> dict:
    """Determine which of two records with the same identity is newer.

    Compares by ``eval_library.version`` (semver-ish, higher wins),
    tie-broken by ``retrieved_timestamp`` (higher/newer wins).

    Args:
        record_a:
            First record in EEE format.
        record_b:
            Second record in EEE format.

    Returns:
        The newer record (either record_a or record_b).

    Raises:
        ValueError:
            If the records do not share the same identity.
    """
    identity_a = identity_from_eee_record(record_a)
    identity_b = identity_from_eee_record(record_b)

    if identity_a != identity_b:
        raise ValueError(
            f"Records have different identities: {identity_a} vs {identity_b}"
        )

    version_a = _extract_version(record_a)
    version_b = _extract_version(record_b)

    version_cmp = _compare_versions(version_a, version_b)
    if version_cmp > 0:
        return record_a
    if version_cmp < 0:
        return record_b

    timestamp_a = _extract_timestamp(record_a)
    timestamp_b = _extract_timestamp(record_b)

    if timestamp_a >= timestamp_b:
        return record_a
    return record_b


def _extract_version(record: dict) -> str | None:
    """Extract the EuroEval version from a record.

    Strips any ``.dev`` suffix.

    Args:
        record:
            A record in EEE format.

    Returns:
        Version string or None if not found.
    """
    version = record.get("eval_library", {}).get("version")
    if version:
        return re.sub(r"\.dev\d+", "", version)
    return None


def _extract_timestamp(record: dict) -> int:
    """Extract the retrieved_timestamp from a record.

    Args:
        record:
            A record in EEE format.

    Returns:
        Timestamp as integer, or 0 if not found.
    """
    ts = record.get("retrieved_timestamp")
    if ts is None:
        return 0
    try:
        return int(ts)
    except (ValueError, TypeError):
        return 0


def _compare_versions(version_a: str | None, version_b: str | None) -> int:
    """Compare two version strings.

    Performs a component-wise numeric comparison. Higher version wins.
    If either version is None, the non-None version wins. If both are None,
    returns 0 (tie).

    Args:
        version_a:
            First version string.
        version_b:
            Second version string.

    Returns:
        Positive if version_a > version_b, negative if version_a < version_b,
        0 if equal or both None.
    """
    if version_a is None and version_b is None:
        return 0
    if version_a is None:
        return -1
    if version_b is None:
        return 1

    parts_a = _parse_version(version_a)
    parts_b = _parse_version(version_b)

    for pa, pb in zip(parts_a, parts_b, strict=False):
        if pa > pb:
            return 1
        if pa < pb:
            return -1

    if len(parts_a) > len(parts_b):
        return 1
    if len(parts_a) < len(parts_b):
        return -1

    return 0


def _parse_version(version: str) -> list[int]:
    """Parse a version string into numeric components.

    Splits on non-digit characters and converts each component to int.
    Non-numeric parts are treated as 0.

    Args:
        version:
            Version string (e.g., "1.2.3", "2.0.1rc1").

    Returns:
        List of integer components.
    """
    parts = re.split(r"[^0-9]+", version)
    result = []
    for part in parts:
        if part:
            try:
                result.append(int(part))
            except ValueError:
                result.append(0)
    return result


def raise_on_collision(identity_a: ResultIdentity, identity_b: ResultIdentity) -> None:
    """Raise an error if two distinct identities would sanitise to the same path.

    Args:
        identity_a:
            First identity tuple.
        identity_b:
            Second identity tuple.

    Raises:
        ValueError:
            If the identities are different but would produce the same path.
    """
    if identity_a != identity_b:
        path_a = identity_to_path(identity_a)
        path_b = identity_to_path(identity_b)
        if path_a == path_b:
            raise ValueError(
                f"Identity collision detected: {identity_a} and {identity_b} "
                f"both sanitise to {path_a}"
            )
