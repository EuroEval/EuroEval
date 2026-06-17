"""Tests for leaderboard EEE result validation helpers."""

from __future__ import annotations

import pytest

from leaderboards.eee_validation import (
    PreciousMetadataCache,
    normalise_record_to_eee,
    validate_eee_record,
)


def test_validate_rejects_old_format_output() -> None:
    """Old-format records are invalid as leaderboard input or output."""
    record = _old_record()

    with pytest.raises(ValueError, match="not an EEE-format record"):
        validate_eee_record(record=record)


def test_old_format_can_be_migrated_with_existing_metadata() -> None:
    """Old-format migration preserves precious metadata from the source."""
    record = _old_record()

    migrated = normalise_record_to_eee(record=record)

    validate_eee_record(record=migrated)
    assert migrated["schema_version"] == "0.2.1"
    assert migrated["commercially_licensed"] is True
    assert migrated["open"] is True
    assert migrated["trained_from_scratch"] is False


def test_missing_precious_metadata_is_not_generated() -> None:
    """Migration fails validation when precious metadata cannot be recovered."""
    record = _old_record()
    for field_name in ("commercially_licensed", "open", "trained_from_scratch"):
        record.pop(field_name)

    migrated = normalise_record_to_eee(record=record)

    with pytest.raises(ValueError, match="missing precious metadata"):
        validate_eee_record(record=migrated)


def test_missing_precious_metadata_can_be_recovered_from_cache() -> None:
    """Migration may recover precious metadata from existing records only."""
    record = _old_record()
    for field_name in ("commercially_licensed", "open", "trained_from_scratch"):
        record.pop(field_name)

    migrated = normalise_record_to_eee(
        record=record,
        precious_metadata_cache=PreciousMetadataCache(
            commercially_licensed={"org/model": False},
            open={"org/model": True},
            trained_from_scratch={"org/model": True},
        ),
    )

    validate_eee_record(record=migrated)
    assert migrated["commercially_licensed"] is False
    assert migrated["open"] is True
    assert migrated["trained_from_scratch"] is True


def _old_record() -> dict[str, object]:
    return {
        "dataset": "test-dataset",
        "task": "classification",
        "languages": ["en"],
        "model": "org/model",
        "results": {
            "raw": [{"test_mcc": 75.0}],
            "total": {"test_mcc": 75.0, "test_mcc_se": 2.0},
        },
        "num_model_parameters": 1,
        "max_sequence_length": 128,
        "vocabulary_size": 32000,
        "merge": False,
        "generative": False,
        "generative_type": None,
        "few_shot": False,
        "validation_split": False,
        "euroeval_version": "13.0.0",
        "commercially_licensed": True,
        "open": True,
        "trained_from_scratch": False,
    }
