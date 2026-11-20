"""Tests for the `leaderboards.record_fields` module."""

import typing as t

import pytest

from leaderboards.record_fields import deduplicate_records


def test_deduplicate_collapses_generative_flag_variants() -> None:
    """Records differing only in the ``generative`` flag collapse to one.

    This is the Apertus v1.1 regression (issue #1970): the two records render
    on the same leaderboard row, so only one must survive deduplication.
    """
    records = [_record(generative=True), _record(generative=None)]
    assert len(deduplicate_records(records=records)) == 1


def _record(
    name: str = "org/model",
    dataset: str = "angry-tweets",
    *,
    version: str | None = None,
    generative: bool | None = None,
) -> dict[str, t.Any]:
    """Build a minimal EEE-style record.

    Args:
        name (optional):
            The model name to store in ``model_info.name``. Defaults to "org/model".
        dataset (optional):
            The dataset name to store in the record. Defaults to "angry-tweets".
        version (optional):
            The EuroEval version to store, or None to omit it. Defaults to None.
        generative (optional):
            Value for the ``generative`` flag, or None to omit it entirely.
            Defaults to None.

    Returns:
        A minimal EEE-style record.
    """
    additional: dict[str, t.Any] = {"dataset": dataset}
    if generative is not None:
        additional["generative"] = generative
    model_info: dict[str, t.Any] = {"name": name, "additional_details": additional}
    library_additional: dict[str, t.Any] = {"dataset": dataset}
    if generative is not None:
        library_additional["generative"] = generative
    library: dict[str, t.Any] = {"additional_details": library_additional}
    if version is not None:
        library["version"] = version
    return {"model_info": model_info, "eval_library": library}


def test_deduplicate_does_not_rank_by_metadata_completeness() -> None:
    """Metadata completeness does not rank leaderboard-equivalent records."""
    # Enriched record with full metadata
    enriched = _record(name="Qwen/Qwen3.6-27B-FP8 (val)", version="17.6.0")
    enriched["model_info"]["additional_details"].update(
        {
            "commercially_licensed": True,
            "open": True,
            "merge": False,
            "trained_from_scratch": True,
            "generative_type": "instruction_tuned",
            "model_url": "https://huggingface.co/Qwen/Qwen3.6-27B-FP8",
        }
    )

    # Stale record with missing metadata
    stale = _record(name="Qwen/Qwen3.6-27B-FP8 (val)", version="17.6.0")
    # No additional metadata - defaults only

    # Stale record comes last in input order
    deduped = deduplicate_records(records=[enriched, stale])

    assert len(deduped) == 1
    assert deduped[0] is enriched


def test_deduplicate_does_not_rank_by_release_date() -> None:
    """Metadata completeness does not rank leaderboard-equivalent records."""
    without_date = _record(version="18.0.0")
    with_date = _record(version="18.0.0")
    with_date["model_info"]["additional_details"]["release_date"] = "2024-02-03"

    deduped = deduplicate_records(records=[without_date, with_date])

    assert len(deduped) == 1
    assert "release_date" not in deduped[0]["model_info"]["additional_details"]


def test_deduplicate_does_not_rank_by_version() -> None:
    """Leaderboard-equivalent records are not ranked by library version."""
    old = _record(version="17.5.0", generative=True)
    new = _record(version="17.6.0", generative=None)

    deduped = deduplicate_records(records=[old, new])

    assert len(deduped) == 1
    assert deduped[0] is old


@pytest.mark.parametrize("version", ["18.1.0.dev", "18.1.0.dev0"])
def test_deduplicate_does_not_rank_development_versions(version: str) -> None:
    """Development suffixes do not rank leaderboard-equivalent records.

    Args:
        version:
            Version supplied to the test.
    """
    old = _record(version="18.0.0")
    new = _record(version=version)

    deduped = deduplicate_records(records=[old, new])

    assert deduped[0] is old


def test_deduplicate_does_not_rank_malformed_release_metadata() -> None:
    """Release metadata does not rank leaderboard-equivalent records."""
    malformed = _record(version="18.0.0")
    malformed["model_info"]["additional_details"]["release_date"] = "not-a-date"
    valid = _record(version="18.0.0")
    valid["model_info"]["additional_details"]["release_date"] = "2024-02-03"

    deduped = deduplicate_records(records=[malformed, valid])

    assert len(deduped) == 1
    assert deduped[0] is malformed


def test_deduplicate_equal_richness_preserves_first_record() -> None:
    """Regression test: same version/richness keeps first record, not later-wins.

    Before the fix, when two records had equal version and richness, the later
    one in input order won. This allowed a later duplicate with True to replace
    an earlier record with explicit legitimate False metadata.
    """
    # First record with explicit False values
    first_false = _record(version="17.6.0")
    first_false["model_info"]["additional_details"].update(
        {
            "commercially_licensed": False,
            "open": False,
            "merge": False,
            "trained_from_scratch": False,
        }
    )

    # Second record with True values (same richness score)
    second_true = _record(version="17.6.0")
    second_true["model_info"]["additional_details"].update(
        {
            "commercially_licensed": True,
            "open": True,
            "merge": True,
            "trained_from_scratch": True,
        }
    )

    # Second record comes last
    deduped = deduplicate_records(records=[first_false, second_true])

    assert len(deduped) == 1
    additional = deduped[0]["model_info"]["additional_details"]
    # First record's False values should be preserved
    assert additional.get("commercially_licensed") is False
    assert additional.get("open") is False
    assert additional.get("merge") is False
    assert additional.get("trained_from_scratch") is False


def test_deduplicate_false_counts_as_rich_metadata() -> None:
    """Explicit False values count as rich metadata and win against missing."""
    # Record with explicit False values (legitimate metadata)
    with_false = _record(version="17.6.0")
    with_false["model_info"]["additional_details"].update(
        {
            "commercially_licensed": False,
            "open": False,
            "merge": False,
            "trained_from_scratch": False,
        }
    )

    # Record with missing metadata
    missing = _record(version="17.6.0")
    # No additional metadata

    # Record with missing metadata comes last
    deduped = deduplicate_records(records=[with_false, missing])

    assert len(deduped) == 1
    additional = deduped[0]["model_info"]["additional_details"]
    # False values should be preserved (they count as rich metadata)
    assert additional.get("commercially_licensed") is False
    assert additional.get("open") is False
    assert additional.get("merge") is False
    assert additional.get("trained_from_scratch") is False


def test_deduplicate_preserves_distinct_datasets() -> None:
    """Records for genuinely distinct rows are all preserved."""
    records = [
        _record(dataset="angry-tweets"),
        _record(dataset="dansk"),
        _record(name="org/other", dataset="angry-tweets"),
    ]
    assert len(deduplicate_records(records=records)) == 3


def test_deduplicate_preserves_first_metadata_variant() -> None:
    """Metadata completeness does not override input order."""
    poor = _record(version="17.6.0")
    rich = _record(version="17.6.0")
    rich["model_info"]["additional_details"]["open"] = True

    # Poor record comes last
    deduped = deduplicate_records(records=[rich, poor])

    assert len(deduped) == 1
    assert deduped[0] is rich
