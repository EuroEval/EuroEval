"""Tests for the `leaderboards.record_fields` module."""

import typing as t

from leaderboards.record_fields import _metadata_richness_score, deduplicate_records


def _record(
    name: str = "org/model",
    dataset: str = "angry-tweets",
    *,
    version: str | None = None,
    generative: bool | None = None,
) -> dict[str, t.Any]:
    """Build a minimal EEE-style record.

    Args:
        name:
            The model name to store in ``model_info.name``.
        dataset:
            The dataset name to store in the record.
        version:
            The EuroEval version to store, or None to omit it.
        generative:
            Value for the ``generative`` flag, or None to omit it entirely.

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


def test_deduplicate_collapses_generative_flag_variants() -> None:
    """Records differing only in the ``generative`` flag collapse to one.

    This is the Apertus v1.1 regression (issue #1970): the two records render
    on the same leaderboard row, so only one must survive deduplication.
    """
    records = [_record(generative=True), _record(generative=None)]
    assert len(deduplicate_records(records=records)) == 1


def test_deduplicate_keeps_newest_version() -> None:
    """Among duplicates, the newest EuroEval version is kept."""
    old = _record(version="17.5.0", generative=True)
    new = _record(version="17.6.0", generative=None)

    deduped = deduplicate_records(records=[old, new])

    assert len(deduped) == 1
    assert deduped[0]["eval_library"]["version"] == "17.6.0"


def test_deduplicate_preserves_distinct_datasets() -> None:
    """Records for genuinely distinct rows are all preserved."""
    records = [
        _record(dataset="angry-tweets"),
        _record(dataset="dansk"),
        _record(name="org/other", dataset="angry-tweets"),
    ]
    assert len(deduplicate_records(records=records)) == 3


def test_metadata_richness_score_empty() -> None:
    """A record with no metadata gets a score of 0."""
    record = _record()
    assert _metadata_richness_score(record=record) == 0


def test_metadata_richness_score_commercial() -> None:
    """A record with commercially_licensed gets +1 (True or False both count)."""
    record_true = _record()
    record_true["model_info"]["additional_details"]["commercially_licensed"] = True
    assert _metadata_richness_score(record=record_true) == 1

    record_false = _record()
    record_false["model_info"]["additional_details"]["commercially_licensed"] = False
    assert _metadata_richness_score(record=record_false) == 1


def test_metadata_richness_score_open() -> None:
    """A record with open gets +1 (True or False both count)."""
    record_true = _record()
    record_true["model_info"]["additional_details"]["open"] = True
    assert _metadata_richness_score(record=record_true) == 1

    record_false = _record()
    record_false["model_info"]["additional_details"]["open"] = False
    assert _metadata_richness_score(record=record_false) == 1


def test_metadata_richness_score_merge() -> None:
    """A record with merge gets +1 (True or False both count)."""
    record_true = _record()
    record_true["model_info"]["additional_details"]["merge"] = True
    assert _metadata_richness_score(record=record_true) == 1

    record_false = _record()
    record_false["model_info"]["additional_details"]["merge"] = False
    assert _metadata_richness_score(record=record_false) == 1


def test_metadata_richness_score_trained_from_scratch() -> None:
    """A record with trained_from_scratch gets +1 (True or False both count)."""
    record_true = _record()
    record_true["model_info"]["additional_details"]["trained_from_scratch"] = True
    assert _metadata_richness_score(record=record_true) == 1

    record_false = _record()
    record_false["model_info"]["additional_details"]["trained_from_scratch"] = False
    assert _metadata_richness_score(record=record_false) == 1


def test_metadata_richness_score_generative_type() -> None:
    """A record with generative_type gets +1."""
    record = _record()
    record["model_info"]["additional_details"]["generative_type"] = "instruction_tuned"
    assert _metadata_richness_score(record=record) == 1


def test_metadata_richness_score_model_url() -> None:
    """A record with model_url gets +1."""
    record = _record()
    record["model_info"]["additional_details"]["model_url"] = (
        "https://example.com/model"
    )
    assert _metadata_richness_score(record=record) == 1


def test_metadata_richness_score_full() -> None:
    """A record with all metadata fields gets a score of 6."""
    record = _record()
    record["model_info"]["additional_details"].update(
        {
            "commercially_licensed": True,
            "open": True,
            "merge": True,
            "trained_from_scratch": True,
            "generative_type": "instruction_tuned",
            "model_url": "https://example.com/model",
        }
    )
    assert _metadata_richness_score(record=record) == 6


def test_deduplicate_prefers_richer_metadata_same_version() -> None:
    """Among same-version duplicates, the one with richer metadata wins.

    Regression test for issue where misfiled results would override
    enriched metadata during deduplication.
    """
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
    # Enriched record should win despite being first in input order
    result = deduped[0]
    additional = result["model_info"]["additional_details"]
    assert additional.get("commercially_licensed") is True
    assert additional.get("open") is True
    assert additional.get("trained_from_scratch") is True


def test_deduplicate_richness_beats_input_order() -> None:
    """Richer metadata wins even when the poorer record appears later."""
    poor = _record(version="17.6.0")
    rich = _record(version="17.6.0")
    rich["model_info"]["additional_details"]["open"] = True

    # Poor record comes last
    deduped = deduplicate_records(records=[rich, poor])

    assert len(deduped) == 1
    assert deduped[0]["model_info"]["additional_details"].get("open") is True


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
