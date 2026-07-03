"""Tests for the `leaderboards.record_fields` module."""

from leaderboards.record_fields import deduplicate_records


def _record(
    name: str = "org/model",
    dataset: str = "angry-tweets",
    *,
    version: str | None = None,
    generative: bool | None = None,
) -> dict[str, object]:
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
    additional: dict[str, object] = {"dataset": dataset}
    if generative is not None:
        additional["generative"] = generative
    library: dict[str, object] = {"additional_details": additional}
    if version is not None:
        library["version"] = version
    return {"model_info": {"name": name}, "eval_library": library}


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
