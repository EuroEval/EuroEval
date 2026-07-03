"""Tests for the `leaderboards.records` module."""

from leaderboards.records import extract_model_ids_from_record, get_record_hash


def _record(name: str, dataset: str = "angry-tweets") -> dict:
    """Build a minimal EEE-style record with the given model name and dataset.

    Args:
        name:
            The model name to store in ``model_info.name``.
        dataset:
            The dataset name to store in the record.

    Returns:
        A minimal EEE-style record.
    """
    return {
        "model_info": {"name": name},
        "eval_library": {"additional_details": {"dataset": dataset}},
    }


def test_anchored_and_plain_names_hash_identically() -> None:
    """Anchored and plain model names must deduplicate to the same hash.

    Otherwise both records survive deduplication yet collapse onto a single
    leaderboard row (``extract_model_ids_from_record`` strips the anchor),
    showing multiple scores for one model+benchmark combination (issue #1970).
    """
    anchored = _record(
        "<a href='https://ollama.com/library/gemma3'>ollama_chat/gemma3</a>"
    )
    plain = _record("ollama_chat/gemma3")

    assert get_record_hash(record=anchored) == get_record_hash(record=plain)
    # The two forms must also collapse to the same leaderboard row identity.
    assert extract_model_ids_from_record(record=anchored) == (
        extract_model_ids_from_record(record=plain)
    )


def test_distinct_datasets_hash_differently() -> None:
    """Records for different datasets must not deduplicate together."""
    assert get_record_hash(
        record=_record("org/model", dataset="angry-tweets")
    ) != get_record_hash(record=_record("org/model", dataset="dansk"))
