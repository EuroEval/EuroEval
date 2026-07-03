"""Tests for the `leaderboards.records` module."""

from leaderboards.records import extract_model_ids_from_record, get_record_hash


def _record(
    name: str,
    dataset: str = "angry-tweets",
    *,
    generative: bool | None = None,
    few_shot: bool = True,
    validation_split: bool = False,
) -> dict[str, object]:
    """Build a minimal EEE-style record.

    Args:
        name:
            The model name to store in ``model_info.name``.
        dataset:
            The dataset name to store in the record.
        generative:
            Value for the ``generative`` flag, or None to omit it entirely.
        few_shot:
            Value for the ``few_shot`` flag.
        validation_split:
            Value for the ``validation_split`` flag.

    Returns:
        A minimal EEE-style record.
    """
    additional: dict[str, object] = {
        "dataset": dataset,
        "few_shot": few_shot,
        "validation_split": validation_split,
    }
    if generative is not None:
        additional["generative"] = generative
    return {
        "model_info": {"name": name},
        "eval_library": {"additional_details": additional},
    }


def test_anchored_and_plain_names_hash_identically() -> None:
    """Anchored and plain model names must deduplicate to the same hash.

    Otherwise both records survive deduplication yet collapse onto a single
    leaderboard row (``extract_model_ids_from_record`` strips the anchor),
    showing multiple scores for one model+benchmark combination (issue #1970).
    """
    anchored = _record(
        name="<a href='https://ollama.com/library/gemma3'>ollama_chat/gemma3</a>"
    )
    plain = _record(name="ollama_chat/gemma3")

    assert get_record_hash(record=anchored) == get_record_hash(record=plain)
    # The two forms must also collapse to the same leaderboard row identity.
    assert extract_model_ids_from_record(record=anchored) == (
        extract_model_ids_from_record(record=plain)
    )


def test_distinct_datasets_hash_differently() -> None:
    """Records for different datasets must not deduplicate together."""
    assert get_record_hash(
        record=_record(name="org/model", dataset="angry-tweets")
    ) != get_record_hash(record=_record(name="org/model", dataset="dansk"))


def test_generative_flag_does_not_split_hash() -> None:
    """A differing (or missing) ``generative`` flag must not defeat dedup.

    Two records for the same model+dataset+split that differ only in the
    ``generative`` flag render on the same leaderboard row, so they must hash
    identically — otherwise both survive deduplication and the row shows the
    metric twice (issue #1970, Apertus v1.1).
    """
    with_flag = _record(name="org/model", generative=True)
    without_flag = _record(name="org/model", generative=None)

    assert get_record_hash(record=with_flag) == get_record_hash(record=without_flag)
    assert extract_model_ids_from_record(record=with_flag) == (
        extract_model_ids_from_record(record=without_flag)
    )


def test_few_shot_and_validation_split_change_hash() -> None:
    """Zero-shot and validation-split variants must remain distinct.

    These map to distinct leaderboard rows (``(zero-shot)`` / ``(val)`` notes),
    so their hashes must differ to avoid collapsing genuinely separate rows.
    """
    base = _record(name="org/model", few_shot=True, validation_split=False)
    zero_shot = _record(name="org/model", few_shot=False, validation_split=False)
    val = _record(name="org/model", few_shot=True, validation_split=True)

    hashes = {
        get_record_hash(record=base),
        get_record_hash(record=zero_shot),
        get_record_hash(record=val),
    }
    assert len(hashes) == 3
