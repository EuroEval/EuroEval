"""Helpers for loading and parsing result records and model identifiers."""

from __future__ import annotations

import re

from .constants import ANCHOR_RE, VARIANT_SUFFIX_RE


def strip_anchor(model_id: str) -> str:
    """Strip any surrounding HTML anchor tag from a model id.

    Unlike :func:`plain_model_id`, this preserves any ``(zero-shot)`` / ``(val)``
    variant suffix — it only unwraps the ``<a href=...>...</a>`` tag.

    Args:
        model_id:
            The (possibly anchored) identifier.

    Returns:
        The identifier with any anchor tag removed.
    """
    match = ANCHOR_RE.search(model_id)
    return match.group("inner").strip() if match else model_id


def plain_model_id(model_id: str) -> str:
    """Strip the HTML anchor and variant-suffix from a result-record model id.

    Records label few-shot vs zero-shot and test vs validation by appending
    ``(zero-shot)`` / ``(val)`` / ``(zero-shot, val)`` to the model id.
    For the core-model list we collapse all those variants down to the
    canonical ``org/repo`` slug — we don't want to list the same model
    several times.

    Preserves ``#no-thinking``, ``#thinking`` and ``@revision`` suffixes
    used for parameterised model IDs — these are meaningful for upload
    filenames and core/per-model grouping.

    Args:
        model_id:
            The (possibly anchored, possibly variant-suffixed) identifier.

    Returns:
        The canonical ``org/repo`` slug with variant suffix removed, but
        ``#param`` and ``@revision`` preserved.
    """
    # Strip HTML anchors and (val)/(zero-shot) variant suffixes only
    return VARIANT_SUFFIX_RE.sub("", strip_anchor(model_id))


def convert_to_float(value: str | float) -> float | str:
    """Convert a value to float if possible.

    Args:
        value:
            The value to convert, can be a string or a float.

    Returns:
        The value converted to float if possible, otherwise the original value.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def get_model_name(record: dict) -> str:
    """Get the model name from an EEE record.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The model name, or ``"unknown"`` if absent.
    """
    return record.get("model_info", {}).get("name", "unknown")


def extract_model_ids_from_record(record: dict) -> list[str]:
    """Extract the model ID candidates from a record.

    Args:
        record:
            The record.

    Returns:
        The model ID candidates.
    """
    # Strip any anchor tag so that records stored with an already-anchored name
    # (``<a ...>org/repo</a>``) and records stored with the plain ``org/repo``
    # name collapse to the same identifier — otherwise the model is split into
    # two leaderboard rows. The anchor is re-applied at render time.
    model_id = strip_anchor(get_model_name(record))

    few_shot = get_bool_field(record, "few_shot", True)
    validation_split = get_bool_field(record, "validation_split", False)

    note = [] if few_shot else ["zero-shot"]
    if validation_split:
        note.append("val")
    if not note:
        return [model_id]

    return [f"{model_id} ({', '.join(note)})"]


def get_dataset(record: dict) -> str | None:
    """Get the dataset from an EEE record.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The dataset name, or None if not found.
    """
    additional = record.get("eval_library", {}).get("additional_details", {})
    if "dataset" in additional:
        return additional["dataset"]
    eval_results = record.get("evaluation_results", [])
    if eval_results and isinstance(eval_results, list):
        source_data = eval_results[0].get("source_data", {})
        if "dataset_name" in source_data:
            return source_data["dataset_name"]
    return None


def get_record_hash(record: dict) -> str:
    """Returns a hash value for a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        A hash value for the record.

    Raises:
        ValueError:
            If no dataset is found in the record.
    """
    # The hash must identify records that render on the same leaderboard row, so
    # that duplicates collapse during deduplication. A row is identified by
    # ``extract_model_ids_from_record`` (the stripped model name plus its
    # zero-shot/val note) and the dataset — nothing else. We therefore hash
    # exactly those components:
    #
    # * strip the anchor tag, so an already-anchored name (``<a ...>org/repo</a>``)
    #   and the plain ``org/repo`` collapse together;
    # * include ``validation_split`` and ``few_shot``, which the row note encodes;
    # * do NOT include ``generative`` — the row label ignores it, so hashing it
    #   would let two records (e.g. one tagged ``generative: true`` and one
    #   missing the flag) survive deduplication yet land on the same row, showing
    #   multiple scores for a single model+benchmark combination.
    model = strip_anchor(get_model_name(record))
    dataset = get_dataset(record)
    if dataset is None:
        raise ValueError(f"No dataset found in record: {record}")
    validation_split = get_bool_field(record, "validation_split", False)
    few_shot = get_bool_field(record, "few_shot", True)
    return f"{model}{dataset}{int(validation_split)}{int(few_shot)}"


def get_bool_field(record: dict, field: str, default: bool) -> bool:
    """Get a boolean field from an EEE record.

    The value lives in ``eval_library.additional_details`` and may be stored as
    a bool or a ``"true"``/``"false"`` string.

    Args:
        record:
            A result record in EEE format.
        field:
            The field name to extract (e.g., "validation_split", "few_shot").
        default:
            Default value if field not found.

    Returns:
        The boolean value, or the default if not found.
    """
    additional = record.get("eval_library", {}).get("additional_details", {})
    if field in additional:
        val = additional[field]
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
    return default


def strip_val_suffix(model_id: str) -> str | None:
    """Return the model ID with the 'val' note removed, or None if absent.

    Args:
        model_id:
            The model ID, possibly wrapped in an anchor tag and possibly
            carrying a parenthesised note like ``(val)`` or ``(zero-shot, val)``.

    Returns:
        The model ID with ``val`` removed from its note, or ``None`` if the
        model ID did not contain a ``val`` note.
    """
    match = re.match(r"^(.*)\s*\(([^()]+)\)(\s*</a>)?$", model_id)
    if not match:
        return None
    prefix, note, suffix = match.group(1), match.group(2), match.group(3) or ""
    items = [item.strip() for item in note.split(",")]
    if "val" not in items:
        return None
    items = [item for item in items if item != "val"]
    if not items:
        return f"{prefix.rstrip()}{suffix}"
    return f"{prefix.rstrip()} ({', '.join(items)}){suffix}"


def drop_val_duplicates(
    model_results: dict[str, dict[str, list[tuple[list[float], float, float]]]],
) -> dict[str, dict[str, list[tuple[list[float], float, float]]]]:
    """Drop validation-split variants when the full test-split variant exists.

    When a model has been evaluated on both the validation and full test split,
    only show the test-split row if it covers at least as many datasets.
    Otherwise keep the validation-split version (which may have more data).

    Args:
        model_results:
            The grouped model results, keyed by model ID.

    Returns:
        The model results with ``(val)``-suffixed entries removed whenever the
        corresponding full test-split entry is also present and covers at least
        as many datasets.
    """
    filtered: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {}
    for model_id, results in model_results.items():
        equivalent = strip_val_suffix(model_id=model_id)
        if equivalent is not None and equivalent in model_results:
            # Only drop the (val) version if the test-split version has >= datasets
            equivalent_count = len(model_results[equivalent])
            if equivalent_count >= len(results):
                continue
        filtered[model_id] = results
    return filtered
