"""Helpers for parsing result records and model identifiers."""

from __future__ import annotations

import re

from .constants import ANCHOR_RE, VARIANT_SUFFIX_RE


def plain_model_id(model_id: str) -> str:
    """Strip the HTML anchor and variant-suffix from a result-record model id.

    Records label few-shot vs zero-shot and test vs validation by appending
    ``(zero-shot)`` / ``(val)`` / ``(zero-shot, val)`` to the model id.
    For the core-model list we collapse all those variants down to the
    canonical ``org/repo`` slug — we don't want to list the same model
    several times.

    Args:
        model_id:
            The (possibly anchored, possibly variant-suffixed) identifier.

    Returns:
        The canonical ``org/repo`` slug.
    """
    match = ANCHOR_RE.search(model_id)
    inner = match.group("inner").strip() if match else model_id
    return VARIANT_SUFFIX_RE.sub("", inner)


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
    """Get model name from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The model name.
    """
    if "model_info" in record and "name" in record["model_info"]:
        return record["model_info"]["name"]
    return record.get("model", "unknown")


def extract_model_ids_from_record(record: dict) -> list[str]:
    """Extract the model ID candidates from a record.

    Args:
        record:
            The record.

    Returns:
        The model ID candidates.
    """
    model_id = get_model_name(record)

    # _get_bool_field normalises both the EEE and old record formats.
    few_shot = _get_bool_field(record, "few_shot", True)
    validation_split = _get_bool_field(record, "validation_split", False)

    base_note = [] if few_shot else ["zero-shot"]
    note = base_note + ["val"] if validation_split else base_note
    all_model_notes: list[list[str]] = [note]

    has_anchor = model_id.endswith("</a>")
    base = re.sub(r"</a>$", "", model_id) if has_anchor else model_id
    suffix = "</a>" if has_anchor else ""

    model_id_candidates = [
        f"{base} ({', '.join(note)}){suffix}" if note != [] else model_id
        for note in all_model_notes
    ]
    return model_id_candidates


def get_dataset(record: dict) -> str | None:
    """Get dataset from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.

    Returns:
        The dataset name, or None if not found.
    """
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        if "dataset" in additional:
            return additional["dataset"]
        eval_results = record.get("evaluation_results", [])
        if eval_results and isinstance(eval_results, list):
            source_data = eval_results[0].get("source_data", {})
            if "dataset_name" in source_data:
                return source_data["dataset_name"]
    return record.get("dataset")


def _get_bool_field(record: dict, field: str, default: bool) -> bool:
    """Get a boolean field from record, supporting both EEE and old formats.

    Args:
        record:
            A result record in either EEE or old EuroEval format.
        field:
            The field name to extract (e.g., "validation_split", "few_shot").
        default:
            Default value if field not found.

    Returns:
        The boolean value, or the default if not found.
    """
    # Check EEE format first: nested in eval_library.additional_details
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        if field in additional:
            val = additional[field]
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() == "true"
    # Old format: top-level
    if field in record:
        val = record[field]
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
    return default


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
    model = get_model_name(record)
    dataset = get_dataset(record)
    if dataset is None:
        raise ValueError(f"No dataset found in record: {record}")
    validation_split = _get_bool_field(record, "validation_split", False)
    few_shot = _get_bool_field(record, "few_shot", True)
    # Check EEE format for generative
    if "eval_library" in record:
        additional = record.get("eval_library", {}).get("additional_details", {})
        generative_val = additional.get("generative", False)
        if isinstance(generative_val, str):
            generative_val = generative_val.lower() == "true"
        generative = int(generative_val)
    else:
        generative = int(record.get("generative", False))
    return f"{model}{dataset}{int(validation_split)}{generative * (int(few_shot) + 1)}"


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
