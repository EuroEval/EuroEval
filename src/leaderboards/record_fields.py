"""Accessors for fields on EEE result records."""

from __future__ import annotations

import json
import re

from .records import get_bool_field


def get_task(record: dict) -> str | None:
    """Get the task name from an EEE record.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The task name, or None if not found.
    """
    return record.get("eval_library", {}).get("additional_details", {}).get("task")


def get_raw_results(record: dict) -> list[dict] | None:
    """Get the raw per-iteration results from an EEE record.

    The raw results live in ``eval_library.additional_details.raw_results`` as a
    JSON string (or, occasionally, an already-parsed list).

    Args:
        record:
            A result record in EEE format.

    Returns:
        The raw results as a list of dicts, or None if absent or unparseable.
    """
    additional = record.get("eval_library", {}).get("additional_details", {})
    raw = additional.get("raw_results")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, list):
        return raw
    return None


def get_total_scores(record: dict) -> dict[str, float] | None:
    """Get the aggregated total scores from an EEE record.

    Scores are aggregated from ``evaluation_results``, keyed by their
    ``evaluation_name`` (e.g. ``"test_mcc"``).

    Args:
        record:
            A result record in EEE format.

    Returns:
        Dict mapping score names to values, or None if not found.
    """
    scores: dict[str, float] = {}
    eval_results = record.get("evaluation_results", [])
    if isinstance(eval_results, list):
        for er in eval_results:
            if isinstance(er, dict):
                name = er.get("evaluation_name", "")
                score_details = er.get("score_details", {})
                if isinstance(score_details, dict):
                    score = score_details.get("score")
                    if score is not None and isinstance(name, str) and name:
                        scores[name] = float(score)
    return scores if scores else None


def get_num_failed_instances(record: dict) -> int | None:
    """Get the number of failed instances from an EEE record.

    The count is stored per evaluation result at
    ``score_details.details.num_failed_instances`` and is identical across the
    primary/secondary metric entries of a dataset, so the first parseable value
    is returned.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The number of failed instances, or None if not found or unparseable.
    """
    eval_results = record.get("evaluation_results", [])
    if not isinstance(eval_results, list):
        return None
    for er in eval_results:
        if not isinstance(er, dict):
            continue
        score_details = er.get("score_details", {})
        if not isinstance(score_details, dict):
            continue
        details = score_details.get("details", {})
        if not isinstance(details, dict) or "num_failed_instances" not in details:
            continue
        try:
            count = int(float(details["num_failed_instances"]))
        except (TypeError, ValueError):
            continue
        if count < 0:
            continue
        return count
    return None


def get_version(record: dict) -> str | None:
    """Get the EuroEval version from an EEE record.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The ``eval_library.version`` string with any ``.dev`` suffix stripped,
        or None if not found.
    """
    version = record.get("eval_library", {}).get("version")
    if version:
        return re.sub(r"\.dev\d+", "", version)
    return None


def get_few_shot(record: dict) -> bool:
    """Get the few-shot flag from an EEE record.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The few_shot boolean value, defaulting to True.
    """
    return get_bool_field(record, "few_shot", True)


def get_validation_split(record: dict) -> bool:
    """Get the validation-split flag from an EEE record.

    Args:
        record:
            A result record in EEE format.

    Returns:
        The validation_split boolean value, defaulting to False.
    """
    return get_bool_field(record, "validation_split", False)
