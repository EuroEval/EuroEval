"""Utility functions for the project."""

import logging
import re
import warnings
from functools import cache

from scipy import stats

logger = logging.getLogger(__name__)


def convert_to_float(value: str | float) -> float | str:
    """Convert a value to float if possible.

    Args:
        value:
            The value to convert, can be a string or a float.

    Returns:
        The value converted to float if possible, otherwise returns the original value.
    """
    try:
        return float(value)
    except Exception:
        return value


def significantly_better(
    score_values_1: list[float], score_values_2: list[float]
) -> float:
    """Compute one-tailed t-statistic for the difference between two sets of scores.

    Args:
        score_values_1:
            The first set of scores.
        score_values_2:
            The second set of scores.

    Returns:
        The t-statistic of the difference between the two sets of scores, where
        a positive t-statistic indicates that the first set of scores is
        statistically better than the second set of scores.
    """
    assert len(score_values_1) == len(score_values_2), (
        f"Length of score values must be equal, but got {len(score_values_1)} and "
        f"{len(score_values_2)}."
    )
    if score_values_1 == score_values_2:
        return 0
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        test_result = stats.ttest_ind(
            a=score_values_1, b=score_values_2, alternative="greater", equal_var=False
        )
    return test_result.pvalue < 0.05  # type: ignore[attr-defined]


def extract_model_ids_from_record(record: dict) -> list[str]:
    """Extract the model ID candidates from a record.

    Args:
        record:
            The record.

    Returns:
        The model ID candidates.
    """
    model_id = record["model"]
    all_model_notes: list[list[str]] = [[]]

    match record.get("few_shot", True):
        case False:
            all_model_notes = [note + ["zero-shot"] for note in all_model_notes]
        case None:
            all_model_notes += [note + ["zero-shot"] for note in all_model_notes]

    match record.get("validation_split", False):
        case True:
            all_model_notes = [note + ["val"] for note in all_model_notes]
        case None:
            all_model_notes += [note + ["val"] for note in all_model_notes]

    model_id_candidates = [
        f"{re.sub(r'</a>$', '', model_id)} ({', '.join(note)})</a>"
        if note != []
        else model_id
        for note in all_model_notes
    ]
    return model_id_candidates


def get_record_hash(record: dict) -> str:
    """Returns a hash value for a record.

    Args:
        record:
            A record from the JSONL file.

    Returns:
        A hash value for the record.
    """
    model = record["model"]
    dataset = record["dataset"]
    validation_split = (
        int(record_validation_split)
        if (record_validation_split := record.get("validation_split", False))
        is not None
        else 0
    )
    few_shot = (
        int(record_few_shot)
        if (record_few_shot := record.get("few_shot", True)) is not None
        else 1
    )
    generative = int(record.get("generative", False))
    return f"{model}{dataset}{validation_split}{generative * (few_shot + 1)}"


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

    Args:
        model_results:
            The grouped model results, keyed by model ID.

    Returns:
        The model results with ``(val)``-suffixed entries removed whenever the
        corresponding full test-split entry is also present.
    """
    filtered: dict[str, dict[str, list[tuple[list[float], float, float]]]] = {}
    for model_id, results in model_results.items():
        equivalent = strip_val_suffix(model_id=model_id)
        if equivalent is not None and equivalent in model_results:
            continue
        filtered[model_id] = results
    return filtered


@cache
def log_once(message: str, logging_level: int) -> None:
    """Log a message only once.

    Args:
        message:
            The message to log.
        logging_level:
            The logging level to use for the message.

    Raises:
        ValueError:
            If the logging level is invalid.
    """
    match logging_level:
        case logging.DEBUG:
            logger.debug(message)
        case logging.INFO:
            logger.info(message)
        case logging.WARNING:
            logger.warning(message)
        case logging.ERROR:
            logger.error(message)
        case logging.CRITICAL:
            logger.critical(message)
        case _:
            raise ValueError(f"Invalid logging level: {logging_level}")
